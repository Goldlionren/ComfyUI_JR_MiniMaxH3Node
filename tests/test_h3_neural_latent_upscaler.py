from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_neural_latent_upscaler import (
    JR_MiniMaxH3NeuralLatentUpscaler,
)
from ComfyUI_JR_MiniMaxH3Node.utils.h3_av_latent_builder import build_h3_av_latent
from ComfyUI_JR_MiniMaxH3Node.utils.h3_neural_latent_upscaler import (
    ERROR_PREFIX,
    H3NeuralLatentUpscalerError,
    H3SpatialContract,
    _CachedModel,
    _H3CheckpointNetwork,
    _run_checkpoint_backend,
    _run_temporally_chunked,
    _select_checkpoint,
    build_network_from_state_dict,
    get_h3_spatial_contract,
    plan_h3_latent_upscale,
    upscale_h3_video_latent,
)

CONTRACT = H3SpatialContract(
    vae_compression=16,
    latent_alignment_h=2,
    latent_alignment_w=2,
    pixel_alignment_h=32,
    pixel_alignment_w=32,
)


def _video(*, batch=1, channels=24, temporal=37, height=48, width=64, dtype=torch.float32):
    return torch.zeros((batch, channels, temporal, height, width), dtype=dtype)


def _audio(*, batch=1, temporal=207, dtype=torch.float32):
    return torch.zeros((batch, 32, 2, temporal), dtype=dtype)


def _runner(samples, plan):
    return torch.zeros(
        (samples.shape[0], samples.shape[1], samples.shape[2], plan.output_h, plan.output_w),
        dtype=samples.dtype,
        device=samples.device,
    )


def _assert_error(latent, expected, *, mode="scale", scale=1.5, target=2.0):
    with pytest.raises(H3NeuralLatentUpscalerError) as exc:
        upscale_h3_video_latent(
            latent,
            mode,
            scale,
            target,
            neural_runner=_runner,
            contract=CONTRACT,
        )
    assert str(exc.value).startswith(ERROR_PREFIX)
    assert expected in str(exc.value)


def test_node_schema_is_plain_video_latent_and_size_only_controls():
    node = JR_MiniMaxH3NeuralLatentUpscaler()
    schema = node.INPUT_TYPES()
    assert list(schema["required"]) == ["video_latent", "resize_mode", "scale", "target_megapixels"]
    assert schema["required"]["video_latent"] == ("LATENT",)
    assert schema["required"]["resize_mode"][0] == ["scale", "megapixels"]
    assert schema["required"]["scale"][1]["default"] == 1.5
    assert schema["required"]["target_megapixels"][1]["default"] == 2.0
    assert node.RETURN_TYPES == ("LATENT", "STRING")
    assert node.RETURN_NAMES == ("video_latent", "status")
    assert node.CATEGORY == "JR MiniMax H3/Latent"


def test_native_spatial_contract_is_discovered_from_current_comfyui():
    contract = get_h3_spatial_contract()
    assert contract.vae_compression == 16
    assert contract.latent_alignment_h == 2
    assert contract.latent_alignment_w == 2
    assert contract.pixel_alignment_h == 32
    assert contract.pixel_alignment_w == 32


@pytest.mark.parametrize(
    ("scale", "expected_h", "expected_w"),
    [(1.0, 48, 64), (1.25, 60, 80), (1.5, 72, 96), (2.0, 96, 128)],
)
def test_scale_mode_uses_linear_scale_and_legal_alignment(scale, expected_h, expected_w):
    plan = plan_h3_latent_upscale(
        input_h=48,
        input_w=64,
        resize_mode="scale",
        scale=scale,
        target_megapixels=2.0,
        contract=CONTRACT,
    )
    assert (plan.output_h, plan.output_w) == (expected_h, expected_w)
    assert plan.output_h % 2 == 0
    assert plan.output_w % 2 == 0
    assert plan.output_pixel_h % 32 == 0
    assert plan.output_pixel_w % 32 == 0


@pytest.mark.parametrize("target_mp", [0.9, 1.4, 2.0])
def test_megapixel_mode_is_pixel_space_and_preserves_aspect(target_mp):
    plan = plan_h3_latent_upscale(
        input_h=48,
        input_w=64,
        resize_mode="megapixels",
        scale=1.5,
        target_megapixels=target_mp,
        contract=CONTRACT,
    )
    assert abs(plan.actual_megapixels - target_mp) < 0.05
    assert abs((plan.output_w / plan.output_h) / (64 / 48) - 1.0) < 0.04
    assert plan.output_pixel_h == plan.output_h * 16
    assert plan.output_pixel_w == plan.output_w * 16
    assert plan.output_h % 2 == 0
    assert plan.output_w % 2 == 0


def test_output_preserves_bct_metadata_dtype_device_and_container_immutability():
    samples = _video(dtype=torch.float16)
    metadata = {"batch_index": [4], "custom": {"keep": True}}
    latent = {"samples": samples, **metadata}
    output, status = upscale_h3_video_latent(
        latent,
        "scale",
        1.5,
        2.0,
        neural_runner=_runner,
        contract=CONTRACT,
    )
    out = output["samples"]
    assert output is not latent
    assert latent["samples"] is samples
    assert output["batch_index"] is metadata["batch_index"]
    assert output["custom"] is metadata["custom"]
    assert out.shape == (1, 24, 37, 72, 96)
    assert out.shape[:3] == samples.shape[:3]
    assert out.dtype == samples.dtype
    assert out.device == samples.device
    assert "temporal: 37 -> 37" in status
    assert "output pixels: 1536x1152" in status


def test_scale_one_returns_new_container_without_loading_checkpoint():
    latent = {"samples": _video(), "custom": "preserved"}
    output, status = upscale_h3_video_latent(
        latent,
        "scale",
        1.0,
        2.0,
        neural_runner=None,
        contract=CONTRACT,
    )
    assert output is not latent
    assert output["samples"] is latent["samples"]
    assert output["custom"] == "preserved"
    assert "checkpoint not loaded" in status


@pytest.mark.parametrize(
    ("latent", "expected"),
    [
        ({}, "LATENT dictionary"),
        ({"samples": torch.zeros((1, 24, 48, 64))}, "[B,24,T,H,W]"),
        ({"samples": torch.zeros((1, 16, 37, 48, 64))}, "[B,24,T,H,W]"),
        ({"samples": torch.zeros((1, 32, 37, 48, 64))}, "[B,24,T,H,W]"),
        ({"samples": torch.zeros((1, 24, 1, 37, 48, 64))}, "[B,24,T,H,W]"),
        ({"samples": torch.zeros((1, 24, 37, 48, 64), dtype=torch.float64)}, "dtype must be"),
    ],
)
def test_rank_channel_container_and_dtype_validation(latent, expected):
    _assert_error(latent, expected)


def test_full_av_nested_tensor_is_rejected_with_splitter_instruction():
    av_latent, _status = build_h3_av_latent({"samples": _video()}, {"samples": _audio()})
    assert isinstance(av_latent["samples"], NestedTensor)
    _assert_error(av_latent, "Use JR MiniMax H3 Split AV Latent")


def test_input_alignment_and_resize_bounds_fail_fast():
    _assert_error({"samples": _video(height=47)}, "not aligned")
    _assert_error({"samples": _video()}, "scale must be finite", scale=0.5)
    _assert_error({"samples": _video()}, "would downscale", mode="megapixels", target=0.1)
    _assert_error({"samples": _video()}, "supports at most", mode="megapixels", target=20.0)


@pytest.mark.parametrize(("stream", "value"), [("input", float("nan")), ("input", float("inf"))])
def test_non_finite_input_is_rejected(stream, value):
    samples = _video()
    samples[0, 0, 0, 0, 0] = value
    _assert_error({"samples": samples}, "Input video latent contains NaN or Inf")


def test_non_finite_and_wrong_shape_backend_outputs_are_rejected():
    def wrong_shape(samples, plan):
        return torch.zeros((1, 24, samples.shape[2] + 1, plan.output_h, plan.output_w))

    with pytest.raises(H3NeuralLatentUpscalerError, match="returned shape"):
        upscale_h3_video_latent(
            {"samples": _video()},
            "scale",
            1.5,
            2.0,
            neural_runner=wrong_shape,
            contract=CONTRACT,
        )

    def non_finite(samples, plan):
        output = _runner(samples, plan)
        output[0, 0, 0, 0, 0] = float("nan")
        return output

    with pytest.raises(H3NeuralLatentUpscalerError, match="produced NaN or Inf"):
        upscale_h3_video_latent(
            {"samples": _video()},
            "scale",
            1.5,
            2.0,
            neural_runner=non_finite,
            contract=CONTRACT,
        )


def test_upscaled_video_rejoins_unchanged_audio_in_existing_builder():
    video = _video()
    audio = _audio()
    upscaled, _status = upscale_h3_video_latent(
        {"samples": video},
        "scale",
        1.5,
        2.0,
        neural_runner=_runner,
        contract=CONTRACT,
    )
    av_latent, builder_status = build_h3_av_latent(upscaled, {"samples": audio})
    joined_video, joined_audio = av_latent["samples"].unbind()
    assert joined_video is upscaled["samples"]
    assert joined_audio is audio
    assert joined_video.shape == (1, 24, 37, 72, 96)
    assert "H3 AV latent: valid" in builder_status


def _tiny_network():
    return _H3CheckpointNetwork(
        in_channels=24,
        channels=32,
        embedding_channels=16,
        input_layout=(("residual", 0), ("temporal", 3)),
        output_layout=(("residual", 0), ("temporal", 3)),
    ).eval()


def test_synthetic_checkpoint_rebuild_and_neural_forward():
    original = _tiny_network()
    rebuilt = build_network_from_state_dict(original.state_dict())
    latent = torch.randn((1, 24, 3, 4, 6))
    with torch.inference_mode():
        expected = original(latent, 1.5, 6, 10)
        actual = rebuilt(latent, 1.5, 6, 10)
    assert actual.shape == (1, 24, 3, 6, 10)
    assert torch.equal(actual, expected)
    assert actual.shape[:3] == latent.shape[:3]


def test_temporal_chunking_keeps_exact_bct_and_target_spatial_shape():
    model = _tiny_network()
    latent = torch.randn((1, 24, 29, 4, 6))
    plan = plan_h3_latent_upscale(
        input_h=4,
        input_w=6,
        resize_mode="scale",
        scale=1.5,
        target_megapixels=2.0,
        contract=CONTRACT,
    )
    with torch.inference_mode():
        output = _run_temporally_chunked(model, latent, plan, temporal_context=3)
    assert output.shape == (1, 24, 29, plan.output_h, plan.output_w)


def test_incompatible_checkpoint_is_rejected():
    state = _tiny_network().state_dict()
    state.pop("conv_in.weight")
    with pytest.raises(H3NeuralLatentUpscalerError, match="missing: conv_in.weight"):
        build_network_from_state_dict(state)


def test_missing_checkpoint_error_names_comfy_model_folder(monkeypatch):
    import ComfyUI_JR_MiniMaxH3Node.utils.h3_neural_latent_upscaler as module
    import folder_paths

    monkeypatch.setattr(module, "_candidate_checkpoint_names", lambda: [])
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda _name: ["X:/ComfyUI/models/latent_upscale_models"])
    with pytest.raises(H3NeuralLatentUpscalerError) as exc:
        _select_checkpoint(torch.float16)
    expected_folder = str(Path("X:/ComfyUI/models/latent_upscale_models"))
    assert expected_folder in str(exc.value)
    assert "Automatic download and interpolation fallback are intentionally disabled" in str(exc.value)


def test_checkpoint_backend_uses_model_specific_comfy_offload(monkeypatch):
    import comfy.model_management as model_management
    import ComfyUI_JR_MiniMaxH3Node.utils.h3_neural_latent_upscaler as module

    patcher = types.SimpleNamespace(load_device=torch.device("cpu"))
    cached = _CachedModel(
        path="synthetic.safetensors",
        patcher=patcher,
        model=object(),
        dtype=torch.float32,
        temporal_context=1,
    )
    calls = []
    monkeypatch.setattr(module, "_select_checkpoint", lambda _dtype: ("synthetic.safetensors", "synthetic"))
    monkeypatch.setattr(module, "_load_cached_model", lambda _path: cached)
    monkeypatch.setattr(module, "_normalization_tensors", lambda device, dtype: (torch.zeros((1, 24, 1, 1, 1)), torch.ones((1, 24, 1, 1, 1))))
    monkeypatch.setattr(module, "_run_temporally_chunked", lambda model, latent, plan, context: latent)
    monkeypatch.setattr(model_management, "load_models_gpu", lambda models, force_full_load: calls.append(("load", models, force_full_load)))
    monkeypatch.setattr(
        model_management,
        "unload_model_and_clones",
        lambda model, unload_additional_models, all_devices: calls.append(
            ("unload", model, unload_additional_models, all_devices)
        ),
    )
    samples = _video(height=48, width=64)
    plan = plan_h3_latent_upscale(
        input_h=48,
        input_w=64,
        resize_mode="scale",
        scale=1.5,
        target_megapixels=2.0,
        contract=CONTRACT,
    )
    output, name = _run_checkpoint_backend(samples, plan)
    assert torch.equal(output, samples)
    assert output.shape == samples.shape
    assert name == "synthetic.safetensors"
    assert calls[0] == ("load", [patcher], True)
    assert calls[-1] == ("unload", patcher, False, False)
