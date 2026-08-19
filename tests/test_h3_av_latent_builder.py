from __future__ import annotations

import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_av_latent_builder import JR_MiniMaxH3AVLatentBuilder
from ComfyUI_JR_MiniMaxH3Node.utils.h3_av_latent_builder import (
    ERROR_PREFIX,
    H3AVLatentBuilderError,
    build_h3_av_latent,
)


def _video(*, batch=1, channels=24, temporal=37, height=2, width=3, dtype=torch.float32, device="cpu"):
    return torch.zeros((batch, channels, temporal, height, width), dtype=dtype, device=device)


def _audio(*, batch=1, channels=32, streams=2, temporal=207, dtype=torch.float32, device="cpu"):
    return torch.zeros((batch, channels, streams, temporal), dtype=dtype, device=device)


def _assert_builder_error(video_latent, audio_latent, expected):
    with pytest.raises(H3AVLatentBuilderError) as exc:
        build_h3_av_latent(video_latent, audio_latent)
    message = str(exc.value)
    assert message.startswith(ERROR_PREFIX)
    assert expected in message


def test_schema_and_successful_official_nested_tensor():
    from comfy.nested_tensor import NestedTensor

    node = JR_MiniMaxH3AVLatentBuilder()
    schema = node.INPUT_TYPES()
    assert schema == {"required": {"video_latent": ("LATENT",), "audio_latent": ("LATENT",)}}
    assert node.RETURN_TYPES == ("LATENT", "STRING")
    assert node.RETURN_NAMES == ("latent", "status")

    video = _video()
    audio = _audio()
    latent, status = node.build({"samples": video}, {"samples": audio})

    assert isinstance(latent, dict)
    assert isinstance(latent["samples"], NestedTensor)
    streams = latent["samples"].unbind()
    assert len(streams) == 2
    assert streams[0] is video
    assert streams[1] is audio
    assert "video: [1,24,37,2,3]" in status
    assert "audio: [1,32,2,207]" in status
    assert "124 frames @ 24 fps" in status
    assert "temporal check: passed" in status
    assert status.endswith("H3 AV latent: valid")


@pytest.mark.parametrize(
    ("latent", "expected"),
    [
        ({}, "Missing required 'samples'"),
        ({"samples": "not-a-tensor"}, "must be a torch.Tensor"),
        ({"samples": torch.zeros((1, 24, 2, 2))}, "shape [B,24,T,H,W]"),
        ({"samples": _video(channels=16)}, "shape [B,24,T,H,W]"),
        ({"samples": _video(batch=0)}, "Batch size must be greater than zero"),
        ({"samples": _video(temporal=0)}, "Temporal dimension T must be greater than zero"),
        ({"samples": _video(height=0)}, "spatial dimensions H and W must be greater than zero"),
        ({"samples": _video(width=0)}, "spatial dimensions H and W must be greater than zero"),
        ({"samples": torch.zeros((1, 24, 37, 2, 3), dtype=torch.int32)}, "floating-point tensor"),
    ],
)
def test_video_latent_validation(latent, expected):
    _assert_builder_error(latent, {"samples": _audio()}, expected)


@pytest.mark.parametrize(
    ("latent", "expected"),
    [
        ({}, "Missing required 'samples'"),
        ({"samples": object()}, "must be a torch.Tensor"),
        ({"samples": torch.zeros((1, 32, 207))}, "shape [B,32,2,T]"),
        ({"samples": _audio(channels=24)}, "shape [B,32,2,T]"),
        ({"samples": _audio(streams=1)}, "shape [B,32,2,T]"),
        ({"samples": _audio(batch=0)}, "Batch size must be greater than zero"),
        ({"samples": _audio(temporal=0)}, "Temporal dimension T must be greater than zero"),
        ({"samples": torch.zeros((1, 32, 2, 207), dtype=torch.int32)}, "floating-point tensor"),
    ],
)
def test_audio_latent_validation(latent, expected):
    _assert_builder_error({"samples": _video()}, latent, expected)


@pytest.mark.parametrize(("stream", "bad_value"), [("video", float("nan")), ("video", float("inf")), ("audio", float("nan")), ("audio", float("-inf"))])
def test_non_finite_values_are_rejected(stream, bad_value):
    video = _video()
    audio = _audio()
    if stream == "video":
        video[0, 0, 0, 0, 0] = bad_value
    else:
        audio[0, 0, 0, 0] = bad_value
    _assert_builder_error({"samples": video}, {"samples": audio}, "contains NaN or Inf")


def test_batch_dtype_and_temporal_mismatches_are_rejected():
    _assert_builder_error(
        {"samples": _video(batch=2)},
        {"samples": _audio(batch=1)},
        "video/audio batch mismatch",
    )
    _assert_builder_error(
        {"samples": _video(dtype=torch.float32)},
        {"samples": _audio(dtype=torch.float16)},
        "video/audio dtype mismatch",
    )
    _assert_builder_error(
        {"samples": _video()},
        {"samples": _audio(temporal=400)},
        "video/audio temporal mismatch",
    )
    _assert_builder_error(
        {"samples": _video(temporal=36)},
        {"samples": _audio()},
        "invalid H3 temporal grid",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to construct a real device mismatch")
def test_device_mismatch_is_rejected():
    _assert_builder_error(
        {"samples": _video(device="cpu")},
        {"samples": _audio(device="cuda")},
        "video/audio device mismatch",
    )


@pytest.mark.parametrize("audio_t", [206, 207, 208])
def test_one_audio_tick_boundary_tolerance(audio_t):
    latent, status = build_h3_av_latent({"samples": _video()}, {"samples": _audio(temporal=audio_t)})
    assert latent["samples"].unbind()[1].shape[-1] == audio_t
    assert f"audio delta={audio_t - 207:+d}" in status


def test_non_latent_inputs_and_meta_storage_are_rejected():
    _assert_builder_error([], {"samples": _audio()}, "Expected a LATENT mapping")
    meta_video = _video(device="meta")
    _assert_builder_error({"samples": meta_video}, {"samples": _audio()}, "Meta tensors")
