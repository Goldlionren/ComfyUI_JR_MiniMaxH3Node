from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_temporal_chunk_sampler import JR_H3_TemporalChunkSampler
from ComfyUI_JR_MiniMaxH3Node.utils.h3_temporal_chunk_sampler import (
    H3TemporalChunkSamplerError,
    _apply_native_previous_frame_guide,
    _decode_terminal_frame,
    sample_h3_temporal_chunks,
)


def _latent(*, video_t=22, audio_t=122):
    video = torch.arange(24 * video_t, dtype=torch.float32).reshape(1, 24, video_t, 1, 1)
    audio = torch.arange(32 * 2 * audio_t, dtype=torch.float32).reshape(1, 32, 2, audio_t)
    return {"samples": NestedTensor((video, audio)), "custom": "preserved"}


def _positive(*, keyframes=None):
    metadata = {"prompt": "original"}
    if keyframes is not None:
        metadata["minimax_keyframes"] = keyframes
    return [[torch.ones((1, 1, 1)), metadata]]


def _empty_noise():
    from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise

    return Noise_EmptyNoise()


def _run(*, positive=None, chunk_seconds=1.0, sampled=None):
    state = {"built": [], "guided": [], "decoded": []}

    def build_guider(model, chunk_positive):
        guider = SimpleNamespace(serial=len(state["built"]), model=model, positive=chunk_positive)
        state["built"].append(guider)
        return guider

    def apply_guide(*, positive, latent, vae, image):
        metadata = dict(positive[0][1])
        metadata["minimax_keyframes"] = [
            {"resolved_frame_index": 0, "image_value": float(image.mean().item())}
        ]
        guided = [[positive[0][0], metadata]]
        state["guided"].append((latent, vae, image.clone(), guided))
        return guided

    def decode_last_frame(vae, video):
        value = float(video[:, :, -1].mean().item())
        frame = torch.full((1, 2, 2, 3), value, dtype=torch.float32)
        state["decoded"].append((vae, video.shape, frame.clone()))
        return frame

    def default_sampled(**kwargs):
        video, audio = kwargs["latent_image"]["samples"].unbind()
        serial = kwargs["guider"].serial
        return {"samples": NestedTensor((video.clone() + serial, audio.clone() + serial))}

    output, status = sample_h3_temporal_chunks(
        model="model",
        positive=positive if positive is not None else _positive(),
        vae="vae",
        noise=_empty_noise(),
        sampler="sampler",
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=_latent(),
        chunk_duration_seconds=chunk_seconds,
        sample_chunk=sampled or default_sampled,
        build_guider=build_guider,
        apply_guide=apply_guide,
        decode_last_frame=decode_last_frame,
    )
    return output, status, state


def test_previous_last_frame_flow_rebuilds_guider_per_chunk_without_mutating_positive():
    original_positive = _positive()
    snapshot = copy.deepcopy(original_positive)

    output, status, state = _run(positive=original_positive)
    output_video, output_audio = output["samples"].unbind()
    source_video, source_audio = _latent()["samples"].unbind()

    assert len(state["built"]) == 4
    assert len({id(item) for item in state["built"]}) == 4
    assert state["built"][0].positive is original_positive
    assert "minimax_keyframes" not in state["built"][0].positive[0][1]
    assert all(
        item.positive[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 0
        for item in state["built"][1:]
    )
    assert len(state["decoded"]) == 3
    assert len(state["guided"]) == 3
    assert torch.equal(output_video[:, :, :5], source_video[:, :, :5])
    assert torch.equal(output_video[:, :, 5:10], source_video[:, :, 5:10] + 1)
    assert torch.equal(output_audio[:, :, :, -1], source_audio[:, :, :, -1] + 3)
    assert torch.equal(original_positive[0][0], snapshot[0][0])
    assert original_positive[0][1] == snapshot[0][1]
    assert "MiniMaxH3AddGuide(frame_idx=0)" in status
    assert "guides_applied=3" in status
    assert "rebuilt per chunk" in status
    assert output["custom"] == "preserved"


def test_guide_receives_previous_decoded_terminal_frame_and_current_local_latent():
    _output, _status, state = _run()
    assert torch.equal(state["guided"][0][2], state["decoded"][0][2])
    assert torch.equal(state["guided"][1][2], state["decoded"][1][2])
    assert [item[0]["samples"].unbind()[0].shape[2] for item in state["guided"]] == [5, 5, 7]


def test_single_chunk_uses_original_positive_without_decode_or_guide():
    keyframed = _positive(keyframes=[{"resolved_frame_index": 0, "latent": torch.ones(1)}])
    _output, status, state = _run(positive=keyframed, chunk_seconds=99.0)
    assert len(state["built"]) == 1
    assert state["built"][0].positive is keyframed
    assert state["guided"] == []
    assert state["decoded"] == []
    assert "guides_applied=0" in status


def test_existing_keyframes_fail_closed_for_multiple_chunks():
    keyframed = _positive(keyframes=[{"resolved_frame_index": 0, "latent": torch.ones(1)}])
    with pytest.raises(H3TemporalChunkSamplerError, match="without existing minimax_keyframes"):
        _run(positive=keyframed)


def test_native_add_guide_adapter_calls_official_frame_zero_api(monkeypatch):
    from comfy_extras import nodes_minimax_h3

    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(result=(["guided"],))

    monkeypatch.setattr(nodes_minimax_h3.MiniMaxH3AddGuide, "execute", execute)
    image = torch.zeros((1, 2, 2, 3))
    latent = _latent(video_t=7, audio_t=37)
    result = _apply_native_previous_frame_guide(
        positive=["original"], latent=latent, vae="vae", image=image
    )
    assert result == ["guided"]
    assert calls[0]["frame_idx"] == 0
    assert calls[0]["positive"] == ["original"]
    assert calls[0]["latent"] is latent
    assert calls[0]["vae"] == "vae"
    assert calls[0]["image"] is image
    assert calls[0]["audio"] is None
    assert calls[0]["audio_vae"] is None


@pytest.mark.parametrize("five_dimensional", [False, True])
def test_decode_terminal_frame_returns_last_rgb_frame_on_cpu(five_dimensional):
    images = torch.arange(2 * 2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 2, 4)
    decoded = images.reshape(1, 2, 2, 2, 4) if five_dimensional else images

    class VAE:
        def decode(self, video):
            assert video.shape == (1, 24, 7, 1, 1)
            return decoded

    frame = _decode_terminal_frame(VAE(), torch.zeros((1, 24, 7, 1, 1)))
    assert frame.shape == (1, 2, 2, 3)
    assert frame.device.type == "cpu"
    assert frame.dtype == torch.float32
    assert torch.equal(frame[0], images[-1, ..., :3])


def test_guided_inputs_are_all_required_together():
    with pytest.raises(H3TemporalChunkSamplerError, match="requires model, positive, and vae together"):
        sample_h3_temporal_chunks(
            model="model",
            positive=_positive(),
            noise=_empty_noise(),
            sampler="sampler",
            sigmas=torch.tensor([1.0, 0.0]),
            latent_image=_latent(),
            chunk_duration_seconds=1.0,
            sample_chunk=lambda **kwargs: kwargs["latent_image"],
        )


def test_node_schema_exposes_model_positive_vae_and_removes_bc_and_external_guider():
    node = JR_H3_TemporalChunkSampler()
    schema = node.INPUT_TYPES()["required"]
    assert list(schema) == [
        "model",
        "positive",
        "vae",
        "noise",
        "sampler",
        "sigmas",
        "latent_image",
        "chunk_duration_seconds",
        "aggressive_memory_cleanup",
    ]
    assert "temporal_mode" not in schema
    assert "guider" not in schema
    assert node.RETURN_TYPES == ("LATENT", "STRING")
