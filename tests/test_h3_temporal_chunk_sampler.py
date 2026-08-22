from __future__ import annotations

import gc
import weakref

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_temporal_chunk_sampler import JR_H3_TemporalChunkSampler
from ComfyUI_JR_MiniMaxH3Node.utils.h3_temporal_chunk_sampler import (
    ERROR_PREFIX,
    H3TemporalChunkSamplerError,
    frame_boundary_for_video_token,
    plan_h3_temporal_chunks,
    sample_h3_temporal_chunks,
)


def _latent(*, video_t=22, audio_t=122, dtype=torch.float32):
    video = torch.arange(1 * 24 * video_t * 1 * 1, dtype=dtype).reshape(1, 24, video_t, 1, 1)
    audio = torch.arange(1 * 32 * 2 * audio_t, dtype=dtype).reshape(1, 32, 2, audio_t)
    return {"samples": NestedTensor((video, audio)), "batch_index": [7], "custom": "preserve-me"}


def _identity_native(**kwargs):
    latent = kwargs["latent_image"]
    video, audio = latent["samples"].unbind()
    return {"samples": NestedTensor((video.clone(), audio.clone()))}


@pytest.mark.parametrize(
    ("token_index", "frame_boundary"),
    [(0, 0), (1, 1), (2, 5), (5, 17), (7, 22), (105, 357), (427, 1450)],
)
def test_global_video_token_to_frame_boundary(token_index, frame_boundary):
    assert frame_boundary_for_video_token(token_index) == frame_boundary


def test_sixty_second_plan_uses_aligned_sequential_av_ranges():
    plan = plan_h3_temporal_chunks(427, 2417, 15.0)

    assert plan.frame_count == 1450
    assert plan.target_video_tokens == 105
    assert [(chunk.video_start, chunk.video_end) for chunk in plan.chunks] == [
        (0, 105),
        (105, 210),
        (210, 315),
        (315, 427),
    ]
    assert plan.chunks[0].frame_end == 357
    assert plan.chunks[-1].frame_end == 1450
    assert plan.chunks[0].audio_end == round(357 * 40 / 24)
    assert plan.chunks[-1].audio_end == 2417
    assert all(left.video_end == right.video_start for left, right in zip(plan.chunks, plan.chunks[1:]))
    assert all(left.audio_end == right.audio_start for left, right in zip(plan.chunks, plan.chunks[1:]))


@pytest.mark.parametrize("audio_t", [121, 122, 123])
def test_audio_encoder_tick_tolerance_is_preserved_at_final_boundary(audio_t):
    plan = plan_h3_temporal_chunks(22, audio_t, 1.0)
    assert plan.expected_audio_t == 122
    assert plan.audio_delta == audio_t - 122
    assert plan.chunks[-1].audio_end == audio_t


def test_exact_target_plus_terminal_tokens_stays_one_chunk():
    plan = plan_h3_temporal_chunks(7, 37, 1.0)
    assert len(plan.chunks) == 1
    assert plan.chunks[0].video_tokens == 7
    assert plan.chunks[0].frames == 22


def test_remainder_is_kept_without_truncating_the_timeline():
    plan = plan_h3_temporal_chunks(377, 2133, 15.0)
    assert [chunk.frames for chunk in plan.chunks] == [357, 357, 357, 209]
    assert plan.chunks[-1].video_end == 377
    assert plan.chunks[-1].audio_end == 2133


def test_short_timeline_calls_native_sampler_once():
    calls = []

    def sampled(**kwargs):
        calls.append(kwargs["latent_image"])
        return _identity_native(**kwargs)

    output, _ = sample_h3_temporal_chunks(
        noise=None,
        guider=None,
        sampler=None,
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=_latent(),
        chunk_duration_seconds=99.0,
        sample_chunk=sampled,
    )
    assert len(calls) == 1
    assert output["samples"].unbind()[0].shape[2] == 22


def test_sequential_sampling_reassembles_without_mutating_input_or_metadata(monkeypatch):
    latent = _latent()
    original_samples = latent["samples"]
    original_video, original_audio = original_samples.unbind()
    original_video_copy = original_video.clone()
    original_audio_copy = original_audio.clone()
    calls = []

    def sampled(**kwargs):
        chunk_video, chunk_audio = kwargs["latent_image"]["samples"].unbind()
        calls.append((chunk_video.shape[2], chunk_audio.shape[3]))
        return {"samples": NestedTensor((chunk_video.clone() + 10, chunk_audio.clone() + 20))}

    def forbidden_cat(*args, **kwargs):
        raise AssertionError("temporal sampler must not concatenate retained chunk outputs")

    monkeypatch.setattr(torch, "cat", forbidden_cat)
    output, status = sample_h3_temporal_chunks(
        noise=object(),
        guider=object(),
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=latent,
        chunk_duration_seconds=1.0,
        sample_chunk=sampled,
    )

    output_video, output_audio = output["samples"].unbind()
    assert calls == [(5, 28), (5, 29), (5, 28), (7, 37)]
    assert output_video.device.type == "cpu"
    assert output_audio.device.type == "cpu"
    assert torch.equal(output_video, original_video_copy + 10)
    assert torch.equal(output_audio, original_audio_copy + 20)
    assert output["batch_index"] == [7]
    assert output["custom"] == "preserve-me"
    assert output is not latent
    assert latent["samples"] is original_samples
    assert torch.equal(original_video, original_video_copy)
    assert torch.equal(original_audio, original_audio_copy)
    assert "chunks: 4" in status
    assert "CPU preallocated" in status
    assert "no temporal hidden-state carry" in status


def test_completed_sampled_chunks_are_not_retained_between_calls():
    latent = _latent()
    previous_refs = []
    calls = 0

    def sampled(**kwargs):
        nonlocal calls
        gc.collect()
        if previous_refs:
            assert previous_refs[-1]() is None
        video, audio = kwargs["latent_image"]["samples"].unbind()
        sampled_video = video.clone()
        sampled_audio = audio.clone()
        previous_refs.append(weakref.ref(sampled_video))
        calls += 1
        return {"samples": NestedTensor((sampled_video, sampled_audio))}

    output, _ = sample_h3_temporal_chunks(
        noise=None,
        guider=None,
        sampler=None,
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=latent,
        chunk_duration_seconds=1.0,
        sample_chunk=sampled,
    )

    assert calls == 4
    assert output["samples"].unbind()[0].shape == (1, 24, 22, 1, 1)


def test_aggressive_cleanup_runs_only_after_each_completed_chunk():
    cleanup_calls = []
    sample_h3_temporal_chunks(
        noise=None,
        guider=None,
        sampler=None,
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=_latent(),
        chunk_duration_seconds=1.0,
        aggressive_memory_cleanup=True,
        sample_chunk=_identity_native,
        cleanup=lambda: cleanup_calls.append("cleanup"),
    )
    assert cleanup_calls == ["cleanup"] * 4


def test_output_preallocation_uses_native_sample_dtype():
    def half_output(**kwargs):
        video, audio = kwargs["latent_image"]["samples"].unbind()
        return {"samples": NestedTensor((video.half(), audio.half()))}

    output, _ = sample_h3_temporal_chunks(
        noise=None,
        guider=None,
        sampler=None,
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=_latent(dtype=torch.float32),
        chunk_duration_seconds=99.0,
        sample_chunk=half_output,
    )
    video, audio = output["samples"].unbind()
    assert video.dtype == torch.float16
    assert audio.dtype == torch.float16


def test_detected_absolute_h3_keyframes_fail_closed_for_multiple_chunks():
    class Guider:
        original_conds = {"positive": [{"minimax_keyframes": [{"resolved_frame_index": 0}]}]}

    with pytest.raises(H3TemporalChunkSamplerError, match="cannot safely consume minimax_keyframes"):
        sample_h3_temporal_chunks(
            noise=None,
            guider=Guider(),
            sampler=None,
            sigmas=torch.tensor([1.0, 0.0]),
            latent_image=_latent(),
            chunk_duration_seconds=1.0,
            sample_chunk=_identity_native,
        )


@pytest.mark.parametrize(
    ("latent", "expected"),
    [
        ({}, "missing required 'samples'"),
        ({"samples": torch.zeros(1)}, "two-stream H3 NestedTensor"),
        ({"samples": NestedTensor((torch.zeros((1, 16, 22, 1, 1)), torch.zeros((1, 32, 2, 122))))}, "[B,24,T,H,W]"),
        ({"samples": NestedTensor((torch.zeros((1, 24, 22, 1, 1)), torch.zeros((1, 32, 1, 122))))}, "[B,32,2,T]"),
        ({"samples": NestedTensor((torch.zeros((1, 24, 21, 1, 1)), torch.zeros((1, 32, 2, 122))))}, "T_video = 5k + 2"),
        ({"samples": NestedTensor((torch.zeros((1, 24, 22, 1, 1)), torch.zeros((1, 32, 2, 100))))}, "temporal mismatch"),
        ({"samples": _latent()["samples"], "noise_mask": torch.ones(1)}, "noise_mask is not supported"),
    ],
)
def test_invalid_h3_latents_fail_closed(latent, expected):
    with pytest.raises(H3TemporalChunkSamplerError) as exc:
        sample_h3_temporal_chunks(
            noise=None,
            guider=None,
            sampler=None,
            sigmas=torch.tensor([1.0, 0.0]),
            latent_image=latent,
            chunk_duration_seconds=1.0,
            sample_chunk=_identity_native,
        )
    assert str(exc.value).startswith(ERROR_PREFIX)
    assert expected in str(exc.value)


@pytest.mark.parametrize("stream", ["video", "audio"])
def test_non_finite_streams_fail_closed(stream):
    latent = _latent()
    video, audio = latent["samples"].unbind()
    if stream == "video":
        video[0, 0, 0, 0, 0] = float("nan")
    else:
        audio[0, 0, 0, 0] = float("inf")
    with pytest.raises(H3TemporalChunkSamplerError, match="contains NaN or Inf"):
        sample_h3_temporal_chunks(
            noise=None,
            guider=None,
            sampler=None,
            sigmas=torch.tensor([1.0, 0.0]),
            latent_image=latent,
            chunk_duration_seconds=1.0,
            sample_chunk=_identity_native,
        )


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), True, "15"])
def test_invalid_chunk_duration_fails_closed(seconds):
    with pytest.raises(H3TemporalChunkSamplerError) as exc:
        plan_h3_temporal_chunks(22, 70, seconds)
    assert "finite positive number" in str(exc.value)


def test_native_output_shape_change_is_rejected():
    def wrong_shape(**kwargs):
        video, audio = kwargs["latent_image"]["samples"].unbind()
        return {"samples": NestedTensor((video[:, :, :-1], audio))}

    with pytest.raises(H3TemporalChunkSamplerError, match="changed the video shape"):
        sample_h3_temporal_chunks(
            noise=None,
            guider=None,
            sampler=None,
            sigmas=torch.tensor([1.0, 0.0]),
            latent_image=_latent(),
            chunk_duration_seconds=1.0,
            sample_chunk=wrong_shape,
        )


def test_node_schema_matches_advanced_sampler_contract():
    node = JR_H3_TemporalChunkSampler()
    schema = node.INPUT_TYPES()["required"]
    assert list(schema) == [
        "noise",
        "guider",
        "sampler",
        "sigmas",
        "latent_image",
        "chunk_duration_seconds",
        "aggressive_memory_cleanup",
    ]
    assert schema["noise"] == ("NOISE",)
    assert schema["guider"] == ("GUIDER",)
    assert schema["sampler"] == ("SAMPLER",)
    assert schema["sigmas"] == ("SIGMAS",)
    assert schema["latent_image"] == ("LATENT",)
    assert schema["chunk_duration_seconds"][1]["default"] == 15.0
    assert schema["aggressive_memory_cleanup"][1]["default"] is False
    assert node.RETURN_TYPES == ("LATENT", "STRING")
    assert node.RETURN_NAMES == ("output", "status")
