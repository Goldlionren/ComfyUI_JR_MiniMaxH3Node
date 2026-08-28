from __future__ import annotations

import gc
import weakref

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_temporal_chunk_sampler import JR_H3_TemporalChunkSampler
from ComfyUI_JR_MiniMaxH3Node.utils.h3_temporal_chunk_sampler import (
    ERROR_PREFIX,
    TEMPORAL_MODE_A,
    TEMPORAL_MODE_B,
    TEMPORAL_MODE_C,
    H3TemporalChunkSamplerError,
    derive_chunk_seed,
    frame_boundary_for_video_token,
    plan_h3_temporal_chunks,
    plan_h3_temporal_overlap_windows,
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


def _temporal_values_latent(*, video_t=217, audio_t=1227):
    video_values = torch.arange(video_t, dtype=torch.float32).reshape(1, 1, video_t, 1, 1)
    audio_values = torch.arange(audio_t, dtype=torch.float32).reshape(1, 1, 1, audio_t)
    video = video_values.expand(1, 24, video_t, 1, 1).clone()
    audio = audio_values.expand(1, 32, 2, audio_t).clone()
    return {"samples": NestedTensor((video, audio)), "custom": "preserve-me"}


def _empty_noise():
    from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise

    return Noise_EmptyNoise()


def _random_noise(seed):
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

    return Noise_RandomNoise(seed)


def test_native_standard_random_noise_repeats_same_shape_with_same_seed():
    latent = _latent(video_t=7, audio_t=37)
    noise = _random_noise(123456789)

    first = noise.generate_noise(latent)
    second = noise.generate_noise(latent)

    first_video, first_audio = first.unbind()
    second_video, second_audio = second.unbind()
    assert torch.equal(first_video, second_video)
    assert torch.equal(first_audio, second_audio)


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


def test_explicit_a_mode_matches_legacy_canonical_plan():
    plan = plan_h3_temporal_chunks(217, 1227, 5.0)
    assert [(chunk.video_start, chunk.video_end) for chunk in plan.chunks] == [
        (0, 35),
        (35, 70),
        (70, 105),
        (105, 140),
        (140, 175),
        (175, 217),
    ]
    assert [(chunk.audio_start, chunk.audio_end) for chunk in plan.chunks] == [
        (0, 198),
        (198, 397),
        (397, 595),
        (595, 793),
        (793, 992),
        (992, 1227),
    ]


def test_bc_canonical_plan_matches_hardcoded_global_window_oracle():
    plan = plan_h3_temporal_overlap_windows(217, 1227, 5.0)

    assert plan.stride_video_tokens == 35
    assert plan.stride_frames == 119
    assert plan.window_video_tokens == 42
    assert plan.window_frames == 141
    assert plan.window_audio_tokens == 235
    assert plan.overlap_video_tokens == 7
    assert plan.overlap_frames == 22
    assert [
        (
            window.sample.video_start,
            window.sample.video_end,
            window.keep_video_start,
            window.keep_video_end,
            window.sample.audio_start,
            window.sample.audio_end,
            window.keep_audio_start,
            window.keep_audio_end,
            window.sample.frame_start,
        )
        for window in plan.windows
    ] == [
        (0, 42, 0, 42, 0, 235, 0, 235, 0),
        (35, 77, 42, 77, 198, 433, 235, 433, 119),
        (70, 112, 77, 112, 397, 632, 433, 632, 238),
        (105, 147, 112, 147, 595, 830, 632, 830, 357),
        (140, 182, 147, 182, 793, 1028, 830, 1028, 476),
        (175, 217, 182, 217, 992, 1227, 1028, 1227, 595),
    ]


def test_bc_final_window_is_back_aligned_and_keep_ranges_cover_once():
    plan = plan_h3_temporal_overlap_windows(202, 1142, 5.0)
    assert [(window.sample.video_start, window.sample.video_end) for window in plan.windows] == [
        (0, 42),
        (35, 77),
        (70, 112),
        (105, 147),
        (140, 182),
        (160, 202),
    ]
    final = plan.windows[-1]
    assert (final.keep_video_start, final.keep_video_end) == (182, 202)
    assert (final.sample.audio_start, final.sample.audio_end) == (907, 1142)
    assert (final.keep_audio_start, final.keep_audio_end) == (1028, 1142)
    assert final.sample.frame_start == 544

    video_coverage = torch.zeros(202, dtype=torch.int32)
    audio_coverage = torch.zeros(1142, dtype=torch.int32)
    for window in plan.windows:
        video_coverage[window.keep_video_start : window.keep_video_end] += 1
        audio_coverage[window.keep_audio_start : window.keep_audio_end] += 1
        assert window.sample.video_tokens == 42
        assert window.sample.frames == 141
        assert window.sample.audio_tokens == 235
    assert torch.equal(video_coverage, torch.ones_like(video_coverage))
    assert torch.equal(audio_coverage, torch.ones_like(audio_coverage))


@pytest.mark.parametrize(
    ("requested", "expected_window"),
    [
        (5.0, (42, 141, 235)),
        (8.0, (57, 192, 320)),
        (10.125, (72, 243, 405)),
        (12.25, (87, 294, 490)),
        (14.375, (102, 345, 575)),
    ],
)
def test_exact_window_math_derives_safe_windows_and_cross_checks_sequential_presets(
    requested, expected_window
):
    from ComfyUI_JR_MiniMaxH3Node.utils.h3_sequential_audio import CHUNK_PRESETS

    plan = plan_h3_temporal_overlap_windows(217, 1227, requested)
    assert (plan.window_video_tokens, plan.window_frames, plan.window_audio_tokens) == expected_window
    sequential = {(item.video_latent_t, item.frames, item.audio_ticks) for item in CHUNK_PRESETS}
    if expected_window != (87, 294, 490):
        assert expected_window in sequential


@pytest.mark.parametrize("audio_t", [1226, 1228])
@pytest.mark.parametrize("mode", [TEMPORAL_MODE_B, TEMPORAL_MODE_C])
def test_bc_preserves_official_terminal_audio_tick_delta_without_padding(audio_t, mode):
    plan = plan_h3_temporal_overlap_windows(217, audio_t, 5.0)
    final = plan.windows[-1]
    assert plan.audio_delta == audio_t - 1227
    assert final.sample.audio_end == audio_t
    assert final.sample.audio_tokens == 235 + plan.audio_delta
    assert final.keep_audio_end == audio_t

    latent = _temporal_values_latent(audio_t=audio_t)
    source_video, source_audio = latent["samples"].unbind()
    output, status = sample_h3_temporal_chunks(
        noise=_empty_noise(),
        guider=None,
        sampler=None,
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=latent,
        chunk_duration_seconds=5.0,
        temporal_mode=mode,
        sample_chunk=_identity_native,
    )
    output_video, output_audio = output["samples"].unbind()
    assert torch.equal(output_video, source_video)
    assert torch.equal(output_audio, source_audio)
    assert f"audio timeline delta: {plan.audio_delta:+d} tick(s)" in status


@pytest.mark.parametrize("seconds", [14.875, 15.0, 99.0])
def test_bc_requested_duration_above_safe_window_fails_closed(seconds):
    with pytest.raises(H3TemporalChunkSamplerError, match="maximum 102 video tokens"):
        plan_h3_temporal_overlap_windows(217, 1227, seconds)


def test_bc_short_timeline_without_real_overlap_fails_closed():
    with pytest.raises(H3TemporalChunkSamplerError, match="requires a global timeline longer"):
        plan_h3_temporal_overlap_windows(42, 235, 5.0)


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
        noise=_empty_noise(),
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
    assert "noise_mode=native_zero" in status
    assert "no temporal hidden-state carry" in status


def test_b_uses_original_source_overlap_and_reassembles_identity_once():
    latent = _temporal_values_latent()
    original_video, original_audio = latent["samples"].unbind()
    captured = []

    def sampled(**kwargs):
        video, audio = kwargs["latent_image"]["samples"].unbind()
        captured.append((video.clone(), audio.clone()))
        return _identity_native(**kwargs)

    output, status = sample_h3_temporal_chunks(
        noise=_empty_noise(),
        guider=object(),
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=latent,
        chunk_duration_seconds=5.0,
        temporal_mode=TEMPORAL_MODE_B,
        sample_chunk=sampled,
    )

    output_video, output_audio = output["samples"].unbind()
    assert len(captured) == 6
    assert torch.equal(captured[1][0][:, :, :7], original_video[:, :, 35:42])
    assert torch.equal(captured[1][0][:, :, 7:], original_video[:, :, 42:77])
    assert torch.equal(captured[1][1][:, :, :, :37], original_audio[:, :, :, 198:235])
    assert torch.equal(captured[1][1][:, :, :, 37:], original_audio[:, :, :, 235:433])
    assert torch.equal(output_video, original_video)
    assert torch.equal(output_audio, original_audio)
    assert output_video.device.type == "cpu"
    assert output_audio.device.type == "cpu"
    assert output["custom"] == "preserve-me"
    assert f"temporal_mode={TEMPORAL_MODE_B}" in status
    assert "context=source" in status
    assert "requested_chunk=5s" in status
    assert "global: video T=217, audio T=1227, frames=736" in status
    assert "stride: 35 video tokens / 119 frames / 4.958s" in status
    assert "window: 42 video tokens / 141 frames / 235 audio ticks" in status
    assert "nominal overlap: 7 video tokens / 22 frames / 0.917s" in status
    assert "windows: 6" in status
    assert "#1 sample v[0:42] f[0:141] a[0:235] keep v[0:42] a[0:235]" in status
    assert "#6 sample v[175:217] f[595:736] a[992:1227] keep v[182:217] a[1028:1227]" in status


def test_c_uses_only_previous_refined_av_overlap_and_discards_resampled_prefix(monkeypatch):
    latent = _temporal_values_latent()
    original_video, original_audio = latent["samples"].unbind()
    captured = []

    def forbidden_cat(*args, **kwargs):
        raise AssertionError("exact-overlap sampler must not concatenate retained window outputs")

    original_to = torch.Tensor.to

    def reject_full_output_transfer(tensor, *args, **kwargs):
        if (tensor.ndim == 5 and tensor.shape[2] == 217) or (tensor.ndim == 4 and tensor.shape[3] == 1227):
            raise AssertionError("C must not move a full global CPU output back to the sampling device")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", forbidden_cat)
    monkeypatch.setattr(torch.Tensor, "to", reject_full_output_transfer)

    def refined(**kwargs):
        assert "noise_mask" not in kwargs["latent_image"]
        video, audio = kwargs["latent_image"]["samples"].unbind()
        captured.append((video.clone(), audio.clone()))
        return {"samples": NestedTensor((video.clone() + 10000, audio.clone() + 10000))}

    output, status = sample_h3_temporal_chunks(
        noise=_empty_noise(),
        guider=object(),
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=latent,
        chunk_duration_seconds=5.0,
        temporal_mode=TEMPORAL_MODE_C,
        sample_chunk=refined,
    )

    output_video, output_audio = output["samples"].unbind()
    assert len(captured) == 6
    assert torch.equal(captured[1][0][:, :, :7], original_video[:, :, 35:42] + 10000)
    assert torch.equal(captured[1][0][:, :, 7:], original_video[:, :, 42:77])
    assert torch.equal(captured[1][1][:, :, :, :37], original_audio[:, :, :, 198:235] + 10000)
    assert torch.equal(captured[1][1][:, :, :, 37:], original_audio[:, :, :, 235:433])
    assert torch.equal(output_video, original_video + 10000)
    assert torch.equal(output_audio, original_audio + 10000)
    assert f"temporal_mode={TEMPORAL_MODE_C}" in status
    assert "context=previous_refined" in status


def test_b_keep_discard_assigns_each_global_position_to_one_window():
    latent = _temporal_values_latent()
    plan = plan_h3_temporal_overlap_windows(217, 1227, 5.0)
    call_index = 0

    def marked(**kwargs):
        nonlocal call_index
        call_index += 1
        video, audio = kwargs["latent_image"]["samples"].unbind()
        return {
            "samples": NestedTensor(
                (torch.full_like(video, call_index), torch.full_like(audio, call_index))
            )
        }

    output, _ = sample_h3_temporal_chunks(
        noise=_empty_noise(),
        guider=None,
        sampler=None,
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=latent,
        chunk_duration_seconds=5.0,
        temporal_mode=TEMPORAL_MODE_B,
        sample_chunk=marked,
    )
    output_video, output_audio = output["samples"].unbind()
    expected_video = torch.empty_like(output_video)
    expected_audio = torch.empty_like(output_audio)
    for owner, window in enumerate(plan.windows, start=1):
        expected_video[:, :, window.keep_video_start : window.keep_video_end] = owner
        expected_audio[:, :, :, window.keep_audio_start : window.keep_audio_end] = owner
    assert torch.equal(output_video, expected_video)
    assert torch.equal(output_audio, expected_audio)


def test_c_final_tail_uses_entire_larger_previous_refined_av_overlap():
    latent = _temporal_values_latent(video_t=202, audio_t=1142)
    original_video, original_audio = latent["samples"].unbind()
    captured_final = None
    calls = 0

    def refined(**kwargs):
        nonlocal calls, captured_final
        calls += 1
        video, audio = kwargs["latent_image"]["samples"].unbind()
        if calls == 6:
            captured_final = (video.clone(), audio.clone())
        return {"samples": NestedTensor((video.clone() + 10000, audio.clone() + 10000))}

    output, status = sample_h3_temporal_chunks(
        noise=_empty_noise(),
        guider=None,
        sampler=None,
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=latent,
        chunk_duration_seconds=5.0,
        temporal_mode=TEMPORAL_MODE_C,
        sample_chunk=refined,
    )
    assert captured_final is not None
    assert torch.equal(captured_final[0][:, :, :22], original_video[:, :, 160:182] + 10000)
    assert torch.equal(captured_final[0][:, :, 22:], original_video[:, :, 182:202])
    assert torch.equal(captured_final[1][:, :, :, :121], original_audio[:, :, :, 907:1028] + 10000)
    assert torch.equal(captured_final[1][:, :, :, 121:], original_audio[:, :, :, 1028:1142])
    output_video, output_audio = output["samples"].unbind()
    assert torch.equal(output_video, original_video + 10000)
    assert torch.equal(output_audio, original_audio + 10000)
    assert "#6 sample v[160:202]" in status
    assert "keep v[182:202] a[1028:1142]" in status


@pytest.mark.parametrize("mode", [TEMPORAL_MODE_B, TEMPORAL_MODE_C])
def test_overlap_sampled_windows_are_released_before_the_next_call(mode):
    previous_refs = []

    def sampled(**kwargs):
        gc.collect()
        if previous_refs:
            assert previous_refs[-1][0]() is None
            assert previous_refs[-1][1]() is None
        video, audio = kwargs["latent_image"]["samples"].unbind()
        sampled_video = video.clone()
        sampled_audio = audio.clone()
        previous_refs.append((weakref.ref(sampled_video), weakref.ref(sampled_audio)))
        return {"samples": NestedTensor((sampled_video, sampled_audio))}

    sample_h3_temporal_chunks(
        noise=_empty_noise(),
        guider=None,
        sampler=None,
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=_temporal_values_latent(),
        chunk_duration_seconds=5.0,
        temporal_mode=mode,
        sample_chunk=sampled,
    )
    gc.collect()
    assert all(video_ref() is None and audio_ref() is None for video_ref, audio_ref in previous_refs)


def test_default_and_explicit_a_sampling_are_identical():
    kwargs = dict(
        noise=_empty_noise(),
        guider=object(),
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        chunk_duration_seconds=1.0,
        sample_chunk=_identity_native,
    )
    default_output, default_status = sample_h3_temporal_chunks(latent_image=_latent(), **kwargs)
    explicit_output, explicit_status = sample_h3_temporal_chunks(
        latent_image=_latent(), temporal_mode=TEMPORAL_MODE_A, **kwargs
    )
    for default_stream, explicit_stream in zip(
        default_output["samples"].unbind(), explicit_output["samples"].unbind()
    ):
        assert torch.equal(default_stream, explicit_stream)
    assert default_status == explicit_status


def test_unknown_temporal_mode_fails_closed_before_sampling():
    with pytest.raises(H3TemporalChunkSamplerError, match="temporal_mode must be one of"):
        sample_h3_temporal_chunks(
            noise=_empty_noise(),
            guider=None,
            sampler=None,
            sigmas=torch.tensor([1.0, 0.0]),
            latent_image=_latent(),
            chunk_duration_seconds=1.0,
            temporal_mode="experimental typo",
            sample_chunk=_identity_native,
        )


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
        noise=_empty_noise(),
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
        noise=_empty_noise(),
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


def _capture_chunk_random_noise(base_seed=123456789):
    generated = []
    derived_seeds = []

    def sampled(**kwargs):
        chunk_noise = kwargs["noise"]
        noise_samples = chunk_noise.generate_noise(kwargs["latent_image"])
        video_noise, audio_noise = noise_samples.unbind()
        generated.append((video_noise.clone(), audio_noise.clone()))
        derived_seeds.append(chunk_noise.seed)
        return _identity_native(**kwargs)

    _, status = sample_h3_temporal_chunks(
        noise=_random_noise(base_seed),
        guider=None,
        sampler=None,
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=_latent(),
        chunk_duration_seconds=1.0,
        sample_chunk=sampled,
    )
    return generated, derived_seeds, status


def test_standard_random_noise_is_distinct_across_equal_shape_chunks_and_remainder():
    generated, derived_seeds, status = _capture_chunk_random_noise()

    assert len(generated) == 4
    assert len(set(derived_seeds)) == 4
    for left, right in zip(generated[:3], generated[1:3]):
        assert left[0].shape == right[0].shape
        assert not torch.equal(left[0], right[0])
    assert generated[0][1].shape == generated[2][1].shape
    assert not torch.equal(generated[0][1], generated[2][1])
    assert generated[-1][0].shape[2] == 7
    assert generated[-1][1].shape[3] == 37
    assert "noise_mode=chunk_derived" in status


def test_chunk_derived_random_noise_is_deterministic_across_runs():
    first_noise, first_seeds, _ = _capture_chunk_random_noise()
    second_noise, second_seeds, _ = _capture_chunk_random_noise()

    assert first_seeds == second_seeds
    for first, second in zip(first_noise, second_noise):
        assert torch.equal(first[0], second[0])
        assert torch.equal(first[1], second[1])


def test_a_b_and_c_use_identical_canonical_global_start_noise_identity():
    def capture(mode=None):
        seeds = []

        def sampled(**kwargs):
            seeds.append(kwargs["noise"].seed)
            return _identity_native(**kwargs)

        kwargs = dict(
            noise=_random_noise(123456789),
            guider=None,
            sampler=None,
            sigmas=torch.tensor([1.0, 0.0]),
            latent_image=_temporal_values_latent(),
            chunk_duration_seconds=5.0,
            sample_chunk=sampled,
        )
        if mode is not None:
            kwargs["temporal_mode"] = mode
        sample_h3_temporal_chunks(**kwargs)
        return seeds

    expected = [derive_chunk_seed(123456789, frame) for frame in (0, 119, 238, 357, 476, 595)]
    assert capture() == expected
    assert capture(TEMPORAL_MODE_B) == expected
    assert capture(TEMPORAL_MODE_C) == expected


@pytest.mark.parametrize("mode", [TEMPORAL_MODE_B, TEMPORAL_MODE_C])
def test_final_shifted_window_noise_uses_actual_global_frame_start(mode):
    seeds = []

    def sampled(**kwargs):
        seeds.append(kwargs["noise"].seed)
        return _identity_native(**kwargs)

    sample_h3_temporal_chunks(
        noise=_random_noise(987654321),
        guider=None,
        sampler=None,
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=_temporal_values_latent(video_t=202, audio_t=1142),
        chunk_duration_seconds=5.0,
        temporal_mode=mode,
        sample_chunk=sampled,
    )
    assert seeds[-1] == derive_chunk_seed(987654321, 544)


def test_registered_runtime_random_noise_identity_is_supported(monkeypatch):
    import nodes as comfy_nodes

    class RuntimeRandomNoise:
        def __init__(self, seed):
            self.seed = seed

        def generate_noise(self, input_latent):
            return _random_noise(self.seed).generate_noise(input_latent)

    class RegisteredRandomNoiseNode:
        @classmethod
        def execute(cls, noise_seed):
            return (RuntimeRandomNoise(noise_seed),)

    monkeypatch.setattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}, raising=False)
    monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, "RandomNoise", RegisteredRandomNoiseNode)
    original_noise = RuntimeRandomNoise(123456789)
    seen = []

    def sampled(**kwargs):
        seen.append(kwargs["noise"])
        return _identity_native(**kwargs)

    _, status = sample_h3_temporal_chunks(
        noise=original_noise,
        guider=None,
        sampler=None,
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=_latent(),
        chunk_duration_seconds=1.0,
        sample_chunk=sampled,
    )

    plan = plan_h3_temporal_chunks(22, 122, 1.0)
    assert [provider.seed for provider in seen] == [
        derive_chunk_seed(original_noise.seed, chunk.frame_start) for chunk in plan.chunks
    ]
    assert all(type(provider) is RuntimeRandomNoise for provider in seen)
    assert "noise_mode=chunk_derived" in status


def test_registered_runtime_disable_noise_identity_is_supported(monkeypatch):
    import nodes as comfy_nodes

    class RuntimeEmptyNoise:
        def __init__(self):
            self.seed = 0

        def generate_noise(self, input_latent):
            return _empty_noise().generate_noise(input_latent)

    class RegisteredDisableNoiseNode:
        @classmethod
        def execute(cls):
            return (RuntimeEmptyNoise(),)

    monkeypatch.setattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}, raising=False)
    monkeypatch.setitem(comfy_nodes.NODE_CLASS_MAPPINGS, "DisableNoise", RegisteredDisableNoiseNode)
    original_noise = RuntimeEmptyNoise()
    seen = []

    def sampled(**kwargs):
        seen.append(kwargs["noise"])
        return _identity_native(**kwargs)

    _, status = sample_h3_temporal_chunks(
        noise=original_noise,
        guider=None,
        sampler=None,
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=_latent(),
        chunk_duration_seconds=1.0,
        sample_chunk=sampled,
    )

    assert seen == [original_noise] * 4
    assert "noise_mode=native_zero" in status


def test_derived_seed_uses_stable_absolute_temporal_identity():
    first_plan = plan_h3_temporal_chunks(427, 2417, 15.0)
    second_plan = plan_h3_temporal_chunks(427, 2417, 15.0)
    first = [derive_chunk_seed(0xFFFFFFFFFFFFFFFF, chunk.frame_start) for chunk in first_plan.chunks]
    second = [derive_chunk_seed(0xFFFFFFFFFFFFFFFF, chunk.frame_start) for chunk in second_plan.chunks]

    assert first == second
    assert len(set(first)) == len(first_plan.chunks)
    assert all(0 <= seed <= 0xFFFFFFFFFFFFFFFF for seed in first)


def test_single_chunk_keeps_original_random_noise_object_and_native_seed():
    original_noise = _random_noise(987654321)
    seen = []

    def sampled(**kwargs):
        seen.append(kwargs["noise"])
        return _identity_native(**kwargs)

    _, status = sample_h3_temporal_chunks(
        noise=original_noise,
        guider=None,
        sampler=None,
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=_latent(),
        chunk_duration_seconds=99.0,
        sample_chunk=sampled,
    )

    assert seen == [original_noise]
    assert seen[0].seed == 987654321
    assert "noise_mode=native_single" in status


def test_multi_chunk_generic_noise_fails_closed_without_mutation():
    class CustomNoise:
        def generate_noise(self, input_latent):
            return _empty_noise().generate_noise(input_latent)

    custom_noise = CustomNoise()
    with pytest.raises(H3TemporalChunkSamplerError, match="generic/custom NOISE"):
        sample_h3_temporal_chunks(
            noise=custom_noise,
            guider=None,
            sampler=None,
            sigmas=torch.tensor([1.0, 0.0]),
            latent_image=_latent(),
            chunk_duration_seconds=1.0,
            sample_chunk=_identity_native,
        )
    assert not hasattr(custom_noise, "seed")


def test_seeded_custom_noise_is_not_mistaken_for_registered_random_noise():
    class Noise_RandomNoise:
        def __init__(self):
            self.seed = 123

        def generate_noise(self, input_latent):
            return _empty_noise().generate_noise(input_latent)

    with pytest.raises(H3TemporalChunkSamplerError, match="generic/custom NOISE"):
        sample_h3_temporal_chunks(
            noise=Noise_RandomNoise(),
            guider=None,
            sampler=None,
            sigmas=torch.tensor([1.0, 0.0]),
            latent_image=_latent(),
            chunk_duration_seconds=1.0,
            sample_chunk=_identity_native,
        )


def test_detected_absolute_h3_keyframes_fail_closed_for_multiple_chunks():
    class Guider:
        original_conds = {"positive": [{"minimax_keyframes": [{"resolved_frame_index": 0}]}]}

    with pytest.raises(H3TemporalChunkSamplerError, match="cannot safely consume minimax_keyframes"):
        sample_h3_temporal_chunks(
            noise=_empty_noise(),
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
            noise=_empty_noise(),
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
            noise=_empty_noise(),
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
        "temporal_mode",
    ]
    assert schema["noise"] == ("NOISE",)
    assert schema["guider"] == ("GUIDER",)
    assert schema["sampler"] == ("SAMPLER",)
    assert schema["sigmas"] == ("SIGMAS",)
    assert schema["latent_image"] == ("LATENT",)
    assert schema["chunk_duration_seconds"][1]["default"] == 15.0
    assert schema["aggressive_memory_cleanup"][1]["default"] is False
    assert schema["temporal_mode"][0] == [TEMPORAL_MODE_A, TEMPORAL_MODE_B, TEMPORAL_MODE_C]
    assert schema["temporal_mode"][1]["default"] == TEMPORAL_MODE_A
    assert node.RETURN_TYPES == ("LATENT", "STRING")
    assert node.RETURN_NAMES == ("output", "status")
