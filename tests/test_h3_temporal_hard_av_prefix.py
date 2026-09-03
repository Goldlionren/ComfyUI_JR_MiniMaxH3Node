from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_temporal_chunk_sampler import JR_H3_TemporalChunkSampler
from ComfyUI_JR_MiniMaxH3Node.utils.h3_temporal_chunk_sampler import (
    CONTINUITY_MODES,
    DEFAULT_HARD_CHUNK_PRESET,
    HARD_AUDIO_FRESH_T,
    HARD_AUDIO_PREFIX_T,
    HARD_AUDIO_WINDOW_T,
    HARD_AV_PREFIX_MODE,
    HARD_CHUNK_PRESET_LABELS,
    HARD_OVERLAP_FRAMES,
    HARD_STRIDE_FRAMES,
    HARD_VIDEO_FRESH_T,
    HARD_VIDEO_PREFIX_T,
    HARD_VIDEO_WINDOW_T,
    HARD_WINDOW_FRAMES,
    LEGACY_INDEPENDENT_MODE,
    H3TemporalChunkSamplerError,
    derive_chunk_seed,
    frame_boundary_for_video_token,
    plan_h3_hard_av_prefix_windows,
    sample_h3_temporal_chunks,
)

MAX_HARD_PRESET = "14.375s / 345 frames / 575 ticks"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _latent(*, video_t=282, audio_t=1595, noise_mask=None):
    video = torch.arange(24 * video_t, dtype=torch.float32).reshape(1, 24, video_t, 1, 1)
    audio = torch.arange(32 * 2 * audio_t, dtype=torch.float32).reshape(1, 32, 2, audio_t)
    latent = {"samples": NestedTensor((video, audio)), "custom": "preserved"}
    if noise_mask is not None:
        latent["noise_mask"] = noise_mask
    return latent


def _positive(*, keyframes=None):
    metadata = {"prompt": "normal H3 AV"}
    if keyframes is not None:
        metadata["minimax_keyframes"] = keyframes
    return [[torch.ones((1, 1, 1)), metadata]]


def _empty_noise():
    from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise

    return Noise_EmptyNoise()


def _run_hard(latent, sample_chunk, *, noise=None, positive=None, preset=MAX_HARD_PRESET):
    built = []

    def build_guider(model, chunk_positive):
        guider = SimpleNamespace(index=len(built), model=model, positive=chunk_positive)
        built.append(guider)
        return guider

    output, status = sample_h3_temporal_chunks(
        model="model",
        positive=_positive() if positive is None else positive,
        vae="unused-in-hard-mode",
        noise=_empty_noise() if noise is None else noise,
        sampler="sampler",
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=latent,
        chunk_duration_seconds=15.0,
        aggressive_memory_cleanup=False,
        continuity_mode=HARD_AV_PREFIX_MODE,
        hard_chunk_preset=preset,
        sample_chunk=sample_chunk,
        build_guider=build_guider,
        apply_guide=lambda **_kwargs: pytest.fail("Hard mode must not call AddGuide"),
        decode_last_frame=lambda *_args: pytest.fail("Hard mode must not decode/re-encode"),
    )
    return output, status, built


def test_ui_defaults_to_hard_av_prefix_and_exposes_only_two_production_modes():
    schema = JR_H3_TemporalChunkSampler.INPUT_TYPES()["required"]
    assert schema["continuity_mode"][0] == list(CONTINUITY_MODES)
    assert schema["continuity_mode"][0] == [HARD_AV_PREFIX_MODE, LEGACY_INDEPENDENT_MODE]
    assert schema["continuity_mode"][1]["default"] == HARD_AV_PREFIX_MODE
    assert schema["chunk_duration_seconds"][1]["default"] == 15.0
    assert schema["chunk_duration_seconds"][1]["step"] == 0.5
    assert schema["hard_chunk_preset"][0] == list(HARD_CHUNK_PRESET_LABELS)
    assert schema["hard_chunk_preset"][1]["default"] == DEFAULT_HARD_CHUNK_PRESET
    assert DEFAULT_HARD_CHUNK_PRESET == "5.875s / 141 frames / 235 ticks"
    assert not any(option.startswith(("A -", "B -", "C -")) for option in schema["continuity_mode"][0])


def test_frontend_switches_between_hard_preset_and_legacy_duration_widgets():
    script = (PACKAGE_ROOT / "js" / "temporal_chunk_sampler.js").read_text(encoding="utf-8")
    assert 'widget.name === "continuity_mode"' in script
    assert 'widget.name === "chunk_duration_seconds"' in script
    assert 'widget.name === "hard_chunk_preset"' in script
    assert "setWidgetVisible(legacyDuration, !hard)" in script
    assert "setWidgetVisible(hardPreset, hard)" in script
    assert "widget.hidden = false" in script
    assert "const selectedMode = arguments[0]" in script
    assert "queueMicrotask(() => {" in script
    assert "function installModeCallback(node)" in script
    assert "nodeType.prototype.onDrawForeground" in script
    assert "this.__jrTemporalLastMode !== mode.value" in script


@pytest.mark.parametrize(
    ("preset", "window_frames", "window_video_t", "window_audio_t", "stride_frames", "fresh_video_t", "fresh_audio_t"),
    [
        ("5.875s / 141 frames / 235 ticks", 141, 42, 235, 102, 30, 170),
        ("8.000s / 192 frames / 320 ticks", 192, 57, 320, 153, 45, 255),
        ("10.125s / 243 frames / 405 ticks", 243, 72, 405, 204, 60, 340),
        (MAX_HARD_PRESET, 345, 102, 575, 306, 90, 510),
    ],
)
def test_all_hard_presets_keep_exact_prefix_and_timeline(
    preset,
    window_frames,
    window_video_t,
    window_audio_t,
    stride_frames,
    fresh_video_t,
    fresh_audio_t,
):
    assert HARD_WINDOW_FRAMES == 345
    assert HARD_VIDEO_WINDOW_T == 102
    assert HARD_AUDIO_WINDOW_T == 575
    assert HARD_OVERLAP_FRAMES == 39
    assert HARD_VIDEO_PREFIX_T == 12
    assert HARD_AUDIO_PREFIX_T == 65
    assert HARD_STRIDE_FRAMES == 306
    assert HARD_VIDEO_FRESH_T == 90
    assert HARD_AUDIO_FRESH_T == 510
    assert frame_boundary_for_video_token(HARD_VIDEO_PREFIX_T) == HARD_OVERLAP_FRAMES
    assert round(HARD_OVERLAP_FRAMES * 40 / 24) == HARD_AUDIO_PREFIX_T
    assert frame_boundary_for_video_token(HARD_VIDEO_FRESH_T) == HARD_STRIDE_FRAMES
    assert round(HARD_STRIDE_FRAMES * 40 / 24) == HARD_AUDIO_FRESH_T

    global_video_t = window_video_t + 2 * fresh_video_t
    global_audio_t = window_audio_t + 2 * fresh_audio_t
    plan = plan_h3_hard_av_prefix_windows(global_video_t, global_audio_t, preset)
    assert plan.preset_label == preset
    assert [window.sample.frame_start for window in plan.windows] == [0, stride_frames, 2 * stride_frames]
    assert [window.sample.frame_end for window in plan.windows] == [
        window_frames,
        window_frames + stride_frames,
        window_frames + 2 * stride_frames,
    ]
    assert [window.sample.video_start for window in plan.windows] == [0, fresh_video_t, 2 * fresh_video_t]
    assert [window.sample.audio_start for window in plan.windows] == [0, fresh_audio_t, 2 * fresh_audio_t]
    assert [window.local_keep_video_start for window in plan.windows] == [0, 12, 12]
    assert [window.local_keep_audio_start for window in plan.windows] == [0, 65, 65]
    assert [(window.keep_video_start, window.keep_video_end) for window in plan.windows] == [
        (0, window_video_t),
        (window_video_t, window_video_t + fresh_video_t),
        (window_video_t + fresh_video_t, global_video_t),
    ]
    assert [(window.keep_audio_start, window.keep_audio_end) for window in plan.windows] == [
        (0, window_audio_t),
        (window_audio_t, window_audio_t + fresh_audio_t),
        (window_audio_t + fresh_audio_t, global_audio_t),
    ]


@pytest.mark.parametrize(
    ("preset", "video_t", "audio_t", "window_video_t", "window_audio_t"),
    [
        ("5.875s / 141 frames / 235 ticks", 72, 405, 42, 235),
        ("8.000s / 192 frames / 320 ticks", 102, 575, 57, 320),
        ("10.125s / 243 frames / 405 ticks", 132, 745, 72, 405),
        (MAX_HARD_PRESET, 192, 1085, 102, 575),
    ],
)
def test_each_hard_preset_executes_two_exact_windows(preset, video_t, audio_t, window_video_t, window_audio_t):
    calls = []

    def sample_chunk(**kwargs):
        video, audio = kwargs["latent_image"]["samples"].unbind()
        calls.append((video.shape, audio.shape, kwargs["latent_image"].get("noise_mask")))
        return kwargs["latent_image"]

    output, status, _built = _run_hard(_latent(video_t=video_t, audio_t=audio_t), sample_chunk, preset=preset)
    assert len(calls) == 2
    assert calls[0] == ((1, 24, window_video_t, 1, 1), (1, 32, 2, window_audio_t), None)
    assert calls[1][0] == (1, 24, window_video_t, 1, 1)
    assert calls[1][1] == (1, 32, 2, window_audio_t)
    assert calls[1][2] is not None
    assert output["samples"].unbind()[0].shape[2] == video_t
    assert output["samples"].unbind()[1].shape[3] == audio_t
    assert f"hard_chunk_preset={preset}" in status


def test_short_preset_pads_only_the_final_local_window_for_real_world_t107_timeline():
    latent = _latent(video_t=107, audio_t=603)
    source_video, source_audio = latent["samples"].unbind()
    calls = []

    def sample_chunk(**kwargs):
        video, audio = kwargs["latent_image"]["samples"].unbind()
        calls.append((video.clone(), audio.clone()))
        return kwargs["latent_image"]

    output, status, _built = _run_hard(
        latent,
        sample_chunk,
        preset="5.875s / 141 frames / 235 ticks",
    )
    output_video, output_audio = output["samples"].unbind()
    plan = plan_h3_hard_av_prefix_windows(107, 603, "5.875s / 141 frames / 235 ticks")

    assert [window.sample.video_start for window in plan.windows] == [0, 30, 60, 90]
    assert [window.sample.audio_start for window in plan.windows] == [0, 170, 340, 510]
    assert plan.windows[-1].sample.video_end == 132
    assert plan.windows[-1].sample.audio_end == 745
    assert plan.windows[-1].keep_video_end == 107
    assert plan.windows[-1].keep_audio_end == 603
    assert len(calls) == 4
    assert calls[-1][0].shape[2] == 42
    assert calls[-1][1].shape[3] == 235
    assert torch.count_nonzero(calls[-1][0][:, :, 17:]) == 0
    assert torch.count_nonzero(calls[-1][1][:, :, :, 93:]) == 0
    assert torch.equal(output_video, source_video)
    assert torch.equal(output_audio, source_audio)
    assert "tail_pad v=25 a=142" in status


def test_hard_prefix_uses_previous_sampled_av_tails_masks_and_fresh_only_global_writes():
    latent = _latent()
    calls = []
    sampled_chunks = []

    def sample_chunk(**kwargs):
        chunk_video, chunk_audio = kwargs["latent_image"]["samples"].unbind()
        mask = kwargs["latent_image"].get("noise_mask")
        index = len(calls)
        if index == 0:
            assert mask is None
            sampled_video = chunk_video.clone() + 100_000
            sampled_audio = chunk_audio.clone() + 200_000
            mask_streams = None
        else:
            video_mask, audio_mask = mask.unbind()
            assert torch.count_nonzero(video_mask[:, :, :HARD_VIDEO_PREFIX_T]) == 0
            assert torch.count_nonzero(audio_mask[:, :, :, :HARD_AUDIO_PREFIX_T]) == 0
            assert torch.all(video_mask[:, :, HARD_VIDEO_PREFIX_T:] == 1)
            assert torch.all(audio_mask[:, :, :, HARD_AUDIO_PREFIX_T:] == 1)
            assert torch.equal(
                chunk_video[:, :, :HARD_VIDEO_PREFIX_T],
                sampled_chunks[-1][0][:, :, -HARD_VIDEO_PREFIX_T:],
            )
            assert torch.equal(
                chunk_audio[:, :, :, :HARD_AUDIO_PREFIX_T],
                sampled_chunks[-1][1][:, :, :, -HARD_AUDIO_PREFIX_T:],
            )
            sampled_video = chunk_video.clone() + video_mask * ((index + 1) * 100_000)
            sampled_audio = chunk_audio.clone() + audio_mask * ((index + 1) * 200_000)
            mask_streams = (video_mask.clone(), audio_mask.clone())
        sampled_chunks.append((sampled_video.clone(), sampled_audio.clone()))
        calls.append((chunk_video.clone(), chunk_audio.clone(), mask_streams, kwargs["guider"]))
        return {"samples": NestedTensor((sampled_video, sampled_audio))}

    output, status, built = _run_hard(latent, sample_chunk)
    output_video, output_audio = output["samples"].unbind()
    expected_video = torch.full_like(output_video, torch.nan)
    expected_audio = torch.full_like(output_audio, torch.nan)
    expected_video[:, :, :102] = sampled_chunks[0][0]
    expected_video[:, :, 102:192] = sampled_chunks[1][0][:, :, 12:]
    expected_video[:, :, 192:282] = sampled_chunks[2][0][:, :, 12:]
    expected_audio[:, :, :, :575] = sampled_chunks[0][1]
    expected_audio[:, :, :, 575:1085] = sampled_chunks[1][1][:, :, :, 65:]
    expected_audio[:, :, :, 1085:1595] = sampled_chunks[2][1][:, :, :, 65:]

    assert len(calls) == 3
    assert len(built) == 3
    assert all(item.positive is built[0].positive for item in built)
    assert torch.equal(output_video, expected_video)
    assert torch.equal(output_audio, expected_audio)
    assert not torch.isnan(output_video).any()
    assert not torch.isnan(output_audio).any()
    assert output_video.shape == (1, 24, 282, 1, 1)
    assert output_audio.shape == (1, 32, 2, 1595)
    assert output_video.device.type == "cpu"
    assert output_audio.device.type == "cpu"
    assert output["custom"] == "preserved"
    assert "noise_mask" not in output
    assert "prefixes_applied=2" in status
    assert "prefix_source=previous sampled AV tail" in status
    assert "continuation_guide=none" in status
    assert "overlap copied once" in status


def test_chunk_zero_is_native_and_needs_no_previous_prefix():
    seen = []

    def sample_chunk(**kwargs):
        seen.append(kwargs["latent_image"])
        return kwargs["latent_image"]

    output, status, _built = _run_hard(_latent(video_t=102, audio_t=575), sample_chunk)
    assert len(seen) == 1
    assert "noise_mask" not in seen[0]
    assert output["samples"].unbind()[0].shape[2] == 102
    assert "prefixes_applied=0" in status
    assert "noise_mode=native_single" in status


def test_hard_mode_seed_derivation_uses_absolute_frame_starts():
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

    seeds = []

    def sample_chunk(**kwargs):
        seeds.append(kwargs["noise"].seed)
        return kwargs["latent_image"]

    _run_hard(_latent(), sample_chunk, noise=Noise_RandomNoise(1234))
    assert seeds == [derive_chunk_seed(1234, frame_start) for frame_start in (0, 306, 612)]


def test_hard_mode_relocks_native_sampler_prefix_drift_bit_identically():
    call_index = 0
    returned_chunks = []
    expected_video_prefix = None
    expected_audio_prefix = None

    def sample_chunk(**kwargs):
        nonlocal call_index, expected_video_prefix, expected_audio_prefix
        video, audio = kwargs["latent_image"]["samples"].unbind()
        output_video = video.clone()
        output_audio = audio.clone()
        if call_index == 0:
            expected_video_prefix = output_video[:, :, -HARD_VIDEO_PREFIX_T:].clone()
            expected_audio_prefix = output_audio[:, :, :, -HARD_AUDIO_PREFIX_T:].clone()
        if call_index == 1:
            output_video[:, :, 0] += 1
            output_audio[:, :, :, :HARD_AUDIO_PREFIX_T] -= 2
        returned_chunks.append((output_video, output_audio))
        call_index += 1
        return {"samples": NestedTensor((output_video, output_audio))}

    _output, status, _built = _run_hard(_latent(video_t=192, audio_t=1085), sample_chunk)

    assert expected_video_prefix is not None
    assert expected_audio_prefix is not None
    assert torch.equal(returned_chunks[1][0][:, :, :HARD_VIDEO_PREFIX_T], expected_video_prefix)
    assert torch.equal(returned_chunks[1][1][:, :, :, :HARD_AUDIO_PREFIX_T], expected_audio_prefix)
    assert "post_sample_prefix_relock=bit-identical" in status
    assert "native_drift_corrected video_chunks=1 audio_chunks=1" in status


def test_hard_mode_accepts_only_absent_none_or_exact_all_one_input_mask():
    base = _latent(video_t=102, audio_t=575)
    video, audio = base["samples"].unbind()
    all_one = NestedTensor((torch.ones_like(video), torch.ones_like(audio)))
    all_one_latent = _latent(video_t=102, audio_t=575, noise_mask=all_one)
    output, _status, _built = _run_hard(all_one_latent, lambda **kwargs: kwargs["latent_image"])
    assert "noise_mask" not in output

    nontrivial = NestedTensor((torch.ones_like(video), torch.zeros_like(audio)))
    with pytest.raises(H3TemporalChunkSamplerError, match="existing nontrivial noise_mask"):
        _run_hard(
            _latent(video_t=102, audio_t=575, noise_mask=nontrivial),
            lambda **kwargs: kwargs["latent_image"],
        )

    malformed = NestedTensor((torch.ones_like(video), torch.ones((1, 32, 2, 574))))
    with pytest.raises(H3TemporalChunkSamplerError, match="shapes must match"):
        _run_hard(
            _latent(video_t=102, audio_t=575, noise_mask=malformed),
            lambda **kwargs: kwargs["latent_image"],
        )


@pytest.mark.parametrize(
    ("video_t", "audio_t", "preset", "message"),
    [
        (42, 236, "5.875s / 141 frames / 235 ticks", "cannot cover the permitted video/audio boundary"),
        (282, 1595, "unsupported", "hard_chunk_preset must be one of"),
    ],
)
def test_hard_mode_fails_closed_for_nonfixed_timeline_or_preset(video_t, audio_t, preset, message):
    with pytest.raises(H3TemporalChunkSamplerError, match=message):
        plan_h3_hard_av_prefix_windows(video_t, audio_t, preset)


def test_hard_mode_rejects_existing_keyframes_instead_of_adding_another_guide():
    positive = _positive(keyframes=[{"resolved_frame_index": 0, "latent": torch.ones(1)}])
    with pytest.raises(H3TemporalChunkSamplerError, match="cannot be combined with minimax_keyframes/AddGuide"):
        _run_hard(_latent(video_t=192, audio_t=1085), lambda **kwargs: kwargs["latent_image"], positive=positive)
