import json
from pathlib import Path

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_sequential_audio import JR_H3_SequentialAudioChunkDriver
from ComfyUI_JR_MiniMaxH3Node.utils import h3_sequential_audio as module
from ComfyUI_JR_MiniMaxH3Node.utils.h3_sequential_audio import (
    CHUNK_PRESETS,
    HARD_AUDIO_OVERLAP_TICKS,
    HARD_AUDIO_STRIDE_TICKS,
    HARD_CONTEXT_FRAMES,
    HARD_CONTEXT_LATENT_STEPS,
    HARD_LATENT_PREFIX_MODE,
    HARD_STRIDE_FRAMES,
    H3SequentialAudioError,
    apply_continuation_guide,
    checkpoint_sampled_latent,
    commit_decoded_chunk,
    load_manifest,
    prepare_audio_chunk,
)


class _AudioVAE:
    audio_sample_rate = 32000

    def encode(self, waveform):
        ticks = round(waveform.shape[1] * 40 / self.audio_sample_rate)
        marker = waveform[:, :1, :1].mean().item()
        return torch.full((1, 32, 2, ticks), marker, dtype=torch.float32)


def _audio_frames(frame_count, sample_rate=32000):
    sample_count = round(frame_count * sample_rate / 24)
    values = torch.arange(sample_count, dtype=torch.float32).reshape(1, 1, sample_count)
    values = values / max(1, sample_count)
    return {"waveform": values, "sample_rate": sample_rate}


def _av_latent(*, preset=None, height=2, width=3, video_value=0.0, video_mask=None):
    preset = preset or CHUNK_PRESETS[0]
    video = torch.full((1, 24, preset.video_latent_t, height, width), video_value)
    audio = torch.zeros((1, 32, 2, preset.audio_ticks))
    result = {"samples": NestedTensor((video, audio)), "preserved": "metadata"}
    if video_mask is not None:
        audio_mask = torch.ones_like(audio)
        result["noise_mask"] = NestedTensor((video_mask, audio_mask))
    return result


def _prepare(
    tmp_path,
    source,
    *,
    run_id=1,
    av_latent=None,
    mode=HARD_LATENT_PREFIX_MODE,
    preset=None,
):
    preset = preset or CHUNK_PRESETS[0]
    return prepare_audio_chunk(
        av_latent=av_latent or _av_latent(preset=preset),
        audio=source,
        audio_vae=_AudioVAE(),
        chunk_preset=preset.label,
        cache_path="jobs",
        job_name="hard-prefix",
        run_id=run_id,
        continuity_mode=mode,
        seed_mode="Derived per chunk",
        base_seed=99,
        output_directory=tmp_path,
    )


def _fake_video_io(monkeypatch, tmp_path, captured=None):
    def encode(images, path, **kwargs):
        if captured is not None:
            captured.append(images.detach().clone())
        Path(path).write_bytes(b"segment")
        return kwargs.get("required_encoder") or "libx264"

    monkeypatch.setattr(module, "_encode_segment", encode)
    monkeypatch.setattr(module, "_allocate_final_output", lambda *_args: tmp_path / "final.mp4")
    monkeypatch.setattr(
        module,
        "_concat_and_mux",
        lambda *, output_path, **_kwargs: Path(output_path).write_bytes(b"final"),
    )


def _commit(context, images=None):
    if images is None:
        images = torch.zeros((context.generated_frames, 2, 3, 3))
    return commit_decoded_chunk(
        images=images,
        context=context,
        quality=20,
        bit_depth="8-bit",
        audio_bitrate="192k",
        filename_prefix="video/test",
    )


def _checkpoint_with_video(latent, context, video, audio=None):
    _current_video, current_audio = latent["samples"].unbind()
    sampled = dict(latent)
    sampled["samples"] = NestedTensor((video, current_audio if audio is None else audio))
    return checkpoint_sampled_latent(sampled, context)[0]


def test_hard_profile_manifest_and_absolute_chunk_grid(tmp_path, monkeypatch):
    captured = []
    _fake_video_io(monkeypatch, tmp_path, captured)
    source = _audio_frames(700)
    contexts = []
    slices = []
    for _ in range(3):
        latent, context, _seed, audio_slice, _status = _prepare(tmp_path, source)
        contexts.append(context)
        slices.append(audio_slice)
        checkpoint_sampled_latent(latent, context)
        _commit(context)

    assert [context.frame_start for context in contexts] == [0, 306, 612]
    assert [context.stride_frames for context in contexts] == [HARD_STRIDE_FRAMES] * 3
    assert [context.trim_head_frames for context in contexts] == [0, 39, 39]
    assert [context.raw_real_frames for context in contexts] == [345, 345, 88]
    assert [context.real_frames for context in contexts] == [345, 306, 49]
    assert sum(context.real_frames for context in contexts) == 700
    assert [int(segment.shape[0]) for segment in captured] == [345, 306, 49]
    assert contexts[1].source_sample_start < contexts[0].source_sample_end
    assert slices[1]["waveform"][0, 0, 0] == source["waveform"][0, 0, contexts[1].source_sample_start]

    manifest = load_manifest(Path(contexts[-1].job_dir))
    assert manifest["continuation_mode"] == HARD_LATENT_PREFIX_MODE
    assert manifest["hard_context_frames"] == HARD_CONTEXT_FRAMES
    assert manifest["hard_context_latent_steps"] == HARD_CONTEXT_LATENT_STEPS
    assert manifest["stride_frames"] == HARD_STRIDE_FRAMES
    assert HARD_AUDIO_OVERLAP_TICKS == 65
    assert HARD_AUDIO_STRIDE_TICKS == 510
    assert round(HARD_CONTEXT_FRAMES * 40 / 24) == HARD_AUDIO_OVERLAP_TICKS
    assert round(HARD_STRIDE_FRAMES * 40 / 24) == HARD_AUDIO_STRIDE_TICKS


@pytest.mark.parametrize(
    ("preset_index", "expected_stride_frames", "expected_audio_stride_ticks"),
    (
        (0, 306, 510),
        (1, 204, 340),
        (2, 153, 255),
        (3, 102, 170),
    ),
)
def test_hard_prefix_supports_every_chunk_preset(
    tmp_path,
    monkeypatch,
    preset_index,
    expected_stride_frames,
    expected_audio_stride_ticks,
):
    captured = []
    _fake_video_io(monkeypatch, tmp_path, captured)
    preset = CHUNK_PRESETS[preset_index]
    source = _audio_frames(preset.frames + expected_stride_frames)
    latent0, context0, *_ = _prepare(tmp_path, source, preset=preset)
    previous_video = torch.arange(
        1 * 24 * preset.video_latent_t * 2 * 3,
        dtype=torch.float32,
    ).reshape(1, 24, preset.video_latent_t, 2, 3)
    _checkpoint_with_video(latent0, context0, previous_video)
    _commit(context0)

    latent1, context1, *_ = _prepare(tmp_path, source, preset=preset)
    _positive, output, _status = apply_continuation_guide(
        positive=("base",),
        latent=latent1,
        context=context1,
        vae=object(),
    )
    output_video, output_audio = output["samples"].unbind()

    assert context0.generated_frames == preset.frames
    assert context0.stride_frames == expected_stride_frames
    assert context1.frame_start == expected_stride_frames
    assert context1.trim_head_frames == HARD_CONTEXT_FRAMES
    assert context1.real_frames == expected_stride_frames
    assert torch.equal(
        output_video[..., :HARD_CONTEXT_LATENT_STEPS, :, :],
        previous_video[..., -HARD_CONTEXT_LATENT_STEPS:, :, :],
    )
    assert output_audio.shape[-1] == preset.audio_ticks
    assert round(expected_stride_frames * 40 / 24) == expected_audio_stride_ticks
    assert preset.audio_ticks - HARD_AUDIO_OVERLAP_TICKS == expected_audio_stride_ticks

    images = torch.arange(preset.frames, dtype=torch.float32).reshape(-1, 1, 1, 1).expand(-1, 2, 3, 3)
    _commit(context1, images)
    assert captured[-1].shape[0] == expected_stride_frames
    assert captured[-1][0, 0, 0, 0].item() == HARD_CONTEXT_FRAMES
    assert captured[-1][-1, 0, 0, 0].item() == preset.frames - 1

    manifest = load_manifest(Path(context1.job_dir))
    assert manifest["chunk"]["frames"] == preset.frames
    assert manifest["hard_context_frames"] == HARD_CONTEXT_FRAMES
    assert manifest["hard_context_latent_steps"] == HARD_CONTEXT_LATENT_STEPS
    assert manifest["stride_frames"] == expected_stride_frames


def test_hard_prefix_is_bit_identical_and_masks_only_video_prefix(tmp_path, monkeypatch):
    _fake_video_io(monkeypatch, tmp_path)
    source = _audio_frames(700)
    incoming_video_mask = torch.linspace(0.1, 1.0, 102).reshape(1, 1, 102, 1, 1)
    template = _av_latent(video_mask=incoming_video_mask)
    latent0, context0, *_ = _prepare(tmp_path, source, av_latent=template)
    unchanged_positive, unchanged_latent, status0 = apply_continuation_guide(
        positive=("base",), latent=latent0, context=context0, vae=object()
    )
    assert unchanged_positive == ("base",)
    assert unchanged_latent is latent0
    assert "no previous sampled latent" in status0

    previous_video = torch.arange(1 * 24 * 102 * 2 * 3, dtype=torch.float32).reshape(1, 24, 102, 2, 3)
    previous_audio = torch.full((1, 32, 2, 575), -123.0)
    sampled0 = _checkpoint_with_video(latent0, context0, previous_video, previous_audio)
    previous_audio = sampled0["samples"].unbind()[1]
    _commit(context0)

    current_template = _av_latent(video_value=7.0, video_mask=incoming_video_mask)
    latent1, context1, *_ = _prepare(tmp_path, source, av_latent=current_template)
    current_video, current_audio = latent1["samples"].unbind()
    expected_generation = current_video[..., HARD_CONTEXT_LATENT_STEPS:, :, :].clone()
    expected_audio = current_audio.clone()
    positive = [(torch.zeros(1), {"source": "base"})]
    output_positive, output, status = apply_continuation_guide(
        positive=positive,
        latent=latent1,
        context=context1,
        vae=object(),
        initial_frame=torch.ones((1, 2, 3, 3)),
    )
    output_video, output_audio = output["samples"].unbind()
    output_video_mask, output_audio_mask = output["noise_mask"].unbind()

    assert output_positive is positive
    assert torch.equal(output_video[..., :12, :, :], previous_video[..., -12:, :, :])
    assert torch.equal(output_video[..., 12:, :, :], expected_generation)
    assert torch.count_nonzero(output_video_mask[..., :12, :, :]) == 0
    assert torch.equal(output_video_mask[..., 12:, :, :], incoming_video_mask[..., 12:, :, :])
    assert torch.count_nonzero(output_audio_mask) == 0
    assert output_audio is current_audio
    assert torch.equal(output_audio, expected_audio)
    assert not torch.equal(output_audio, previous_audio)
    assert "PNG/VAE continuation guide: not applied" in status


def test_hard_output_trims_exactly_39_frames(tmp_path, monkeypatch):
    captured = []
    _fake_video_io(monkeypatch, tmp_path, captured)
    source = _audio_frames(700)
    latent0, context0, *_ = _prepare(tmp_path, source)
    checkpoint_sampled_latent(latent0, context0)
    _commit(context0)
    _latent1, context1, *_ = _prepare(tmp_path, source)
    images = torch.arange(345, dtype=torch.float32).reshape(345, 1, 1, 1).expand(-1, 2, 3, 3)
    _commit(context1, images)
    assert captured[-1].shape[0] == 306
    assert captured[-1][0, 0, 0, 0].item() == 39
    assert captured[-1][-1, 0, 0, 0].item() == 344


@pytest.mark.parametrize("damage", ["missing", "corrupt", "incompatible"])
def test_hard_prefix_checkpoint_failures_are_closed(tmp_path, monkeypatch, damage):
    _fake_video_io(monkeypatch, tmp_path)
    source = _audio_frames(700)
    latent0, context0, *_ = _prepare(tmp_path, source)
    if damage == "incompatible":
        bad_video = torch.zeros((1, 24, 102, 4, 3))
        checkpoint_sampled_latent(
            {"samples": NestedTensor((bad_video, latent0["samples"].unbind()[1]))},
            context0,
        )
    elif damage == "corrupt":
        checkpoint_sampled_latent(latent0, context0)
    _commit(context0)
    latent1, context1, *_ = _prepare(tmp_path, source)
    checkpoint_path = Path(context1.job_dir) / "latents" / "chunk_00000.safetensors"
    if damage == "corrupt":
        checkpoint_path.write_bytes(b"not safetensors")
    with pytest.raises(H3SequentialAudioError, match="missing|corrupt|incompatible"):
        apply_continuation_guide(positive=("base",), latent=latent1, context=context1, vae=object())


def test_legacy_manifest_schema_requires_new_run_id(tmp_path):
    source = _audio_frames(100)
    _latent, context, *_ = _prepare(tmp_path, source)
    manifest_path = Path(context.job_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(H3SequentialAudioError, match="Increment run_id"):
        _prepare(tmp_path, source)


def test_hard_mode_is_node_default():
    continuity_input = JR_H3_SequentialAudioChunkDriver.INPUT_TYPES()["required"]["continuity_mode"]
    assert continuity_input[0][0] == HARD_LATENT_PREFIX_MODE
    assert continuity_input[1]["default"] == HARD_LATENT_PREFIX_MODE
