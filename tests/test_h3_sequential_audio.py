from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_sequential_audio import (
    JR_H3_SequentialAudioChunkDriver,
    JR_H3_SequentialContinuationGuide,
    JR_H3_SequentialLatentCheckpoint,
    JR_H3_SequentialVideoOutput,
)
from ComfyUI_JR_MiniMaxH3Node.utils import h3_sequential_audio as module
from ComfyUI_JR_MiniMaxH3Node.utils.h3_sequential_audio import (
    CHUNK_PRESETS,
    H3SequentialAudioError,
    apply_continuation_guide,
    checkpoint_sampled_latent,
    commit_decoded_chunk,
    load_manifest,
    manifest_fingerprint,
    prepare_audio_chunk,
    resolve_job_directory,
)


class _AudioVAE:
    audio_sample_rate = 32000

    def encode(self, waveform):
        assert waveform.ndim == 3 and waveform.shape[0] == 1
        ticks = round(waveform.shape[1] * 40 / self.audio_sample_rate)
        return torch.zeros((1, 32, 2, ticks), dtype=torch.float32)


def _av_latent(preset):
    video = torch.zeros((1, 24, preset.video_latent_t, 2, 3), dtype=torch.float32)
    audio = torch.zeros((1, 32, 2, preset.audio_ticks), dtype=torch.float32)
    return {"samples": NestedTensor((video, audio)), "custom": {"preserved": True}}


def _audio(seconds, sample_rate=32000, channels=2):
    samples = round(seconds * sample_rate)
    values = torch.arange(channels * samples, dtype=torch.float32).reshape(1, channels, samples)
    values = values / max(1, values.numel())
    return {"waveform": values, "sample_rate": sample_rate}


def _prepare(tmp_path, audio, *, preset=None, continuity="Previous Last Frame", run_id=1):
    preset = preset or CHUNK_PRESETS[-1]
    return prepare_audio_chunk(
        av_latent=_av_latent(preset),
        audio=audio,
        audio_vae=_AudioVAE(),
        chunk_preset=preset.label,
        cache_path="jobs",
        job_name="测试 sequence",
        run_id=run_id,
        continuity_mode=continuity,
        seed_mode="Derived per chunk",
        base_seed=1234,
        output_directory=tmp_path,
    )


def _fake_video_io(monkeypatch, tmp_path):
    def encode(images, path, **kwargs):
        assert images.ndim == 4
        Path(path).write_bytes(b"segment")
        return kwargs.get("required_encoder") or "libx264"

    def allocate(_prefix, _width, _height):
        return tmp_path / "final_audio_sequence.mp4"

    def concat(*, output_path, **kwargs):
        assert kwargs["manifest"]["source"]["samples"] > 0
        Path(output_path).write_bytes(b"final")

    monkeypatch.setattr(module, "_encode_segment", encode)
    monkeypatch.setattr(module, "_allocate_final_output", allocate)
    monkeypatch.setattr(module, "_concat_and_mux", concat)


def test_presets_are_exact_h3_frame_and_audio_tick_grids():
    expected = {
        345: (14.375, 102, 575),
        243: (10.125, 72, 405),
        192: (8.0, 57, 320),
        141: (5.875, 42, 235),
    }
    assert len(CHUNK_PRESETS) == 4
    for preset in CHUNK_PRESETS:
        seconds, video_t, audio_t = expected[preset.frames]
        assert preset.seconds == seconds
        assert preset.video_latent_t == video_t
        assert preset.audio_ticks == audio_t
        assert round(preset.frames * 40 / 24) == audio_t


def test_job_path_is_bounded_relative_to_output_and_run_id_is_non_destructive(tmp_path):
    first = resolve_job_directory("temp/JR jobs", "a/b:c", 1, output_directory=tmp_path)
    second = resolve_job_directory("temp/JR jobs", "a/b:c", 2, output_directory=tmp_path)
    assert first.parent == second.parent
    assert first.name == "run_0001"
    assert second.name == "run_0002"
    assert first.is_dir() and second.is_dir()
    with pytest.raises(H3SequentialAudioError, match="must not contain"):
        resolve_job_directory("../escape", "job", 1, output_directory=tmp_path)


def test_two_chunk_job_uses_global_exact_audio_boundaries_and_final_mux_once(tmp_path, monkeypatch):
    _fake_video_io(monkeypatch, tmp_path)
    source = _audio(8.0)
    latent1, context1, seed1, slice1, status1 = _prepare(tmp_path, source)
    assert context1.chunk_index == 0
    assert context1.total_chunks == 2
    assert context1.real_frames == 141
    assert slice1["waveform"].shape[-1] == 188000
    assert seed1 == context1.seed
    assert "Same Audio Reactive Prompt" in status1
    assert latent1["custom"] is not source

    same = _prepare(tmp_path, source)
    assert same[1].chunk_index == 0
    assert same[1].generation_token == context1.generation_token

    images1 = torch.linspace(0, 1, 141 * 2 * 3 * 3).reshape(141, 2, 3, 3)
    filename1, _status, has_next = commit_decoded_chunk(
        images=images1,
        context=context1,
        quality=20,
        bit_depth="8-bit",
        audio_bitrate="192k",
        filename_prefix="video/test",
    )
    assert filename1 == ""
    assert has_next
    assert (Path(context1.job_dir) / "continuation" / "last_00000.png").is_file()

    _latent2, context2, seed2, slice2, _status2 = _prepare(tmp_path, source)
    assert context2.chunk_index == 1
    assert context2.real_frames == 51
    assert context2.source_sample_start == context1.source_sample_end
    assert context2.source_sample_end == source["waveform"].shape[-1]
    assert seed2 != seed1
    joined = torch.cat((slice1["waveform"], slice2["waveform"]), dim=-1)
    assert torch.equal(joined, source["waveform"])

    images2 = torch.ones((141, 2, 3, 3), dtype=torch.float32)
    filename2, status2, has_next2 = commit_decoded_chunk(
        images=images2,
        context=context2,
        quality=20,
        bit_depth="8-bit",
        audio_bitrate="192k",
        filename_prefix="video/test",
    )
    assert not has_next2
    assert Path(filename2).read_bytes() == b"final"
    assert "original continuous PCM muxed once" in status2
    manifest = load_manifest(Path(context2.job_dir))
    assert manifest["status"] == "complete"
    assert manifest["current_index"] == 2
    assert len(manifest["segments"]) == 2

    duplicate = commit_decoded_chunk(
        images=images2,
        context=context2,
        quality=20,
        bit_depth="8-bit",
        audio_bitrate="192k",
        filename_prefix="video/test",
    )
    assert duplicate[2] is False
    assert "duplicate" in duplicate[1].lower()


def test_manifest_fingerprint_advances_only_after_commit(tmp_path, monkeypatch):
    _fake_video_io(monkeypatch, tmp_path)
    source = _audio(7.0)
    _latent, context, _seed, _slice, _status = _prepare(tmp_path, source)
    before = manifest_fingerprint("jobs", "测试 sequence", 1, output_directory=tmp_path)
    commit_decoded_chunk(
        images=torch.zeros((141, 2, 3, 3)),
        context=context,
        quality=20,
        bit_depth="8-bit",
        audio_bitrate="192k",
        filename_prefix="video/test",
    )
    after = manifest_fingerprint("jobs", "测试 sequence", 1, output_directory=tmp_path)
    assert before != after


def test_continuation_guide_uses_initial_then_previous_terminal_frame(tmp_path, monkeypatch):
    _fake_video_io(monkeypatch, tmp_path)
    source = _audio(8.0)
    latent1, context1, *_ = _prepare(tmp_path, source)
    calls = []

    class Guide:
        @staticmethod
        def execute(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(result=(("guided", len(calls)),))

    native = SimpleNamespace(MiniMaxH3AddGuide=Guide)
    initial = torch.full((1, 2, 3, 3), 0.25)
    positive1, output1, status1 = apply_continuation_guide(
        positive=("original",),
        latent=latent1,
        context=context1,
        vae=object(),
        initial_frame=initial,
        native_module=native,
    )
    assert positive1 == ("guided", 1)
    assert output1 is latent1
    assert torch.equal(calls[-1]["image"], initial)
    assert "initial_frame" in status1

    terminal = torch.linspace(0, 1, 141 * 2 * 3 * 3).reshape(141, 2, 3, 3)
    commit_decoded_chunk(
        images=terminal,
        context=context1,
        quality=20,
        bit_depth="8-bit",
        audio_bitrate="192k",
        filename_prefix="video/test",
    )
    latent2, context2, *_ = _prepare(tmp_path, source)
    positive2, output2, status2 = apply_continuation_guide(
        positive=("original",),
        latent=latent2,
        context=context2,
        vae=object(),
        initial_frame=torch.zeros_like(initial),
        native_module=native,
    )
    assert positive2 == ("guided", 2)
    assert output2 is latent2
    assert "last_00000.png" in status2
    assert calls[-1]["frame_idx"] == 0
    assert calls[-1]["image"].shape == initial.shape


def test_independent_mv_passes_conditioning_without_guide(tmp_path):
    source = _audio(1.0)
    latent, context, *_ = _prepare(tmp_path, source, continuity="Independent MV", run_id=2)
    positive = [(torch.zeros(1), {})]
    output_positive, output_latent, status = apply_continuation_guide(
        positive=positive,
        latent=latent,
        context=context,
        vae=object(),
    )
    assert output_positive is positive
    assert output_latent is latent
    assert "Independent MV" in status


def test_checkpoint_is_atomic_safetensors_and_returns_cpu_nested_latent(tmp_path):
    latent, context, *_ = _prepare(tmp_path, _audio(1.0), continuity="Independent MV", run_id=3)
    output, status = checkpoint_sampled_latent(latent, context)
    video, audio = output["samples"].unbind()
    assert video.device.type == "cpu" and audio.device.type == "cpu"
    target = Path(context.job_dir) / "latents" / "chunk_00000.safetensors"
    assert target.is_file() and target.stat().st_size > 0
    assert "CPU-backed" in status


def test_final_mux_failure_does_not_advance_manifest(tmp_path, monkeypatch):
    _fake_video_io(monkeypatch, tmp_path)
    _latent, context, *_ = _prepare(tmp_path, _audio(1.0), continuity="Independent MV", run_id=4)

    def fail(**kwargs):
        raise RuntimeError("mux failed")

    monkeypatch.setattr(module, "_concat_and_mux", fail)
    with pytest.raises(RuntimeError, match="mux failed"):
        commit_decoded_chunk(
            images=torch.zeros((141, 2, 3, 3)),
            context=context,
            quality=20,
            bit_depth="8-bit",
            audio_bitrate="192k",
            filename_prefix="video/test",
        )
    manifest = load_manifest(Path(context.job_dir))
    assert manifest["status"] == "active"
    assert manifest["current_index"] == 0
    assert manifest["segments"] == []


def test_node_contracts_and_frontend_requeue_contract():
    driver = JR_H3_SequentialAudioChunkDriver.INPUT_TYPES()
    assert list(driver["required"]) == [
        "av_latent",
        "audio",
        "audio_vae",
        "chunk_preset",
        "continuity_mode",
        "seed_mode",
        "base_seed",
        "cache_path",
        "job_name",
        "run_id",
    ]
    assert JR_H3_SequentialAudioChunkDriver.RETURN_TYPES == (
        "LATENT",
        "JR_H3_AUDIO_CHUNK_CONTEXT",
        "INT",
        "AUDIO",
        "STRING",
    )
    assert JR_H3_SequentialContinuationGuide.RETURN_TYPES[:2] == ("CONDITIONING", "LATENT")
    assert JR_H3_SequentialLatentCheckpoint.RETURN_TYPES[0] == "LATENT"
    assert JR_H3_SequentialVideoOutput.OUTPUT_NODE is True
    script = (Path(__file__).parents[1] / "js" / "sequential_audio.js").read_text(encoding="utf-8")
    assert "execution_success" in script
    assert "api.queuePrompt(0, prompt)" in script
    assert "execution_error" in script and "execution_interrupted" in script


def test_existing_job_rejects_changed_audio_or_settings_without_overwrite(tmp_path):
    _prepare(tmp_path, _audio(1.0), continuity="Independent MV", run_id=5)
    with pytest.raises(H3SequentialAudioError, match="Increment run_id"):
        _prepare(tmp_path, _audio(1.25), continuity="Independent MV", run_id=5)
    with pytest.raises(H3SequentialAudioError, match="Increment run_id"):
        _prepare(tmp_path, _audio(1.0), continuity="Previous Last Frame", run_id=5)
