"""Disk-backed sequential audio-driven generation helpers for MiniMax H3.

The workflow deliberately advances one ComfyUI prompt at a time.  A compact
JSON manifest is the only persistent control plane; tensors and PCM remain in
separate runtime files and are never serialized into workflow JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .h3_audio_driven_latent_builder import build_h3_audio_driven_latent
from .h3_temporal_chunk_sampler import derive_chunk_seed, frame_boundary_for_video_token

ERROR_PREFIX = "JR MiniMax H3 Sequential Audio:"
SCHEMA_VERSION = 1
H3_FPS = 24
AUDIO_LATENT_FPS = 40
DEFAULT_CACHE_PATH = "temp/JR_H3_audio_jobs"
CONTINUITY_MODES = ("Previous Last Frame", "Independent MV")
SEED_MODES = ("Derived per chunk", "Fixed")


@dataclass(frozen=True, slots=True)
class ChunkPreset:
    label: str
    frames: int
    video_latent_t: int
    audio_ticks: int

    @property
    def seconds(self) -> float:
        return self.frames / H3_FPS


CHUNK_PRESETS = (
    ChunkPreset("14.375s / 345 frames / 575 ticks", 345, 102, 575),
    ChunkPreset("10.125s / 243 frames / 405 ticks", 243, 72, 405),
    ChunkPreset("8.000s / 192 frames / 320 ticks", 192, 57, 320),
    ChunkPreset("5.875s / 141 frames / 235 ticks", 141, 42, 235),
)
CHUNK_PRESET_LABELS = tuple(item.label for item in CHUNK_PRESETS)
_PRESET_BY_LABEL = {item.label: item for item in CHUNK_PRESETS}
_JOB_LOCKS: dict[str, threading.RLock] = {}
_JOB_LOCKS_GUARD = threading.Lock()


class H3SequentialAudioError(ValueError):
    """Raised when a sequential audio job cannot advance safely."""


@dataclass(frozen=True, slots=True)
class H3AudioChunkContext:
    """In-memory token carried through one chunk's sampling/decode path."""

    schema_version: int
    job_id: str
    job_dir: str
    generation_token: str
    manifest_revision: int
    chunk_index: int
    total_chunks: int
    preset_label: str
    frame_start: int
    frame_end: int
    generated_frames: int
    real_frames: int
    source_sample_start: int
    source_sample_end: int
    source_sample_rate: int
    continuity_mode: str
    seed: int


def _error(message: str) -> H3SequentialAudioError:
    return H3SequentialAudioError(f"{ERROR_PREFIX}\n{message}")


def preset_from_label(label: str) -> ChunkPreset:
    try:
        return _PRESET_BY_LABEL[str(label)]
    except KeyError:
        raise _error(f"Unknown chunk preset: {label!r}.") from None


def _safe_job_name(value: str) -> str:
    text = str(value or "audio_sequence").strip()
    text = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", text)
    text = text.strip(" ._")[:96]
    return text or "audio_sequence"


def _output_directory() -> Path:
    try:
        import folder_paths
    except ImportError:
        raise RuntimeError(f"{ERROR_PREFIX}\nComfyUI folder_paths is unavailable.") from None
    return Path(folder_paths.get_output_directory()).resolve()


def resolve_job_directory(
    cache_path: str,
    job_name: str,
    run_id: int,
    *,
    output_directory: str | Path | None = None,
    create: bool = True,
) -> Path:
    """Resolve one bounded job directory without deleting existing data."""

    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 1:
        raise _error("run_id must be an integer greater than or equal to 1.")
    output_root = Path(output_directory).resolve() if output_directory is not None else _output_directory()
    raw = str(cache_path or DEFAULT_CACHE_PATH).strip()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        cache_root = candidate.resolve()
    else:
        if ".." in candidate.parts:
            raise _error("Relative cache_path must not contain '..'.")
        cache_root = (output_root / candidate).resolve()
        try:
            cache_root.relative_to(output_root)
        except ValueError:
            raise _error("Relative cache_path resolves outside the ComfyUI output directory.") from None
    job_dir = (cache_root / _safe_job_name(job_name) / f"run_{run_id:04d}").resolve()
    try:
        job_dir.relative_to(cache_root)
    except ValueError:
        raise _error("The resolved job directory escapes cache_path.") from None
    if create:
        job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def _job_lock(job_dir: Path) -> threading.RLock:
    key = os.path.normcase(str(job_dir.resolve()))
    with _JOB_LOCKS_GUARD:
        return _JOB_LOCKS.setdefault(key, threading.RLock())


def _manifest_path(job_dir: Path) -> Path:
    return job_dir / "manifest.json"


def load_manifest(job_dir: Path) -> dict[str, Any] | None:
    path = _manifest_path(job_dir)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise _error(f"Manifest is unreadable or corrupt: {path}.") from None
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise _error(f"Unsupported sequential job manifest schema in {path}.")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _shape_text(tensor: torch.Tensor) -> str:
    return "[" + ",".join(str(int(value)) for value in tensor.shape) + "]"


def _normalise_audio(audio: Any) -> tuple[torch.Tensor, int]:
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise _error("audio must be a ComfyUI AUDIO value containing waveform and sample_rate.")
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        raise _error("audio.waveform must be a torch.Tensor.")
    if waveform.ndim != 3 or waveform.shape[0] < 1 or waveform.shape[1] not in {1, 2} or waveform.shape[2] < 1:
        raise _error(
            "audio.waveform must have shape [B,1|2,T] with a non-empty timeline; "
            f"received {_shape_text(waveform)}."
        )
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise _error("audio.sample_rate must be positive.")
    values = waveform[:1].detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(values).all().item()):
        raise _error("audio.waveform contains NaN or Inf values.")
    return values, sample_rate


def _quick_audio_hash(values: torch.Tensor, sample_rate: int) -> str:
    total = int(values.shape[-1])
    count = min(total, 2048)
    if count == total:
        sample = values
    else:
        positions = torch.linspace(0, total - 1, count, dtype=torch.float64).round().to(torch.long)
        sample = values.index_select(-1, positions)
    digest = hashlib.sha256()
    digest.update(f"{sample_rate}:{tuple(values.shape)}".encode())
    digest.update(sample.numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _interleaved_bytes(values: torch.Tensor) -> bytes:
    return (
        values[0]
        .transpose(0, 1)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
        .tobytes()
    )


def _atomic_pcm(path: Path, values: torch.Tensor) -> str:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as handle:
            # Stream the spool so long AUDIO inputs do not create a second
            # full-duration interleaved bytes object in RAM.
            samples_per_write = 1_048_576
            for start in range(0, int(values.shape[-1]), samples_per_write):
                payload = _interleaved_bytes(values[..., start : start + samples_per_write])
                handle.write(payload)
                digest.update(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def _read_pcm(path: Path, *, channels: int, start: int, end: int) -> torch.Tensor:
    if start < 0 or end <= start or channels not in {1, 2}:
        raise _error("Invalid PCM slice request.")
    byte_count = (end - start) * channels * 4
    try:
        with path.open("rb") as handle:
            handle.seek(start * channels * 4)
            payload = handle.read(byte_count)
    except OSError:
        raise _error(f"Could not read cached PCM data: {path}.") from None
    if len(payload) != byte_count:
        raise _error(f"Cached PCM data is truncated: {path}.")
    values = np.frombuffer(payload, dtype="<f4").copy().reshape(end - start, channels)
    return torch.from_numpy(values).transpose(0, 1).unsqueeze(0).contiguous()


def _resample_audio(values: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return values
    try:
        import torchaudio
    except ImportError:
        raise RuntimeError(f"{ERROR_PREFIX}\ntorchaudio is required to resample the driving audio.") from None
    return torchaudio.functional.resample(values, source_rate, target_rate).contiguous()


def _total_chunks(total_samples: int, sample_rate: int, frames_per_chunk: int) -> int:
    return max(1, math.ceil(total_samples * H3_FPS / (sample_rate * frames_per_chunk)))


def _sample_boundary(index: int, frames_per_chunk: int, sample_rate: int) -> int:
    return round(index * frames_per_chunk * sample_rate / H3_FPS)


def _initial_manifest(
    *,
    job_dir: Path,
    values: torch.Tensor,
    source_rate: int,
    vae_rate: int,
    preset: ChunkPreset,
    continuity_mode: str,
    seed_mode: str,
    base_seed: int,
) -> dict[str, Any]:
    if continuity_mode not in CONTINUITY_MODES:
        raise _error(f"Unsupported continuity_mode: {continuity_mode!r}.")
    if seed_mode not in SEED_MODES:
        raise _error(f"Unsupported seed_mode: {seed_mode!r}.")
    channels = int(values.shape[1])
    total_samples = int(values.shape[-1])
    source_pcm = job_dir / "source_audio.f32le"
    vae_pcm = job_dir / "audio_vae_32k.f32le"
    source_sha256 = _atomic_pcm(source_pcm, values)
    vae_values = _resample_audio(values, source_rate, vae_rate)
    vae_sha256 = _atomic_pcm(vae_pcm, vae_values)
    total = _total_chunks(total_samples, source_rate, preset.frames)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "job_id": f"{job_dir.parent.name}/{job_dir.name}",
        "status": "active",
        "revision": 1,
        "current_index": 0,
        "total_chunks": total,
        "active_token": None,
        "source": {
            "sample_rate": source_rate,
            "channels": channels,
            "samples": total_samples,
            "duration_seconds": total_samples / source_rate,
            "quick_hash": _quick_audio_hash(values, source_rate),
            "sha256": source_sha256,
            "pcm_file": source_pcm.name,
        },
        "vae_audio": {
            "sample_rate": vae_rate,
            "channels": channels,
            "samples": int(vae_values.shape[-1]),
            "sha256": vae_sha256,
            "pcm_file": vae_pcm.name,
        },
        "chunk": asdict(preset),
        "continuity_mode": continuity_mode,
        "seed_mode": seed_mode,
        "base_seed": int(base_seed),
        "segments": [],
        "final_output": None,
    }
    _atomic_json(_manifest_path(job_dir), manifest)
    return manifest


def _validate_manifest_inputs(
    manifest: dict[str, Any],
    values: torch.Tensor,
    source_rate: int,
    vae_rate: int,
    preset: ChunkPreset,
    continuity_mode: str,
    seed_mode: str,
    base_seed: int,
) -> None:
    expected = {
        "sample_rate": source_rate,
        "channels": int(values.shape[1]),
        "samples": int(values.shape[-1]),
        "quick_hash": _quick_audio_hash(values, source_rate),
    }
    source = manifest.get("source", {})
    mismatches = [key for key, value in expected.items() if source.get(key) != value]
    chunk = manifest.get("chunk", {})
    if chunk.get("label") != preset.label:
        mismatches.append("chunk preset")
    if manifest.get("continuity_mode") != continuity_mode:
        mismatches.append("continuity_mode")
    if manifest.get("seed_mode") != seed_mode or manifest.get("base_seed") != int(base_seed):
        mismatches.append("seed settings")
    if manifest.get("vae_audio", {}).get("sample_rate") != vae_rate:
        mismatches.append("audio VAE sample rate")
    if mismatches:
        detail = ", ".join(dict.fromkeys(mismatches))
        raise _error(
            f"Existing job settings do not match the current workflow ({detail}). "
            "Increment run_id to start a new recoverable job; existing cache files were not modified."
        )
    if manifest.get("status") == "complete":
        raise _error(
            "This sequential job is already complete. Increment run_id to start a new job; "
            "the completed output and cache remain untouched."
        )


def _template_frame_count(av_latent: Any) -> int:
    if not isinstance(av_latent, dict) or "samples" not in av_latent:
        raise _error("av_latent must be a MiniMax H3 LATENT mapping.")
    samples = av_latent["samples"]
    if not getattr(samples, "is_nested", False) or not callable(getattr(samples, "unbind", None)):
        raise _error("av_latent must contain the official MiniMax H3 AV NestedTensor.")
    streams = tuple(samples.unbind())
    if len(streams) != 2 or not isinstance(streams[0], torch.Tensor):
        raise _error("av_latent must contain exactly two H3 tensor streams.")
    return frame_boundary_for_video_token(int(streams[0].shape[2]))


def prepare_audio_chunk(
    *,
    av_latent: Any,
    audio: Any,
    audio_vae: Any,
    chunk_preset: str,
    cache_path: str,
    job_name: str,
    run_id: int,
    continuity_mode: str,
    seed_mode: str,
    base_seed: int,
    output_directory: str | Path | None = None,
) -> tuple[dict[str, Any], H3AudioChunkContext, int, dict[str, Any], str]:
    """Prepare exactly one padded audio slice and lock it into an H3 AV latent."""

    preset = preset_from_label(chunk_preset)
    if _template_frame_count(av_latent) != preset.frames:
        raise _error(
            f"Directed conditioning latent length does not match {preset.label}. "
            f"Set its length widget to {preset.frames} frames."
        )
    values, source_rate = _normalise_audio(audio)
    vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if vae_rate <= 0:
        raise _error("audio_vae.audio_sample_rate must be positive.")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or not 0 <= base_seed <= (1 << 64) - 1:
        raise _error("base_seed must be an unsigned 64-bit integer.")
    job_dir = resolve_job_directory(
        cache_path,
        job_name,
        run_id,
        output_directory=output_directory,
    )
    with _job_lock(job_dir):
        manifest = load_manifest(job_dir)
        if manifest is None:
            manifest = _initial_manifest(
                job_dir=job_dir,
                values=values,
                source_rate=source_rate,
                vae_rate=vae_rate,
                preset=preset,
                continuity_mode=continuity_mode,
                seed_mode=seed_mode,
                base_seed=base_seed,
            )
        else:
            _validate_manifest_inputs(
                manifest,
                values,
                source_rate,
                vae_rate,
                preset,
                continuity_mode,
                seed_mode,
                base_seed,
            )

        index = int(manifest["current_index"])
        total = int(manifest["total_chunks"])
        if not 0 <= index < total:
            raise _error("Manifest current_index is outside the job timeline.")
        token = manifest.get("active_token")
        if not isinstance(token, str) or not token:
            token = uuid.uuid4().hex
            manifest["active_token"] = token
            manifest["revision"] = int(manifest["revision"]) + 1
            _atomic_json(_manifest_path(job_dir), manifest)

        source = manifest["source"]
        source_start = _sample_boundary(index, preset.frames, source_rate)
        source_end = min(
            int(source["samples"]),
            _sample_boundary(index + 1, preset.frames, source_rate),
        )
        if source_end <= source_start:
            raise _error("The current source-audio slice is empty.")
        vae = manifest["vae_audio"]
        vae_start = _sample_boundary(index, preset.frames, vae_rate)
        vae_real_end = min(
            int(vae["samples"]),
            _sample_boundary(index + 1, preset.frames, vae_rate),
        )
        vae_target_samples = _sample_boundary(1, preset.frames, vae_rate)
        vae_values = _read_pcm(
            job_dir / vae["pcm_file"],
            channels=int(vae["channels"]),
            start=vae_start,
            end=vae_real_end,
        )
        if vae_values.shape[-1] < vae_target_samples:
            vae_values = torch.nn.functional.pad(
                vae_values,
                (0, vae_target_samples - int(vae_values.shape[-1])),
            )
        elif vae_values.shape[-1] > vae_target_samples:
            vae_values = vae_values[..., :vae_target_samples]

        encoded = audio_vae.encode(vae_values.movedim(1, -1))
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 4 or encoded.shape[1:3] != (32, 2):
            shape = getattr(encoded, "shape", None)
            raise _error(f"audio_vae returned an incompatible H3 audio latent: {shape}.")
        driven_latent, fit_status = build_h3_audio_driven_latent(av_latent, {"samples": encoded})
        frame_start = index * preset.frames
        frame_end = frame_start + preset.frames
        real_samples = source_end - source_start
        real_frames = min(preset.frames, max(1, math.ceil(real_samples * H3_FPS / source_rate)))
        seed = int(base_seed) if seed_mode == "Fixed" else derive_chunk_seed(int(base_seed), frame_start)
        context = H3AudioChunkContext(
            schema_version=SCHEMA_VERSION,
            job_id=str(manifest["job_id"]),
            job_dir=str(job_dir),
            generation_token=token,
            manifest_revision=int(manifest["revision"]),
            chunk_index=index,
            total_chunks=total,
            preset_label=preset.label,
            frame_start=frame_start,
            frame_end=frame_end,
            generated_frames=preset.frames,
            real_frames=real_frames,
            source_sample_start=source_start,
            source_sample_end=source_end,
            source_sample_rate=source_rate,
            continuity_mode=continuity_mode,
            seed=seed,
        )
        source_slice = _read_pcm(
            job_dir / source["pcm_file"],
            channels=int(source["channels"]),
            start=source_start,
            end=source_end,
        )
        audio_slice = {"waveform": source_slice, "sample_rate": source_rate}
        status = "\n".join(
            (
                "JR MiniMax H3 Sequential Audio Chunk Driver",
                "",
                f"Job: {manifest['job_id']}",
                f"Chunk: {index + 1}/{total}",
                f"Preset: {preset.label}",
                f"Source samples: [{source_start}:{source_end}] @ {source_rate} Hz",
                f"Real output frames: {real_frames}/{preset.frames}",
                f"Continuity: {continuity_mode}",
                f"Seed: {seed} ({seed_mode})",
                f"Cache: {job_dir}",
                "Prompt: Same Audio Reactive Prompt (shared across chunks)",
                "",
                fit_status,
            )
        )
        return driven_latent, context, seed, audio_slice, status


def manifest_fingerprint(
    cache_path: str,
    job_name: str,
    run_id: int,
    *,
    output_directory: str | Path | None = None,
) -> str:
    """Return a stable cache fingerprint that changes after each committed chunk."""

    try:
        job_dir = resolve_job_directory(
            cache_path,
            job_name,
            run_id,
            output_directory=output_directory,
            create=False,
        )
        manifest = load_manifest(job_dir)
    except Exception as error:
        return f"unresolved:{type(error).__name__}:{cache_path}:{job_name}:{run_id}"
    if manifest is None:
        return f"new:{job_dir}"
    return ":".join(
        (
            str(job_dir),
            str(manifest.get("revision")),
            str(manifest.get("current_index")),
            str(manifest.get("status")),
        )
    )


def validate_chunk_context(context: Any) -> tuple[H3AudioChunkContext, Path, dict[str, Any]]:
    if not isinstance(context, H3AudioChunkContext) or context.schema_version != SCHEMA_VERSION:
        raise _error("chunk_context is not a current JR H3 sequential audio context.")
    job_dir = Path(context.job_dir).resolve()
    manifest = load_manifest(job_dir)
    if manifest is None:
        raise _error("The chunk manifest no longer exists.")
    if manifest.get("job_id") != context.job_id:
        raise _error("chunk_context job_id does not match the manifest.")
    current = int(manifest.get("current_index", -1))
    if current != context.chunk_index:
        if current > context.chunk_index:
            return context, job_dir, manifest
        raise _error("chunk_context is ahead of the manifest and cannot be committed.")
    if manifest.get("active_token") != context.generation_token:
        raise _error("chunk_context generation token does not match the active manifest chunk.")
    return context, job_dir, manifest


def checkpoint_sampled_latent(
    sampled_latent: Any,
    context: H3AudioChunkContext,
) -> tuple[dict[str, Any], str]:
    """Persist one sampled H3 AV latent and return a CPU-backed pass-through."""

    context, job_dir, manifest = validate_chunk_context(context)
    if int(manifest.get("current_index", -1)) > context.chunk_index:
        return sampled_latent, "Chunk was already committed; checkpoint write skipped."
    if not isinstance(sampled_latent, dict) or "samples" not in sampled_latent:
        raise _error("sampled_latent must be a LATENT mapping.")
    nested = sampled_latent["samples"]
    if not getattr(nested, "is_nested", False) or not callable(getattr(nested, "unbind", None)):
        raise _error("sampled_latent must contain the official H3 AV NestedTensor.")
    streams = tuple(nested.unbind())
    if len(streams) != 2 or not all(isinstance(item, torch.Tensor) for item in streams):
        raise _error("sampled_latent must contain video and audio tensor streams.")
    video, audio = (item.detach().to("cpu").contiguous() for item in streams)
    checkpoint_dir = job_dir / "latents"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / f"chunk_{context.chunk_index:05d}.safetensors"
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        from safetensors.torch import save_file

        save_file(
            {"video": video, "audio": audio},
            str(temporary),
            metadata={
                "job_id": context.job_id,
                "chunk_index": str(context.chunk_index),
                "generation_token": context.generation_token,
            },
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    try:
        from comfy.nested_tensor import NestedTensor
    except ImportError:
        raise RuntimeError(f"{ERROR_PREFIX}\nComfyUI NestedTensor is unavailable.") from None
    output = dict(sampled_latent)
    output["samples"] = NestedTensor((video, audio))
    incoming_mask = sampled_latent.get("noise_mask")
    if getattr(incoming_mask, "is_nested", False) and callable(getattr(incoming_mask, "unbind", None)):
        masks = tuple(item.detach().to("cpu").contiguous() for item in incoming_mask.unbind())
        output["noise_mask"] = NestedTensor(masks)
    status = (
        f"Saved chunk {context.chunk_index + 1}/{context.total_chunks} sampled latent to {target}; "
        "downstream latent is CPU-backed."
    )
    return output, status


def _load_rgb_image(path: Path) -> torch.Tensor:
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except (OSError, ValueError):
        raise _error(f"Could not load continuation frame: {path}.") from None
    return torch.from_numpy(array.copy()).unsqueeze(0)


def apply_continuation_guide(
    *,
    positive: Any,
    latent: Any,
    context: H3AudioChunkContext,
    vae: Any,
    initial_frame: Any = None,
    native_module: Any = None,
) -> tuple[Any, Any, str]:
    """Anchor the initial/previous terminal frame at local frame zero."""

    context, job_dir, manifest = validate_chunk_context(context)
    if manifest.get("continuity_mode") == "Independent MV":
        return positive, latent, "Independent MV: no previous-frame guide was applied."
    if context.chunk_index == 0:
        if initial_frame is None:
            raise _error("Previous Last Frame mode requires initial_frame for chunk 1.")
        image = initial_frame
        source = "connected initial_frame"
    else:
        path = job_dir / "continuation" / f"last_{context.chunk_index - 1:05d}.png"
        if not path.is_file():
            raise _error(
                f"Previous chunk continuation frame is missing: {path}. "
                "Re-run the preceding chunk or use Independent MV mode."
            )
        image = _load_rgb_image(path)
        source = str(path)
    shape = getattr(image, "shape", ())
    if len(shape) != 4 or shape[0] < 1 or shape[-1] < 3:
        raise _error("Continuation image must be a non-empty ComfyUI IMAGE batch.")
    if native_module is None:
        try:
            from comfy_extras import nodes_minimax_h3 as native_module
        except ImportError:
            raise RuntimeError(
                f"{ERROR_PREFIX}\nCurrent ComfyUI MiniMax H3 guide implementation is unavailable."
            ) from None
    guide = getattr(native_module, "MiniMaxH3AddGuide", None)
    if not callable(getattr(guide, "execute", None)):
        raise RuntimeError(f"{ERROR_PREFIX}\nMiniMaxH3AddGuide API is incompatible.")
    result = guide.execute(
        positive=positive,
        vae=vae,
        audio_vae=None,
        latent=latent,
        image=image[:1],
        audio=None,
        frame_idx=0,
    )
    values = getattr(result, "result", None) or getattr(result, "args", None)
    if values is None and isinstance(result, tuple):
        values = result
    if not isinstance(values, tuple) or len(values) != 1:
        raise RuntimeError(f"{ERROR_PREFIX}\nMiniMaxH3AddGuide returned an incompatible output.")
    return values[0], latent, f"Applied local frame-0 continuation guide from {source}."


def _save_continuation_frame(images: torch.Tensor, path: Path) -> None:
    rgb = images[-1, ..., :3].detach().to(device="cpu", dtype=torch.float32).clamp(0, 1)
    target = np.rint(rgb.numpy() * 255).astype(np.uint8)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.png")
    try:
        Image.fromarray(target, mode="RGB").save(temporary, "PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ffmpeg_error(result: subprocess.CompletedProcess[bytes], action: str) -> RuntimeError:
    lines = result.stderr.decode("utf-8", errors="replace").splitlines()
    detail = " | ".join(lines[-8:])[-2000:] or f"return code {result.returncode}"
    return RuntimeError(f"{ERROR_PREFIX}\nFFmpeg {action} failed: {detail}")


def _probe_video(ffmpeg: str, path: Path) -> None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        command = [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height",
            "-of", "json",
            str(path),
        ]
    else:
        command = [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False)
    if result.returncode != 0 or not path.is_file() or path.stat().st_size <= 0:
        raise _ffmpeg_error(result, "segment validation")


def _encode_segment(
    images: torch.Tensor,
    path: Path,
    *,
    quality: int,
    bit_depth: int,
    required_encoder: str | None,
) -> str:
    from ..nodes.enhanced_video_combine import (
        _available_video_encoders,
        _encode_video,
        _iter_raw_chunks,
        find_ffmpeg,
    )

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(f"{ERROR_PREFIX}\nFFmpeg was not found.")
    available = _available_video_encoders(ffmpeg)
    if required_encoder:
        if required_encoder not in available:
            raise _error(f"Previously selected segment encoder {required_encoder!r} is no longer available.")
        available = {required_encoder}
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.partial.mp4")
    try:
        encoder = _encode_video(
            ffmpeg,
            available,
            "H.264",
            "MP4",
            bit_depth,
            int(images.shape[2]),
            int(images.shape[1]),
            float(H3_FPS),
            lambda: _iter_raw_chunks(images, bit_depth, False),
            temporary,
            int(quality),
            None,
            None,
            None,
            False,
            "Auto",
            "192k",
            None,
        )
        _probe_video(ffmpeg, temporary)
        os.replace(temporary, path)
        return encoder
    finally:
        temporary.unlink(missing_ok=True)


def _allocate_final_output(filename_prefix: str, width: int, height: int) -> Path:
    from ..nodes.enhanced_video_combine import _allocate_output, _output_name

    output_dir = str(_output_directory())
    folder, basename, counter, _subfolder = _allocate_output(
        filename_prefix,
        output_dir,
        width,
        height,
        filename_tag="audio_sequence",
    )
    return Path(folder) / _output_name(basename, counter, ".mp4")


def _concat_and_mux(
    *,
    job_dir: Path,
    manifest: dict[str, Any],
    output_path: Path,
    audio_bitrate: str,
) -> None:
    from ..nodes.enhanced_video_combine import find_ffmpeg

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(f"{ERROR_PREFIX}\nFFmpeg was not found.")
    segments = manifest.get("segments", [])
    if len(segments) != int(manifest.get("total_chunks", -1)):
        raise _error("Cannot finalize: not every expected segment has been committed.")
    concat_path = job_dir / "segments.ffconcat"
    lines = ["ffconcat version 1.0"]
    for item in segments:
        segment = (job_dir / str(item["file"])).resolve()
        try:
            segment.relative_to(job_dir)
        except ValueError:
            raise _error("Manifest segment path escapes the job directory.") from None
        escaped = str(segment).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    source = manifest["source"]
    pcm = (job_dir / source["pcm_file"]).resolve()
    duration = int(source["samples"]) / int(source["sample_rate"])
    temporary = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.partial.mp4")
    command = [
        ffmpeg,
        "-y",
        "-v", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_path),
        "-f", "f32le",
        "-ar", str(source["sample_rate"]),
        "-ac", str(source["channels"]),
        "-i", str(pcm),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", str(audio_bitrate),
        "-t", f"{duration:.9f}",
        "-movflags", "+faststart",
        str(temporary),
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=3600,
            check=False,
        )
        if result.returncode != 0:
            raise _ffmpeg_error(result, "final concat/mux")
        _probe_video(ffmpeg, temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def commit_decoded_chunk(
    *,
    images: Any,
    context: H3AudioChunkContext,
    quality: int,
    bit_depth: str,
    audio_bitrate: str,
    filename_prefix: str,
) -> tuple[str, str, bool]:
    """Encode, validate, atomically commit one segment, and finalize when complete."""

    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] < context.real_frames:
        shape = getattr(images, "shape", None)
        raise _error(
            f"decoded images must be [frames,H,W,C] with at least {context.real_frames} frames; received {shape}."
        )
    if images.shape[-1] < 3 or images.shape[1] < 1 or images.shape[2] < 1:
        raise _error("decoded images have invalid channel or spatial dimensions.")
    if bit_depth not in {"8-bit", "10-bit"}:
        raise _error("bit_depth must be '8-bit' or '10-bit'.")
    if not 0 <= int(quality) <= 51:
        raise _error("quality must be between 0 and 51.")
    context, job_dir, manifest = validate_chunk_context(context)
    with _job_lock(job_dir):
        manifest = load_manifest(job_dir)
        if manifest is None:
            raise _error("The chunk manifest no longer exists.")
        current = int(manifest.get("current_index", -1))
        if current > context.chunk_index:
            final = manifest.get("final_output") or ""
            return str(final), "Chunk was already committed; duplicate finalizer call ignored.", False
        if current != context.chunk_index or manifest.get("active_token") != context.generation_token:
            raise _error("The finalizer received a stale or mismatched chunk context.")

        segment_dir = job_dir / "segments"
        continuation_dir = job_dir / "continuation"
        segment_dir.mkdir(parents=True, exist_ok=True)
        continuation_dir.mkdir(parents=True, exist_ok=True)
        selected = images[: context.real_frames]
        segment_path = segment_dir / f"segment_{context.chunk_index:05d}.mp4"
        required_encoder = manifest.get("segment_encoder")
        encoder = _encode_segment(
            selected,
            segment_path,
            quality=int(quality),
            bit_depth=10 if bit_depth == "10-bit" else 8,
            required_encoder=required_encoder if isinstance(required_encoder, str) else None,
        )
        if required_encoder is not None and encoder != required_encoder:
            raise _error("Segment encoder changed inside one job; concat safety cannot be guaranteed.")
        _save_continuation_frame(
            selected,
            continuation_dir / f"last_{context.chunk_index:05d}.png",
        )
        relative_segment = segment_path.relative_to(job_dir).as_posix()
        segments = list(manifest.get("segments", []))
        if len(segments) != context.chunk_index:
            raise _error("Manifest segment order is inconsistent; no files were deleted.")
        segments.append(
            {
                "index": context.chunk_index,
                "file": relative_segment,
                "frames": context.real_frames,
                "encoder": encoder,
                "width": int(images.shape[2]),
                "height": int(images.shape[1]),
            }
        )
        candidate_manifest = dict(manifest)
        candidate_manifest["segments"] = segments
        candidate_manifest["segment_encoder"] = encoder
        candidate_manifest["current_index"] = context.chunk_index + 1
        candidate_manifest["active_token"] = None
        candidate_manifest["revision"] = int(manifest["revision"]) + 1
        has_next = candidate_manifest["current_index"] < int(candidate_manifest["total_chunks"])
        filename = ""
        if has_next:
            candidate_manifest["status"] = "active"
            _atomic_json(_manifest_path(job_dir), candidate_manifest)
        else:
            output_path = _allocate_final_output(
                filename_prefix,
                int(images.shape[2]),
                int(images.shape[1]),
            )
            _concat_and_mux(
                job_dir=job_dir,
                manifest=candidate_manifest,
                output_path=output_path,
                audio_bitrate=audio_bitrate,
            )
            candidate_manifest["status"] = "complete"
            candidate_manifest["final_output"] = str(output_path)
            candidate_manifest["revision"] = int(candidate_manifest["revision"]) + 1
            _atomic_json(_manifest_path(job_dir), candidate_manifest)
            filename = str(output_path)

        status = "\n".join(
            (
                "JR MiniMax H3 Sequential Video Output",
                f"Committed chunk: {context.chunk_index + 1}/{context.total_chunks}",
                f"Segment: {segment_path}",
                f"Encoder: {encoder}",
                f"Frames: {context.real_frames} @ {H3_FPS} fps",
                f"Next chunk: {'yes' if has_next else 'no'}",
                f"Final output: {filename or 'pending'}",
                "Audio: original continuous PCM muxed once at finalization",
            )
        )
        return filename, status, has_next


__all__ = [
    "AUDIO_LATENT_FPS",
    "CHUNK_PRESETS",
    "CHUNK_PRESET_LABELS",
    "CONTINUITY_MODES",
    "DEFAULT_CACHE_PATH",
    "ERROR_PREFIX",
    "H3AudioChunkContext",
    "H3SequentialAudioError",
    "SCHEMA_VERSION",
    "SEED_MODES",
    "apply_continuation_guide",
    "checkpoint_sampled_latent",
    "commit_decoded_chunk",
    "load_manifest",
    "manifest_fingerprint",
    "prepare_audio_chunk",
    "preset_from_label",
    "resolve_job_directory",
    "validate_chunk_context",
]
