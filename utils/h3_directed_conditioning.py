"""Pure routing plus lazy adapters for MiniMax H3 native conditioning nodes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .director_pipe import DirectorPipe, RuntimeMedia, RuntimeMediaFile, validate_director_pipe

NATIVE_FPS = 24
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
MAX_REFERENCE_VIDEO_SECONDS = 15.0
MAX_DECODED_VIDEO_PIXELS = 200_000_000
MAX_AUDIO_FILE_SECONDS = 180.0
MAX_AUDIO_FILE_BYTES = 128 * 1024 * 1024
MAX_DECODED_AUDIO_SAMPLES = 40_000_000


@dataclass(frozen=True, slots=True)
class DirectedInputs:
    mode: str
    prompt: str
    width: int
    height: int
    length: int
    first_frame: Any = None
    last_frame: Any = None
    ref_images: tuple[tuple[str, Any], ...] = ()
    ref_videos: tuple[tuple[str, Any], ...] = ()
    ref_audios: tuple[tuple[str, Any], ...] = ()


def _media_map(pipe: DirectorPipe) -> dict[str, RuntimeMedia]:
    return {media.item_id: media for media in pipe.runtime_media}


def validated_runtime_media(pipe: DirectorPipe, item_id: str, family: str) -> RuntimeMedia:
    """Return one runtime payload after revalidating file-backed media."""

    media = _media_map(pipe).get(item_id)
    if media is None:
        raise ValueError(f"Director PIP {family} item {item_id!r} has no runtime media payload.")
    if isinstance(media.payload, RuntimeMediaFile):
        from .director_media import probe_asset, resolve_asset_path

        item = next(
            candidate
            for candidate in (*pipe.visual_items, *pipe.audio_items)
            if candidate.id == item_id
        )
        expected = resolve_asset_path(item.asset)
        try:
            actual = Path(media.payload.path).resolve()
        except (OSError, RuntimeError):
            raise ValueError(f"Director PIP {family} runtime path is invalid.") from None
        if actual != expected:
            raise ValueError(f"Director PIP {family} runtime path does not match its asset descriptor.")
        _, current_metadata = probe_asset(item.asset)
        original_metadata = media.metadata_dict()
        for key in ("size_bytes", "mtime_ns"):
            if key not in original_metadata:
                raise ValueError(f"Director PIP {family} runtime media is missing its file fingerprint.")
            if original_metadata[key] != current_metadata.get(key):
                raise ValueError(
                    f"Director PIP {family} asset changed after Director Desk execution; queue again."
                )
    return media


def _runtime(pipe: DirectorPipe, item_id: str, family: str) -> RuntimeMedia:
    """Backward-compatible private alias retained for focused adapter tests."""

    return validated_runtime_media(pipe, item_id, family)


def _visual_item(pipe: DirectorPipe, item_id: str):
    return next(item for item in pipe.visual_items if item.id == item_id)


def _audio_item(pipe: DirectorPipe, item_id: str):
    return next(item for item in pipe.audio_items if item.id == item_id)


def _source_window(item: Any) -> tuple[float, float]:
    if item.source_in is None:
        return 0.0, 0.0
    return float(item.source_in), float(item.source_out - item.source_in)


def _load_video(media: RuntimeMedia, item: Any, max_duration: float = MAX_REFERENCE_VIDEO_SECONDS) -> Any:
    payload = media.payload
    if isinstance(payload, RuntimeMediaFile):
        if payload.kind != "video":
            raise ValueError(f"Reference video {item.asset.display_name!r} has an invalid runtime kind.")
        start, duration = _source_window(item)
        source_duration = float(media.metadata_dict().get("duration_seconds", 0.0) or 0.0)
        if duration <= 0 and source_duration > start:
            duration = source_duration - start
        duration = min(
            duration if duration > 0 else max_duration,
            max_duration,
            MAX_REFERENCE_VIDEO_SECONDS,
        )
        metadata = media.metadata_dict()
        width = int(metadata.get("width", 0) or 0)
        height = int(metadata.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Reference video {item.asset.display_name!r} is missing bounded width/height metadata; "
                "install ffprobe and queue Director Desk again."
            )
        decoded_pixels = width * height * math.ceil(duration * NATIVE_FPS)
        if decoded_pixels > MAX_DECODED_VIDEO_PIXELS:
            raise ValueError(
                f"Reference video {item.asset.display_name!r} exceeds the bounded decode budget; "
                "trim or downscale it before conditioning."
            )
        try:
            from comfy_api.latest import VideoFromFile

            components = VideoFromFile(payload.path, start_time=start, duration=duration).get_components()
        except Exception as error:
            raise RuntimeError(
                f"Could not decode Director reference video {item.asset.display_name!r}: {type(error).__name__}."
            ) from None
        frame_rate = float(components.frame_rate)
        if not math.isclose(frame_rate, NATIVE_FPS, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError(
                f"MiniMax H3 reference video {item.asset.display_name!r} must be 24 fps; got {frame_rate:g} fps."
            )
        frames = components.images
    elif callable(getattr(payload, "get_components", None)):
        start, duration = _source_window(item)
        source_duration = float(media.metadata_dict().get("duration_seconds", 0.0) or 0.0)
        if duration <= 0 and source_duration > start:
            duration = source_duration - start
        duration = min(
            duration if duration > 0 else max_duration,
            max_duration,
            MAX_REFERENCE_VIDEO_SECONDS,
        )
        try:
            selected = payload
            if callable(getattr(payload, "as_trimmed", None)):
                selected = payload.as_trimmed(
                    start_time=start,
                    duration=duration,
                    strict_duration=False,
                )
                if selected is None:
                    raise ValueError("trim produced no video")
            components = selected.get_components()
        except Exception as error:
            raise RuntimeError(
                f"Could not decode standard VIDEO {item.asset.display_name!r}: {type(error).__name__}."
            ) from None
        frame_rate = float(components.frame_rate)
        if not math.isclose(frame_rate, NATIVE_FPS, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError(
                f"MiniMax H3 reference video {item.asset.display_name!r} must be 24 fps; got {frame_rate:g} fps."
            )
        frames = components.images
    else:
        if payload is None:
            raise ValueError(f"Reference video {item.asset.display_name!r} has no runtime payload.")
        frames = payload
    shape = getattr(frames, "shape", ())
    if len(shape) != 4 or shape[0] < 5:
        raise ValueError(
            f"MiniMax H3 reference video {item.asset.display_name!r} must decode to at least 5 IMAGE frames."
        )
    return frames


def _load_audio(media: RuntimeMedia, item: Any) -> Any:
    payload = media.payload
    if not isinstance(payload, RuntimeMediaFile):
        if not isinstance(payload, dict) or "waveform" not in payload or "sample_rate" not in payload:
            raise ValueError(f"Reference audio {item.asset.display_name!r} has no valid AUDIO payload.")
        audio = payload
    else:
        if payload.kind != "audio":
            raise ValueError(f"Reference audio {item.asset.display_name!r} has an invalid runtime kind.")
        metadata = media.metadata_dict()
        size_bytes = int(metadata.get("size_bytes", 0) or 0)
        duration_seconds = float(metadata.get("duration_seconds", 0.0) or 0.0)
        sample_rate = int(metadata.get("sample_rate", 0) or 0)
        channels = int(metadata.get("channels", 0) or 0)
        if size_bytes <= 0 or duration_seconds <= 0 or sample_rate <= 0 or channels <= 0:
            raise ValueError(
                f"Reference audio {item.asset.display_name!r} is missing bounded media metadata; "
                "install ffprobe and queue Director Desk again."
            )
        if size_bytes > MAX_AUDIO_FILE_BYTES:
            raise ValueError(
                f"Reference audio {item.asset.display_name!r} exceeds the bounded decode size; trim it first."
            )
        if duration_seconds > MAX_AUDIO_FILE_SECONDS:
            raise ValueError(
                f"Reference audio {item.asset.display_name!r} exceeds {MAX_AUDIO_FILE_SECONDS:g} seconds; "
                "trim it before conditioning."
            )
        if channels > 2 or duration_seconds * sample_rate * channels > MAX_DECODED_AUDIO_SAMPLES:
            raise ValueError(
                f"Reference audio {item.asset.display_name!r} exceeds the bounded decoded-sample budget; "
                "convert it to mono/stereo at a lower sample rate or trim it first."
            )
        try:
            from comfy_extras.nodes_audio import load

            waveform, sample_rate = load(payload.path)
        except Exception as error:
            raise RuntimeError(
                f"Could not decode Director reference audio {item.asset.display_name!r}: {type(error).__name__}."
            ) from None
        start, duration = _source_window(item)
        if duration > 0:
            begin = round(start * sample_rate)
            finish = begin + round(duration * sample_rate)
            waveform = waveform[..., begin:finish]
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    waveform = audio["waveform"]
    shape = getattr(waveform, "shape", ())
    if len(shape) != 3 or shape[0] < 1 or shape[2] < 1:
        raise ValueError(f"Reference audio {item.asset.display_name!r} has an invalid waveform shape.")
    if shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif shape[1] != 2:
        raise ValueError(f"MiniMax H3 reference audio must be mono or stereo; got {shape[1]} channels.")
    return {"waveform": waveform, "sample_rate": int(audio["sample_rate"])}


def materialize_runtime_audio(pipe: DirectorPipe, item_id: str) -> Any:
    """Return one standard AUDIO value from an in-memory or file-backed PIPE item."""

    pipe = validate_director_pipe(pipe)
    return _load_audio(
        validated_runtime_media(pipe, item_id, "Audio"),
        _audio_item(pipe, item_id),
    )


def _pipe_dimensions(pipe: DirectorPipe) -> tuple[int, int] | None:
    by_item = _media_map(pipe)
    for record in pipe.reference_registry:
        if record.family not in {"Picture", "Video"}:
            continue
        item = _visual_item(pipe, record.item_id)
        media = by_item.get(record.item_id)
        metadata = media.metadata_dict() if media is not None else {}
        width = metadata.get("width", item.asset.width)
        height = metadata.get("height", item.asset.height)
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return width, height
    return None


def _resolve_dimensions(
    pipe: DirectorPipe,
    dimension_source: str,
    width: int,
    height: int,
    length: int,
    native_module: Any,
) -> tuple[int, int, int]:
    if dimension_source not in {"Prefer Pipe", "Prefer Node"}:
        raise ValueError(f"Unsupported dimension_source: {dimension_source!r}.")
    if dimension_source == "Prefer Pipe":
        dimensions = _pipe_dimensions(pipe)
        if dimensions is not None:
            width, height = native_module.adapt_canvas(*dimensions)
        length = max(5, math.ceil(pipe.timeline.duration_seconds * NATIVE_FPS))
    maximum = int(getattr(getattr(native_module, "nodes", None), "MAX_RESOLUTION", 16384))
    for name, value in (("width", width), ("height", height)):
        if int(value) < 32 or int(value) > maximum or int(value) % 32:
            raise ValueError(f"MiniMax H3 {name} must be a multiple of 32 between 32 and {maximum}.")
    if int(length) < 5 or int(length) > 3600:
        raise ValueError("MiniMax H3 length must be between 5 and 3600 frames.")
    return int(width), int(height), int(length)


def _aligned_native_frame_count(length: int) -> int:
    frame_count = max(5, int(length))
    while frame_count % 17 != 5:
        frame_count += 1
    return frame_count


def prepare_directed_inputs(
    pipe: DirectorPipe,
    *,
    mode_override: str,
    dimension_source: str,
    width: int,
    height: int,
    length: int,
    native_module: Any,
) -> DirectedInputs:
    """Resolve mode, prompt, dimensions and real runtime media in registry order."""

    pipe = validate_director_pipe(pipe)
    prompt = pipe.final_prompt()
    if not prompt.strip():
        raise ValueError("Director PIP has no reviewed, optimized, or director prompt.")

    records = pipe.reference_registry
    reference_only = any(
        record.role in {"reference_image", "reference_video", "reference_audio", "driving_audio"}
        for record in records
    )
    anchor_records = tuple(record for record in records if record.role in {"first_frame", "last_frame"})
    if mode_override == "Auto":
        mode = "Reference to Video" if reference_only else "Image to Video"
    elif mode_override in {"Image to Video", "Reference to Video"}:
        mode = mode_override
    else:
        raise ValueError(f"Unsupported mode_override: {mode_override!r}.")
    if mode == "Image to Video" and reference_only:
        raise ValueError(
            "Image to Video conflicts with Director reference image/video/audio media; use Auto or Reference to Video."
        )
    if mode == "Reference to Video" and not records:
        raise ValueError("Reference to Video requires at least one Director reference or frame image.")

    width, height, length = _resolve_dimensions(
        pipe, dimension_source, width, height, length, native_module
    )
    if mode == "Image to Video":
        frames = {
            record.role: validated_runtime_media(pipe, record.item_id, "Picture")
            for record in anchor_records
        }
        for role, media in frames.items():
            if media is None or media.payload is None:
                raise ValueError(f"Director PIP {role} has no runtime IMAGE payload.")
            shape = getattr(media.payload, "shape", ())
            if len(shape) != 4 or shape[0] != 1:
                raise ValueError(f"Director PIP {role} must contain exactly one IMAGE.")
        return DirectedInputs(
            mode=mode,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            first_frame=frames.get("first_frame").payload if frames.get("first_frame") else None,
            last_frame=frames.get("last_frame").payload if frames.get("last_frame") else None,
        )

    pictures = [record for record in records if record.family == "Picture"]
    videos = [record for record in records if record.family == "Video"]
    audios = [record for record in records if record.family == "Audio"]
    if len(pictures) > MAX_REF_IMAGES:
        raise ValueError(f"MiniMax H3 supports at most {MAX_REF_IMAGES} reference images.")
    if len(videos) > MAX_REF_VIDEOS:
        raise ValueError(f"MiniMax H3 supports at most {MAX_REF_VIDEOS} reference videos.")
    if len(audios) > MAX_REF_AUDIOS:
        raise ValueError(f"MiniMax H3 supports at most {MAX_REF_AUDIOS} standalone reference audios.")
    ref_images = []
    for index, record in enumerate(pictures):
        media = validated_runtime_media(pipe, record.item_id, "Picture")
        if media.payload is None:
            raise ValueError(f"Director PIP {record.label} has no runtime IMAGE payload.")
        shape = getattr(media.payload, "shape", ())
        if len(shape) != 4 or shape[0] != 1:
            raise ValueError(f"Director PIP {record.label} must contain exactly one IMAGE.")
        ref_images.append((f"ref_image_{index}", media.payload))
    ref_videos = []
    for index, record in enumerate(videos):
        item = _visual_item(pipe, record.item_id)
        maximum_duration = min(
            MAX_REFERENCE_VIDEO_SECONDS,
            _aligned_native_frame_count(length) / NATIVE_FPS,
        )
        ref_videos.append((
            f"ref_video_{index}",
            _load_video(validated_runtime_media(pipe, record.item_id, "Video"), item, maximum_duration),
        ))
    ref_audios = []
    for index, record in enumerate(audios):
        item = _audio_item(pipe, record.item_id)
        ref_audios.append((
            f"ref_audio_{index}",
            _load_audio(validated_runtime_media(pipe, record.item_id, "Audio"), item),
        ))
    return DirectedInputs(
        mode=mode,
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        ref_images=tuple(ref_images),
        ref_videos=tuple(ref_videos),
        ref_audios=tuple(ref_audios),
    )


def normalize_native_output(output: Any) -> tuple[Any, Any]:
    values = getattr(output, "result", None) or getattr(output, "args", None)
    if values is None and isinstance(output, tuple):
        values = output
    if not isinstance(values, tuple) or len(values) != 2:
        raise RuntimeError("MiniMax H3 native conditioning returned an incompatible output shape.")
    return values[0], values[1]


__all__ = [
    "DirectedInputs",
    "NATIVE_FPS",
    "materialize_runtime_audio",
    "normalize_native_output",
    "prepare_directed_inputs",
    "validated_runtime_media",
]
