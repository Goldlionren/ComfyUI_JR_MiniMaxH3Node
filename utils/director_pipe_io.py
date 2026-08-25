"""Standard ComfyUI media adapters for building and inspecting Director PIPE values."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .director_pipe import (
    MAX_STAGE_PROMPT_CHARS,
    DirectorPipe,
    RuntimeMedia,
    RuntimeMediaFile,
    build_director_pipe,
    validate_director_pipe,
)
from .director_state import (
    AssetDescriptor,
    AudioState,
    DirectorState,
    DirectorUIState,
    ShotState,
    TimelineState,
    VisualState,
    canonical_time,
)
from .h3_directed_conditioning import materialize_runtime_audio, validated_runtime_media

MAX_REFERENCE_IMAGES = 9


@dataclass(frozen=True, slots=True)
class UnpackedDirectorPipe:
    pipe: DirectorPipe
    prompt: str
    director_prompt: str
    optimized_prompt: str
    reviewed_prompt: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    first_frame: Any
    last_frame: Any
    reference_image: Any
    reference_video: Any
    reference_audio: Any
    driving_audio: Any
    registry_json: str
    status: str


def _prompt(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("prompt must be text.")
    if not value.strip():
        raise ValueError("prompt must not be empty.")
    if len(value) > MAX_STAGE_PROMPT_CHARS or "\x00" in value:
        raise ValueError("prompt is too large or contains a NUL character.")
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _image_shape(value: Any, field: str, *, require_single: bool) -> tuple[int, int, int]:
    shape = tuple(getattr(value, "shape", ()))
    if len(shape) != 4 or shape[0] < 1 or shape[1] < 1 or shape[2] < 1 or shape[3] != 3:
        raise ValueError(f"{field} must be an RGB IMAGE tensor shaped [B,H,W,3].")
    if require_single and shape[0] != 1:
        raise ValueError(f"{field} must contain exactly one IMAGE.")
    return int(shape[0]), int(shape[2]), int(shape[1])


def _audio_metadata(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or "waveform" not in value or "sample_rate" not in value:
        raise ValueError(f"{field} must be a standard ComfyUI AUDIO value.")
    waveform = value["waveform"]
    shape = tuple(getattr(waveform, "shape", ()))
    if len(shape) != 3 or shape[0] != 1 or shape[1] not in {1, 2} or shape[2] < 1:
        raise ValueError(f"{field}.waveform must be shaped [1,1|2,T] with at least one sample.")
    sample_rate = value["sample_rate"]
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"{field}.sample_rate must be a positive integer.")
    return {
        "channels": int(shape[1]),
        "duration_seconds": float(shape[2]) / sample_rate,
        "sample_rate": sample_rate,
    }


def _video_metadata(value: Any) -> dict[str, Any]:
    required = ("get_components", "get_dimensions", "get_duration", "get_frame_rate")
    if any(not callable(getattr(value, name, None)) for name in required):
        raise ValueError("reference_video must be a standard ComfyUI VIDEO value.")
    try:
        width, height = value.get_dimensions()
        duration = float(value.get_duration())
        frame_rate = float(value.get_frame_rate())
    except Exception as error:
        raise ValueError(f"reference_video metadata is unavailable: {type(error).__name__}.") from None
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("reference_video dimensions must be positive integers.")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("reference_video duration must be a positive finite number.")
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("reference_video frame rate must be a positive finite number.")
    return {
        "duration_seconds": duration,
        "fps": frame_rate,
        "height": height,
        "width": width,
    }


def _asset(
    key: str,
    kind: str,
    *,
    width: int | None = None,
    height: int | None = None,
    duration_seconds: float | None = None,
) -> AssetDescriptor:
    extension = {"image": "png", "video": "mp4", "audio": "wav"}[kind]
    display_kind = kind.replace("_", " ").title()
    return AssetDescriptor(
        id=f"runtime-asset-{key}",
        kind=kind,
        filename=f"runtime-{key}.{extension}",
        subfolder="jr_h3_runtime",
        folder_type="temp",
        display_name=f"Runtime {display_kind} {key}",
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        status="probe_unavailable",
    )


def _runtime(asset: AssetDescriptor, item_id: str, payload: Any, metadata: dict[str, Any]) -> RuntimeMedia:
    return RuntimeMedia(
        asset_id=asset.id,
        item_id=item_id,
        kind=asset.kind,
        payload=payload,
        metadata=tuple(sorted(metadata.items())),
    )


def build_pipe_from_standard_inputs(
    *,
    prompt: str,
    duration_seconds: float,
    fps: float,
    first_frame: Any = None,
    last_frame: Any = None,
    reference_images: Any = None,
    reference_video: Any = None,
    reference_audio: Any = None,
    driving_audio: Any = None,
) -> DirectorPipe:
    """Build one immutable runtime-only Director PIPE from standard ComfyUI values."""

    prompt = _prompt(prompt)
    duration = canonical_time(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be greater than zero.")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise ValueError("fps must be a finite number between 1 and 240.")
    fps = float(fps)
    if not math.isfinite(fps) or fps < 1 or fps > 240:
        raise ValueError("fps must be a finite number between 1 and 240.")

    visuals: list[VisualState] = []
    audios: list[AudioState] = []
    runtime: list[RuntimeMedia] = []

    def add_image(key: str, role: str, payload: Any, order: int, start: float, end: float) -> None:
        _, width, height = _image_shape(payload, key, require_single=True)
        asset = _asset(key, "image", width=width, height=height)
        item_id = f"runtime-{key}"
        visuals.append(VisualState(
            id=item_id,
            kind="image",
            role=role,
            start=start,
            end=end,
            source_in=None,
            source_out=None,
            direction="",
            notes="",
            registry_order=order,
            asset=asset,
        ))
        runtime.append(_runtime(asset, item_id, payload, {"height": height, "width": width}))

    if first_frame is not None:
        add_image("first-frame", "first_frame", first_frame, 1, 0.0, 0.0)
    if last_frame is not None:
        add_image("last-frame", "last_frame", last_frame, 2, duration, duration)

    reference_count = 0
    if reference_images is not None:
        reference_count, _, _ = _image_shape(
            reference_images,
            "reference_images",
            require_single=False,
        )
        if reference_count > MAX_REFERENCE_IMAGES:
            raise ValueError(f"reference_images supports at most {MAX_REFERENCE_IMAGES} IMAGEs.")
        picture_total = reference_count + int(first_frame is not None) + int(last_frame is not None)
        if picture_total > MAX_REFERENCE_IMAGES:
            raise ValueError(
                "first_frame, last_frame and reference_images together exceed the native 9-Picture limit."
            )
        for index in range(reference_count):
            add_image(
                f"reference-image-{index + 1}",
                "reference_image",
                reference_images[index:index + 1],
                index + 1,
                0.0,
                duration,
            )

    if reference_video is not None:
        metadata = _video_metadata(reference_video)
        asset = _asset(
            "reference-video-1",
            "video",
            width=metadata["width"],
            height=metadata["height"],
            duration_seconds=metadata["duration_seconds"],
        )
        item_id = "runtime-reference-video-1"
        visuals.append(VisualState(
            id=item_id,
            kind="video",
            role="reference_video",
            start=0.0,
            end=duration,
            source_in=None,
            source_out=None,
            direction="",
            notes="",
            registry_order=1,
            asset=asset,
        ))
        runtime.append(_runtime(asset, item_id, reference_video, metadata))

    def add_audio(key: str, role: str, payload: Any, order: int) -> None:
        metadata = _audio_metadata(payload, key)
        asset = _asset(key, "audio", duration_seconds=metadata["duration_seconds"])
        item_id = f"runtime-{key}"
        audios.append(AudioState(
            id=item_id,
            role=role,
            start=0.0,
            end=duration,
            source_in=None,
            source_out=None,
            direction="",
            notes="",
            registry_order=order,
            asset=asset,
        ))
        runtime.append(_runtime(asset, item_id, payload, metadata))

    if reference_audio is not None:
        add_audio("reference-audio-1", "reference_audio", reference_audio, 1)
    if driving_audio is not None:
        add_audio("driving-audio-1", "driving_audio", driving_audio, 2)

    state = DirectorState(
        schema="jr_h3_director_state",
        schema_version=1,
        timeline=TimelineState(duration_seconds=duration, fps=fps),
        global_direction=prompt,
        shots=(ShotState("runtime-shot-1", 0.0, duration, "", ""),),
        visual_items=tuple(visuals),
        audio_items=tuple(audios),
        ui=DirectorUIState(),
    )
    pipe = build_director_pipe(state, runtime_resolver=lambda _state: tuple(runtime))
    return pipe.derive(optimized_prompt=prompt, reviewed_prompt="")


def _records(pipe: DirectorPipe, *, family: str, role: str | None = None) -> tuple[Any, ...]:
    return tuple(
        record
        for record in pipe.reference_registry
        if record.family == family and (role is None or record.role == role)
    )


def _selected(records: tuple[Any, ...], index: int) -> Any | None:
    return records[index - 1] if 1 <= index <= len(records) else None


def _image_for_record(pipe: DirectorPipe, record: Any | None) -> Any:
    if record is None:
        return None
    media = validated_runtime_media(pipe, record.item_id, "Picture")
    _image_shape(media.payload, record.label, require_single=True)
    return media.payload


def _video_for_record(pipe: DirectorPipe, record: Any | None) -> Any:
    if record is None:
        return None
    media = validated_runtime_media(pipe, record.item_id, "Video")
    payload = media.payload
    if isinstance(payload, RuntimeMediaFile):
        item = next(item for item in pipe.visual_items if item.id == record.item_id)
        start = float(item.source_in or 0.0)
        duration = 0.0 if item.source_out is None else float(item.source_out - start)
        try:
            from comfy_api.latest import VideoFromFile

            return VideoFromFile(payload.path, start_time=start, duration=duration)
        except ImportError:
            raise RuntimeError("Standard VIDEO output requires a current ComfyUI build.") from None
    if callable(getattr(payload, "get_components", None)):
        return payload
    shape = tuple(getattr(payload, "shape", ()))
    if len(shape) == 4 and shape[0] > 0:
        metadata = media.metadata_dict()
        raw_rate = metadata.get("fps", pipe.timeline.fps)
        try:
            frame_rate = Fraction(str(raw_rate)).limit_denominator(100_000)
            from comfy_api.latest import InputImpl, Types

            components = Types.VideoComponents(images=payload, frame_rate=frame_rate)
            return InputImpl.VideoFromComponents(components)
        except (ImportError, TypeError, ValueError, ZeroDivisionError):
            raise RuntimeError("Could not convert PIPE video frames into a standard VIDEO value.") from None
    raise ValueError(f"Director PIP {record.label} has no standard VIDEO-compatible payload.")


def _audio_for_record(pipe: DirectorPipe, record: Any | None) -> Any:
    return None if record is None else materialize_runtime_audio(pipe, record.item_id)


def _dimensions(pipe: DirectorPipe) -> tuple[int, int]:
    for record in pipe.reference_registry:
        if record.family not in {"Picture", "Video"}:
            continue
        item = next(item for item in pipe.visual_items if item.id == record.item_id)
        media = pipe.media_for_item(record.item_id)
        metadata = media.metadata_dict() if media is not None else {}
        width = metadata.get("width", item.asset.width)
        height = metadata.get("height", item.asset.height)
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return width, height
    return 0, 0


def unpack_director_pipe(
    pipe: DirectorPipe,
    *,
    reference_image_index: int,
    reference_video_index: int,
    reference_audio_index: int,
    driving_audio_index: int,
) -> UnpackedDirectorPipe:
    """Read a PIPE without mutation and expose selected standard media values."""

    pipe = validate_director_pipe(pipe)
    anchors_first = _records(pipe, family="Picture", role="first_frame")
    anchors_last = _records(pipe, family="Picture", role="last_frame")
    reference_images = _records(pipe, family="Picture", role="reference_image")
    reference_videos = _records(pipe, family="Video")
    reference_audios = _records(pipe, family="Audio", role="reference_audio")
    driving_audios = _records(pipe, family="Audio", role="driving_audio")

    selected_image = _selected(reference_images, int(reference_image_index))
    selected_video = _selected(reference_videos, int(reference_video_index))
    selected_reference_audio = _selected(reference_audios, int(reference_audio_index))
    selected_driving_audio = _selected(driving_audios, int(driving_audio_index))
    width, height = _dimensions(pipe)
    registry_json = json.dumps(
        [
            {
                "label": record.label,
                "family": record.family,
                "role": record.role,
                "start": record.start,
                "end": record.end,
                "direction": record.direction,
                "notes": record.notes,
            }
            for record in pipe.reference_registry
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    status = (
        f"Pictures={len(reference_images)} (selected {reference_image_index}: "
        f"{'yes' if selected_image else 'none'}); Videos={len(reference_videos)} "
        f"(selected {reference_video_index}: {'yes' if selected_video else 'none'}); "
        f"ReferenceAudio={len(reference_audios)} (selected {reference_audio_index}: "
        f"{'yes' if selected_reference_audio else 'none'}); DrivingAudio={len(driving_audios)} "
        f"(selected {driving_audio_index}: {'yes' if selected_driving_audio else 'none'})"
    )
    return UnpackedDirectorPipe(
        pipe=pipe,
        prompt=pipe.final_prompt(),
        director_prompt=pipe.compiled_director_prompt,
        optimized_prompt=pipe.optimized_prompt,
        reviewed_prompt=pipe.reviewed_prompt,
        duration_seconds=float(pipe.timeline.duration_seconds),
        fps=float(pipe.timeline.fps),
        width=width,
        height=height,
        first_frame=_image_for_record(pipe, anchors_first[0] if anchors_first else None),
        last_frame=_image_for_record(pipe, anchors_last[0] if anchors_last else None),
        reference_image=_image_for_record(pipe, selected_image),
        reference_video=_video_for_record(pipe, selected_video),
        reference_audio=_audio_for_record(pipe, selected_reference_audio),
        driving_audio=_audio_for_record(pipe, selected_driving_audio),
        registry_json=registry_json,
        status=status,
    )


__all__ = [
    "MAX_REFERENCE_IMAGES",
    "UnpackedDirectorPipe",
    "build_pipe_from_standard_inputs",
    "unpack_director_pipe",
]
