"""Adapt an immutable Director PIP to the existing H3 optimizer pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .director_pipe import DirectorPipe, validate_director_pipe
from .h3_prompt_builder import extract_protected_dialogues
from .h3_reference_registry import ReferenceRegistry
from .image_conversion import image_batch_to_jpeg_data_urls


@dataclass(frozen=True, slots=True)
class DirectorOptimizerContext:
    original_prompt: str
    duration_seconds: float
    registry: ReferenceRegistry
    encoded_images: tuple[tuple[str, str], ...]
    reference_instructions: str
    has_first_frame: bool
    has_last_frame: bool
    reference_image_count: int
    shot_starts: tuple[float, ...]
    protected_dialogues: tuple[str, ...]


def _media_instruction(record) -> str:
    interval = (
        f"{record.start:.1f}s point anchor"
        if record.role in {"first_frame", "last_frame"}
        else f"{record.start:.1f}s-{record.end:.1f}s"
    )
    direction = record.direction.strip() or "No additional local direction."
    notes = record.notes.strip() or "No additional notes."
    return (
        f"{record.label} is a Director Desk {record.role} active at {interval}. "
        f"Direction: {direction} Notes: {notes}"
    )


def pipe_to_optimizer_context(pipe: DirectorPipe, image_send_size: int) -> DirectorOptimizerContext:
    pipe = validate_director_pipe(pipe)
    registry = ReferenceRegistry()
    encoded = []
    instructions = []
    has_first = False
    has_last = False
    reference_images = 0
    runtime_by_item = {item.item_id: item for item in pipe.runtime_media}

    for record in pipe.reference_registry:
        source = f"director_pipe:{record.item_id}"
        if record.family == "Picture":
            registry_role = (
                record.role if record.role in {"first_frame", "last_frame"} else "reference"
            )
            entry = registry.register_picture(
                source, registry_role, source_key=record.item_id, identifier=record.label,
            )
            media = runtime_by_item.get(record.item_id)
            if media is None or media.kind != "image" or media.payload is None:
                raise ValueError(f"Director PIP image payload is missing for {record.label}.")
            urls = image_batch_to_jpeg_data_urls(media.payload, int(image_send_size))
            if len(urls) != 1:
                raise ValueError(f"Director PIP {record.label} must contain exactly one IMAGE.")
            encoded.append((entry.label, urls[0]))
            if record.role == "first_frame":
                has_first = True
            elif record.role == "last_frame":
                has_last = True
            else:
                reference_images += 1
                instructions.append(_media_instruction(record))
        elif record.family == "Video":
            registry.register_video(source, "source", source_key=record.item_id, identifier=record.label)
            instructions.append(_media_instruction(record))
        elif record.family == "Audio":
            registry.register_audio(source, "source", source_key=record.item_id, identifier=record.label)
            instructions.append(_media_instruction(record))
        else:
            raise ValueError(f"Director PIP reference family is unsupported: {record.family!r}.")

    reference_text = "\n".join(instructions)
    registry.validate_references(reference_text)
    ordered_shots = sorted(pipe.shots, key=lambda item: (item.start, item.end, item.id))
    protected_dialogues = tuple(
        literal
        for shot in ordered_shots
        for literal in extract_protected_dialogues(f"{shot.direction}\n{shot.notes}")
    )
    return DirectorOptimizerContext(
        original_prompt=pipe.compiled_director_prompt,
        duration_seconds=pipe.timeline.duration_seconds,
        registry=registry,
        encoded_images=tuple(encoded),
        reference_instructions=reference_text,
        has_first_frame=has_first,
        has_last_frame=has_last,
        reference_image_count=reference_images,
        shot_starts=tuple(shot.start for shot in ordered_shots),
        protected_dialogues=protected_dialogues,
    )


def validate_legacy_conflicts(
    pipe: DirectorPipe,
    *,
    prompt: str,
    reference_instructions: str,
    first_frame,
    last_frame,
    reference_image_count: int,
    duration_seconds: float,
) -> None:
    authoritative = pipe.compiled_director_prompt
    if str(prompt).strip() and str(prompt) != authoritative:
        raise ValueError(
            "Director PIP conflict: prompt must be empty or exactly equal to pip.compiled_director_prompt."
        )
    conflicts = []
    if not math.isclose(
        float(duration_seconds), float(pipe.timeline.duration_seconds), rel_tol=0.0, abs_tol=1e-9
    ):
        conflicts.append(
            f"duration_seconds ({float(duration_seconds):g} != {pipe.timeline.duration_seconds:g})"
        )
    if str(reference_instructions or "").strip():
        conflicts.append("reference_instructions")
    if first_frame is not None:
        conflicts.append("first_frame")
    if last_frame is not None:
        conflicts.append("last_frame")
    if reference_image_count:
        conflicts.append("ref_image_1..9")
    if conflicts:
        raise ValueError(
            "Director PIP conflict: legacy media/reference inputs cannot be combined with pip: "
            + ", ".join(conflicts)
            + "."
        )


__all__ = [
    "DirectorOptimizerContext", "pipe_to_optimizer_context", "validate_legacy_conflicts",
]
