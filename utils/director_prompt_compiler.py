"""Pure validation, registry, and prompt compilation for Director Desk."""

from __future__ import annotations

from dataclasses import dataclass

from .director_state import (
    AudioState,
    DirectorState,
    DirectorStateError,
    VisualState,
    director_state_from_dict,
    director_state_to_dict,
)


class DirectorValidationError(ValueError):
    """Raised when a timeline cannot be compiled deterministically."""


@dataclass(frozen=True, slots=True)
class ReferenceRecord:
    label: str
    family: str
    item_id: str
    asset_id: str
    role: str
    start: float
    end: float
    direction: str
    notes: str


def _ranges_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return start_a < end_b and start_b < end_a


def _validate_source_range(item: VisualState | AudioState, field: str) -> None:
    source_in, source_out = item.source_in, item.source_out
    if (source_in is None) != (source_out is None):
        raise DirectorValidationError(f"{field} source_in and source_out must be set together.")
    if source_in is None:
        return
    if source_out <= source_in:
        raise DirectorValidationError(f"{field} source_out must be greater than source_in.")
    duration = item.asset.duration_seconds
    if duration is not None and source_out > duration + 1e-9:
        raise DirectorValidationError(
            f"{field} source_out {source_out:.1f}s exceeds source duration {duration:.3f}s."
        )


def validate_director_state(state: DirectorState) -> None:
    """Validate timeline semantics without reading files or invoking media tools."""

    if not isinstance(state, DirectorState):
        raise DirectorValidationError("Director state must be a parsed DirectorState value.")
    try:
        canonical = director_state_from_dict(director_state_to_dict(state))
    except DirectorStateError as error:
        raise DirectorValidationError(str(error)) from error
    if (
        director_state_to_dict(canonical, include_ui=False)
        != director_state_to_dict(state, include_ui=False)
    ):
        raise DirectorValidationError(
            "Director state contains non-canonical values; parse persisted state before compilation."
        )

    duration = state.timeline.duration_seconds
    if duration <= 0:
        raise DirectorValidationError("Timeline duration_seconds must be greater than zero.")
    if state.timeline.fps <= 0:
        raise DirectorValidationError("Timeline fps must be greater than zero.")
    if not state.shots:
        raise DirectorValidationError("Director Desk requires at least one Shot.")

    item_ids: set[str] = set()
    ordered_shots = sorted(state.shots, key=lambda item: (item.start, item.end, item.id))
    previous = None
    for index, shot in enumerate(ordered_shots, 1):
        if shot.id in item_ids:
            raise DirectorValidationError(f"Duplicate timeline item id: {shot.id}.")
        item_ids.add(shot.id)
        if shot.end <= shot.start:
            raise DirectorValidationError(f"Shot {index} end must be greater than start.")
        if shot.end > duration:
            raise DirectorValidationError(f"Shot {index} ends after the timeline duration.")
        if previous is not None and shot.start < previous.end:
            raise DirectorValidationError(
                f"Shots {previous.id} and {shot.id} overlap; Shot intervals may only touch."
            )
        previous = shot

    first_frames = []
    for index, item in enumerate(state.visual_items, 1):
        field = f"Visual item {item.id or index}"
        if item.id in item_ids:
            raise DirectorValidationError(f"Duplicate timeline item id: {item.id}.")
        item_ids.add(item.id)
        if item.kind != item.asset.kind:
            raise DirectorValidationError(f"{field} kind does not match its asset kind.")
        if item.role == "first_frame":
            first_frames.append(item)
            if item.kind != "image":
                raise DirectorValidationError("First Frame must use an IMAGE asset.")
            if item.start != 0.0 or item.end != 0.0:
                raise DirectorValidationError("First Frame is a point marker fixed at 0.0 seconds.")
        else:
            expected = "image" if item.role == "reference_image" else "video"
            if item.kind != expected:
                raise DirectorValidationError(f"{field} role requires a {expected.upper()} asset.")
            if item.end <= item.start:
                raise DirectorValidationError(f"{field} end must be greater than start.")
            if item.end > duration:
                raise DirectorValidationError(f"{field} ends after the timeline duration.")
        if item.kind == "video":
            _validate_source_range(item, field)
        elif item.source_in is not None or item.source_out is not None:
            raise DirectorValidationError(f"{field} IMAGE assets cannot define a source range.")
        if item.asset.status in {"missing", "invalid"}:
            raise DirectorValidationError(f"{field} asset is {item.asset.status}: {item.asset.display_name}.")
    if len(first_frames) > 1:
        raise DirectorValidationError("Only one First Frame may exist in a Director Desk timeline.")
    if first_frames and ordered_shots[0].start != 0.0:
        raise DirectorValidationError("A timeline with a First Frame must have its first Shot start at 0.0 seconds.")

    driving = []
    for index, item in enumerate(state.audio_items, 1):
        field = f"Audio item {item.id or index}"
        if item.id in item_ids:
            raise DirectorValidationError(f"Duplicate timeline item id: {item.id}.")
        item_ids.add(item.id)
        if item.asset.kind != "audio":
            raise DirectorValidationError(f"{field} requires an AUDIO asset.")
        if item.end <= item.start:
            raise DirectorValidationError(f"{field} end must be greater than start.")
        if item.end > duration:
            raise DirectorValidationError(f"{field} ends after the timeline duration.")
        _validate_source_range(item, field)
        if item.asset.status in {"missing", "invalid"}:
            raise DirectorValidationError(f"{field} asset is {item.asset.status}: {item.asset.display_name}.")
        if item.role == "driving_audio":
            driving.append(item)

    driving.sort(key=lambda item: (item.start, item.end, item.id))
    for left, right in zip(driving, driving[1:]):
        if _ranges_overlap(left.start, left.end, right.start, right.end):
            raise DirectorValidationError(
                f"Driving Audio items {left.id} and {right.id} overlap; the active driving source is ambiguous."
            )


def build_reference_registry(state: DirectorState) -> tuple[ReferenceRecord, ...]:
    """Build stable labels independent of visual lane layout or item array order."""

    validate_director_state(state)
    pictures = sorted(
        (item for item in state.visual_items if item.kind == "image"),
        key=lambda item: (0 if item.role == "first_frame" else 1, item.registry_order, item.id),
    )
    videos = sorted(
        (item for item in state.visual_items if item.kind == "video"),
        key=lambda item: (item.registry_order, item.id),
    )
    audios = sorted(state.audio_items, key=lambda item: (item.registry_order, item.id))
    records = []
    for family, items in (("Picture", pictures), ("Video", videos), ("Audio", audios)):
        for index, item in enumerate(items, 1):
            records.append(ReferenceRecord(
                label=f"<{family} {index}>", family=family, item_id=item.id,
                asset_id=item.asset.id, role=item.role, start=item.start, end=item.end,
                direction=item.direction, notes=item.notes,
            ))
    return tuple(records)


def _time(value: float) -> str:
    return f"{value:.1f}s"


def _interval(record: ReferenceRecord) -> str:
    if record.role == "first_frame":
        return "0.0s point anchor"
    return f"{_time(record.start)}-{_time(record.end)}"


def _text_block(lines: list[str], title: str, value: str, indent: str = "") -> None:
    lines.append(f"{indent}{title}:")
    if value.strip():
        lines.extend(f"{indent}  {line}" for line in value.splitlines())
    else:
        lines.append(f"{indent}  (none)")


def compile_director_prompt(
    state: DirectorState,
    registry: tuple[ReferenceRecord, ...] | None = None,
) -> str:
    """Compile a byte-stable raw Director Prompt from validated state."""

    validate_director_state(state)
    records = registry if registry is not None else build_reference_registry(state)
    by_item = {record.item_id: record for record in records}
    if len(by_item) != len(records):
        raise DirectorValidationError("Reference registry contains a duplicate item_id.")

    lines = [
        "GLOBAL DIRECTION",
        f"timeline_duration: {_time(state.timeline.duration_seconds)}",
        f"timeline_fps: {state.timeline.fps:g}",
    ]
    _text_block(lines, "direction", state.global_direction)
    lines.extend(["", "REFERENCE MEDIA"])
    if not records:
        lines.append("(none)")
    else:
        all_items = {item.id: item for item in (*state.visual_items, *state.audio_items)}
        for record in records:
            item = all_items[record.item_id]
            asset = item.asset
            lines.append(
                f"{record.label} | role={record.role} | timeline={_interval(record)} | "
                f"asset={asset.display_name} | item_id={record.item_id}"
            )
            if item.source_in is not None:
                lines.append(f"  source_range: {_time(item.source_in)}-{_time(item.source_out)}")
            _text_block(lines, "direction", record.direction, "  ")
            _text_block(lines, "notes", record.notes, "  ")

    lines.extend(["", "TIMELINE"])
    ordered_shots = sorted(state.shots, key=lambda item: (item.start, item.end, item.id))
    media_records = tuple(records)
    for shot_index, shot in enumerate(ordered_shots, 1):
        lines.append(f"[Shot {shot_index} | {_time(shot.start)}-{_time(shot.end)} | id={shot.id}]")
        _text_block(lines, "direction", shot.direction, "  ")
        _text_block(lines, "notes", shot.notes, "  ")
        active = [
            record for record in media_records
            if (record.role == "first_frame" and shot_index == 1)
            or _ranges_overlap(shot.start, shot.end, record.start, record.end)
        ]
        lines.append("  active_references:")
        if active:
            for record in active:
                lines.append(f"    - {record.label} ({record.role}, {_interval(record)})")
        else:
            lines.append("    - (none)")

    lines.extend(["", "END STATE", f"final_time: {_time(state.timeline.duration_seconds)}"])
    last = ordered_shots[-1]
    lines.append(f"final_shot: {last.id} ({_time(last.start)}-{_time(last.end)})")
    return "\n".join(lines).rstrip()


__all__ = [
    "DirectorValidationError", "ReferenceRecord", "build_reference_registry",
    "compile_director_prompt", "validate_director_state",
]
