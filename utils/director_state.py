"""Schema-versioned, JSON-only state for the Director Desk editor."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping

STATE_SCHEMA = "jr_h3_director_state"
STATE_VERSION = 1
MAX_STATE_BYTES = 512 * 1024
MAX_ITEMS = 200
MAX_DIRECTION_CHARS = 32_768
MAX_GLOBAL_DIRECTION_CHARS = 65_536
MAX_TIMELINE_SECONDS = 3600.0

VISUAL_ROLES = {"reference_image", "first_frame", "reference_video"}
AUDIO_ROLES = {"reference_audio", "driving_audio"}
ASSET_KINDS = {"image", "video", "audio"}
FOLDER_TYPES = {"input", "output", "temp"}
ASSET_STATUSES = {"ready", "missing", "invalid", "probe_unavailable"}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class DirectorStateError(ValueError):
    """Raised when persisted Director Desk state is malformed or unsafe."""


@dataclass(frozen=True, slots=True)
class TimelineState:
    duration_seconds: float
    fps: float


@dataclass(frozen=True, slots=True)
class AssetDescriptor:
    id: str
    kind: str
    filename: str
    subfolder: str
    folder_type: str
    display_name: str
    mime_type: str = ""
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    status: str = "ready"


@dataclass(frozen=True, slots=True)
class ShotState:
    id: str
    start: float
    end: float
    direction: str
    notes: str


@dataclass(frozen=True, slots=True)
class VisualState:
    id: str
    kind: str
    role: str
    start: float
    end: float
    source_in: float | None
    source_out: float | None
    direction: str
    notes: str
    registry_order: int
    asset: AssetDescriptor


@dataclass(frozen=True, slots=True)
class AudioState:
    id: str
    role: str
    start: float
    end: float
    source_in: float | None
    source_out: float | None
    direction: str
    notes: str
    registry_order: int
    asset: AssetDescriptor


@dataclass(frozen=True, slots=True)
class DirectorUIState:
    selected_item_id: str | None = None
    inspector_tab: str = "item"
    zoom: float = 1.0
    visual_lane_order: tuple[str, ...] = ()
    audio_lane_order: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DirectorState:
    schema: str
    schema_version: int
    timeline: TimelineState
    global_direction: str
    shots: tuple[ShotState, ...]
    visual_items: tuple[VisualState, ...]
    audio_items: tuple[AudioState, ...]
    ui: DirectorUIState


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DirectorStateError(f"{field} must be an object.")
    return value


def _items(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise DirectorStateError(f"{field} must be an array.")
    if len(value) > MAX_ITEMS:
        raise DirectorStateError(f"{field} exceeds the {MAX_ITEMS}-item limit.")
    return value


def _identifier_order(value: Any, field: str) -> tuple[str, ...]:
    entries = _items(value, field)
    result = tuple(_identifier(item, f"{field}[{index}]") for index, item in enumerate(entries))
    if len(set(result)) != len(result):
        raise DirectorStateError(f"{field} contains duplicate item ids.")
    return result


def _text(value: Any, field: str, limit: int = MAX_DIRECTION_CHARS) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise DirectorStateError(f"{field} must be text.")
    if len(value) > limit:
        raise DirectorStateError(f"{field} exceeds the {limit}-character limit.")
    if "\x00" in value:
        raise DirectorStateError(f"{field} contains a NUL character.")
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _identifier(value: Any, field: str) -> str:
    text = _text(value, field, 128)
    if not _ID_RE.fullmatch(text):
        raise DirectorStateError(f"{field} is not a valid stable identifier.")
    return text


def _number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DirectorStateError(f"{field} must be a finite number.")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise DirectorStateError(f"{field} must be a finite number.") from None
    if not math.isfinite(result):
        raise DirectorStateError(f"{field} must be a finite number.")
    if minimum is not None and result < minimum:
        raise DirectorStateError(f"{field} must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise DirectorStateError(f"{field} must be at most {maximum}.")
    return result


def canonical_time(value: Any, field: str) -> float:
    """Return the canonical 0.1-second value shared with the frontend."""

    number = _number(value, field, minimum=0.0)
    try:
        return float(Decimal(str(number)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, OverflowError):
        raise DirectorStateError(f"{field} cannot be represented at 0.1-second precision.") from None


def _optional_time(value: Any, field: str) -> float | None:
    return None if value is None else canonical_time(value, field)


def _relative_component(value: Any, field: str, *, allow_empty: bool) -> str:
    text = _text(value, field, 1024).replace("\\", "/")
    if not text and allow_empty:
        return ""
    if not text:
        raise DirectorStateError(f"{field} must not be empty.")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise DirectorStateError(f"{field} must be a relative ComfyUI asset path.")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise DirectorStateError(f"{field} contains an unsafe path component.")
    return posix.as_posix()


def _asset(value: Any, field: str) -> AssetDescriptor:
    raw = _mapping(value, field)
    kind = _text(raw.get("kind"), f"{field}.kind", 16).lower()
    if kind not in ASSET_KINDS:
        raise DirectorStateError(f"{field}.kind must be image, video, or audio.")
    filename = _relative_component(raw.get("filename"), f"{field}.filename", allow_empty=False)
    if "/" in filename:
        raise DirectorStateError(f"{field}.filename must be a basename; use subfolder separately.")
    folder_type = _text(raw.get("type", raw.get("folder_type", "input")), f"{field}.type", 16).lower()
    if folder_type not in FOLDER_TYPES:
        raise DirectorStateError(f"{field}.type must be input, output, or temp.")
    status = _text(raw.get("status", "ready"), f"{field}.status", 32).lower()
    if status not in ASSET_STATUSES:
        raise DirectorStateError(f"{field}.status is unsupported.")
    width = raw.get("width")
    height = raw.get("height")
    return AssetDescriptor(
        id=_identifier(raw.get("id"), f"{field}.id"),
        kind=kind,
        filename=filename,
        subfolder=_relative_component(raw.get("subfolder", ""), f"{field}.subfolder", allow_empty=True),
        folder_type=folder_type,
        display_name=_text(raw.get("display_name", filename), f"{field}.display_name", 512),
        mime_type=_text(raw.get("mime_type", ""), f"{field}.mime_type", 128),
        duration_seconds=(
            None if raw.get("duration_seconds") is None
            else _number(raw["duration_seconds"], f"{field}.duration_seconds", minimum=0.0)
        ),
        width=(None if width is None else int(_number(width, f"{field}.width", minimum=1.0))),
        height=(None if height is None else int(_number(height, f"{field}.height", minimum=1.0))),
        status=status,
    )


def asset_descriptor_from_dict(value: Mapping[str, Any]) -> AssetDescriptor:
    """Parse one public ComfyUI asset descriptor without timeline state."""

    return _asset(value, "asset")


def default_director_state() -> DirectorState:
    return DirectorState(
        schema=STATE_SCHEMA,
        schema_version=STATE_VERSION,
        timeline=TimelineState(duration_seconds=10.0, fps=24.0),
        global_direction="",
        shots=(ShotState("shot-1", 0.0, 10.0, "", ""),),
        visual_items=(),
        audio_items=(),
        ui=DirectorUIState(),
    )


def director_state_from_dict(value: Mapping[str, Any]) -> DirectorState:
    if not value:
        return default_director_state()
    raw = _mapping(value, "director_state")
    schema = raw.get("schema", STATE_SCHEMA)
    version = raw.get("schema_version", STATE_VERSION)
    if schema != STATE_SCHEMA:
        raise DirectorStateError(f"Unsupported Director state schema: {schema!r}.")
    if isinstance(version, bool) or not isinstance(version, int) or version != STATE_VERSION:
        raise DirectorStateError(f"Unsupported Director state schema_version: {version!r}.")

    timeline_raw = _mapping(raw.get("timeline", {}), "timeline")
    timeline = TimelineState(
        duration_seconds=canonical_time(timeline_raw.get("duration_seconds", 10.0), "timeline.duration_seconds"),
        fps=_number(timeline_raw.get("fps", 24.0), "timeline.fps", minimum=1.0, maximum=240.0),
    )
    if timeline.duration_seconds > MAX_TIMELINE_SECONDS:
        raise DirectorStateError(
            f"timeline.duration_seconds must not exceed {MAX_TIMELINE_SECONDS:g}."
        )
    shots = []
    for index, item in enumerate(_items(raw.get("shots", []), "shots")):
        entry = _mapping(item, f"shots[{index}]")
        shots.append(ShotState(
            id=_identifier(entry.get("id"), f"shots[{index}].id"),
            start=canonical_time(entry.get("start"), f"shots[{index}].start"),
            end=canonical_time(entry.get("end"), f"shots[{index}].end"),
            direction=_text(entry.get("direction", ""), f"shots[{index}].direction"),
            notes=_text(entry.get("notes", ""), f"shots[{index}].notes"),
        ))

    visual_items = []
    for index, item in enumerate(_items(raw.get("visual_items", []), "visual_items")):
        entry = _mapping(item, f"visual_items[{index}]")
        asset = _asset(entry.get("asset"), f"visual_items[{index}].asset")
        kind = _text(entry.get("kind", asset.kind), f"visual_items[{index}].kind", 16).lower()
        role = _text(entry.get("role"), f"visual_items[{index}].role", 32).lower()
        if role not in VISUAL_ROLES:
            raise DirectorStateError(f"visual_items[{index}].role is unsupported.")
        visual_items.append(VisualState(
            id=_identifier(entry.get("id"), f"visual_items[{index}].id"),
            kind=kind,
            role=role,
            start=canonical_time(entry.get("start", 0.0), f"visual_items[{index}].start"),
            end=canonical_time(entry.get("end", 0.0), f"visual_items[{index}].end"),
            source_in=_optional_time(entry.get("source_in"), f"visual_items[{index}].source_in"),
            source_out=_optional_time(entry.get("source_out"), f"visual_items[{index}].source_out"),
            direction=_text(entry.get("direction", ""), f"visual_items[{index}].direction"),
            notes=_text(entry.get("notes", ""), f"visual_items[{index}].notes"),
            registry_order=int(_number(entry.get("registry_order", index + 1), f"visual_items[{index}].registry_order", minimum=0)),
            asset=asset,
        ))

    audio_items = []
    for index, item in enumerate(_items(raw.get("audio_items", []), "audio_items")):
        entry = _mapping(item, f"audio_items[{index}]")
        role = _text(entry.get("role"), f"audio_items[{index}].role", 32).lower()
        if role not in AUDIO_ROLES:
            raise DirectorStateError(f"audio_items[{index}].role is unsupported.")
        audio_items.append(AudioState(
            id=_identifier(entry.get("id"), f"audio_items[{index}].id"),
            role=role,
            start=canonical_time(entry.get("start", 0.0), f"audio_items[{index}].start"),
            end=canonical_time(entry.get("end", timeline.duration_seconds), f"audio_items[{index}].end"),
            source_in=_optional_time(entry.get("source_in"), f"audio_items[{index}].source_in"),
            source_out=_optional_time(entry.get("source_out"), f"audio_items[{index}].source_out"),
            direction=_text(entry.get("direction", ""), f"audio_items[{index}].direction"),
            notes=_text(entry.get("notes", ""), f"audio_items[{index}].notes"),
            registry_order=int(_number(entry.get("registry_order", index + 1), f"audio_items[{index}].registry_order", minimum=0)),
            asset=_asset(entry.get("asset"), f"audio_items[{index}].asset"),
        ))

    ui_raw = _mapping(raw.get("ui", {}), "ui")
    selected = ui_raw.get("selected_item_id")
    lane_raw = _mapping(ui_raw.get("lane_order", {}), "ui.lane_order")
    visual_ids = tuple(item.id for item in visual_items)
    audio_ids = tuple(item.id for item in audio_items)
    visual_order = _identifier_order(lane_raw.get("visual", []), "ui.lane_order.visual")
    audio_order = _identifier_order(lane_raw.get("audio", []), "ui.lane_order.audio")
    if not set(visual_order).issubset(visual_ids):
        raise DirectorStateError("ui.lane_order.visual contains an unknown visual item id.")
    if not set(audio_order).issubset(audio_ids):
        raise DirectorStateError("ui.lane_order.audio contains an unknown audio item id.")
    visual_order += tuple(item_id for item_id in visual_ids if item_id not in visual_order)
    audio_order += tuple(item_id for item_id in audio_ids if item_id not in audio_order)
    ui = DirectorUIState(
        selected_item_id=None if selected is None or selected == "" else _identifier(selected, "ui.selected_item_id"),
        inspector_tab=_text(ui_raw.get("inspector_tab", "item"), "ui.inspector_tab", 32),
        zoom=_number(ui_raw.get("zoom", 1.0), "ui.zoom", minimum=0.75, maximum=3.0),
        visual_lane_order=visual_order,
        audio_lane_order=audio_order,
    )
    if len(shots) + len(visual_items) + len(audio_items) > MAX_ITEMS:
        raise DirectorStateError(f"Director timeline exceeds the {MAX_ITEMS}-item total limit.")
    return DirectorState(
        schema=STATE_SCHEMA,
        schema_version=STATE_VERSION,
        timeline=timeline,
        global_direction=_text(raw.get("global_direction", ""), "global_direction", MAX_GLOBAL_DIRECTION_CHARS),
        shots=tuple(shots),
        visual_items=tuple(visual_items),
        audio_items=tuple(audio_items),
        ui=ui,
    )


def director_state_from_json(value: Any) -> DirectorState:
    if isinstance(value, DirectorState):
        return value
    if not isinstance(value, str):
        raise DirectorStateError("director_state_json must be UTF-8 JSON text.")
    if len(value.encode("utf-8")) > MAX_STATE_BYTES:
        raise DirectorStateError(f"director_state_json exceeds the {MAX_STATE_BYTES}-byte limit.")
    try:
        decoded = json.loads(value or "{}")
    except json.JSONDecodeError as error:
        raise DirectorStateError(f"director_state_json is invalid JSON at character {error.pos}.") from error
    except ValueError:
        raise DirectorStateError(
            "director_state_json contains an unsupported JSON value."
        ) from None
    except RecursionError:
        raise DirectorStateError("director_state_json nesting is too deep.") from None
    return director_state_from_dict(_mapping(decoded, "director_state_json"))


def asset_descriptor_to_dict(asset: AssetDescriptor) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": asset.id,
        "kind": asset.kind,
        "filename": asset.filename,
        "subfolder": asset.subfolder,
        "type": asset.folder_type,
        "display_name": asset.display_name,
        "mime_type": asset.mime_type,
        "status": asset.status,
    }
    for key in ("duration_seconds", "width", "height"):
        value = getattr(asset, key)
        if value is not None:
            result[key] = value
    return result


def director_state_to_dict(state: DirectorState, *, include_ui: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": state.schema,
        "schema_version": state.schema_version,
        "timeline": {
            "duration_seconds": state.timeline.duration_seconds,
            "fps": state.timeline.fps,
        },
        "global_direction": state.global_direction,
        "shots": [
            {"id": item.id, "start": item.start, "end": item.end, "direction": item.direction, "notes": item.notes}
            for item in state.shots
        ],
        "visual_items": [
            {
                "id": item.id, "kind": item.kind, "role": item.role,
                "start": item.start, "end": item.end,
                "source_in": item.source_in, "source_out": item.source_out,
                "direction": item.direction, "notes": item.notes,
                "registry_order": item.registry_order, "asset": asset_descriptor_to_dict(item.asset),
            }
            for item in state.visual_items
        ],
        "audio_items": [
            {
                "id": item.id, "role": item.role,
                "start": item.start, "end": item.end,
                "source_in": item.source_in, "source_out": item.source_out,
                "direction": item.direction, "notes": item.notes,
                "registry_order": item.registry_order, "asset": asset_descriptor_to_dict(item.asset),
            }
            for item in state.audio_items
        ],
    }
    if include_ui:
        result["ui"] = {
            "selected_item_id": state.ui.selected_item_id,
            "inspector_tab": state.ui.inspector_tab,
            "zoom": state.ui.zoom,
            "lane_order": {
                "visual": list(state.ui.visual_lane_order),
                "audio": list(state.ui.audio_lane_order),
            },
        }
    return result


def director_state_to_json(state: DirectorState) -> str:
    return json.dumps(director_state_to_dict(state), ensure_ascii=False, separators=(",", ":"))


DEFAULT_DIRECTOR_STATE_JSON = director_state_to_json(default_director_state())


__all__ = [
    "ASSET_KINDS", "AUDIO_ROLES", "AssetDescriptor", "AudioState",
    "DEFAULT_DIRECTOR_STATE_JSON", "DirectorState", "DirectorStateError", "DirectorUIState",
    "FOLDER_TYPES", "MAX_STATE_BYTES", "STATE_SCHEMA", "STATE_VERSION", "ShotState",
    "TimelineState", "VISUAL_ROLES", "VisualState", "asset_descriptor_to_dict",
    "asset_descriptor_from_dict",
    "canonical_time", "default_director_state", "director_state_from_dict",
    "director_state_from_json", "director_state_to_dict", "director_state_to_json",
]
