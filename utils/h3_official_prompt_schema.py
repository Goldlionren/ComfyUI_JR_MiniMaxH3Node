"""Strict semantic JSON schema for the deterministic MiniMax H3 formatter."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Iterable

from .h3_official_resources import AUDIO_RETENTION_VALUES, VISIBLE_RETENTION_VALUES

MAX_SEMANTIC_RESPONSE_BYTES = 512 * 1024
TASK_TYPES = (
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
)
_LABEL_RE = re.compile(r"^<(Subject|Picture|Video|Audio) ([1-9]\d*)>$")
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.I | re.S)
_SPECULATIVE_PROSE_RE = re.compile(
    r"\b(?:or|either|likely|possibly|perhaps|maybe|apparently)\b|"
    r"\b(?:appears?|seems?)\s+to\b",
    re.I,
)


class H3SemanticError(ValueError):
    """The model response is not a valid H3 semantic object."""


@dataclass(frozen=True, slots=True)
class SemanticDialogue:
    literal_index: int
    speaker_key: str
    speaker_description: str
    delivery: str


@dataclass(frozen=True, slots=True)
class SemanticShot:
    description: str
    start_seconds: float | None
    dialogues: tuple[SemanticDialogue, ...]


@dataclass(frozen=True, slots=True)
class SemanticReference:
    label: str
    definition: str
    retention: str
    retention_detail: str


@dataclass(frozen=True, slots=True)
class H3SemanticPrompt:
    style: str
    shots: tuple[SemanticShot, ...]
    overall_soundscape: str
    non_diegetic_music: str
    task_types: tuple[str, ...] = ()
    summary: str = ""
    references: tuple[SemanticReference, ...] = ()


def _object(value: Any, field: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise H3SemanticError(f"{field} must be a JSON object.")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise H3SemanticError(f"{field} contains unknown field(s): {', '.join(unknown)}.")
    return value


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise H3SemanticError(f"{field} must be a JSON array.")
    return value


def _text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise H3SemanticError(f"{field} must be text.")
    result = value.strip()
    if not allow_empty and not result:
        raise H3SemanticError(f"{field} must not be empty.")
    if "\x00" in result or len(result) > 65536:
        raise H3SemanticError(f"{field} is too large or contains a NUL character.")
    return result


def _plain_semantic_text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    result = _text(value, field, allow_empty=allow_empty)
    if "<d>" in result.casefold() or "</d>" in result.casefold():
        raise H3SemanticError(f"{field} must contain semantics only, not final <d> formatting.")
    if _SPECULATIVE_PROSE_RE.search(result):
        raise H3SemanticError(
            f"{field} must be decisive and must not contain alternatives or speculative wording."
        )
    return result


def _optional_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise H3SemanticError(f"{field} must be a JSON number or null.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise H3SemanticError(f"{field} must be a finite non-negative number.")
    return result


def _parse_dialogue(value: Any, field: str) -> SemanticDialogue:
    data = _object(
        value, field,
        {"literal_index", "speaker_key", "speaker_description", "delivery"},
    )
    index = data.get("literal_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise H3SemanticError(f"{field}.literal_index must be a positive integer.")
    return SemanticDialogue(
        literal_index=index,
        speaker_key=_text(data.get("speaker_key"), f"{field}.speaker_key"),
        speaker_description=_plain_semantic_text(
            data.get("speaker_description"), f"{field}.speaker_description"
        ),
        delivery=_plain_semantic_text(
            data.get("delivery", "says"), f"{field}.delivery"
        ),
    )


def _parse_shot(value: Any, index: int) -> SemanticShot:
    field = f"shots[{index}]"
    data = _object(value, field, {"description", "start_seconds", "dialogues"})
    dialogues = tuple(
        _parse_dialogue(item, f"{field}.dialogues[{dialogue_index}]")
        for dialogue_index, item in enumerate(_array(data.get("dialogues", []), f"{field}.dialogues"))
    )
    return SemanticShot(
        description=_plain_semantic_text(data.get("description"), f"{field}.description"),
        start_seconds=_optional_number(data.get("start_seconds"), f"{field}.start_seconds"),
        dialogues=dialogues,
    )


def _normalize_reference_aliases(value: Any, index: int) -> Any:
    """Normalize only known model-produced reference aliases.

    The formatter contract always uses ``label`` and ``retention``. Some
    OpenAI-compatible models nevertheless expand those names to
    ``reference_label`` plus family-specific retention fields. Accepting that
    narrow, deterministic spelling variation is safe; conflicting canonical
    and aliased values remain an error.
    """

    if not isinstance(value, dict):
        return value
    field = f"references[{index}]"
    normalized = dict(value)

    if "reference_label" in normalized:
        alias = normalized.pop("reference_label")
        if "label" in normalized and normalized["label"] != alias:
            raise H3SemanticError(
                f"{field} has conflicting values for label and reference_label."
            )
        normalized.setdefault("label", alias)

    label = normalized.get("label")
    match = _LABEL_RE.fullmatch(label) if isinstance(label, str) else None
    relevant_alias = "audio_retention" if match and match.group(1) == "Audio" else "visible_retention"
    irrelevant_alias = "visible_retention" if relevant_alias == "audio_retention" else "audio_retention"

    if relevant_alias in normalized:
        alias = normalized.pop(relevant_alias)
        if "retention" in normalized and normalized["retention"] != alias:
            raise H3SemanticError(
                f"{field} has conflicting values for retention and {relevant_alias}."
            )
        normalized.setdefault("retention", alias)

    # Models sometimes emit both family-specific fields from a generic JSON
    # template. The field for the other media family is inapplicable, so it is
    # discarded rather than treated as a semantic value or an unknown field.
    normalized.pop(irrelevant_alias, None)
    return normalized


def _parse_reference(value: Any, index: int) -> SemanticReference:
    field = f"references[{index}]"
    value = _normalize_reference_aliases(value, index)
    data = _object(value, field, {"label", "definition", "retention", "retention_detail"})
    label = _text(data.get("label"), f"{field}.label")
    match = _LABEL_RE.fullmatch(label)
    if match is None:
        raise H3SemanticError(f"{field}.label is not a canonical H3 reference label.")
    retention = _text(data.get("retention"), f"{field}.retention")
    allowed = AUDIO_RETENTION_VALUES if match.group(1) == "Audio" else VISIBLE_RETENTION_VALUES
    if retention not in allowed:
        raise H3SemanticError(
            f"{field}.retention must be one of: {', '.join(allowed)}."
        )
    return SemanticReference(
        label=label,
        definition=_plain_semantic_text(data.get("definition"), f"{field}.definition"),
        retention=retention,
        retention_detail=_plain_semantic_text(
            data.get("retention_detail"), f"{field}.retention_detail"
        ),
    )


def parse_semantic_response(
    value: Any,
    *,
    mode: str,
    allowed_labels: Iterable[str] = (),
    protected_dialogue_count: int = 0,
    protected_dialogues: Iterable[str] = (),
    expected_shot_count: int | None = None,
) -> H3SemanticPrompt:
    """Parse and validate one model-produced semantic JSON object."""

    if not isinstance(value, str):
        raise H3SemanticError("Semantic response must be text containing one JSON object.")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise H3SemanticError("Semantic response must be valid UTF-8.") from None
    if len(encoded) > MAX_SEMANTIC_RESPONSE_BYTES:
        raise H3SemanticError("Semantic response exceeds the 512 KiB limit.")
    text = value.strip()
    fence = _FENCE_RE.fullmatch(text)
    if fence:
        text = fence.group(1).strip()
    try:
        raw = json.loads(text)
    except (JSONDecodeError, RecursionError, ValueError) as error:
        raise H3SemanticError(f"Semantic response is not valid JSON: {error}.") from None
    data = _object(
        raw, "semantic response",
        {"style", "shots", "overall_soundscape", "non_diegetic_music", "task_types", "summary", "references"},
    )
    shots = tuple(
        _parse_shot(item, index)
        for index, item in enumerate(_array(data.get("shots"), "shots"))
    )
    if not shots:
        raise H3SemanticError("shots must contain at least one semantic shot.")
    if expected_shot_count is not None and len(shots) != expected_shot_count:
        raise H3SemanticError(
            f"shots must contain exactly {expected_shot_count} entries from the Director timeline."
        )

    dialogues = [dialogue for shot in shots for dialogue in shot.dialogues]
    literal_indices = [dialogue.literal_index for dialogue in dialogues]
    expected_indices = list(range(1, protected_dialogue_count + 1))
    if sorted(literal_indices) != expected_indices:
        raise H3SemanticError(
            "dialogue literal_index values must reference every protected dialogue exactly once: "
            + (", ".join(map(str, expected_indices)) or "none")
            + "."
        )

    task_types = tuple(_text(item, f"task_types[{index}]") for index, item in enumerate(
        _array(data.get("task_types", []), "task_types")
    ))
    if len(set(task_types)) != len(task_types) or any(item not in TASK_TYPES for item in task_types):
        raise H3SemanticError("task_types contains an unknown or duplicate official task type.")
    references = tuple(
        _parse_reference(item, index)
        for index, item in enumerate(_array(data.get("references", []), "references"))
    )
    labels = tuple(item.label for item in references)
    allowed = tuple(str(item) for item in allowed_labels)
    if mode == "Ref2VA":
        if labels != allowed:
            raise H3SemanticError(
                "references must follow the exact registered label order: "
                + (", ".join(allowed) or "none")
                + "."
            )
        if not task_types:
            raise H3SemanticError("Ref2VA task_types must contain at least one official task type.")
    elif references or task_types or str(data.get("summary", "")).strip():
        raise H3SemanticError("Base modes must not return Ref2VA task_types, summary, or references.")

    semantic = H3SemanticPrompt(
        style=_plain_semantic_text(data.get("style", ""), "style", allow_empty=mode != "Ref2VA"),
        shots=shots,
        overall_soundscape=_plain_semantic_text(
            data.get("overall_soundscape"), "overall_soundscape"
        ),
        non_diegetic_music=_plain_semantic_text(
            data.get("non_diegetic_music"), "non_diegetic_music"
        ),
        task_types=task_types,
        summary=_plain_semantic_text(
            data.get("summary", ""), "summary", allow_empty=mode != "Ref2VA"
        ),
        references=references,
    )
    protected = tuple(str(value) for value in protected_dialogues)
    if protected and len(protected) != protected_dialogue_count:
        raise H3SemanticError(
            "protected_dialogues length must match protected_dialogue_count."
        )
    prose_fields = [
        ("style", semantic.style),
        ("overall_soundscape", semantic.overall_soundscape),
        ("non_diegetic_music", semantic.non_diegetic_music),
        ("summary", semantic.summary),
    ]
    for shot_index, shot in enumerate(semantic.shots):
        prose_fields.append((f"shots[{shot_index}].description", shot.description))
        for dialogue_index, dialogue in enumerate(shot.dialogues):
            prose_fields.extend(
                (
                    (
                        f"shots[{shot_index}].dialogues[{dialogue_index}].speaker_description",
                        dialogue.speaker_description,
                    ),
                    (
                        f"shots[{shot_index}].dialogues[{dialogue_index}].delivery",
                        dialogue.delivery,
                    ),
                )
            )
    for reference_index, reference in enumerate(semantic.references):
        prose_fields.extend(
            (
                (f"references[{reference_index}].definition", reference.definition),
                (f"references[{reference_index}].retention_detail", reference.retention_detail),
            )
        )
    for field, prose in prose_fields:
        for literal in protected:
            if literal and literal in prose:
                raise H3SemanticError(
                    f"{field} must not copy protected dialogue literal {literal!r}."
                )
    return semantic


def detect_dialogue_language(text: str) -> str:
    """Choose the official language tag deterministically from Unicode ranges."""

    if re.search(r"[\u3040-\u30ff]", text):
        return "Japanese"
    if re.search(r"[\uac00-\ud7af]", text):
        return "Korean"
    if re.search(r"[\u3400-\u9fff]", text):
        return "Chinese"
    return "English"


__all__ = [
    "H3SemanticError", "H3SemanticPrompt", "SemanticDialogue", "SemanticReference",
    "SemanticShot", "TASK_TYPES", "detect_dialogue_language", "parse_semantic_response",
]
