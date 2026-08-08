"""Deterministic cleanup and validation for MiniMax H3 prompts.

The validator intentionally performs only structural checks.  It does not try to
rewrite a prompt or infer missing content; this makes it suitable for a final
static gate after an optimiser has produced text.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .h3_official_resources import (
    AUDIO_RETENTION_VALUES,
    BASE_SECTION_ORDER,
    REF_SECTION_ORDER,
    VISIBLE_RETENTION_VALUES,
)

BASE_MODES = ("T2VA", "I2VA", "FL2VA", "L2VA")
REF_MODE = "Ref2VA"
MODES = BASE_MODES + (REF_MODE,)

AUDIO_VALUES = AUDIO_RETENTION_VALUES

_PREFIX_RE = re.compile(
    r"^(?:Final Answer:|Answer:|Here is the prompt:|Optimized prompt:)[ \t]*(?:\r?\n[ \t]*)?",
    re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_MARKER_RE = re.compile(r"</?think\b", re.IGNORECASE)
_SECTION_NAMES = frozenset(BASE_SECTION_ORDER + REF_SECTION_ORDER)
_SHOT_RE = re.compile(r"\[Shot (?P<number>\d+)\](?P<rest>.*?)(?=\[Shot \d+\]|\Z)", re.DOTALL)
_SHOT_MARKER_RE = re.compile(r"\[Shot\b", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(r"^ At (?P<minute>\d{2}):(?P<second>\d{2})\.(?P<millis>\d{3}),", re.ASCII)
_TIMESTAMP_ANY_RE = re.compile(r"\b\d{2}:\d{2}\.\d{3}\b")
_TAG_RE = re.compile(r"<(?P<inner>[^<>\r\n]+)>")
_OPEN_TAG_RE = re.compile(
    r"<(?P<name>Picture|Subject|Video|Audio|Image|Reference|Ref)\b[^<>\r\n]*(?:>|$)",
    re.IGNORECASE | re.MULTILINE,
)
_EXACT_TAG_RE = re.compile(r"^<(Picture|Subject|Video|Audio) ([1-9]\d*)>$")
_TAG_INNER_RE = re.compile(r"^(Picture|Subject|Video|Audio)\s+([1-9]\d*)$", re.ASCII)
_RETENTION_VALUE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_VALUE_SEPARATOR_RE = re.compile(r"(?:[:=]|\s+[-–—>]\s*|\s+→\s*)")
_FIELD_VALUE_RE = re.compile(
    r"(?:[:=]|\s+(?:->|[-–—]|→)\s*)([A-Za-z][A-Za-z0-9_-]*)"
)
_VISIBLE_RETENTION_LINE_RE = re.compile(
    r"^\s*<(?P<family>Subject|Picture|Video) [1-9]\d*>\s*(?:\([^\r\n:]*\))?\s*:\s*(?P<value>[A-Za-z][A-Za-z0-9_-]*)",
    re.ASCII,
)
_AUDIO_RETENTION_LINE_RE = re.compile(
    r"^\s*<Audio [1-9]\d*>\s*(?:\([^\r\n:]*\))?\s*:\s*(?P<value>[A-Za-z][A-Za-z0-9_-]*)",
    re.ASCII,
)
_SUBJECT_DEFINITION_RE = re.compile(r"^\s*<Subject ([1-9]\d*)>\s+is\b", re.MULTILINE)


@dataclass(frozen=True)
class ValidationResult:
    """Result returned by :func:`validate_prompt`.

    ``cleaned_prompt`` is always included so callers can display exactly what
    was checked.  ``errors`` is a tuple of descriptive, deterministic strings.
    """

    cleaned_prompt: str
    valid: bool
    errors: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.valid


def _remove_outer_fence(text: str) -> tuple[str, bool]:
    """Remove one complete outer Markdown fence, if present."""

    lines = text.splitlines()
    if len(lines) < 2:
        return text, False
    opening = lines[0].strip()
    closing = lines[-1].strip()
    if not re.fullmatch(r"```(?:[A-Za-z0-9_+.-]+)?", opening):
        return text, False
    if closing != "```":
        return text, False
    return "\n".join(lines[1:-1]), True


def _remove_prefixes(text: str) -> str:
    # Prefixes are wrappers and can occasionally be repeated by a provider.
    # Removing only these fixed strings is deliberately conservative.
    while True:
        match = _PREFIX_RE.match(text)
        if not match:
            return text
        text = text[match.end() :]


def cleanup_prompt(prompt: Any) -> str:
    """Remove permitted transport wrappers without changing prompt semantics.

    The function removes surrounding whitespace, one outer Markdown fence,
    complete ``<think>...</think>`` blocks, and the documented leading answer
    prefixes.  Interior content, including reference markers and literals, is
    left byte-for-byte unchanged.
    """

    text = "" if prompt is None else str(prompt)
    text = text.strip()

    # A provider may put a prefix before a fence.  Track whether a fence was
    # already removed so nested fences are not silently normalised away.
    fence_removed = False
    text, removed = _remove_outer_fence(text)
    fence_removed = fence_removed or removed
    text = _THINK_RE.sub("", text).strip()
    text = _remove_prefixes(text).strip()
    text = _THINK_RE.sub("", text).strip()
    if not fence_removed:
        text, removed = _remove_outer_fence(text)
        fence_removed = fence_removed or removed
        text = _THINK_RE.sub("", text).strip()
    # A prefix can become visible after removing a think block.  It is still a
    # permitted leading wrapper, so remove it once more.
    text = _remove_prefixes(text).strip()
    return text


def _normalise_mode(mode: Any) -> str | None:
    if mode is None:
        return None
    value = str(mode).strip()
    for candidate in MODES:
        if value.casefold() == candidate.casefold():
            return candidate
    return value


def _section_order_for(mode: str) -> tuple[str, ...]:
    return REF_SECTION_ORDER if mode == REF_MODE else BASE_SECTION_ORDER


def _section_spans(text: str, expected: tuple[str, ...]) -> tuple[list[tuple[str, int, int, str]], list[str]]:
    """Return ``(name, start, end, body)`` entries and structural errors."""

    found: list[tuple[str, int, int, str]] = []
    errors: list[str] = []
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    # splitlines() returns no line for an empty string; no section can exist.
    for index, line in enumerate(lines):
        raw = line.rstrip("\r\n")
        for name in expected:
            # The official field names are exact lowercase identifiers followed
            # by ``:``.  A body may start on the same line (for example,
            # ``integrated_multimodal_description: [Shot 1] ...``).
            if re.match(re.escape(name) + r":(?:[ \t].*)?$", raw):
                start = offsets[index]
                found.append((name, start, -1, ""))
                break

    for position, item in enumerate(found):
        name, start, _, _ = item
        end = found[position + 1][1] if position + 1 < len(found) else len(text)
        body = text[start + len(name) + 1 : end]
        found[position] = (name, start, end, body)

    positions = {name: [] for name in expected}
    for name, _, _, body in found:
        positions[name].append(body)
    for name in expected:
        count = len(positions[name])
        if count == 0:
            errors.append(f"missing required section '{name}'")
        elif count > 1:
            errors.append(f"duplicate section '{name}' ({count} occurrences)")
    if found:
        actual = [name for name, _, _, _ in found]
        # Compare the first occurrence sequence so duplicate diagnostics remain
        # useful while still reporting order errors.
        first_seen: list[str] = []
        for name in actual:
            if name not in first_seen:
                first_seen.append(name)
        if first_seen != [name for name in expected if name in first_seen]:
            errors.append(
                "sections out of order: expected "
                + ", ".join(expected)
                + "; found "
                + ", ".join(actual)
            )
    for name, _, _, body in found:
        if not body.strip():
            errors.append(f"section '{name}' is empty")
    return found, errors


def _as_label_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return list(value.items())
    try:
        return list(value)
    except TypeError:
        return [value]


def _canonical_tag(value: Any, family_hint: str | None = None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    exact = _EXACT_TAG_RE.fullmatch(text)
    if exact:
        return f"<{exact.group(1)} {int(exact.group(2))}>"
    if family_hint and re.fullmatch(r"[1-9]\d*", text):
        return f"<{family_hint.title()} {int(text)}>"
    bare = _TAG_INNER_RE.fullmatch(text)
    if bare:
        return f"<{bare.group(1)} {int(bare.group(2))}>"
    return None


def _registry_from_values(*values: Any) -> tuple[set[str], list[str]]:
    """Canonicalise supplied labels and return ``(labels, duplicate_errors)``."""

    labels: set[str] = set()
    errors: list[str] = []

    def add(value: Any, family_hint: str | None = None) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_tag = _canonical_tag(key)
                if key_tag:
                    add(key_tag)
                    # A mapping from label to metadata has no child labels.
                    continue
                family = str(key).strip().title()
                if family.casefold() in {"picture", "subject", "video", "audio"}:
                    for item in _as_label_items(child):
                        add(item, family)
                else:
                    # Unknown mapping keys are treated as labels rather than
                    # making the validator depend on a particular registry API.
                    add(key)
            return
        if isinstance(value, str):
            tag = _canonical_tag(value, family_hint)
            if tag is not None:
                if tag in labels:
                    errors.append(f"duplicate allowed reference label '{tag}'")
                labels.add(tag)
            return
        if isinstance(value, Iterable):
            for item in value:
                add(item, family_hint)
            return
        tag = _canonical_tag(value, family_hint)
        if tag is not None:
            if tag in labels:
                errors.append(f"duplicate allowed reference label '{tag}'")
            labels.add(tag)

    for value in values:
        add(value)
    return labels, errors


def _extract_alias(kwargs: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in kwargs:
            return kwargs[name]
    return None


def _reference_errors(
    text: str,
    mode: str,
    allowed_labels: set[str],
    section_spans: list[tuple[str, int, int, str]],
) -> list[str]:
    errors: list[str] = []
    tags: list[tuple[str, int, int]] = []
    for match in _TAG_RE.finditer(text):
        token = match.group(0)
        inner = match.group("inner")
        family = re.match(r"^(Picture|Subject|Video|Audio)", inner, re.IGNORECASE)
        alias = re.match(r"^(Image|Reference|Ref)", inner, re.IGNORECASE)
        if not family and not alias:
            continue
        if not _EXACT_TAG_RE.fullmatch(token):
            errors.append(
                f"invalid reference syntax '{token}'; use '<Picture N>', '<Subject N>', '<Video N>', or '<Audio N>'"
            )
            continue
        canonical = _canonical_tag(token)
        assert canonical is not None
        tags.append((canonical, match.start(), match.end()))
        family_name = canonical[1:].split(" ", 1)[0]
        if mode != REF_MODE and family_name != "Picture":
            errors.append(f"reference label '{canonical}' is not allowed in {mode}; base modes only resolve Picture labels")
        elif family_name != "Subject" and canonical not in allowed_labels:
            errors.append(f"unresolved reference label '{canonical}'")

    # Detect malformed/open reference-like tags (for example ``<Picture 1`` or
    # ``<picture 1>``) that the closed-tag scan cannot see.
    for match in _OPEN_TAG_RE.finditer(text):
        token = match.group(0)
        if token.endswith(">") and _EXACT_TAG_RE.fullmatch(token):
            continue
        # Closed malformed tags have already been reported above.
        if ">" in token:
            continue
        errors.append(
            f"invalid reference syntax '{token}'; use '<Picture N>', '<Subject N>', '<Video N>', or '<Audio N>'"
        )

    if mode == REF_MODE:
        subject_section = next((item for item in section_spans if item[0] == "subject_definitions"), None)
        subject_start = subject_section[1] + len("subject_definitions:") if subject_section else -1
        definitions: dict[str, list[int]] = {}
        if subject_section:
            # Only a line-leading ``<Subject N> is ...`` introduces a subject.
            # References embedded in descriptive prose are uses, not duplicate
            # definitions.
            for definition in _SUBJECT_DEFINITION_RE.finditer(subject_section[3]):
                start = subject_start + definition.start()
                canonical = f"<Subject {int(definition.group(1))}>"
                definitions.setdefault(canonical, []).append(start)
        for canonical, start, _ in tags:
            if not canonical.startswith("<Subject "):
                continue
            definition_positions = definitions.get(canonical, [])
            # A marker at the exact start of a definition is the declaration;
            # every other occurrence must follow at least one declaration.
            if start in definition_positions:
                continue
            if not any(position < start for position in definition_positions):
                errors.append(f"subject reference '{canonical}' is used before definition")
        for canonical, positions in definitions.items():
            if len(positions) > 1:
                errors.append(f"duplicate subject definition '{canonical}' ({len(positions)} occurrences)")
    return errors


def _taxonomy_errors(text: str, mode: str, section_spans: list[tuple[str, int, int, str]]) -> list[str]:
    if mode != REF_MODE:
        return []
    errors: list[str] = []
    retention_section = next((item for item in section_spans if item[0] == "retention_analysis"), None)
    if retention_section:
        body = retention_section[3]
        for line in body.splitlines():
            if not line.strip():
                continue
            visible_match = _VISIBLE_RETENTION_LINE_RE.match(line)
            audio_match = _AUDIO_RETENTION_LINE_RE.match(line)
            field_match = re.match(
                r"^\s*(?:visible_)?retention(?:_analysis|_status|_value)?\s*:\s*"
                r"(?P<value>[A-Za-z][A-Za-z0-9_-]*)",
                line,
                re.IGNORECASE,
            )
            # Ref2VA retention lines put the taxonomy value immediately after
            # the marker's colon.  Keep a small key/value fallback for callers
            # that spell the field as ``retention: ...``.
            candidate_match = visible_match or audio_match or field_match
            if candidate_match is None:
                continue
            candidate = candidate_match.group("value")
            allowed = AUDIO_VALUES if audio_match else VISIBLE_RETENTION_VALUES
            if candidate not in allowed:
                label = "audio" if audio_match else "visible retention"
                errors.append(
                    f"invalid {label} value '{candidate}'; expected one of {', '.join(allowed)}"
                )
    return errors


def _shot_errors(text: str, duration_seconds: Any) -> list[str]:
    errors: list[str] = []
    shots: list[tuple[int, float | None, int]] = []
    exact_markers = list(_SHOT_RE.finditer(text))
    for marker in exact_markers:
        line_number = text.count("\n", 0, marker.start()) + 1
        number = int(marker.group("number"))
        rest = marker.group("rest")
        timestamp: float | None = None
        if number == 1:
            # Only a timestamp immediately following the marker is a cut
            # timestamp.  Times mentioned later in scene prose are literals.
            if re.match(r"^\s+At\b", rest):
                errors.append("shot 1 must use '[Shot 1]' with no timestamp")
        else:
            match = _TIMESTAMP_RE.match(rest)
            if not match:
                errors.append(
                    f"shot {number} must use '[Shot N] At MM:SS.mmm,' syntax"
                )
            else:
                minute = int(match.group("minute"))
                second = int(match.group("second"))
                millis = int(match.group("millis"))
                if second >= 60:
                    errors.append(f"shot {number} timestamp has invalid seconds {second:02d}")
                timestamp = minute * 60.0 + second + millis / 1000.0
                if duration_seconds is not None:
                    try:
                        duration = float(duration_seconds)
                    except (TypeError, ValueError):
                        duration = math.nan
                    if math.isfinite(duration) and (timestamp < 0 or timestamp >= duration):
                        errors.append(
                            f"shot {number} timestamp {match.group(0).strip()[3:-1]} is outside duration_seconds={duration_seconds}"
                        )
        shots.append((number, timestamp, line_number))

    # Report shot-like markers that do not match the exact ``[Shot N]`` form.
    # Exact markers have already been consumed above.
    exact_starts = {marker.start() for marker in exact_markers}
    for malformed in _SHOT_MARKER_RE.finditer(text):
        if malformed.start() in exact_starts:
            continue
        line_number = text.count("\n", 0, malformed.start()) + 1
        errors.append(f"invalid shot header on line {line_number}; expected '[Shot N]'")

    if not shots:
        errors.append("prompt must contain at least one shot headed '[Shot 1]'")
        return errors
    expected_number = 1
    for number, _, line_number in shots:
        if number != expected_number:
            errors.append(
                f"shot numbering error on line {line_number}: expected [Shot {expected_number}], found [Shot {number}]"
            )
        expected_number = number + 1 if number == expected_number else expected_number
    later = [(number, timestamp, line_number) for number, timestamp, line_number in shots if number > 1 and timestamp is not None]
    previous: float | None = None
    previous_number: int | None = None
    for number, timestamp, line_number in later:
        if previous is not None and timestamp <= previous:
            errors.append(
                f"shot {number} timestamp is not strictly increasing after shot {previous_number}"
            )
        previous = timestamp
        previous_number = number
    return errors


def _alignment_errors(
    text: str,
    mode: str,
    duration_seconds: Any,
    section_spans: list[tuple[str, int, int, str]],
) -> list[str]:
    """Validate the mode-specific first-line keyframe alignment contract."""

    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    if mode == "T2VA":
        if not first_line.startswith("integrated_multimodal_description:"):
            return ["T2VA must begin directly with 'integrated_multimodal_description:'"]
        return []
    if mode == REF_MODE:
        if first_line != "subject_definitions:":
            return ["Ref2VA must begin with the exact section heading 'subject_definitions:'"]
        return []
    try:
        duration = f"{float(duration_seconds):.2f}"
    except (TypeError, ValueError):
        duration = None
    if mode == "I2VA":
        expected = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
        return [] if first_line == expected else ["I2VA first-frame alignment instruction is missing or not exact"]
    if duration is None:
        return [f"{mode} alignment validation requires a numeric duration_seconds"]
    main_name = "detailed_description" if mode == REF_MODE else "integrated_multimodal_description"
    main_body = next((item[3] for item in section_spans if item[0] == main_name), "")
    shot_numbers = [int(match.group("number")) for match in _SHOT_RE.finditer(main_body)]
    final_shot = shot_numbers[-1] if shot_numbers else 1
    if mode == "FL2VA":
        expected = (
            "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with "
            "the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {final_shot}) aligns with the {duration}-second mark of the target video."
        )
        return [] if first_line == expected else ["FL2VA first/last-frame alignment instruction is missing or not exact"]
    expected = (
        "How the reference pictures align with the target video — "
        f"<Picture 1> (from [Shot {final_shot}]) aligns with the {duration}-second mark of the target video."
    )
    return [] if first_line == expected else ["L2VA last-frame alignment instruction is missing or not exact"]


def validate_prompt(
    prompt: Any = None,
    mode: Any = None,
    duration_seconds: Any = None,
    allowed_labels: Any = None,
    preserved_literals: Iterable[Any] | None = None,
    **kwargs: Any,
) -> ValidationResult:
    """Clean and statically validate a MiniMax H3 prompt.

    ``allowed_labels`` accepts a sequence of exact reference labels or a
    mapping such as ``{"Picture": ["1", "2"], "Video": ["1"]}``.  Several
    descriptive aliases (``allowed_picture_labels``,
    ``allowed_registry_labels`` and ``reference_labels``) are accepted for
    integration convenience; they are all treated identically.
    """

    if prompt is None:
        prompt = _extract_alias(kwargs, "cleaned_prompt", "text", "value")
    cleaned = cleanup_prompt(prompt)
    errors: list[str] = []

    if not cleaned:
        errors.append("prompt is empty after cleanup")

    if "duration" in kwargs and duration_seconds is None:
        duration_seconds = kwargs["duration"]
    if "video_duration" in kwargs and duration_seconds is None:
        duration_seconds = kwargs["video_duration"]
    if "model_mode" in kwargs and mode is None:
        mode = kwargs["model_mode"]
    normalized_mode = _normalise_mode(mode)
    if normalized_mode is None:
        # Ref2VA has an unambiguous first required section.  Inference is only
        # a convenience for callers that omit mode; explicit unknown modes are
        # still rejected below.
        normalized_mode = REF_MODE if re.search(r"^subject_definitions:", cleaned, re.MULTILINE) else BASE_MODES[0]
    if normalized_mode not in MODES:
        errors.append(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
        # Use the base order to continue producing useful diagnostics.
        normalized_mode = BASE_MODES[0]

    if "```" in cleaned:
        errors.append("residual markdown fence after cleanup")
    if _THINK_MARKER_RE.search(cleaned):
        errors.append("residual <think> block after cleanup")
    if _PREFIX_RE.match(cleaned):
        errors.append("residual leading answer prefix after cleanup")

    expected_sections = _section_order_for(normalized_mode)
    section_spans, section_errors = _section_spans(cleaned, expected_sections)
    errors.extend(section_errors)
    errors.extend(_alignment_errors(cleaned, normalized_mode, duration_seconds, section_spans))

    # Build a registry from all common keyword spellings.  ``allowed_labels``
    # remains the primary positional argument for concise use.
    label_values = [allowed_labels]
    label_values.extend(
        [
            _extract_alias(kwargs, "allowed_picture_labels", "picture_labels"),
            _extract_alias(kwargs, "allowed_reference_labels", "reference_labels"),
            _extract_alias(kwargs, "allowed_registry_labels", "registry_labels", "allowed_references"),
        ]
    )
    registry, registry_errors = _registry_from_values(*label_values)
    errors.extend(registry_errors)

    errors.extend(_reference_errors(cleaned, normalized_mode, registry, section_spans))
    errors.extend(_taxonomy_errors(cleaned, normalized_mode, section_spans))
    main_section = "detailed_description" if normalized_mode == REF_MODE else "integrated_multimodal_description"
    main_body = next((item[3] for item in section_spans if item[0] == main_section), "")
    errors.extend(_shot_errors(main_body, duration_seconds))

    literals = preserved_literals
    if literals is None:
        literals = _extract_alias(kwargs, "required_literals", "literals", "preserve_literals")
    if literals is not None:
        for literal in literals:
            value = str(literal)
            if value not in cleaned:
                errors.append(f"preserved literal missing: {value!r}")

    # Keep errors deterministic even when the same malformed marker triggers
    # more than one scanner (while preserving the first-occurrence order).
    deduped: list[str] = []
    seen: set[str] = set()
    for error in errors:
        if error not in seen:
            deduped.append(error)
            seen.add(error)
    return ValidationResult(cleaned_prompt=cleaned, valid=not deduped, errors=tuple(deduped))


def clean_and_validate(prompt: Any = None, *args: Any, **kwargs: Any) -> ValidationResult:
    """Compatibility alias for callers wanting a single operation."""

    return validate_prompt(prompt, *args, **kwargs)


__all__ = [
    "AUDIO_VALUES",
    "BASE_MODES",
    "BASE_SECTION_ORDER",
    "REF_MODE",
    "REF_SECTION_ORDER",
    "MODES",
    "VISIBLE_RETENTION_VALUES",
    "ValidationResult",
    "cleanup_prompt",
    "clean_and_validate",
    "validate_prompt",
]
