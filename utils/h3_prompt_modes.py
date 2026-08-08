"""Pure MiniMax H3 input-mode routing and validation helpers.

The node accepts a small set of mutually exclusive input modes.  This module
keeps the routing rules independent from ComfyUI (and therefore from torch),
so callers can pass opaque frame objects while this module only reasons about
their presence.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any


class H3InputMode(str, Enum):
    """Supported MiniMax H3 generation modes.

    Values intentionally use the labels exposed by the node UI.  As a
    ``str`` enum the values continue to compare naturally with strings used by
    existing ComfyUI workflows.
    """

    AUTO = "Auto"
    T2VA = "T2VA"
    I2VA = "I2VA"
    FL2VA = "FL2VA"
    L2VA = "L2VA"
    REF2VA = "Ref2VA"


_MODE_BY_NORMALIZED = {mode.value.casefold(): mode for mode in H3InputMode}
_REFERENCE_TOKEN_RE = re.compile(
    r"<\s*(Picture|Video|Audio|Subject)\s+([1-9][0-9]*)\s*>",
    re.IGNORECASE,
)


def normalize_h3_mode(value: H3InputMode | str | None = H3InputMode.AUTO) -> H3InputMode:
    """Normalize a mode label, accepting surrounding whitespace and case.

    Only the six exact mode names are accepted after normalization.  A
    descriptive :class:`ValueError` is raised for all other values rather
    than silently falling back to ``Auto``.
    """

    if isinstance(value, H3InputMode):
        return value
    if value is None:
        # ``None`` is convenient for optional API payloads and has the same
        # meaning as an omitted mode.  Other non-string values remain errors.
        return H3InputMode.AUTO
    if not isinstance(value, str):
        allowed = ", ".join(mode.value for mode in H3InputMode)
        raise ValueError(f"invalid H3 input mode {value!r}; expected one of: {allowed}")
    normalized = value.strip().casefold()
    mode = _MODE_BY_NORMALIZED.get(normalized)
    if mode is None:
        allowed = ", ".join(item.value for item in H3InputMode)
        raise ValueError(f"invalid H3 input mode {value!r}; expected one of: {allowed}")
    return mode


# Short aliases are useful to callers migrating from early prototypes.
normalize_mode = normalize_h3_mode
normalize_input_mode = normalize_h3_mode


def find_reference_label_tokens(reference_instructions: Any) -> list[str]:
    """Return valid H3 reference labels found in an instruction string.

    Labels are returned in textual order and normalized to the canonical H3
    spelling (for example ``<picture 2>`` becomes ``<Picture 2>``).  Non-string
    values are treated as having no labels; importantly this never evaluates
    opaque tensor-like objects for truthiness.
    """

    if not isinstance(reference_instructions, str):
        return []
    return [
        f"<{match.group(1).title()} {int(match.group(2))}>"
        for match in _REFERENCE_TOKEN_RE.finditer(reference_instructions)
    ]


def contains_reference_label(reference_instructions: Any) -> bool:
    """Whether *reference_instructions* contains at least one valid label."""

    return bool(find_reference_label_tokens(reference_instructions))


# Names used by callers in different revisions of the node.
has_reference_label = contains_reference_label
has_reference_label_token = contains_reference_label
contains_reference_label_token = contains_reference_label
detect_reference_labels = find_reference_label_tokens
find_reference_tokens = find_reference_label_tokens


def _present(value: Any) -> bool:
    """Interpret a presence argument without tensor truthiness.

    Canonical callers pass booleans.  For compatibility, an opaque object is
    considered present solely by testing ``is not None``; its ``__bool__`` is
    never invoked.  ``False`` is the only explicit false sentinel.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        return value != 0.0
    return value is not None


def _image_count(value: Any) -> int:
    """Normalize an image count or image collection to a non-negative count."""

    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value < 0:
            raise ValueError("reference_image_count must be non-negative")
        return value
    if isinstance(value, float):
        if not value.is_integer() or value < 0:
            raise ValueError("reference_image_count must be a non-negative integer")
        return int(value)
    # A collection is accepted as a convenience for direct node callers.  Do
    # not use ``if item`` because tensor-like values may reject truth testing.
    if isinstance(value, (str, bytes)):
        return 1
    try:
        return sum(item is not None for item in value)
    except TypeError:
        # An opaque non-iterable object represents one reference.
        return 1


def _default_image_count(value: Any) -> bool:
    """Whether a count argument is the canonical omitted value (zero)."""

    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _empty_instruction(value: Any) -> bool:
    """Check an optional instruction without evaluating opaque truthiness."""

    return value is None or (isinstance(value, str) and not value.strip())


def _coerce_route_inputs(
    *,
    has_first_frame: Any,
    has_last_frame: Any,
    reference_image_count: Any,
    reference_instructions: Any,
) -> tuple[bool, bool, int, bool]:
    first = _present(has_first_frame)
    last = _present(has_last_frame)
    count = _image_count(reference_image_count)
    has_instruction_reference = contains_reference_label(reference_instructions)
    return first, last, count, has_instruction_reference


def route_h3_mode(
    requested_mode: H3InputMode | str | None = H3InputMode.AUTO,
    *,
    has_first_frame: Any = False,
    has_last_frame: Any = False,
    reference_image_count: Any = 0,
    reference_instructions: Any = "",
    **aliases: Any,
) -> H3InputMode:
    """Resolve and validate the H3 mode for a set of media inputs.

    ``Auto`` follows the fixed priority ``Ref2VA`` (any reference image or a
    labelled instruction) → ``T2VA`` → ``I2VA`` → ``FL2VA`` → ``L2VA`` based
    on the two anchor-frame booleans.  Explicit modes always win over Auto but
    are validated against incompatible inputs and raise descriptive
    :class:`ValueError` instances.

    The optional ``aliases`` accommodate callers that still provide
    ``first_frame``/``last_frame`` objects or ``ref_images`` collections.  The
    canonical boolean/count arguments remain the preferred API.
    """

    # Backwards-compatible object/collection names.  Explicit canonical
    # arguments win, while aliases are used only when their canonical value is
    # still at its default.
    if "first_frame" in aliases and has_first_frame is False:
        has_first_frame = aliases.pop("first_frame")
    if "last_frame" in aliases and has_last_frame is False:
        has_last_frame = aliases.pop("last_frame")
    if "ref_images" in aliases and _default_image_count(reference_image_count):
        reference_image_count = aliases.pop("ref_images")
    if "reference_media_count" in aliases and _default_image_count(reference_image_count):
        reference_image_count = aliases.pop("reference_media_count")
    if "ref_image_count" in aliases and _default_image_count(reference_image_count):
        reference_image_count = aliases.pop("ref_image_count")
    if "instructions" in aliases and _empty_instruction(reference_instructions):
        reference_instructions = aliases.pop("instructions")
    if aliases:
        unexpected = ", ".join(sorted(aliases))
        raise TypeError(f"route_h3_mode() got unexpected keyword argument(s): {unexpected}")

    mode = normalize_h3_mode(requested_mode)
    first, last, image_count, labelled_instruction = _coerce_route_inputs(
        has_first_frame=has_first_frame,
        has_last_frame=has_last_frame,
        reference_image_count=reference_image_count,
        reference_instructions=reference_instructions,
    )

    if mode is H3InputMode.AUTO:
        if image_count > 0 or labelled_instruction:
            mode = H3InputMode.REF2VA
        elif not first and not last:
            mode = H3InputMode.T2VA
        elif first and not last:
            mode = H3InputMode.I2VA
        elif first and last:
            mode = H3InputMode.FL2VA
        else:
            mode = H3InputMode.L2VA

    validate_mode_inputs(
        mode,
        has_first_frame=first,
        has_last_frame=last,
        reference_image_count=image_count,
        reference_instructions=reference_instructions,
    )
    return mode


def validate_mode_inputs(
    mode: H3InputMode | str,
    *,
    has_first_frame: Any = False,
    has_last_frame: Any = False,
    reference_image_count: Any = 0,
    reference_instructions: Any = "",
    **aliases: Any,
) -> None:
    """Validate inputs for an already selected H3 mode.

    ``Auto`` is resolved first, making this function safe to call directly or
    through :func:`route_h3_mode`.  The function returns ``None`` on success
    and raises :class:`ValueError` with the selected mode and offending input
    in the message on failure.
    """

    if "first_frame" in aliases and has_first_frame is False:
        has_first_frame = aliases.pop("first_frame")
    if "last_frame" in aliases and has_last_frame is False:
        has_last_frame = aliases.pop("last_frame")
    if "ref_images" in aliases and _default_image_count(reference_image_count):
        reference_image_count = aliases.pop("ref_images")
    if "reference_media_count" in aliases and _default_image_count(reference_image_count):
        reference_image_count = aliases.pop("reference_media_count")
    if "ref_image_count" in aliases and _default_image_count(reference_image_count):
        reference_image_count = aliases.pop("ref_image_count")
    if "instructions" in aliases and _empty_instruction(reference_instructions):
        reference_instructions = aliases.pop("instructions")
    if aliases:
        unexpected = ", ".join(sorted(aliases))
        raise TypeError(f"validate_mode_inputs() got unexpected keyword argument(s): {unexpected}")

    normalized = normalize_h3_mode(mode)
    first, last, image_count, labelled_instruction = _coerce_route_inputs(
        has_first_frame=has_first_frame,
        has_last_frame=has_last_frame,
        reference_image_count=reference_image_count,
        reference_instructions=reference_instructions,
    )

    if normalized is H3InputMode.AUTO:
        # Validation of Auto means validation of the deterministic selected
        # mode, not an unconditional pass-through.
        selected = route_h3_mode(
            H3InputMode.AUTO,
            has_first_frame=first,
            has_last_frame=last,
            reference_image_count=image_count,
            reference_instructions=reference_instructions,
        )
        if selected is not H3InputMode.AUTO:
            validate_mode_inputs(
                selected,
                has_first_frame=first,
                has_last_frame=last,
                reference_image_count=image_count,
                reference_instructions=reference_instructions,
            )
        return

    def reject(reason: str) -> None:
        raise ValueError(f"{normalized.value} input validation failed: {reason}")

    if normalized is H3InputMode.T2VA:
        if first or last or image_count:
            reject("T2VA requires no first/last frame or reference images")
        if labelled_instruction:
            reject("T2VA forbids labelled reference_instructions (use Ref2VA)")
    elif normalized is H3InputMode.I2VA:
        if not first:
            reject("I2VA requires first_frame")
        if last:
            reject("I2VA forbids last_frame")
        if image_count:
            reject("I2VA forbids reference images")
        if labelled_instruction:
            reject("I2VA forbids labelled reference_instructions (use Ref2VA)")
    elif normalized is H3InputMode.FL2VA:
        if not first or not last:
            reject("FL2VA requires both first_frame and last_frame")
        if image_count:
            reject("FL2VA forbids reference images")
        if labelled_instruction:
            reject("FL2VA forbids labelled reference_instructions (use Ref2VA)")
    elif normalized is H3InputMode.L2VA:
        if not last:
            reject("L2VA requires last_frame")
        if first:
            reject("L2VA forbids first_frame")
        if image_count:
            reject("L2VA forbids reference images")
        if labelled_instruction:
            reject("L2VA forbids labelled reference_instructions (use Ref2VA)")
    elif normalized is H3InputMode.REF2VA:
        if not (first or last or image_count or labelled_instruction):
            reject(
                "Ref2VA requires first_frame, last_frame, a reference image, "
                "or a labelled reference_instructions token"
            )


# A concise alias used in a few integrations.
validate_h3_mode_inputs = validate_mode_inputs


__all__ = [
    "H3InputMode",
    "normalize_h3_mode",
    "normalize_mode",
    "normalize_input_mode",
    "find_reference_label_tokens",
    "find_reference_tokens",
    "detect_reference_labels",
    "contains_reference_label",
    "contains_reference_label_token",
    "has_reference_label",
    "has_reference_label_token",
    "route_h3_mode",
    "validate_mode_inputs",
    "validate_h3_mode_inputs",
]
