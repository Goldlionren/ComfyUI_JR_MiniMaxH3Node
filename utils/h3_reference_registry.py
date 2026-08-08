"""Deterministic, dependency-free registry for MiniMax H3 references.

H3 labels are global per media type: Pictures, Videos, Audio and Subjects
each have an independent counter.  Registration order is explicit and stable;
the helper for node inputs registers the first frame, last frame, then
``ref_image_1`` through ``ref_image_9``.  No tensor operation is performed in
this module, so opaque ComfyUI values are safe to pass as presence sentinels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class H3ReferenceType(str, Enum):
    PICTURE = "Picture"
    SUBJECT = "Subject"
    VIDEO = "Video"
    AUDIO = "Audio"


# Friendly aliases used by integrations that called this enum ``ReferenceType``.
ReferenceType = H3ReferenceType
ReferenceKind = H3ReferenceType

_TYPE_BY_CASEFOLD = {item.value.casefold(): item for item in H3ReferenceType}
_LABEL_RE = re.compile(r"^<\s*(Picture|Subject|Video|Audio)\s+([1-9][0-9]*)\s*>$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"<\s*(Picture|Subject|Video|Audio)\s+([1-9][0-9]*)\s*>", re.IGNORECASE)
_ALLOWED_ROLES = {"first_frame", "last_frame", "reference", "source", "subject"}


def normalize_reference_type(value: H3ReferenceType | str) -> H3ReferenceType:
    if isinstance(value, H3ReferenceType):
        return value
    if not isinstance(value, str):
        raise ValueError("reference type must be one of Picture, Subject, Video, Audio")
    result = _TYPE_BY_CASEFOLD.get(value.strip().casefold())
    if result is None:
        allowed = ", ".join(item.value for item in H3ReferenceType)
        raise ValueError(f"invalid reference type {value!r}; expected one of: {allowed}")
    return result


def normalize_reference_label(identifier: str, *, reference_type: H3ReferenceType | str | None = None) -> str:
    """Normalize ``Picture 1``/``<picture 1>`` to a canonical H3 label."""

    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("reference identifier must be a non-empty label")
    value = identifier.strip()
    match = _LABEL_RE.fullmatch(value)
    if match:
        kind = normalize_reference_type(match.group(1)).value
        number = int(match.group(2))
        if reference_type is not None and kind != normalize_reference_type(reference_type).value:
            raise ValueError(f"reference identifier {identifier!r} has type {kind}, not {normalize_reference_type(reference_type).value}")
        return f"<{kind} {number}>"
    # Accept a bare numeric suffix only when a type is supplied; this is handy
    # for callers that persist ``Picture 1`` without angle brackets.
    if reference_type is not None:
        kind = normalize_reference_type(reference_type).value
        bare = re.fullmatch(r"(?:[1-9][0-9]*)", value)
        if bare:
            return f"<{kind} {int(value)}>"
    raise ValueError(f"invalid H3 reference identifier {identifier!r}; expected <Picture N>, <Video N>, <Audio N>, or <Subject N>")


@dataclass(frozen=True)
class ReferenceEntry:
    """One registered H3 media/reference label.

    ``identifier`` is the canonical angle-bracket label.  ``source_key`` is a
    registry-unique key that distinguishes repeated registrations of the same
    source input (for example separate batch items).
    """

    identifier: str
    type: str
    source_input: str
    role: str = "reference"
    subject_binding: str | None = None
    resolved_status: bool = True
    source_key: str = ""

    @property
    def label(self) -> str:
        return self.identifier

    @property
    def reference_type(self) -> str:
        return self.type

    @property
    def kind(self) -> str:
        return self.type

    @property
    def source(self) -> str:
        return self.source_input

    @property
    def resolved(self) -> bool:
        return self.resolved_status


def _normalize_role(role: str | None) -> str:
    if role is None:
        return "reference"
    if not isinstance(role, str):
        raise ValueError("reference role must be a string")
    normalized = role.strip().casefold().replace("-", "_").replace(" ", "_")
    if not normalized:
        normalized = "reference"
    if normalized not in _ALLOWED_ROLES:
        allowed = ", ".join(sorted(_ALLOWED_ROLES))
        raise ValueError(f"invalid reference role {role!r}; expected one of: {allowed}")
    return normalized


def _normalize_source(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _normalize_status(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"resolved", "ready", "ok", "true", "1"}:
            return True
        if lowered in {"unresolved", "pending", "missing", "false", "0"}:
            return False
    # Opaque values are treated as supplied/resolved without invoking their
    # potentially tensor-like truthiness.
    return True


@dataclass
class ReferenceRegistry:
    """Ordered registry with independent per-type label counters."""

    _entries: list[ReferenceEntry] = field(default_factory=list, init=False, repr=False)
    _source_keys: set[str] = field(default_factory=set, init=False, repr=False)
    _identifiers: set[str] = field(default_factory=set, init=False, repr=False)
    _counters: dict[str, int] = field(
        default_factory=lambda: {item.value: 0 for item in H3ReferenceType}, init=False, repr=False
    )

    def _next_identifier(self, reference_type: H3ReferenceType) -> str:
        number = self._counters[reference_type.value] + 1
        identifier = f"<{reference_type.value} {number}>"
        while identifier in self._identifiers:
            number += 1
            identifier = f"<{reference_type.value} {number}>"
        self._counters[reference_type.value] = number
        return identifier

    def _source_key(self, source_input: str, source_key: str | None) -> str:
        if source_key is not None:
            key = _normalize_source(source_key, "source_key")
            if key in self._source_keys:
                raise ValueError(f"duplicate reference source key: {key}")
            return key
        base = source_input
        if base not in self._source_keys:
            return base
        raise ValueError(
            f"duplicate reference source key: {base}; provide a unique source_key suffix for repeated registrations"
        )

    def _register(
        self,
        reference_type: H3ReferenceType | str,
        source_input: str,
        role: str = "reference",
        source_key: str | None = None,
        subject_binding: str | None = None,
        *,
        identifier: str | None = None,
        resolved_status: Any = True,
    ) -> ReferenceEntry:
        kind = normalize_reference_type(reference_type)
        source = _normalize_source(source_input, "source_input")
        normalized_role = _normalize_role(role)
        key = self._source_key(source, source_key)
        if identifier is None:
            label = self._next_identifier(kind)
        else:
            label = normalize_reference_label(identifier, reference_type=kind)
            if label in self._identifiers:
                raise ValueError(f"duplicate reference label: {label}")
            match = _LABEL_RE.fullmatch(label)
            assert match is not None
            self._counters[kind.value] = max(self._counters[kind.value], int(match.group(2)))

        binding = None
        if subject_binding is not None:
            if not isinstance(subject_binding, str) or not subject_binding.strip():
                raise ValueError("subject_binding must be a non-empty reference label or source key")
            binding = subject_binding.strip()

        entry = ReferenceEntry(
            identifier=label,
            type=kind.value,
            source_input=source,
            role=normalized_role,
            subject_binding=binding,
            resolved_status=_normalize_status(resolved_status),
            source_key=key,
        )
        self._entries.append(entry)
        self._identifiers.add(label)
        self._source_keys.add(key)
        return entry

    def register_picture(
        self,
        source_input: str,
        role: str = "reference",
        source_key: str | None = None,
        subject_binding: str | None = None,
        *,
        identifier: str | None = None,
        resolved_status: Any = True,
    ) -> ReferenceEntry:
        return self._register("Picture", source_input, role, source_key, subject_binding, identifier=identifier, resolved_status=resolved_status)

    def register_subject(
        self,
        source_input: str,
        role: str = "subject",
        source_key: str | None = None,
        subject_binding: str | None = None,
        *,
        identifier: str | None = None,
        resolved_status: Any = True,
    ) -> ReferenceEntry:
        return self._register("Subject", source_input, role, source_key, subject_binding, identifier=identifier, resolved_status=resolved_status)

    def register_video(
        self,
        source_input: str,
        role: str = "reference",
        source_key: str | None = None,
        subject_binding: str | None = None,
        *,
        identifier: str | None = None,
        resolved_status: Any = True,
    ) -> ReferenceEntry:
        return self._register("Video", source_input, role, source_key, subject_binding, identifier=identifier, resolved_status=resolved_status)

    def register_audio(
        self,
        source_input: str,
        role: str = "reference",
        source_key: str | None = None,
        subject_binding: str | None = None,
        *,
        identifier: str | None = None,
        resolved_status: Any = True,
    ) -> ReferenceEntry:
        return self._register("Audio", source_input, role, source_key, subject_binding, identifier=identifier, resolved_status=resolved_status)

    def register(
        self,
        reference_type: H3ReferenceType | str = "Picture",
        source_input: str | None = None,
        role: str = "reference",
        source_key: str | None = None,
        subject_binding: str | None = None,
        *,
        identifier: str | None = None,
        resolved_status: Any = True,
        type: H3ReferenceType | str | None = None,
    ) -> ReferenceEntry:
        """Generic registration convenience method."""

        if type is not None:
            reference_type = type
        else:
            # Accommodate the two natural positional spellings used by early
            # callers: ``register("Picture", "source")`` and
            # ``register("source", "Picture")``.  A lone source defaults to
            # a Picture, matching the typed helper methods.
            try:
                normalize_reference_type(reference_type)
            except ValueError:
                if source_input is not None:
                    try:
                        normalize_reference_type(source_input)
                    except ValueError:
                        pass
                    else:
                        reference_type, source_input = source_input, reference_type
                elif isinstance(reference_type, str):
                    source_input, reference_type = reference_type, "Picture"
        if source_input is None:
            raise ValueError("source_input is required")
        return self._register(
            reference_type,
            source_input,
            role,
            source_key,
            subject_binding,
            identifier=identifier,
            resolved_status=resolved_status,
        )

    register_reference = register
    register_ref_image = register_picture

    def register_inputs(
        self,
        *,
        first_frame: Any = None,
        last_frame: Any = None,
        ref_images: Any = None,
        reference_instructions: Any = None,
        **slots: Any,
    ) -> list[ReferenceEntry]:
        """Register the standard H3 node image inputs in deterministic order.

        Presence is tested with ``is not None`` only.  ``ref_images`` may be a
        mapping keyed by ``ref_image_N`` or an iterable in slot order; explicit
        slot keywords are also accepted.
        """

        registered: list[ReferenceEntry] = []
        if first_frame is not None:
            registered.append(self.register_picture("first_frame", "first_frame"))
        if last_frame is not None:
            registered.append(self.register_picture("last_frame", "last_frame"))

        values: dict[int, Any] = {}
        if isinstance(ref_images, Mapping):
            for key, value in ref_images.items():
                match = re.search(r"(?:ref_image|image)[_ -]?(\d+)$", str(key), re.IGNORECASE)
                if match:
                    values[int(match.group(1))] = value
        elif ref_images is not None:
            try:
                values.update({index: value for index, value in enumerate(ref_images, 1)})
            except TypeError:
                values[1] = ref_images
        for index in range(1, 10):
            key = f"ref_image_{index}"
            if key in slots:
                values[index] = slots[key]
        for index in sorted(values):
            if 1 <= index <= 9 and values[index] is not None:
                registered.append(self.register_picture(f"ref_image_{index}", "reference"))
        return registered

    # Common alternate helper names.
    register_standard_inputs = register_inputs
    register_media = register_inputs

    def entries(self) -> list[ReferenceEntry]:
        """Return a stable copy of entries in registration order."""

        return list(self._entries)

    all_entries = entries

    @property
    def ordered_entries(self) -> tuple[ReferenceEntry, ...]:
        return tuple(self._entries)

    def labels(self, reference_type: H3ReferenceType | str | None = None) -> list[str]:
        if reference_type is None:
            return [entry.identifier for entry in self._entries]
        kind = normalize_reference_type(reference_type).value
        return [entry.identifier for entry in self._entries if entry.type == kind]

    def list_references(
        self,
        reference_type: H3ReferenceType | str | None = None,
        *,
        role: str | None = None,
    ) -> list[ReferenceEntry]:
        selected = self._entries
        if reference_type is not None:
            kind = normalize_reference_type(reference_type).value
            selected = [entry for entry in selected if entry.type == kind]
        if role is not None:
            normalized_role = _normalize_role(role)
            selected = [entry for entry in selected if entry.role == normalized_role]
        return list(selected)

    def resolve(self, identifier: str) -> ReferenceEntry | None:
        """Resolve a label or source key to its registered entry."""

        if not isinstance(identifier, str):
            return None
        candidate = identifier.strip()
        try:
            candidate = normalize_reference_label(candidate)
        except ValueError:
            pass
        for entry in self._entries:
            if entry.identifier == candidate or entry.source_key == identifier or entry.source_input == identifier:
                return entry
        return None

    resolve_label = resolve

    def mark_resolved(self, identifier: str, resolved: bool = True) -> ReferenceEntry:
        """Return a copy of an entry with its resolved status updated.

        Entries are frozen so that the ordered registry remains safe to share;
        replacing the entry in place keeps references returned by prior calls
        immutable while allowing callers to finalize asynchronous resolution.
        """

        entry = self.resolve(identifier)
        if entry is None:
            raise KeyError(f"unknown H3 reference: {identifier}")
        replacement = ReferenceEntry(
            identifier=entry.identifier,
            type=entry.type,
            source_input=entry.source_input,
            role=entry.role,
            subject_binding=entry.subject_binding,
            resolved_status=_normalize_status(resolved),
            source_key=entry.source_key,
        )
        index = self._entries.index(entry)
        self._entries[index] = replacement
        return replacement

    set_resolved = mark_resolved

    def unresolved_entries(self) -> list[ReferenceEntry]:
        """Return entries explicitly marked unresolved, in stable order."""

        unresolved = [entry for entry in self._entries if not entry.resolved_status]
        for entry in self._entries:
            if entry.subject_binding and self.resolve(entry.subject_binding) is None and entry not in unresolved:
                unresolved.append(entry)
        return unresolved

    def unresolved_labels(self, text: Any = None) -> list[str]:
        """Report labels in *text* that are not registered.

        With no text, labels of unresolved entries (including broken subject
        bindings) are returned.  Results are de-duplicated while preserving
        first occurrence order.
        """

        if text is None:
            candidates = [entry.identifier for entry in self.unresolved_entries()]
        elif isinstance(text, str):
            candidates = [
                f"<{match.group(1).title()} {int(match.group(2))}>" for match in _TOKEN_RE.finditer(text)
            ]
        else:
            candidates = []
        result: list[str] = []
        for label in candidates:
            entry = self.resolve(label)
            if (entry is None or not entry.resolved_status) and label not in result:
                result.append(label)
        return result

    report_unresolved = unresolved_labels

    def validate_references(self, text: Any = None, *, raise_on_unresolved: bool = True) -> bool:
        unresolved = self.unresolved_labels(text)
        if unresolved and raise_on_unresolved:
            joined = ", ".join(unresolved)
            raise ValueError(f"unresolved H3 reference label(s): {joined}")
        return not unresolved

    validate = validate_references


def build_reference_registry(
    *,
    first_frame: Any = None,
    last_frame: Any = None,
    ref_images: Any = None,
    reference_instructions: Any = None,
    **slots: Any,
) -> ReferenceRegistry:
    """Build a registry for standard node inputs in deterministic order."""

    registry = ReferenceRegistry()
    registry.register_inputs(
        first_frame=first_frame,
        last_frame=last_frame,
        ref_images=ref_images,
        reference_instructions=reference_instructions,
        **slots,
    )
    return registry


register_input_references = build_reference_registry


__all__ = [
    "H3ReferenceType",
    "ReferenceType",
    "ReferenceKind",
    "ReferenceEntry",
    "ReferenceRegistry",
    "normalize_reference_type",
    "normalize_reference_label",
    "build_reference_registry",
    "register_input_references",
]
