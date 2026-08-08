"""Offline MiniMax H3 official-spec metadata and clean-room format facts.

Only field names, section order, relationship markers, and source metadata are
kept here.  The upstream guide text is not bundled or fetched at runtime.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from types import MappingProxyType
from typing import Any

UPSTREAM_REPOSITORY = "https://github.com/MiniMax-AI/MiniMax-H3"
UPSTREAM_BRANCH = "main"
UPSTREAM_COMMIT = "8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea"

SKILL_SOURCE_PATH = "skills/h3-prompt-writing/SKILL.md"
BASE_SOURCE_PATH = "skills/h3-prompt-writing/references/base-en.txt"
REF_SOURCE_PATH = "skills/h3-prompt-writing/references/ref-en.txt"

SKILL_SOURCE_URL = f"{UPSTREAM_REPOSITORY}/blob/{UPSTREAM_COMMIT}/{SKILL_SOURCE_PATH}"
BASE_SOURCE_URL = f"{UPSTREAM_REPOSITORY}/blob/{UPSTREAM_COMMIT}/{BASE_SOURCE_PATH}"
REF_SOURCE_URL = f"{UPSTREAM_REPOSITORY}/blob/{UPSTREAM_COMMIT}/{REF_SOURCE_PATH}"

BASE_SECTION_ORDER = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
REF_SECTION_ORDER = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
VISIBLE_RETENTION_VALUES = (
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
)
AUDIO_RETENTION_VALUES = (
    "fully_copy",
    "partially_copy",
    "reference",
    "weak_reference",
)

# Short aliases are intentionally tuples so callers cannot mutate the facts.
BASE_SECTIONS = BASE_SECTION_ORDER
REF_SECTIONS = REF_SECTION_ORDER
VISIBLE_RETENTION_MARKERS = VISIBLE_RETENTION_VALUES
AUDIO_RETENTION_MARKERS = AUDIO_RETENTION_VALUES


@dataclass(frozen=True, slots=True)
class OfficialH3Spec:
    """Immutable format facts for one H3 prompt mode."""

    mode: str
    sections: tuple[str, ...]
    visible_retention_values: tuple[str, ...]
    audio_retention_values: tuple[str, ...]
    source_paths: tuple[str, ...]
    source_urls: tuple[str, ...]

    @property
    def section_order(self) -> tuple[str, ...]:
        """Alias for consumers that call the field ``section_order``."""

        return self.sections

    @property
    def visible_retention(self) -> tuple[str, ...]:
        """Alias for the visible-content relationship markers."""

        return self.visible_retention_values

    @property
    def audio_retention(self) -> tuple[str, ...]:
        """Alias for the audio relationship markers."""

        return self.audio_retention_values

    @property
    def retention_values(self) -> tuple[str, ...]:
        """Combined visible and audio markers, preserving source order."""

        return self.visible_retention_values + self.audio_retention_values


class H3OfficialResourcesError(RuntimeError):
    """Base class for deterministic upstream metadata loading errors."""


class H3OfficialMetadataNotFoundError(FileNotFoundError, H3OfficialResourcesError):
    """Raised when the metadata file does not exist."""


class H3OfficialMetadataMalformedError(ValueError, H3OfficialResourcesError):
    """Raised when metadata is not valid UTF-8 JSON or has the wrong shape."""


class H3OfficialMetadataValidationError(ValueError, H3OfficialResourcesError):
    """Raised when required metadata keys or values are absent/invalid."""


_BASE_SPEC = OfficialH3Spec(
    mode="base",
    sections=BASE_SECTION_ORDER,
    visible_retention_values=VISIBLE_RETENTION_VALUES,
    audio_retention_values=AUDIO_RETENTION_VALUES,
    source_paths=(SKILL_SOURCE_PATH, BASE_SOURCE_PATH),
    source_urls=(SKILL_SOURCE_URL, BASE_SOURCE_URL),
)
_REF_SPEC = OfficialH3Spec(
    mode="ref",
    sections=REF_SECTION_ORDER,
    visible_retention_values=VISIBLE_RETENTION_VALUES,
    audio_retention_values=AUDIO_RETENTION_VALUES,
    source_paths=(SKILL_SOURCE_PATH, REF_SOURCE_PATH),
    source_urls=(SKILL_SOURCE_URL, REF_SOURCE_URL),
)
_MODE_ALIASES = MappingProxyType({
    "base": _BASE_SPEC,
    "t2va": _BASE_SPEC,
    "i2va": _BASE_SPEC,
    "fl2va": _BASE_SPEC,
    "l2va": _BASE_SPEC,
    "ref": _REF_SPEC,
    "ref2va": _REF_SPEC,
    "full-reference": _REF_SPEC,
    "full_reference": _REF_SPEC,
    "fullreference": _REF_SPEC,
})

_DEFAULT_METADATA_PATH = Path(__file__).resolve().parent.parent / "resources" / "minimax_h3_spec" / "UPSTREAM.json"
_SHA256_RE = re.compile(r"^[0-9A-Fa-f]{64}$")
_REQUIRED_TOP_LEVEL = (
    "repository",
    "branch",
    "commit",
    "retrieved_at",
    "official_sources",
    "license",
    "redistribution_strategy",
)


def _metadata_error(path: Path, message: str) -> H3OfficialMetadataValidationError:
    return H3OfficialMetadataValidationError(f"Invalid MiniMax H3 upstream metadata at {path}: {message}")


def _validate_metadata(data: Any, path: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise _metadata_error(path, "top level must be a JSON object")
    for key in _REQUIRED_TOP_LEVEL:
        if key not in data:
            raise _metadata_error(path, f"missing required key '{key}'")
    for key in ("repository", "branch", "commit", "retrieved_at", "redistribution_strategy"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise _metadata_error(path, f"'{key}' must be a non-empty string")

    sources = data["official_sources"]
    if not isinstance(sources, list) or not sources:
        raise _metadata_error(path, "'official_sources' must be a non-empty array")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise _metadata_error(path, f"official_sources[{index}] must be an object")
        for key in ("id", "path", "url", "sha256"):
            if key not in source:
                raise _metadata_error(path, f"official_sources[{index}] missing required key '{key}'")
            if not isinstance(source[key], str) or not source[key].strip():
                raise _metadata_error(path, f"official_sources[{index}].{key} must be a non-empty string")
        if source["id"] in seen_ids:
            raise _metadata_error(path, f"duplicate official source id '{source['id']}'")
        if source["path"] in seen_paths:
            raise _metadata_error(path, f"duplicate official source path '{source['path']}'")
        if not _SHA256_RE.fullmatch(source["sha256"]):
            raise _metadata_error(path, f"official_sources[{index}].sha256 must be 64 hexadecimal characters")
        seen_ids.add(source["id"])
        seen_paths.add(source["path"])

    license_info = data["license"]
    if not isinstance(license_info, dict):
        raise _metadata_error(path, "'license' must be an object")
    for key in ("name", "link"):
        if key not in license_info:
            raise _metadata_error(path, f"license missing required key '{key}'")
        if not isinstance(license_info[key], str) or not license_info[key].strip():
            raise _metadata_error(path, f"license.{key} must be a non-empty string")
    return data


def load_upstream_metadata(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate ``UPSTREAM.json`` using strict UTF-8, on demand.

    ``path`` is injectable for tests and tooling.  No file is read during
    module import, and this function never performs network access.
    """

    metadata_path = Path(path) if path is not None else _DEFAULT_METADATA_PATH
    try:
        raw = metadata_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise H3OfficialMetadataNotFoundError(
            f"MiniMax H3 upstream metadata file not found: {metadata_path}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise H3OfficialMetadataMalformedError(
            f"MiniMax H3 upstream metadata is not valid UTF-8: {metadata_path}"
        ) from exc
    except OSError as exc:
        raise H3OfficialResourcesError(
            f"Unable to read MiniMax H3 upstream metadata at {metadata_path}: {exc}"
        ) from exc
    try:
        data = json.loads(raw)
    except JSONDecodeError as exc:
        raise H3OfficialMetadataMalformedError(
            f"MiniMax H3 upstream metadata contains malformed JSON at {metadata_path}: "
            f"line {exc.lineno}, column {exc.colno}"
        ) from exc
    return _validate_metadata(data, metadata_path)


def get_spec_for_mode(mode: str) -> OfficialH3Spec:
    """Return immutable official format facts for a base or reference mode."""

    if not isinstance(mode, str):
        raise TypeError("MiniMax H3 mode must be a string")
    normalized = mode.strip().lower().replace(" ", "-")
    try:
        return _MODE_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_MODE_ALIASES))
        raise ValueError(f"Unknown MiniMax H3 mode '{mode}'. Supported modes: {supported}") from exc


__all__ = [
    "AUDIO_RETENTION_MARKERS",
    "AUDIO_RETENTION_VALUES",
    "BASE_SECTION_ORDER",
    "BASE_SECTIONS",
    "BASE_SOURCE_PATH",
    "BASE_SOURCE_URL",
    "H3OfficialMetadataMalformedError",
    "H3OfficialMetadataNotFoundError",
    "H3OfficialMetadataValidationError",
    "H3OfficialResourcesError",
    "OfficialH3Spec",
    "REF_SECTION_ORDER",
    "REF_SECTIONS",
    "REF_SOURCE_PATH",
    "REF_SOURCE_URL",
    "SKILL_SOURCE_PATH",
    "SKILL_SOURCE_URL",
    "UPSTREAM_BRANCH",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPOSITORY",
    "VISIBLE_RETENTION_MARKERS",
    "VISIBLE_RETENTION_VALUES",
    "get_spec_for_mode",
    "load_upstream_metadata",
]
