"""Deterministic header-first planning for the JR MiniMax H3 Hybrid Loader."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .h3_hybrid_tensor_family import (
    H3HybridCompatibilityError,
    SafetensorsHeader,
    TensorFamily,
    TensorHeader,
    read_safetensors_header,
    resolve_compatible_families,
    selected_block_indices,
    validate_h3_layout,
)

PROFILES = (
    "Recommended",
    "All Block AdaLN",
    "All Block AdaLN + Final",
    "Custom Range",
    "Pure FL",
    "Pure REF",
    "Advanced Custom",
)
HYBRID_PROFILES = frozenset(PROFILES) - {"Pure FL", "Pure REF"}
MAX_CUSTOM_TEXT = 4096
MAX_CUSTOM_PATTERNS = 64


@dataclass(frozen=True, slots=True)
class HybridPlan:
    fl_path: Path
    ref_path: Path
    fl_file_size: int
    fl_mtime_ns: int
    ref_file_size: int
    ref_mtime_ns: int
    profile: str
    selected_families: tuple[TensorFamily, ...]
    selected_tensors: tuple[TensorHeader, ...]
    selected_keys: tuple[str, ...]
    selected_bytes: int
    selected_blocks: tuple[int, ...]
    final_adaln_source: str
    weight_dtype: str
    warnings: tuple[str, ...]
    fingerprint: str


def _patterns(value: str | None) -> tuple[str, ...]:
    text = str(value or "").strip()
    if len(text.encode("utf-8")) > MAX_CUSTOM_TEXT:
        raise H3HybridCompatibilityError("Advanced custom pattern text is too large.")
    parts = tuple(part.strip() for chunk in text.splitlines() for part in chunk.split(",") if part.strip())
    if len(parts) > MAX_CUSTOM_PATTERNS:
        raise H3HybridCompatibilityError(f"Advanced custom supports at most {MAX_CUSTOM_PATTERNS} patterns.")
    if any("\x00" in part for part in parts):
        raise H3HybridCompatibilityError("Advanced custom patterns cannot contain NUL characters.")
    return parts


def _matches(key: str, pattern: str) -> bool:
    if pattern.endswith("."):
        return key.startswith(pattern)
    return fnmatch.fnmatchcase(key, pattern)


def _matching_keys(header: SafetensorsHeader, patterns: tuple[str, ...]) -> set[str]:
    return {key for key in header.tensors for pattern in patterns if _matches(key, pattern)}


def _adaln_block_seeds(header: SafetensorsHeader, start: int, end: int) -> set[str]:
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        raise H3HybridCompatibilityError("Block range values must be integers.")
    if start < 0 or end > 49 or start > end:
        raise H3HybridCompatibilityError(f"Invalid block range {start}..{end}; expected 0..49 with start <= end.")
    seeds = set()
    for index in range(start, end + 1):
        prefix = f"blocks.{index}.adaln_proj.linear."
        seeds.update(key for key in header.tensors if key.startswith(prefix))
    return seeds


def _final_seeds(header: SafetensorsHeader) -> set[str]:
    prefix = "final_layer.adaln_proj.linear."
    return {key for key in header.tensors if key.startswith(prefix)}


def _plan_fingerprint(
    fl_header: SafetensorsHeader,
    ref_header: SafetensorsHeader,
    profile: str,
    selected_keys: tuple[str, ...],
    weight_dtype: str,
) -> str:
    payload = {
        "fl": [fl_header.path.name, fl_header.file_size, fl_header.mtime_ns],
        "ref": [ref_header.path.name, ref_header.file_size, ref_header.mtime_ns],
        "profile": profile,
        "selected_keys": selected_keys,
        "weight_dtype": weight_dtype,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_hybrid_plan(
    fl_path: str | Path,
    ref_path: str | Path,
    profile: str = "Recommended",
    weight_dtype: str = "default",
    block_range_start: int = 25,
    block_range_end: int = 49,
    final_adaln_from_ref: bool = False,
    custom_ref: str = "",
    custom_fl: str = "",
) -> HybridPlan:
    """Build a complete compatibility plan before any checkpoint tensor is read."""

    if profile not in HYBRID_PROFILES:
        raise H3HybridCompatibilityError(f"Profile {profile!r} is not a hybrid profile.")
    if not isinstance(final_adaln_from_ref, bool):
        raise H3HybridCompatibilityError("final_adaln_from_ref must be a boolean.")
    fl_header = read_safetensors_header(fl_path)
    ref_header = read_safetensors_header(ref_path)
    validate_h3_layout(fl_header, "FL checkpoint")
    validate_h3_layout(ref_header, "REF checkpoint")

    fl_patterns: tuple[str, ...] = ()
    if profile == "Recommended":
        seeds = _adaln_block_seeds(ref_header, 25, 49)
        final_source = "FL"
    elif profile == "All Block AdaLN":
        seeds = _adaln_block_seeds(ref_header, 0, 49)
        final_source = "FL"
    elif profile == "All Block AdaLN + Final":
        seeds = _adaln_block_seeds(ref_header, 0, 49) | _final_seeds(ref_header)
        final_source = "REF"
    elif profile == "Custom Range":
        seeds = _adaln_block_seeds(ref_header, block_range_start, block_range_end)
        if final_adaln_from_ref:
            seeds |= _final_seeds(ref_header)
        final_source = "REF" if final_adaln_from_ref else "FL"
    else:
        ref_patterns = _patterns(custom_ref)
        if not ref_patterns:
            raise H3HybridCompatibilityError("Advanced Custom requires at least one custom_ref pattern.")
        seeds = _matching_keys(ref_header, ref_patterns)
        if not seeds:
            raise H3HybridCompatibilityError("Advanced Custom custom_ref patterns matched no REF tensors.")
        if final_adaln_from_ref:
            seeds |= _final_seeds(ref_header)
        fl_patterns = _patterns(custom_fl)
        final_source = "REF" if any(key.startswith("final_layer.adaln_proj.linear.") for key in seeds) else "FL"

    families = resolve_compatible_families(fl_header, ref_header, seeds)
    if fl_patterns:
        families = tuple(
            family
            for family in families
            if not any(
                _matches(key, pattern) or _matches(family.stem, pattern)
                for pattern in fl_patterns
                for key in family.keys
            )
        )
        if not families:
            raise H3HybridCompatibilityError("Advanced Custom custom_fl removed every selected REF family.")
        final_source = (
            "REF"
            if any(family.stem == "final_layer.adaln_proj.linear" for family in families)
            else "FL"
        )
    selected_keys = tuple(sorted({key for family in families for key in family.keys}))
    selected_tensors = tuple(ref_header.tensors[key] for key in selected_keys)
    selected_bytes = sum(ref_header.tensors[key].nbytes for key in selected_keys)
    warnings = ()
    if selected_bytes > 1024**3:
        warnings = ("Selected REF overlay exceeds 1 GiB; this checkpoint representation requires substantial host memory.",)
    return HybridPlan(
        fl_path=fl_header.path,
        ref_path=ref_header.path,
        fl_file_size=fl_header.file_size,
        fl_mtime_ns=fl_header.mtime_ns,
        ref_file_size=ref_header.file_size,
        ref_mtime_ns=ref_header.mtime_ns,
        profile=profile,
        selected_families=families,
        selected_tensors=selected_tensors,
        selected_keys=selected_keys,
        selected_bytes=selected_bytes,
        selected_blocks=selected_block_indices(families),
        final_adaln_source=final_source,
        weight_dtype=weight_dtype,
        warnings=warnings,
        fingerprint=_plan_fingerprint(fl_header, ref_header, profile, selected_keys, weight_dtype),
    )


def validate_plan_file_identity(plan: HybridPlan, *, check_fl: bool = True, check_ref: bool = True) -> None:
    """Reject checkpoint replacement between header planning and tensor reads."""

    checks = []
    if check_fl:
        checks.append(("FL", plan.fl_path, plan.fl_file_size, plan.fl_mtime_ns))
    if check_ref:
        checks.append(("REF", plan.ref_path, plan.ref_file_size, plan.ref_mtime_ns))
    for label, path, expected_size, expected_mtime in checks:
        try:
            current = path.stat()
        except OSError:
            raise H3HybridCompatibilityError(f"{label} checkpoint is no longer available: {path.name!r}.") from None
        if current.st_size != expected_size or current.st_mtime_ns != expected_mtime:
            raise H3HybridCompatibilityError(
                f"{label} checkpoint changed after HybridPlan validation: {path.name!r}."
            )
