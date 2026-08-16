"""Header-driven tensor-family resolution for MiniMax H3 hybrid overlays.

The preset and family-provenance ideas are adapted from Scott Mudge's MIT
ComfyUI_MinimaxH3HybridLoader at commit a44c69b02242e41fbd01e22abe2a492adc853038.
The selective-reader and compatibility implementation is specific to this
project and does not use the upstream dual-safe_open full-state loader.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

MAX_HEADER_BYTES = 100 * 1024 * 1024
_PARAMETER_PREFIXES = ("weight", "bias")
_INTEGER_DTYPES = {"I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64"}
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}


class H3HybridCompatibilityError(ValueError):
    """Raised when selected FL and REF tensor families cannot be overlaid."""


@dataclass(frozen=True, slots=True)
class TensorHeader:
    key: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


@dataclass(frozen=True, slots=True)
class SafetensorsHeader:
    path: Path
    file_size: int
    mtime_ns: int
    metadata: Mapping[str, str]
    tensors: Mapping[str, TensorHeader]


@dataclass(frozen=True, slots=True)
class TensorFamily:
    stem: str
    keys: tuple[str, ...]
    quantized: bool
    nbytes: int


def _parse_tensor_header(key: str, raw: object, data_size: int) -> TensorHeader:
    if not isinstance(raw, dict):
        raise H3HybridCompatibilityError(f"Invalid safetensors entry for {key!r}.")
    dtype = raw.get("dtype")
    shape = raw.get("shape")
    offsets = raw.get("data_offsets")
    if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list):
        raise H3HybridCompatibilityError(f"Invalid safetensors tensor metadata for {key!r}.")
    if len(offsets) != 2 or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets):
        raise H3HybridCompatibilityError(f"Invalid safetensors offsets for {key!r}.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape):
        raise H3HybridCompatibilityError(f"Invalid safetensors shape for {key!r}.")
    start, end = offsets
    if start < 0 or end < start or end > data_size:
        raise H3HybridCompatibilityError(f"Safetensors offsets are out of bounds for {key!r}.")
    element_size = _DTYPE_BYTES.get(dtype)
    if element_size is None:
        raise H3HybridCompatibilityError(f"Unsupported safetensors dtype {dtype!r} for {key!r}.")
    expected_bytes = math.prod(shape) * element_size
    if end - start != expected_bytes:
        raise H3HybridCompatibilityError(f"Safetensors byte size does not match shape/dtype for {key!r}.")
    return TensorHeader(key, dtype, tuple(shape), (start, end))


def read_safetensors_header(path: str | Path) -> SafetensorsHeader:
    """Read and validate only the JSON header of a safetensors checkpoint."""

    candidate = Path(path)
    if candidate.suffix.lower() not in {".safetensors", ".sft"}:
        raise H3HybridCompatibilityError(
            f"Unsupported hybrid checkpoint {candidate.name!r}; Hybrid profiles require safetensors."
        )
    try:
        resolved = candidate.resolve(strict=True)
        stat = resolved.stat()
        file_size = stat.st_size
        with resolved.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise H3HybridCompatibilityError(f"Corrupt safetensors header: {resolved.name}.")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size <= 0 or header_size > MAX_HEADER_BYTES or 8 + header_size > file_size:
                raise H3HybridCompatibilityError(
                    f"Invalid safetensors header size for {resolved.name}: {header_size}."
                )
            payload = handle.read(header_size)
    except H3HybridCompatibilityError:
        raise
    except (OSError, RuntimeError):
        raise H3HybridCompatibilityError(f"Cannot read checkpoint {candidate.name!r}.") from None

    try:
        raw_header = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise H3HybridCompatibilityError(f"Corrupt safetensors JSON header: {resolved.name}.") from None
    if not isinstance(raw_header, dict):
        raise H3HybridCompatibilityError(f"Invalid safetensors header object: {resolved.name}.")

    raw_metadata = raw_header.get("__metadata__", {})
    if raw_metadata is None:
        raw_metadata = {}
    if not isinstance(raw_metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in raw_metadata.items()
    ):
        raise H3HybridCompatibilityError(f"Invalid safetensors metadata: {resolved.name}.")
    data_size = file_size - 8 - header_size
    tensors = {
        key: _parse_tensor_header(key, value, data_size)
        for key, value in raw_header.items()
        if key != "__metadata__" and isinstance(key, str)
    }
    if not tensors:
        raise H3HybridCompatibilityError(f"Checkpoint contains no tensors: {resolved.name}.")
    return SafetensorsHeader(
        path=resolved,
        file_size=file_size,
        mtime_ns=stat.st_mtime_ns,
        metadata=MappingProxyType(dict(raw_metadata)),
        tensors=MappingProxyType(tensors),
    )


def validate_h3_layout(header: SafetensorsHeader, label: str) -> None:
    """Fail early unless the checkpoint has the current 50-block H3 layout."""

    missing = [
        f"blocks.{index}.adaln_proj.linear.weight"
        for index in range(50)
        if f"blocks.{index}.adaln_proj.linear.weight" not in header.tensors
    ]
    final_key = "final_layer.adaln_proj.linear.weight"
    if final_key not in header.tensors:
        missing.append(final_key)
    if missing:
        preview = ", ".join(missing[:3])
        raise H3HybridCompatibilityError(
            f"{label} is not a supported MiniMax H3 checkpoint; missing {preview}."
        )


def _is_parameter_tail(tail: str) -> bool:
    if tail == "comfy_quant":
        return True
    return any(tail == prefix or tail.startswith(prefix + "_") for prefix in _PARAMETER_PREFIXES)


def family_stem_for_key(key: str, _all_keys: Mapping[str, TensorHeader]) -> str:
    """Infer a module stem using the sibling names actually present in the header."""

    if "." not in key:
        return key
    candidate, tail = key.rsplit(".", 1)
    if _is_parameter_tail(tail):
        return candidate
    if any(
        sibling == candidate + ".comfy_quant"
        or sibling.startswith(candidate + ".weight")
        or sibling.startswith(candidate + ".bias")
        for sibling in _all_keys
    ):
        return candidate
    # Unknown future layouts remain singleton families rather than guessing.
    return key


def _family_members(stem: str, tensors: Mapping[str, TensorHeader]) -> tuple[str, ...]:
    members = []
    prefix = stem + "."
    for key in tensors:
        if key.startswith(prefix):
            members.append(key)
    if members:
        # Once a concrete parameter identifies the module stem, every actual
        # descendant in the header belongs to that module representation. This
        # automatically carries future quant metadata without guessing names.
        return tuple(sorted(members))
    if stem in tensors:
        return (stem,)
    return ()


def _quantized(members: tuple[str, ...], tensors: Mapping[str, TensorHeader]) -> bool:
    for key in members:
        tail = key.rsplit(".", 1)[-1]
        if tail == "comfy_quant" or "scale" in tail or "zero" in tail or tensors[key].dtype in _INTEGER_DTYPES:
            return True
    return False


def resolve_compatible_families(
    fl_header: SafetensorsHeader,
    ref_header: SafetensorsHeader,
    selected_seed_keys: set[str],
) -> tuple[TensorFamily, ...]:
    """Expand selected keys to complete header-derived families and compare them."""

    if not selected_seed_keys:
        raise H3HybridCompatibilityError("Hybrid profile selected no REF tensors.")
    missing_ref = sorted(selected_seed_keys - ref_header.tensors.keys())
    if missing_ref:
        raise H3HybridCompatibilityError(f"Missing selected REF tensor: {missing_ref[0]}.")

    stems = sorted({family_stem_for_key(key, ref_header.tensors) for key in selected_seed_keys})
    families: list[TensorFamily] = []
    for stem in stems:
        ref_members = _family_members(stem, ref_header.tensors)
        fl_members = _family_members(stem, fl_header.tensors)
        if not ref_members:
            raise H3HybridCompatibilityError(f"Missing selected REF tensor family: {stem}.")
        if not fl_members:
            raise H3HybridCompatibilityError(f"Missing selected FL tensor family: {stem}.")
        ref_suffixes = {key[len(stem) :] for key in ref_members}
        fl_suffixes = {key[len(stem) :] for key in fl_members}
        if ref_suffixes != fl_suffixes:
            raise H3HybridCompatibilityError(
                f"Quant family mismatch for {stem}: FL={sorted(fl_suffixes)}, REF={sorted(ref_suffixes)}."
            )
        for ref_key in ref_members:
            suffix = ref_key[len(stem) :]
            fl_key = stem + suffix
            ref_info = ref_header.tensors[ref_key]
            fl_info = fl_header.tensors[fl_key]
            if ref_info.shape != fl_info.shape:
                raise H3HybridCompatibilityError(
                    f"Selected family shape mismatch for {ref_key}: FL={fl_info.shape}, REF={ref_info.shape}."
                )
            if ref_info.dtype != fl_info.dtype:
                raise H3HybridCompatibilityError(
                    f"Selected family dtype mismatch for {ref_key}: FL={fl_info.dtype}, REF={ref_info.dtype}."
                )
        families.append(
            TensorFamily(
                stem=stem,
                keys=ref_members,
                quantized=_quantized(ref_members, ref_header.tensors),
                nbytes=sum(ref_header.tensors[key].nbytes for key in ref_members),
            )
        )
    return tuple(families)


def selected_block_indices(families: tuple[TensorFamily, ...]) -> tuple[int, ...]:
    blocks = set()
    for family in families:
        parts = family.stem.split(".")
        if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit():
            blocks.add(int(parts[1]))
    return tuple(sorted(blocks))
