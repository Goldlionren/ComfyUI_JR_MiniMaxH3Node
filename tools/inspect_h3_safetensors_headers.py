"""Inspect MiniMax H3 safetensors headers without materializing tensor data."""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path

DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def read_header(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        length_data = handle.read(8)
        if len(length_data) != 8:
            raise ValueError(f"Invalid safetensors header prefix: {path.name}")
        header_length = struct.unpack("<Q", length_data)[0]
        if header_length > 100 * 1024 * 1024:
            raise ValueError(f"Safetensors header is unexpectedly large: {header_length}")
        payload = handle.read(header_length)
    if len(payload) != header_length:
        raise ValueError(f"Truncated safetensors header: {path.name}")
    return json.loads(payload)


def tensor_bytes(info: dict[str, object]) -> int:
    offsets = info["data_offsets"]
    return int(offsets[1]) - int(offsets[0])


def module_stem(key: str) -> str | None:
    for suffix in (".weight", ".bias", ".weight_scale", ".bias_scale", ".comfy_quant"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def summarize(path: Path) -> None:
    header = read_header(path)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    dtype_counts = Counter(info["dtype"] for info in tensors.values())
    adaln = sorted(key for key in tensors if ".adaln_proj." in key)
    recommended = [key for key in adaln if any(key.startswith(f"blocks.{i}.adaln_proj.") for i in range(25, 50))]
    stems = sorted({stem for key in adaln if (stem := module_stem(key))})
    print(f"FILE={path.name}")
    print(f"SIZE_BYTES={path.stat().st_size}")
    print(f"TENSORS={len(tensors)}")
    print(f"DTYPES={dict(sorted(dtype_counts.items()))}")
    print(f"METADATA={header.get('__metadata__', {})}")
    print(f"ADALN_KEYS={len(adaln)} RECOMMENDED_KEYS={len(recommended)}")
    print(f"RECOMMENDED_BYTES={sum(tensor_bytes(tensors[key]) for key in recommended)}")
    print("ADALN_STEMS_SAMPLE=" + json.dumps(stems[:6] + stems[-3:], ensure_ascii=False))
    for stem in stems[:2] + stems[25:27] + stems[-1:]:
        family = sorted(key for key in tensors if key == stem or key.startswith(stem + "."))
        compact = [(key, tensors[key]["dtype"], tensors[key]["shape"], tensor_bytes(tensors[key])) for key in family]
        print("FAMILY=" + json.dumps(compact, ensure_ascii=False))
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        summarize(path)


if __name__ == "__main__":
    main()
