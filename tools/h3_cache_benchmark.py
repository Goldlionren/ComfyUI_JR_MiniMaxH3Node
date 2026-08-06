"""Create and render manual MiniMax H3 cache benchmark records.

This tool records measurements supplied by a real workflow run; it never loads H3
and is intentionally excluded from pytest's long-running work.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BenchmarkRecord:
    model: str
    quantization: str
    attention_backend: str
    seed: int
    steps: int
    resolution: str
    frames: int
    duration: float
    cache_mode: str
    cache_device: str
    total_time: float
    denoise_time: float
    full_forwards: int
    full_step_hits: int
    block_hits: int
    forced_refreshes: int
    audio_vetoes: int
    video_vetoes: int
    cache_bytes: int


FIELDS = list(BenchmarkRecord.__annotations__)


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Benchmark JSON must contain a list.")
    return data


def write_records(records: list[dict], path: Path):
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
    elif suffix in {".md", ".markdown"}:
        lines = ["| " + " | ".join(FIELDS) + " |", "| " + " | ".join("---" for _ in FIELDS) + " |"]
        lines.extend("| " + " | ".join(str(row.get(field, "")) for field in FIELDS) + " |" for row in records)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        raise ValueError("Output extension must be .json, .csv, or .md.")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Record manually measured MiniMax H3 cache benchmarks.")
    result.add_argument("--database", type=Path, default=Path("h3_cache_benchmarks.json"))
    result.add_argument("--export", type=Path, help="Render the current JSON database to JSON, CSV, or Markdown.")
    result.add_argument("--model")
    result.add_argument("--quantization", default="none")
    result.add_argument("--attention-backend", default="default")
    result.add_argument("--seed", type=int)
    result.add_argument("--steps", type=int)
    result.add_argument("--resolution")
    result.add_argument("--frames", type=int)
    result.add_argument("--duration", type=float)
    result.add_argument("--cache-mode", choices=["No Cache", "ComfyUI EasyCache", "JR Visual Fast",
                                                   "JR Dialogue Safe", "JR Action Safe", "JR Balanced"])
    result.add_argument("--cache-device", default="None")
    result.add_argument("--total-time", type=float)
    result.add_argument("--denoise-time", type=float)
    for name in ("full_forwards", "full_step_hits", "block_hits", "forced_refreshes",
                 "audio_vetoes", "video_vetoes", "cache_bytes"):
        result.add_argument("--" + name.replace("_", "-"), type=int, default=0)
    return result


def main():
    args = parser().parse_args()
    records = load_records(args.database)
    if args.export:
        write_records(records, args.export)
        return
    required = ("model", "seed", "steps", "resolution", "frames", "duration", "cache_mode",
                "total_time", "denoise_time")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit("Missing measurement fields: " + ", ".join(missing))
    record = BenchmarkRecord(**{field: getattr(args, field) for field in FIELDS})
    records.append(asdict(record))
    write_records(records, args.database)


if __name__ == "__main__":
    main()
