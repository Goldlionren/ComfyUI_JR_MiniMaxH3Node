"""Safe, lazy media resolution for Director Desk assets."""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from .director_pipe import RuntimeMedia
from .director_state import AssetDescriptor, DirectorState

IMAGE_MAX_BYTES = 128 * 1024 * 1024
AUDIO_MAX_BYTES = 512 * 1024 * 1024
VIDEO_MAX_BYTES = 4 * 1024 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MAX_TOTAL_IMAGE_PIXELS = 24_000_000
MAX_RUNTIME_ASSETS = 32
PROBE_TIMEOUT_SECONDS = 5
PROBE_OUTPUT_LIMIT = 1024 * 1024


class DirectorMediaError(ValueError):
    """A media descriptor is missing, corrupt, unsafe, or unsupported."""


def _folder_paths():
    try:
        import folder_paths
    except ImportError as error:
        raise RuntimeError("Director Desk media resolution requires ComfyUI folder_paths.") from error
    return folder_paths


def resolve_asset_path(asset: AssetDescriptor) -> Path:
    fp = _folder_paths()
    root_value = fp.get_directory_by_type(asset.folder_type)
    if not root_value:
        raise DirectorMediaError(f"Unsupported ComfyUI asset type for {asset.display_name!r}.")
    try:
        root = Path(root_value).resolve()
        candidate = (root / asset.subfolder / asset.filename).resolve()
    except (OSError, RuntimeError):
        raise DirectorMediaError(f"Director asset path could not be resolved: {asset.display_name}.") from None
    try:
        inside = os.path.commonpath((str(root), str(candidate))) == str(root)
    except ValueError:
        inside = False
    if not inside:
        raise DirectorMediaError(f"Director asset resolves outside the ComfyUI {asset.folder_type} directory.")
    if not candidate.is_file():
        raise DirectorMediaError(f"Director asset is missing: {asset.display_name}.")
    return candidate


def _file_limit(kind: str) -> int:
    return {"image": IMAGE_MAX_BYTES, "audio": AUDIO_MAX_BYTES, "video": VIDEO_MAX_BYTES}[kind]


def _basic_metadata(asset: AssetDescriptor, path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError:
        raise DirectorMediaError(f"Director asset metadata is unavailable: {asset.display_name}.") from None
    if size <= 0:
        raise DirectorMediaError(f"Director asset is empty: {asset.display_name}.")
    if size > _file_limit(asset.kind):
        raise DirectorMediaError(f"Director {asset.kind} asset is too large: {asset.display_name}.")
    return {
        "status": "ready",
        "size_bytes": size,
        "mime_type": mimetypes.guess_type(asset.filename)[0] or asset.mime_type or "application/octet-stream",
    }


def _find_ffprobe() -> str | None:
    configured = os.environ.get("JR_H3_FFPROBE", "").strip()
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("ffprobe")
    if found:
        return found
    try:
        import imageio_ffmpeg

        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None
    candidate = ffmpeg.with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    return str(candidate) if candidate.is_file() else None


def _probe_image(asset: AssetDescriptor, path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise DirectorMediaError(f"Director image dimensions are invalid: {asset.display_name}.")
            image.verify()
    except DirectorMediaError:
        raise
    except Exception:
        raise DirectorMediaError(f"Director image is corrupt or unsupported: {asset.display_name}.") from None
    metadata.update(width=int(width), height=int(height))
    return metadata


def _probe_av(asset: AssetDescriptor, path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    ffprobe = _find_ffprobe()
    if ffprobe is None:
        metadata["status"] = "probe_unavailable"
        if asset.duration_seconds is not None:
            metadata["duration_seconds"] = asset.duration_seconds
        return metadata
    command = [
        ffprobe, "-v", "error", "-probesize", "5M", "-analyzeduration", "5M",
        "-print_format", "json", "-show_entries",
        "format=duration,format_name,size:stream=codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels,duration",
        "--", str(path),
    ]
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=False,
        )
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        overflow = threading.Event()

        def read_limited(name, stream, limit):
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    break
                if len(buffers[name]) + len(chunk) > limit:
                    overflow.set()
                    process.kill()
                    break
                buffers[name].extend(chunk)

        readers = [
            threading.Thread(target=read_limited, args=("stdout", process.stdout, PROBE_OUTPUT_LIMIT), daemon=True),
            threading.Thread(target=read_limited, args=("stderr", process.stderr, 65_536), daemon=True),
        ]
        for reader in readers:
            reader.start()
        try:
            return_code = process.wait(timeout=PROBE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
            raise DirectorMediaError(f"Director media probe timed out: {asset.display_name}.") from None
        finally:
            for reader in readers:
                reader.join(timeout=2)
            process.stdout.close()
            process.stderr.close()
        if overflow.is_set():
            raise DirectorMediaError(f"Director media probe output is too large: {asset.display_name}.")
        stdout = bytes(buffers["stdout"]).decode("utf-8", errors="replace")
    except DirectorMediaError:
        raise
    except OSError:
        raise DirectorMediaError(f"Director media probe could not start: {asset.display_name}.") from None
    if return_code != 0:
        raise DirectorMediaError(f"Director media is corrupt or unsupported: {asset.display_name}.") from None
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        raise DirectorMediaError(f"Director media probe returned invalid metadata: {asset.display_name}.") from None
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list):
        raise DirectorMediaError(f"Director media has no readable streams: {asset.display_name}.")
    wanted = asset.kind
    matches = [stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == wanted]
    if not matches:
        raise DirectorMediaError(f"Director {wanted} asset has no {wanted} stream: {asset.display_name}.")
    stream = matches[0]
    format_data = payload.get("format", {}) if isinstance(payload.get("format"), dict) else {}
    raw_duration = format_data.get("duration", stream.get("duration", asset.duration_seconds))
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError):
        duration = asset.duration_seconds
    if duration is not None and duration > 0:
        metadata["duration_seconds"] = duration
    metadata["codec_name"] = str(stream.get("codec_name", ""))
    if asset.kind == "video":
        for key in ("width", "height"):
            value = stream.get(key)
            if isinstance(value, int) and value > 0:
                metadata[key] = value
        metadata["fps"] = str(stream.get("avg_frame_rate", ""))
    else:
        metadata["sample_rate"] = str(stream.get("sample_rate", ""))
        channels = stream.get("channels")
        if isinstance(channels, int):
            metadata["channels"] = channels
    return metadata


def probe_asset(asset: AssetDescriptor) -> tuple[AssetDescriptor, dict[str, Any]]:
    path = resolve_asset_path(asset)
    metadata = _basic_metadata(asset, path)
    metadata = _probe_image(asset, path, metadata) if asset.kind == "image" else _probe_av(asset, path, metadata)
    updated = replace(
        asset,
        mime_type=str(metadata.get("mime_type", asset.mime_type)),
        duration_seconds=metadata.get("duration_seconds", asset.duration_seconds),
        width=metadata.get("width", asset.width),
        height=metadata.get("height", asset.height),
        status=str(metadata.get("status", "ready")),
    )
    return updated, metadata


def _load_image(path: Path, asset: AssetDescriptor):
    try:
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)
    except Exception:
        raise DirectorMediaError(f"Director image could not be decoded: {asset.display_name}.") from None


def resolve_runtime_media(state: DirectorState) -> tuple[RuntimeMedia, ...]:
    runtime = []
    cache = {}
    total_image_pixels = 0
    unique_assets = {
        (item.asset.kind, item.asset.folder_type, item.asset.subfolder, item.asset.filename)
        for item in (*state.visual_items, *state.audio_items)
    }
    if len(unique_assets) > MAX_RUNTIME_ASSETS:
        raise DirectorMediaError(
            f"Director Desk supports at most {MAX_RUNTIME_ASSETS} unique runtime assets per execution."
        )
    for item in (*state.visual_items, *state.audio_items):
        key = (item.asset.kind, item.asset.folder_type, item.asset.subfolder, item.asset.filename)
        cached = cache.get(key)
        if cached is None:
            path = resolve_asset_path(item.asset)
            _, metadata = probe_asset(item.asset)
            if item.asset.kind == "image":
                pixels = int(metadata.get("width", 0)) * int(metadata.get("height", 0))
                total_image_pixels += pixels
                if total_image_pixels > MAX_TOTAL_IMAGE_PIXELS:
                    raise DirectorMediaError(
                        "Director Desk IMAGE references exceed the aggregate pixel budget."
                    )
                payload = _load_image(path, item.asset)
            else:
                payload = None
            cached = (metadata, payload)
            cache[key] = cached
        metadata, payload = cached
        safe_metadata = {
            key: value for key, value in metadata.items()
            if key in {
                "status", "size_bytes", "mime_type", "duration_seconds", "width", "height",
                "codec_name", "fps", "sample_rate", "channels",
            }
        }
        runtime.append(RuntimeMedia(
            asset_id=item.asset.id, item_id=item.id, kind=item.asset.kind,
            payload=payload, metadata=tuple(sorted(safe_metadata.items())),
        ))
    return tuple(runtime)


__all__ = [
    "DirectorMediaError", "probe_asset", "resolve_asset_path", "resolve_runtime_media",
]
