"""FFmpeg video output node with automatic encoding and in-node preview support."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CODECS = ["Auto", "AV1", "VP9", "H.265 (HEVC)", "H.264"]
CONTAINERS = ["Auto", "WebM", "MKV", "MP4", "Animated WebP", "Animated AVIF"]
BIT_DEPTHS = ["Auto", "8-bit", "10-bit"]
AUDIO_CODECS = ["Auto", "AAC", "Opus", "MP3"]
AUDIO_BITRATES = ["64k", "96k", "128k", "160k", "192k", "256k", "320k"]

_VIDEO_EXTENSIONS = {"WebM": ".webm", "MKV": ".mkv", "MP4": ".mp4"}
_VIDEO_MIME_TYPES = {"WebM": "video/webm", "MKV": "video/x-matroska", "MP4": "video/mp4"}
_ANIMATION_FORMATS = {
    "Animated WebP": (".webp", "image/webp"),
    "Animated AVIF": (".avif", "image/avif"),
}
_VIDEO_ENCODERS = {
    "H.264": ("h264_nvenc", "h264_qsv", "h264_amf", "h264_vaapi", "libx264"),
    "H.265 (HEVC)": ("hevc_nvenc", "hevc_qsv", "hevc_amf", "hevc_vaapi", "libx265"),
    "AV1": ("av1_nvenc", "av1_qsv", "av1_amf", "av1_vaapi", "libsvtav1", "libaom-av1"),
    "VP9": ("vp9_qsv", "vp9_vaapi", "libvpx-vp9"),
}
_AVIF_ENCODERS = ("av1_nvenc", "av1_qsv", "av1_amf", "av1_vaapi", "libsvtav1", "libaom-av1")
_AUDIO_ENCODER_NAMES = {"AAC": "aac", "Opus": "libopus", "MP3": "libmp3lame"}
_MAX_CHUNK_BYTES = 64 * 1024 * 1024
_MAX_ERROR_BYTES = 4 * 1024 * 1024
_ENCODE_TIMEOUT_SECONDS = 3600
_HARDWARE_STARTUP_TIMEOUT_SECONDS = 8
_SOFTWARE_STARTUP_TIMEOUT_SECONDS = 120
_PROGRESS_STALL_TIMEOUT_SECONDS = 120
_OUTPUT_ALLOCATION_LOCK = threading.Lock()


def _log(message: str) -> None:
    print(f"[JR MiniMax H3 Enhanced Video Combine] {message}")


def find_ffmpeg() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None


def _folder_paths():
    try:
        import folder_paths

        return folder_paths
    except ImportError as error:
        raise RuntimeError("ComfyUI folder_paths is unavailable; run this node inside ComfyUI.") from error


def pingpong_frames(images: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled or images.shape[0] < 3:
        return images
    return torch.cat((images, images[1:-1].flip(0)), dim=0)


def _encoded_frame_count(images: torch.Tensor, pingpong: bool) -> int:
    return int(images.shape[0] + (images.shape[0] - 2 if pingpong and images.shape[0] >= 3 else 0))


def detect_bit_depth(images: torch.Tensor) -> int:
    """Estimate whether normalized samples lie more closely on an 8- or 10-bit grid."""
    if not torch.is_floating_point(images):
        return 10 if images.element_size() >= 2 else 8
    samples = images.detach().to(device="cpu", dtype=torch.float32).flatten()
    if samples.numel() == 0:
        return 8
    if samples.numel() > 250_000:
        samples = samples[:: max(1, samples.numel() // 250_000)]
    samples = samples.clamp(0, 1)
    error8 = torch.mean(torch.abs(samples * 255 - torch.round(samples * 255)))
    error10 = torch.mean(torch.abs(samples * 1023 - torch.round(samples * 1023)))
    return 10 if error10 < error8 * 0.8 else 8


def _resolve_bit_depth(codec: str, choice: str, images: torch.Tensor) -> int:
    if choice == "8-bit":
        return 8
    if choice == "10-bit":
        return 10
    return 8 if codec == "Auto" else detect_bit_depth(images)


def _format_date_tokens(value: str, now: dt.datetime | None = None) -> str:
    moment = now or dt.datetime.now()

    def replace(match: re.Match) -> str:
        pattern = match.group(1) or "yyyyMMdd_HHmmss"
        for source, target in (
            ("yyyy", "%Y"), ("yy", "%y"), ("MM", "%m"), ("dd", "%d"),
            ("DD", "%d"), ("HH", "%H"), ("hh", "%H"), ("mm", "%M"), ("ss", "%S"),
        ):
            pattern = pattern.replace(source, target)
        return moment.strftime(pattern)

    return re.sub(r"%date(?::([^%]+))?%", replace, str(value), flags=re.IGNORECASE)


def _safe_relative_prefix(value: str) -> str:
    expanded = value.replace("\\", "/").strip().lstrip("/")
    components = []
    for raw_part in expanded.split("/"):
        part = raw_part.strip().strip(".")
        if not part or part in {".", ".."}:
            continue
        clean = "".join(ch for ch in part if ch not in '<>:"|?*' and ord(ch) >= 32)
        if clean:
            components.append(clean[:120])
    return "/".join(components) or "jr_h3_video"


def _allocate_output(prefix: str, output_dir: str, width: int, height: int, filename_tag: str | None = None):
    fp = _folder_paths()
    safe_prefix = _safe_relative_prefix(_format_date_tokens(prefix))
    if filename_tag:
        parent, separator, name = safe_prefix.rpartition("/")
        safe_prefix = f"{parent}{separator}{name}_{filename_tag}"
    if hasattr(fp, "get_save_image_path"):
        folder, basename, counter, subfolder, _ = fp.get_save_image_path(safe_prefix, output_dir, width, height)
        folder = Path(folder)
        counter = int(counter)
    else:
        relative = Path(*safe_prefix.split("/"))
        folder = Path(output_dir, relative.parent).resolve()
        root = Path(output_dir).resolve()
        if os.path.commonpath((str(folder), str(root))) != str(root):
            raise ValueError("filename_prefix resolves outside the ComfyUI output directory.")
        basename = relative.name
        counter = 1
        subfolder = "" if folder == root else folder.relative_to(root).as_posix()
    folder.mkdir(parents=True, exist_ok=True)
    # ComfyUI's image counter does not recognize names such as
    # ``clip_00001-audio.mp4``. Check every output extension/marker ourselves so
    # repeated video executions never overwrite an earlier result.
    with _OUTPUT_ALLOCATION_LOCK:
        existing = {entry.name.casefold() for entry in folder.iterdir() if entry.is_file()}
        counter = max(1, counter)
        while any(name.startswith(f"{basename}_{counter:05d}".casefold()) for name in existing):
            counter += 1
    return folder, basename, counter, subfolder


def _output_name(basename: str, counter: int, extension: str) -> str:
    return f"{basename}_{counter:05d}{extension}"


def _to_raw_bytes(frames: torch.Tensor, bit_depth: int) -> bytes:
    rgb = frames[..., :3].detach().to(device="cpu", dtype=torch.float32).clamp_(0, 1)
    if bit_depth == 10:
        return torch.round(rgb * 1023).to(torch.int32).mul_(64).to(torch.uint16).numpy().tobytes()
    return torch.round(rgb * 255).to(torch.uint8).numpy().tobytes()


def _iter_raw_chunks(images: torch.Tensor, bit_depth: int, pingpong: bool):
    bytes_per_pixel = 2 if bit_depth == 10 else 1
    frame_bytes = int(images.shape[1] * images.shape[2] * 3 * bytes_per_pixel)
    batch_size = max(1, min(32, _MAX_CHUNK_BYTES // frame_bytes))
    for start in range(0, len(images), batch_size):
        yield _to_raw_bytes(images[start:start + batch_size], bit_depth)
    if pingpong and len(images) >= 3:
        for stop in range(len(images) - 1, 1, -batch_size):
            start = max(1, stop - batch_size)
            yield _to_raw_bytes(images[start:stop].flip(0), bit_depth)


def _write_metadata(prompt, extra_pnginfo) -> str | None:
    if prompt is None and not extra_pnginfo:
        return None
    values = {}
    if prompt is not None:
        values["prompt"] = prompt
    if extra_pnginfo:
        values.update(extra_pnginfo)
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".ffmeta", delete=False, encoding="utf-8")
    try:
        handle.write(";FFMETADATA1\n")
        for key, value in values.items():
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
            encoded = encoded.replace("\\", "\\\\").replace(";", "\\;").replace("#", "\\#")
            encoded = encoded.replace("=", "\\=").replace("\n", "\\\n")
            handle.write(f"{key}={encoded}\n")
    except BaseException:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
        raise
    finally:
        handle.close()
    return handle.name


def _write_audio(audio) -> tuple[tuple[str, int, int] | None, float | None]:
    if audio is None:
        return None, None
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("audio must contain ComfyUI waveform and sample_rate values.")
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        raise ValueError("audio waveform must be a torch.Tensor.")
    values = waveform.detach().to(device="cpu", dtype=torch.float32)
    if values.ndim == 1:
        values = values[None, None, :]
    elif values.ndim == 2:
        values = values[None, :, :]
    if values.ndim != 3:
        raise ValueError("audio waveform must have shape [B,C,T], [C,T], or [T].")
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0 or values.shape[-1] < 1:
        raise ValueError("audio must have a positive sample rate and at least one sample.")
    channels = int(values.shape[1])
    interleaved = values.transpose(1, 2).reshape(-1, channels).clamp_(-1, 1).numpy().astype("<f4")
    handle = tempfile.NamedTemporaryFile(suffix=".f32le", delete=False)
    try:
        handle.write(interleaved.tobytes())
    except BaseException:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
        raise
    finally:
        handle.close()
    return (handle.name, sample_rate, channels), len(interleaved) / sample_rate


def _available_video_encoders(ffmpeg: str) -> set[str]:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20, check=False,
    )
    if result.returncode != 0:
        return set()
    available = set()
    for line in result.stdout.splitlines():
        match = re.match(r"^\s*V\S*\s+(\S+)", line)
        if match:
            available.add(match.group(1))
    return available


def _video_encoder_args(codec: str, encoder: str, bit_depth: int, quality: int) -> list[str]:
    pixel_format = "yuv420p10le" if bit_depth == 10 else "yuv420p"
    if encoder.endswith("_nvenc"):
        return ["-c:v", encoder, "-preset", "p5", "-cq", str(quality), "-pix_fmt", pixel_format]
    if encoder.endswith("_qsv"):
        return ["-c:v", encoder, "-global_quality", str(quality), "-pix_fmt", pixel_format]
    if encoder.endswith("_amf"):
        return ["-c:v", encoder, "-quality", "quality", "-qp_i", str(quality), "-qp_p", str(quality), "-pix_fmt", pixel_format]
    if encoder.endswith("_vaapi"):
        return ["-c:v", encoder, "-qp", str(quality), "-pix_fmt", pixel_format]
    if codec in {"H.264", "H.265 (HEVC)"}:
        return ["-c:v", encoder, "-crf", str(quality), "-preset", "medium", "-pix_fmt", pixel_format]
    if encoder == "libsvtav1":
        return ["-c:v", encoder, "-crf", str(quality), "-preset", "6", "-pix_fmt", pixel_format]
    if codec == "AV1":
        return ["-c:v", encoder, "-crf", str(quality), "-b:v", "0", "-cpu-used", "6", "-pix_fmt", pixel_format]
    return ["-c:v", encoder, "-crf", str(quality), "-b:v", "0", "-deadline", "good", "-pix_fmt", pixel_format]


def _preferred_audio_encoder(audio_codec: str, container: str) -> str:
    if audio_codec == "Auto":
        return "libopus" if container == "WebM" else "aac"
    return _AUDIO_ENCODER_NAMES[audio_codec]


def _audio_encoder_order(audio_codec: str, container: str) -> tuple[str, ...]:
    compatible = {
        "WebM": ("libopus",),
        "MKV": ("aac", "libopus", "libmp3lame", "pcm_s16le"),
        "MP4": ("aac", "libmp3lame"),
    }[container]
    return tuple(dict.fromkeys((_preferred_audio_encoder(audio_codec, container), *compatible)))


def _run_streaming_ffmpeg(command: list[str], chunks, progress=None) -> subprocess.CompletedProcess:
    progress = progress or (lambda _seconds: None)
    process = subprocess.Popen(
        [*command[:-1], "-progress", "pipe:2", "-nostats", command[-1]],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    error_tail = bytearray()
    last_update = 0.0
    last_progress = None
    greatest_encoded_time = 0
    started = time.monotonic()
    watchdog_stop = threading.Event()
    timed_out = threading.Event()

    try:
        encoder = command[command.index("-c:v") + 1]
    except (ValueError, IndexError):
        encoder = ""
    hardware_encoder = encoder.endswith(("_nvenc", "_qsv", "_amf", "_vaapi"))
    startup_timeout = _HARDWARE_STARTUP_TIMEOUT_SECONDS if hardware_encoder else _SOFTWARE_STARTUP_TIMEOUT_SECONDS

    def consume_stderr():
        nonlocal greatest_encoded_time, last_progress, last_update
        assert process.stderr is not None
        for raw_line in iter(process.stderr.readline, b""):
            error_tail.extend(raw_line)
            if len(error_tail) > _MAX_ERROR_BYTES:
                del error_tail[:-_MAX_ERROR_BYTES]
            try:
                key, value = raw_line.decode(errors="replace").strip().split("=", 1)
                if key in {"out_time_us", "out_time_ms"}:
                    encoded_time = int(value)
                    if encoded_time > greatest_encoded_time:
                        greatest_encoded_time = encoded_time
                        last_progress = time.monotonic()
                        if last_progress - last_update >= 0.5:
                            progress(encoded_time / 1_000_000)
                            last_update = last_progress
            except (ValueError, TypeError):
                continue

    def watch_process():
        while not watchdog_stop.wait(0.2):
            if process.poll() is not None:
                return
            now = time.monotonic()
            if last_progress is None:
                expired = now - started > startup_timeout
                reason = f"encoder produced no progress within {startup_timeout} seconds"
            else:
                expired = now - last_progress > _PROGRESS_STALL_TIMEOUT_SECONDS
                reason = f"encoder progress stalled for {_PROGRESS_STALL_TIMEOUT_SECONDS} seconds"
            if expired:
                timed_out.set()
                error_tail.extend(f"\nJR timeout: {reason}.".encode())
                process.kill()
                return

    reader = threading.Thread(target=consume_stderr, name="jr-h3-ffmpeg-progress", daemon=True)
    watchdog = threading.Thread(target=watch_process, name="jr-h3-ffmpeg-watchdog", daemon=True)
    reader.start()
    watchdog.start()
    try:
        if process.stdin is None:
            raise RuntimeError("FFmpeg stdin could not be opened.")
        for chunk in chunks():
            process.stdin.write(chunk)
        process.stdin.close()
        return_code = process.wait(timeout=_ENCODE_TIMEOUT_SECONDS)
    except BrokenPipeError:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        return_code = process.wait(timeout=_ENCODE_TIMEOUT_SECONDS)
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        watchdog_stop.set()
        reader.join(timeout=10)
        watchdog.join(timeout=2)
    return subprocess.CompletedProcess(command, -9 if timed_out.is_set() else return_code, stderr=bytes(error_tail))


def _input_command(metadata_path: str | None, width: int, height: int, frame_rate: float, bit_depth: int):
    command = ["-y", "-v", "error"]
    if metadata_path:
        command.extend(["-f", "ffmetadata", "-i", metadata_path])
    command.extend([
        "-f", "rawvideo", "-pix_fmt", "rgb48le" if bit_depth == 10 else "rgb24",
        "-s", f"{width}x{height}", "-framerate", str(frame_rate), "-i", "-",
    ])
    return command, 1 if metadata_path else 0


def _encode_video(
    ffmpeg: str, available: set[str], codec: str, container: str, bit_depth: int, width: int, height: int,
    frame_rate: float, chunks, output_path: Path, quality: int, metadata_path: str | None,
    audio_info, audio_duration: float | None, crop_to_audio: bool, audio_codec: str, audio_bitrate: str, progress,
) -> str:
    failures = []
    for encoder in _VIDEO_ENCODERS[codec]:
        if encoder not in available:
            continue
        audio_encoders = _audio_encoder_order(audio_codec, container) if audio_info else (None,)
        for selected_audio in audio_encoders:
            input_args, video_input = _input_command(metadata_path, width, height, frame_rate, bit_depth)
            command = [ffmpeg, *input_args]
            if audio_info:
                command.extend(["-f", "f32le", "-ar", str(audio_info[1]), "-ac", str(audio_info[2]), "-i", audio_info[0]])
            if metadata_path:
                command.extend(["-map", f"{video_input}:v:0", "-map_metadata", "0"])
            elif audio_info:
                command.extend(["-map", f"{video_input}:v:0"])
            if audio_info:
                command.extend(["-map", f"{video_input + 1}:a:0", "-c:a", selected_audio, "-b:a", audio_bitrate])
            command.extend(_video_encoder_args(codec, encoder, bit_depth, quality))
            if crop_to_audio and audio_duration is not None:
                command.extend(["-t", f"{audio_duration:.9f}"])
            if container == "MP4":
                command.extend(["-movflags", "+use_metadata_tags"])
            result = _run_streaming_ffmpeg([*command, str(output_path)], chunks, progress)
            if result.returncode == 0:
                if selected_audio and selected_audio != _preferred_audio_encoder(audio_codec, container):
                    _log(f"Audio encoder fallback selected: {selected_audio}.")
                return encoder
            output_path.unlink(missing_ok=True)
            lines = result.stderr.decode("utf-8", errors="replace").splitlines()
            failures.append(f"{encoder}/{selected_audio or 'no-audio'}: {(lines[0] if lines else 'unknown error')[:180]}")
    detail = " | ".join(failures) if failures else "no listed encoder is installed"
    raise RuntimeError(f"No usable {codec}/{container} encoder: {detail}")


def _encode_animation(
    ffmpeg: str, available: set[str], container: str, bit_depth: int, width: int, height: int,
    frame_rate: float, chunks, output_path: Path, quality: int, progress,
) -> str:
    candidates = ("libwebp_anim",) if container == "Animated WebP" else _AVIF_ENCODERS
    failures = []
    for encoder in candidates:
        if encoder not in available:
            continue
        input_args, _ = _input_command(None, width, height, frame_rate, bit_depth)
        command = [ffmpeg, *input_args]
        if container == "Animated WebP":
            command.extend(["-c:v", encoder, "-loop", "0", "-q:v", str(quality)])
        else:
            command.extend(_video_encoder_args("AV1", encoder, bit_depth, quality))
            command.extend(["-still-picture", "0", "-f", "avif"])
        result = _run_streaming_ffmpeg([*command, str(output_path)], chunks, progress)
        if result.returncode == 0:
            return encoder
        output_path.unlink(missing_ok=True)
        lines = result.stderr.decode("utf-8", errors="replace").splitlines()
        failures.append(f"{encoder}: {(lines[0] if lines else 'unknown error')[:180]}")
    detail = " | ".join(failures) if failures else "the required encoder is not installed"
    raise RuntimeError(f"{container} encoding failed: {detail}")


def _codec_order(codec: str) -> tuple[str, ...]:
    return ("AV1", "VP9", "H.264") if codec == "Auto" else (codec,)


def _container_order(codec: str, requested: str, codec_is_auto: bool) -> tuple[str, ...]:
    if requested != "Auto":
        return (requested,)
    if codec_is_auto:
        return {"AV1": ("WebM",), "VP9": ("WebM",), "H.264": ("MP4",)}[codec]
    return ("WebM", "MKV", "MP4") if codec in {"AV1", "VP9"} else ("MP4", "MKV")


def _save_frame_exports(images: torch.Tensor, output_path: Path, first: bool, last: bool, pingpong: bool):
    exports = []
    if not first and not last:
        return exports
    final_frame = images[1] if pingpong and len(images) >= 3 else images[-1]
    for suffix, frame, enabled in (("first", images[0], first), ("last", final_frame, last)):
        if not enabled:
            continue
        target = output_path.with_name(f"{output_path.stem}-{suffix}-frame.png")
        rgb = frame[..., :3].detach().to(device="cpu", dtype=torch.float32).clamp(0, 1)
        Image.fromarray(np.rint(rgb.numpy() * 255).astype(np.uint8), mode="RGB").save(target, "PNG")
        exports.append(target)
    return exports


def _preview_source_path(filename: str, subfolder: str, output_type: str) -> Path | None:
    if not filename or Path(filename).name != filename or output_type not in {"input", "output", "temp"}:
        return None
    if Path(subfolder).is_absolute() or ".." in Path(subfolder).parts:
        return None
    fp = _folder_paths()
    root_value = fp.get_directory_by_type(output_type) if hasattr(fp, "get_directory_by_type") else None
    if not root_value:
        return None
    root = Path(root_value).resolve()
    candidate = Path(root, subfolder, filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


try:
    from aiohttp import web
    from server import PromptServer
except ImportError:
    web = None
    PromptServer = None


if PromptServer is not None and web is not None:

    @PromptServer.instance.routes.get("/jr-h3/enhanced-video-preview")
    async def jr_h3_enhanced_video_preview(request):
        source = _preview_source_path(
            request.rel_url.query.get("filename", ""),
            request.rel_url.query.get("subfolder", ""),
            request.rel_url.query.get("type", "output"),
        )
        ffmpeg = find_ffmpeg()
        if source is None:
            return web.Response(status=404)
        if ffmpeg is None:
            return web.Response(status=503, text="FFmpeg is unavailable")
        process = await asyncio.create_subprocess_exec(
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "24", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", "pipe:1",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        response = web.StreamResponse(headers={"Content-Type": "video/mp4", "Cache-Control": "no-store"})
        await response.prepare(request)
        try:
            assert process.stdout is not None
            while chunk := await asyncio.wait_for(process.stdout.read(256 * 1024), timeout=60):
                await response.write(chunk)
            await asyncio.wait_for(process.wait(), timeout=_ENCODE_TIMEOUT_SECONDS)
        except (ConnectionResetError, asyncio.CancelledError, TimeoutError):
            if process.returncode is None:
                process.kill()
            await process.wait()
        return response


class JR_H3_EnhancedVideoCombine:
    CATEGORY = "JR MiniMax H3/Video"
    FUNCTION = "combine"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "filename")
    OUTPUT_NODE = True
    DESCRIPTION = "Encodes IMAGE batches with automatic GPU/software fallback and an interactive video preview."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"description": "Frames to encode."}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.01}),
                "codec": (CODECS, {"default": "Auto"}),
                "container": (CONTAINERS, {"default": "Auto"}),
                "bit_depth": (BIT_DEPTHS, {"default": "Auto"}),
                "quality": ("INT", {"default": 20, "min": 0, "max": 51}),
                "log_level": (["Standard", "Verbose"], {"default": "Standard"}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_metadata": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "video/%date:yyyy-MM-dd%/%date:hhmmss%"}),
                "save_output": ("BOOLEAN", {"default": True}),
                "pass_frames": ("BOOLEAN", {"default": False}),
                "crop_to_audio": ("BOOLEAN", {"default": False}),
                "audio_codec": (AUDIO_CODECS, {"default": "Auto"}),
                "audio_bitrate": (AUDIO_BITRATES, {"default": "192k"}),
                "save_first_frame": ("BOOLEAN", {"default": False}),
                "save_last_frame": ("BOOLEAN", {"default": False}),
            },
            "optional": {"audio": ("AUDIO", {"description": "Optional audio to mux."})},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def validate_inputs(self, *args, **kwargs):
        return True

    def combine(
        self, images, frame_rate, codec, container, bit_depth, quality, log_level, pingpong,
        save_metadata, filename_prefix, save_output, pass_frames, crop_to_audio=False,
        audio_codec="Auto", audio_bitrate="192k", save_first_frame=False, save_last_frame=False,
        audio=None, prompt=None, extra_pnginfo=None,
    ):
        del log_level
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] < 1 or images.shape[-1] < 3:
            raise ValueError("images must be a non-empty IMAGE batch shaped [frames,height,width,channels].")
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError("FFmpeg was not found. Install FFmpeg or imageio-ffmpeg and restart ComfyUI.")
        fp = _folder_paths()
        output_dir = fp.get_output_directory() if save_output else fp.get_temp_directory()
        output_type = "output" if save_output else "temp"
        height, width = map(int, images.shape[1:3])
        audio_filename_tag = "audio" if audio is not None and container not in _ANIMATION_FORMATS else None
        output_folder, basename, counter, subfolder = _allocate_output(
            filename_prefix, output_dir, width, height, audio_filename_tag,
        )
        selected_depth = _resolve_bit_depth(codec, bit_depth, images)
        available = _available_video_encoders(ffmpeg)
        metadata_path = _write_metadata(prompt, extra_pnginfo) if save_metadata else None
        try:
            audio_info, audio_duration = _write_audio(audio)
        except BaseException:
            if metadata_path:
                Path(metadata_path).unlink(missing_ok=True)
            raise
        def chunks():
            return _iter_raw_chunks(images, selected_depth, pingpong)

        progress_bar = None
        try:
            import comfy.utils

            progress_bar = comfy.utils.ProgressBar(_encoded_frame_count(images, pingpong))
            progress_bar.update_absolute(0)
        except ImportError:
            pass

        def progress(seconds: float):
            if progress_bar is not None:
                progress_bar.update_absolute(min(_encoded_frame_count(images, pingpong), max(0, int(seconds * frame_rate))))

        _log(
            f"Encoding {_encoded_frame_count(images, pingpong)} frames at {width}x{height}, {frame_rate:g} fps, "
            f"{selected_depth}-bit; codec={codec}, container={container}, audio={'yes' if audio_info else 'no'}."
        )
        failures = []
        output_path = None
        selected_codec = None
        selected_container = None
        encoder = None
        try:
            if container in _ANIMATION_FORMATS:
                extension, _ = _ANIMATION_FORMATS[container]
                if audio_info:
                    _log(f"{container} cannot contain audio; the connected audio is omitted.")
                output_path = output_folder / _output_name(basename, counter, extension)
                encoder = _encode_animation(
                    ffmpeg, available, container, selected_depth, width, height, frame_rate,
                    chunks, output_path, int(quality), progress,
                )
                selected_codec = container
                selected_container = container
            else:
                for candidate_codec in _codec_order(codec):
                    for candidate_container in _container_order(candidate_codec, container, codec == "Auto"):
                        extension = _VIDEO_EXTENSIONS.get(candidate_container)
                        if extension is None:
                            failures.append(f"{candidate_codec}/{candidate_container}: incompatible container")
                            continue
                        candidate_path = output_folder / _output_name(basename, counter, extension)
                        if codec == "Auto":
                            _log(f"Testing automatic choice {candidate_codec}/{candidate_container}.")
                        try:
                            candidate_encoder = _encode_video(
                                ffmpeg, available, candidate_codec, candidate_container, selected_depth,
                                width, height, frame_rate, chunks, candidate_path, int(quality), metadata_path,
                                audio_info, audio_duration, bool(crop_to_audio), audio_codec, audio_bitrate, progress,
                            )
                        except RuntimeError as error:
                            failures.append(str(error))
                            _log(f"{candidate_codec}/{candidate_container} failed; trying the next candidate.")
                            continue
                        output_path = candidate_path
                        selected_codec = candidate_codec
                        selected_container = candidate_container
                        encoder = candidate_encoder
                        break
                    if output_path is not None:
                        break
                if output_path is None:
                    fallback_path = output_folder / _output_name(basename, counter, ".mp4")
                    _log("Requested combinations failed; trying mandatory H.264/MP4 fallback.")
                    try:
                        encoder = _encode_video(
                            ffmpeg, available, "H.264", "MP4", selected_depth, width, height, frame_rate,
                            chunks, fallback_path, int(quality), metadata_path, audio_info, audio_duration,
                            bool(crop_to_audio), audio_codec, audio_bitrate, progress,
                        )
                    except RuntimeError as error:
                        failures.append(str(error))
                        raise RuntimeError("All video encoding attempts failed. " + " | ".join(failures)) from error
                    output_path = fallback_path
                    selected_codec = "H.264"
                    selected_container = "MP4"
        finally:
            if metadata_path:
                Path(metadata_path).unlink(missing_ok=True)
            if audio_info:
                Path(audio_info[0]).unlink(missing_ok=True)

        assert output_path is not None and selected_codec is not None and selected_container is not None
        if progress_bar is not None:
            progress_bar.update_absolute(_encoded_frame_count(images, pingpong))
        exports = _save_frame_exports(images, output_path, save_first_frame, save_last_frame, pingpong)
        returned_frames = pingpong_frames(images, pingpong) if pass_frames else images[:0]
        if selected_container in _ANIMATION_FORMATS:
            mime = _ANIMATION_FORMATS[selected_container][1]
        else:
            mime = _VIDEO_MIME_TYPES[selected_container]
        output_stat = output_path.stat()
        video_asset = {
            "filename": output_path.name, "subfolder": subfolder, "type": output_type,
            "format": mime, "width": width, "height": height, "codec": selected_codec,
            "bit_depth": selected_depth, "container": selected_container,
            "preview_id": f"{output_stat.st_mtime_ns:x}-{output_stat.st_size:x}",
        }
        assets = [video_asset]
        assets.extend({
            "filename": path.name, "subfolder": subfolder, "type": output_type,
            "format": "image/png", "width": width, "height": height,
        } for path in exports)
        ui = {"images": assets}
        if selected_container not in _ANIMATION_FORMATS:
            ui["gifs"] = [{**video_asset, "fps": float(frame_rate)}]
        _log(f"Saved {output_path} using {encoder} ({selected_codec}/{selected_container}, {selected_depth}-bit).")
        for exported in exports:
            _log(f"Saved frame export {exported}.")
        return {"ui": ui, "result": (returned_frames, str(output_path))}
