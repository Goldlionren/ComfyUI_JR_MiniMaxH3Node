"""Safe FFmpeg IMAGE-batch encoder for ComfyUI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def pingpong_frames(images: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled or images.shape[0] < 3:
        return images
    return torch.cat((images, images[1:-1].flip(0)), dim=0)


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None


def _safe_prefix(value: str) -> str:
    name = Path(str(value).replace("\\", "/")).name.strip().strip(".")
    name = "".join(ch for ch in name if ch not in '<>:"/\\|?*' and ord(ch) >= 32)
    return name[:120] or "jr_h3_video"


def _folder_paths():
    try:
        import folder_paths
        return folder_paths
    except ImportError as error:
        raise RuntimeError("ComfyUI folder_paths is unavailable; run this node inside ComfyUI.") from error


def _save_png(frame: torch.Tensor, path: Path):
    array = frame.detach().cpu().float().clamp(0, 1).numpy()[..., :3]
    Image.fromarray(np.rint(array * 255).astype(np.uint8), "RGB").save(path, format="PNG")


def _write_audio(audio, path: str) -> float:
    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    rate = audio.get("sample_rate") if isinstance(audio, dict) else None
    if not isinstance(waveform, torch.Tensor) or not isinstance(rate, int) or rate <= 0:
        raise ValueError("audio must be a ComfyUI AUDIO object with waveform and sample_rate.")
    data = waveform.detach().cpu().float()
    if data.ndim == 3:
        data = data[0]
    if data.ndim != 2:
        raise ValueError("audio waveform must have shape [B,C,T] or [C,T].")
    pcm = (data.clamp(-1, 1).t().numpy() * 32767).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(pcm.shape[1]); handle.setsampwidth(2); handle.setframerate(rate); handle.writeframes(pcm.tobytes())
    return pcm.shape[0] / rate


class JR_H3_EnhancedVideoCombine:
    CATEGORY = "JR MiniMax H3/Video"
    FUNCTION = "combine"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "filename")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",), "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0}),
                "codec": (["Auto", "H.264", "H.265 (HEVC)", "VP9"], {"default": "Auto"}),
                "container": (["Auto", "MP4", "WebM", "MKV"], {"default": "Auto"}),
                "quality": ("INT", {"default": 20, "min": 0, "max": 51}),
                "pingpong": ("BOOLEAN", {"default": False}), "save_metadata": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "jr_h3_video"}), "save_output": ("BOOLEAN", {"default": True}),
                "pass_frames": ("BOOLEAN", {"default": False}), "crop_to_audio": ("BOOLEAN", {"default": False}),
                "save_first_frame": ("BOOLEAN", {"default": False}), "save_last_frame": ("BOOLEAN", {"default": False}),
            },
            "optional": {"audio": ("AUDIO",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def combine(self, images, frame_rate, codec, container, quality, pingpong, save_metadata, filename_prefix,
                save_output, pass_frames, crop_to_audio, save_first_frame, save_last_frame,
                audio=None, prompt=None, extra_pnginfo=None):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] < 1 or images.shape[-1] not in (3, 4):
            raise ValueError("images must be a non-empty RGB/RGBA IMAGE batch shaped [B,H,W,C].")
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("FFmpeg was not found. Install FFmpeg and ensure ffmpeg.exe is on PATH.")
        fp = _folder_paths()
        output_dir = Path(fp.get_output_directory() if save_output else fp.get_temp_directory()).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_prefix(filename_prefix)
        codec_name = "H.264" if codec == "Auto" else codec
        container_name = ({"H.264": "MP4", "H.265 (HEVC)": "MP4", "VP9": "WebM"}[codec_name] if container == "Auto" else container)
        if container_name == "WebM" and codec_name not in {"VP9"}:
            raise ValueError("WebM requires VP9 in this node.")
        extension = {"MP4": ".mp4", "WebM": ".webm", "MKV": ".mkv"}[container_name]
        counter = 1
        while True:
            output_path = output_dir / f"{safe_name}_{counter:05d}{extension}"
            if not output_path.exists(): break
            counter += 1
        frames = pingpong_frames(images, pingpong)
        h, w = frames.shape[1:3]
        encoder = {"H.264": "libx264", "H.265 (HEVC)": "libx265", "VP9": "libvpx-vp9"}[codec_name]
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{w}x{h}", "-r", str(float(frame_rate)), "-i", "pipe:0"]
        audio_path = None
        try:
            if audio is not None:
                handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); audio_path = handle.name; handle.close()
                _write_audio(audio, audio_path)
                command.extend(["-i", audio_path])
            command.extend(["-c:v", encoder, "-crf", str(int(quality))])
            command.extend(["-pix_fmt", "yuv420p"] if codec_name == "H.264" else [])
            if audio is not None:
                command.extend(["-c:a", "libopus" if container_name == "WebM" else "aac"])
                if crop_to_audio: command.append("-shortest")
            if save_metadata:
                metadata = json.dumps({"prompt": prompt, "extra_pnginfo": extra_pnginfo}, ensure_ascii=False, default=str)
                command.extend(["-metadata", "comment=" + metadata[:16000]])
            command.append(str(output_path))
            rgb = frames[..., :3].detach().cpu().float().clamp(0, 1)
            raw = np.rint(rgb.numpy() * 255).astype(np.uint8).tobytes()
            result = subprocess.run(command, input=raw, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300, check=False)
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"FFmpeg failed with exit code {result.returncode}: {stderr}")
        except Exception:
            if output_path.exists(): output_path.unlink()
            raise
        finally:
            if audio_path:
                try: os.unlink(audio_path)
                except OSError: pass
        exports = []
        if save_first_frame:
            p = output_path.with_name(output_path.stem + "-first-frame.png"); _save_png(frames[0], p); exports.append(p)
        if save_last_frame:
            p = output_path.with_name(output_path.stem + "-last-frame.png"); _save_png(frames[-1], p); exports.append(p)
        returned = frames if pass_frames else images[:0]
        assets = [{"filename": output_path.name, "subfolder": "", "type": "output" if save_output else "temp"}]
        assets.extend({"filename": p.name, "subfolder": "", "type": "output" if save_output else "temp"} for p in exports)
        return {"ui": {"gifs": assets[:1], "images": assets}, "result": (returned, str(output_path))}
