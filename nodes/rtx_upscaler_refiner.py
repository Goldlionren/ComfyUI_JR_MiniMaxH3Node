"""Optional RTX Video Effects node.

This module intentionally does not import nvvfx until execution.
"""

import importlib
import math

import torch
import torch.nn.functional as F


def target_size(width, height, resize_type, scale, megapixels, target_width, target_height, divisor, ratio_preset):
    divisor = int(divisor)
    if resize_type == "Same Size":
        raw_w, raw_h = width, height
    elif resize_type == "Scale":
        raw_w, raw_h = width * float(scale), height * float(scale)
    elif resize_type in {"Keep Ratio", "Preset Ratio"}:
        ratio = width / height
        if resize_type == "Preset Ratio":
            left, right = ratio_preset.split(":", 1)
            ratio = float(left) / float(right)
        pixels = float(megapixels) * 1_000_000
        raw_w, raw_h = math.sqrt(pixels * ratio), math.sqrt(pixels / ratio)
    elif resize_type == "Manual":
        raw_w, raw_h = target_width, target_height
    else:
        raise ValueError(f"Unknown resize_type: {resize_type}")
    return max(divisor, round(raw_w / divisor) * divisor), max(divisor, round(raw_h / divisor) * divisor)


def _load_nvvfx():
    try:
        return importlib.import_module("nvvfx")
    except ImportError as error:
        raise RuntimeError(
            "JR MiniMax H3 RTX Upscaler & Refiner requires the 'nvvfx' Python package, "
            "a compatible NVIDIA RTX GPU, driver, and NVIDIA Video Effects SDK. "
            "Install requirements-rtx.txt and follow the RTX section in README.md."
        ) from error


def _fit_aspect(frame, target_width, target_height, resize_method):
    _, source_height, source_width = frame.shape
    source_ratio = source_width / source_height
    target_ratio = target_width / target_height
    if abs(source_ratio - target_ratio) < 1e-6:
        return frame
    if resize_method == "Center Crop (Fill)":
        if source_ratio > target_ratio:
            crop_width = max(1, round(source_height * target_ratio))
            left = (source_width - crop_width) // 2
            return frame[:, :, left:left + crop_width]
        crop_height = max(1, round(source_width / target_ratio))
        top = (source_height - crop_height) // 2
        return frame[:, top:top + crop_height, :]
    if source_ratio > target_ratio:
        padded_height = max(source_height, round(source_width / target_ratio))
        top = (padded_height - source_height) // 2
        return F.pad(frame, (0, 0, top, padded_height - source_height - top))
    padded_width = max(source_width, round(source_height * target_ratio))
    left = (padded_width - source_width) // 2
    return F.pad(frame, (left, padded_width - source_width - left, 0, 0))


class JR_H3_RTXUpscalerRefiner:
    CATEGORY = "JR MiniMax H3/Video"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        qualities = ["Low", "Medium", "High", "Ultra"]
        return {"required": {
            "images": ("IMAGE",), "denoise": ("BOOLEAN", {"default": False}),
            "denoise_quality": (qualities, {"default": "Ultra"}),
            "deblur": ("BOOLEAN", {"default": False}), "deblur_quality": (qualities, {"default": "Ultra"}),
            "upscale": (["Off", "VSR", "High Bitrate"], {"default": "VSR"}),
            "upscale_quality": (qualities, {"default": "Ultra"}),
            "resize_type": (["Same Size", "Scale", "Keep Ratio", "Preset Ratio", "Manual"], {"default": "Scale"}),
            "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05}),
            "megapixels": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 64.0}),
            "width": ("INT", {"default": 1920, "min": 64, "max": 8192}),
            "height": ("INT", {"default": 1080, "min": 64, "max": 8192}),
            "divisible_by": (["8", "16", "32", "64", "128"], {"default": "8"}),
            "ratio_preset": (["1:1", "4:3", "3:2", "16:9", "21:9"], {"default": "16:9"}),
            "resize_method": (["Center Crop (Fill)", "Letterbox (Fit)"], {"default": "Center Crop (Fill)"}),
            "device_id": ("INT", {"default": 0, "min": 0, "max": 8}),
        }}

    def execute(self, images, denoise, denoise_quality, deblur, deblur_quality, upscale, upscale_quality,
                resize_type, scale, megapixels, width, height, divisible_by, ratio_preset, resize_method, device_id):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] not in (3, 4):
            raise ValueError("RTX input must be an RGB/RGBA IMAGE batch shaped [B,H,W,C].")
        if not denoise and not deblur and upscale == "Off":
            return (images[..., :3],)
        if not torch.cuda.is_available():
            raise RuntimeError("JR MiniMax H3 RTX processing requires CUDA and a compatible NVIDIA RTX GPU.")
        nvvfx = _load_nvvfx()
        if denoise or deblur:
            raise RuntimeError(
                "The installed nvvfx binding exposes VideoSuperRes only; Denoise/Deblur effects are unavailable. "
                "Disable those passes or install an SDK binding that provides them."
            )
        if upscale == "Off":
            return (images[..., :3],)
        if not hasattr(nvvfx, "VideoSuperRes"):
            raise RuntimeError("Installed nvvfx package does not expose VideoSuperRes; see README.md.")
        source_height, source_width = images.shape[1:3]
        output_width, output_height = target_size(
            source_width, source_height, resize_type, scale, megapixels,
            width, height, divisible_by, ratio_preset,
        )
        quality_type = nvvfx.VideoSuperRes.QualityLevel
        quality = getattr(quality_type, upscale_quality.upper())
        output = []
        cuda_device = torch.device(f"cuda:{device_id}")
        with torch.cuda.device(cuda_device), torch.inference_mode(), nvvfx.VideoSuperRes(quality=quality, device=device_id) as effect:
            effect.output_width = output_width
            effect.output_height = output_height
            effect.load()
            for source in images[..., :3]:
                frame = source.to(device=cuda_device, dtype=torch.float32).permute(2, 0, 1).contiguous()
                frame = _fit_aspect(frame, output_width, output_height, resize_method).contiguous()
                enhanced = torch.from_dlpack(effect.run(frame).image).clone()
                output.append(enhanced.permute(1, 2, 0).clamp(0, 1).to(device=images.device, dtype=images.dtype))
        return (torch.stack(output, dim=0),)
