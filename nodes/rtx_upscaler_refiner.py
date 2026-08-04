"""Optional RTX Video Effects node.

This module intentionally does not import nvvfx until execution.
"""

import contextlib
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
        module = importlib.import_module("nvvfx")
    except ImportError as error:
        raise RuntimeError(
            "JR MiniMax H3 RTX Upscaler & Refiner requires the 'nvvfx' Python package, "
            "a compatible NVIDIA RTX GPU, driver, and NVIDIA Video Effects SDK. "
            "Install requirements-rtx.txt and follow the RTX section in README.md."
        ) from error
    effect_type = getattr(module, "VideoSuperRes", None)
    quality_type = getattr(getattr(module, "effects", None), "QualityLevel", None)
    if quality_type is None and effect_type is not None:
        quality_type = getattr(effect_type, "QualityLevel", None)
    if effect_type is None or quality_type is None:
        raise RuntimeError("nvvfx is installed but VideoSuperRes or QualityLevel is unavailable.")
    return effect_type, quality_type


def _quality_level(quality_type, operation, quality):
    suffix = quality.upper()
    prefixes = {
        "VSR": ("",),
        "High Bitrate": ("HIGHBITRATE_", "HIGH_BITRATE_"),
        "Denoise": ("DENOISE_",),
        "Deblur": ("DEBLUR_",),
    }
    for prefix in prefixes[operation]:
        name = prefix + suffix
        if hasattr(quality_type, name):
            return getattr(quality_type, name), name
    available = ", ".join(sorted(name for name in dir(quality_type) if name.isupper()))
    raise RuntimeError(
        f"The installed nvvfx SDK does not support {operation} quality {quality}. "
        f"Available QualityLevel values: {available or 'none'}."
    )


def _construct_effect(effect_type, level, device_id):
    attempts = (
        ((), {"quality": level, "device": device_id}),
        ((level,), {"device": device_id}),
        ((), {"quality": level}),
        ((level,), {}),
    )
    last_error = None
    for args, kwargs in attempts:
        try:
            return effect_type(*args, **kwargs)
        except TypeError as error:
            last_error = error
    raise RuntimeError(f"Unable to construct nvvfx.VideoSuperRes: {last_error}")


def _close_effect(effect):
    for name in ("close", "destroy", "unload"):
        method = getattr(effect, name, None)
        if callable(method):
            method()
            return


@contextlib.contextmanager
def _effect_context(api, enabled, operation, quality, device_id, width, height):
    if not enabled:
        yield None
        return
    effect_type, quality_type = api
    level, level_name = _quality_level(quality_type, operation, quality)
    effect = _construct_effect(effect_type, level, device_id)
    manager = effect
    entered = False
    try:
        if hasattr(manager, "__enter__"):
            effect = manager.__enter__()
            entered = True
        effect.output_width = int(width)
        effect.output_height = int(height)
        load = getattr(effect, "load", None)
        if callable(load):
            load()
        yield effect
    except Exception as error:
        raise RuntimeError(f"nvvfx {operation} ({level_name}) failed: {error}") from error
    finally:
        if entered and hasattr(manager, "__exit__"):
            manager.__exit__(None, None, None)
        else:
            _close_effect(effect)


def _run_effect(effect, frame, cuda_device):
    if effect is None:
        return frame
    frame = frame.contiguous()
    torch.cuda.current_stream(cuda_device).synchronize()
    result = effect.run(frame)
    torch.cuda.synchronize(cuda_device)
    return torch.from_dlpack(result.image).clone().contiguous()


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
        if device_id < 0 or device_id >= torch.cuda.device_count():
            raise ValueError(f"CUDA device_id {device_id} is unavailable; detected {torch.cuda.device_count()} devices.")
        api = _load_nvvfx()
        upscale_enabled = upscale != "Off"
        source_height, source_width = images.shape[1:3]
        if upscale_enabled:
            output_width, output_height = target_size(
                source_width, source_height, resize_type, scale, megapixels,
                width, height, divisible_by, ratio_preset,
            )
        else:
            output_width, output_height = source_width, source_height
        cuda_device = torch.device(f"cuda:{device_id}")
        output = torch.empty(
            (images.shape[0], output_height, output_width, 3),
            device=images.device,
            dtype=images.dtype,
        )
        with (
            torch.cuda.device(cuda_device),
            torch.inference_mode(),
            _effect_context(api, denoise, "Denoise", denoise_quality, device_id, source_width, source_height) as denoise_effect,
            _effect_context(api, deblur, "Deblur", deblur_quality, device_id, source_width, source_height) as deblur_effect,
            _effect_context(api, upscale_enabled, upscale, upscale_quality, device_id, output_width, output_height) as upscale_effect,
        ):
            for index, source in enumerate(images[..., :3]):
                frame = source.to(device=cuda_device, dtype=torch.float32).permute(2, 0, 1).contiguous()
                frame = _run_effect(denoise_effect, frame, cuda_device)
                frame = _run_effect(deblur_effect, frame, cuda_device)
                if upscale_enabled:
                    frame = _fit_aspect(frame, output_width, output_height, resize_method).contiguous()
                    frame = _run_effect(upscale_effect, frame, cuda_device)
                result = frame.permute(1, 2, 0).clamp(0, 1).to(device=images.device, dtype=images.dtype)
                output[index].copy_(result)
        return (output,)
