"""Clean single-frame H3 latent masters and stage-local spatial copies."""

import torch

MASTER_KEY = "jr_h3_keyframe_master"
CROP_KEY = "jr_h3_keyframe_crop"


def validate_keyframe_latent(value, name="keyframe latent"):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a LATENT dictionary from the H3 video VAE.")
    z = value.get("samples")
    if (not isinstance(z, torch.Tensor) or z.is_nested or z.ndim != 5 or
            tuple(z.shape[:3]) != (1, 24, 1) or min(z.shape[-2:]) <= 0 or
            z.layout != torch.strided or z.device.type == "meta" or not z.is_floating_point()):
        raise ValueError(f"{name} requires a plain single-frame H3 latent [1,24,1,H,W]; not an AV or video clip latent.")
    if not bool(torch.isfinite(z).all()):
        raise ValueError(f"{name} contains NaN or Inf.")
    if value.get("noise_mask") is not None:
        raise ValueError(f"{name} must be a clean, unmasked keyframe, not a masked sampling state.")
    return z


def resize_keyframe_latent(master, width, height, crop="disabled"):
    """Return an independent canvas copy; never resize a previous stage's copy."""
    from comfy.utils import common_upscale

    z = validate_keyframe_latent({"samples": master})
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("H3 keyframe target width/height must be positive multiples of 32.")
    if crop not in {"disabled", "center"}:
        raise ValueError("Unsupported H3 keyframe crop mode.")
    if tuple(z.shape[-2:]) == (height // 16, width // 16):
        return z.detach().clone()
    resized = common_upscale(z[:, :, 0].float(), width // 16, height // 16, "area", crop)
    return resized.unsqueeze(2).to(dtype=z.dtype).contiguous().clone()
