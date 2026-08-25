"""ComfyUI node for neural spatial upscaling of MiniMax H3 video latents."""

from __future__ import annotations

from ..utils.h3_neural_latent_upscaler import upscale_h3_video_latent


class JR_MiniMaxH3NeuralLatentUpscaler:
    CATEGORY = "JR MiniMax H3/Latent"
    FUNCTION = "upscale"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("video_latent", "status")
    DESCRIPTION = (
        "Uses a user-supplied H3-specific 3D neural checkpoint to spatially upscale a plain "
        "24-channel MiniMax H3 video latent while preserving B/C/T and LATENT metadata."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT",),
                "resize_mode": (["scale", "megapixels"], {"default": "scale"}),
                "scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05}),
                "target_megapixels": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.01, "max": 64.0, "step": 0.01},
                ),
            }
        }

    def upscale(self, video_latent, resize_mode, scale, target_megapixels):
        return upscale_h3_video_latent(video_latent, resize_mode, scale, target_megapixels)


__all__ = ["JR_MiniMaxH3NeuralLatentUpscaler"]
