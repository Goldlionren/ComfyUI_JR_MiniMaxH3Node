"""ComfyUI node for assembling separately encoded MiniMax H3 AV latents."""

from __future__ import annotations

from ..utils.h3_av_latent_builder import build_h3_av_latent


class JR_MiniMaxH3AVLatentBuilder:
    CATEGORY = "JR MiniMax H3/Latent"
    FUNCTION = "build"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "status")
    DESCRIPTION = (
        "Validates separately encoded MiniMax H3 video/audio latent tensors and wraps them "
        "as the official two-stream H3 NestedTensor LATENT without encoding, casting or copying."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT",),
                "audio_latent": ("LATENT",),
            }
        }

    def build(self, video_latent, audio_latent):
        return build_h3_av_latent(video_latent, audio_latent)


__all__ = ["JR_MiniMaxH3AVLatentBuilder"]
