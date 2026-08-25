"""ComfyUI node for splitting a MiniMax H3 AV latent into its two streams."""

from __future__ import annotations

from ..utils.h3_av_latent_split import split_h3_av_latent


class JR_H3_SplitAVLatent:
    CATEGORY = "JR MiniMax H3/Latent"
    FUNCTION = "split"
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")
    DESCRIPTION = (
        "Validates and splits an official MiniMax H3 two-stream NestedTensor LATENT into "
        "standard video/audio LATENT mappings without copying, casting or moving tensors."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"av_latent": ("LATENT",)}}

    def split(self, av_latent):
        return split_h3_av_latent(av_latent)


__all__ = ["JR_H3_SplitAVLatent"]
