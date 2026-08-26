"""ComfyUI node for injecting and locking an external MiniMax H3 audio latent."""

from __future__ import annotations

from ..utils.h3_audio_driven_latent_builder import build_h3_audio_driven_latent


class JR_H3_AudioDrivenLatentBuilder:
    CATEGORY = "JR MiniMax H3/Latent"
    FUNCTION = "build"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("audio_driven_av_latent", "status")
    DESCRIPTION = (
        "Replaces the audio stream in an official MiniMax H3 AV LATENT with an externally encoded H3 audio "
        "latent, preserves the video denoise mask, and locks the audio branch with a zero noise mask."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": (
                    "LATENT",
                    {"tooltip": "Official MiniMax H3 joint video/audio LATENT from Directed Video Conditioning."},
                ),
                "audio_drive_latent": (
                    "LATENT",
                    {"tooltip": "Audio LATENT encoded with the appropriate MiniMax H3 Audio VAE."},
                ),
            }
        }

    def build(self, av_latent, audio_drive_latent):
        return build_h3_audio_driven_latent(av_latent, audio_drive_latent)


__all__ = ["JR_H3_AudioDrivenLatentBuilder"]
