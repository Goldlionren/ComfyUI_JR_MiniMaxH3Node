"""ComfyUI node for guided sequential MiniMax H3 AV temporal chunk sampling."""

from __future__ import annotations

from ..utils.h3_temporal_chunk_sampler import (
    CONTINUITY_MODES,
    DEFAULT_HARD_CHUNK_PRESET,
    HARD_AV_PREFIX_MODE,
    HARD_CHUNK_PRESET_LABELS,
    sample_h3_temporal_chunks,
)


class JR_H3_TemporalChunkSampler:
    CATEGORY = "JR MiniMax H3/Sampling"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("output", "status")
    DESCRIPTION = (
        "Sequentially slices an official MiniMax H3 AV NestedTensor along its shared timeline, "
        "with recommended fixed-profile Hard AV Latent Prefix continuity or the preserved legacy path. "
        "Hard mode locks both sampled video and sampled audio overlap without decode/re-encode or AddGuide. "
        "A fresh Basic Guider is built for every chunk before native SamplerCustomAdvanced execution."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "MiniMax H3 MODEL used to build a fresh Basic Guider for every chunk."},
                ),
                "positive": (
                    "CONDITIONING",
                    {"tooltip": "Original H3 positive conditioning. It is never mutated."},
                ),
                "vae": (
                    "VAE",
                    {"tooltip": "Used only by Legacy mode to decode each previous terminal frame."},
                ),
                "noise": ("NOISE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "chunk_duration_seconds": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": 1.0,
                        "max": 3600.0,
                        "step": 0.5,
                        "tooltip": (
                            "Legacy-only approximate maximum duration. Internal cuts align to H3's 17-frame cycle."
                        ),
                    },
                ),
                "aggressive_memory_cleanup": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Run ComfyUI soft_empty_cache after each completed chunk. Slower; normally leave disabled.",
                    },
                ),
                "continuity_mode": (
                    list(CONTINUITY_MODES),
                    {
                        "default": HARD_AV_PREFIX_MODE,
                        "tooltip": (
                            "Hard AV Latent Prefix is recommended and uses the separate preset dropdown. "
                            "Legacy Independent Chunks preserves the previous decoded-last-frame behavior."
                        ),
                    },
                ),
                "hard_chunk_preset": (
                    list(HARD_CHUNK_PRESET_LABELS),
                    {
                        "default": DEFAULT_HARD_CHUNK_PRESET,
                        "tooltip": (
                            "Hard-mode local AV window. The 5.875s preset is recommended for latent upscaling "
                            "and lower VRAM; every preset keeps the same 39-frame hard prefix."
                        ),
                    },
                ),
            }
        }

    def sample(
        self,
        model,
        positive,
        vae,
        noise,
        sampler,
        sigmas,
        latent_image,
        chunk_duration_seconds=15.0,
        aggressive_memory_cleanup=False,
        continuity_mode=HARD_AV_PREFIX_MODE,
        hard_chunk_preset=DEFAULT_HARD_CHUNK_PRESET,
    ):
        return sample_h3_temporal_chunks(
            model=model,
            positive=positive,
            vae=vae,
            noise=noise,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent_image,
            chunk_duration_seconds=chunk_duration_seconds,
            aggressive_memory_cleanup=aggressive_memory_cleanup,
            continuity_mode=continuity_mode,
            hard_chunk_preset=hard_chunk_preset,
        )


__all__ = ["JR_H3_TemporalChunkSampler"]
