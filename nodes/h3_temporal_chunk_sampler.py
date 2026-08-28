"""ComfyUI node for guided sequential MiniMax H3 AV temporal chunk sampling."""

from __future__ import annotations

from ..utils.h3_temporal_chunk_sampler import sample_h3_temporal_chunks


class JR_H3_TemporalChunkSampler:
    CATEGORY = "JR MiniMax H3/Sampling"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("output", "status")
    DESCRIPTION = (
        "Sequentially slices an official MiniMax H3 AV NestedTensor along its shared timeline, "
        "uses the original conditioning for chunk 1, then anchors every later chunk to the previous "
        "decoded terminal frame through ComfyUI's native MiniMaxH3AddGuide at local frame 0. A fresh "
        "Basic Guider is built for every chunk before native SamplerCustomAdvanced execution."
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
                    {"tooltip": "MiniMax H3 video VAE used to decode each previous terminal frame."},
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
                        "tooltip": "Approximate maximum chunk duration. Internal cuts align to H3's 17-frame cycle.",
                    },
                ),
                "aggressive_memory_cleanup": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Run ComfyUI soft_empty_cache after each completed chunk. Slower; normally leave disabled.",
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
        )


__all__ = ["JR_H3_TemporalChunkSampler"]
