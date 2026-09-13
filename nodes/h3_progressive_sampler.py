"""Experimental spatially progressive Euler sampling for native H3 AV."""

from ..utils.h3_progressive_sampler import sample_h3_progressive


class JR_H3_ProgressiveSampler:
    CATEGORY = "JR MiniMax H3/Sampling"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("output", "status")
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Experimental H3 AV sampler: low-resolution Euler steps, JR neural lift of predicted clean video x0, "
        "then remaining steps of the same sigma schedule. Empty AV latent, batch 1, standard Euler only. "
        "Masks, keyframes and audio-driven/continuation latents are not supported. Scale 1 is a native baseline."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "noise": ("NOISE",),
            "sampler": ("SAMPLER",),
            "sigmas": ("SIGMAS",),
            "latent_image": ("LATENT", {"tooltip": "Empty native H3 AV latent at FINAL target resolution."}),
            "transition_step": ("INT", {"default": 3, "min": 1, "max": 10000,
                "tooltip": "Number of low-resolution denoiser evaluations; must be smaller than total steps."}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05,
                "tooltip": "Spatial scale only. 1.0 bypasses the transition for a native Euler A/B baseline."}),
            "transition_seed_offset": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
            "aggressive_memory_cleanup": ("BOOLEAN", {"default": False}),
        }}

    def sample(self, model, positive, noise, sampler, sigmas, latent_image,
               transition_step=3, lowres_scale=0.5, transition_seed_offset=1, aggressive_memory_cleanup=False):
        return sample_h3_progressive(
            model=model, positive=positive, noise=noise, sampler=sampler, sigmas=sigmas,
            latent_image=latent_image, transition_step=transition_step, lowres_scale=lowres_scale,
            transition_seed_offset=transition_seed_offset, aggressive_memory_cleanup=aggressive_memory_cleanup,
        )
