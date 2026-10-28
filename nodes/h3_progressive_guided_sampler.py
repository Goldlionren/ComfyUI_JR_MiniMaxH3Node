"""Reference/keyframe/audio-driven variant, retaining the strict T2VA node."""

from ..utils.h3_progressive_sampler import sample_h3_progressive
from .h3_progressive_sampler import JR_H3_ProgressiveSampler


class JR_H3_ProgressiveGuidedSampler(JR_H3_ProgressiveSampler):
    DESCRIPTION = (
        "Experimental progressive H3 Euler with independent references, first/last keyframes and locked audio. "
        "Connect existing H3 CONDITIONING and final-resolution AV LATENT. For keyframes, connect the same H3 video "
        "VAE to re-encode legacy low-stage guides; Director pre-encoded masters use latent copies without a VAE round trip. "
        "Audio drive uses JR Audio Driven Latent Builder. "
        "Batch 1, full schedule; empty target video or JR Sequential Continuation Guide's 12-token hard prefix "
        "with an empty generation suffix and fully locked audio. No spatial/soft/other partial masks. "
        "Scale 1 bypasses guide re-encoding and the resolution transition for A/B testing."
    )

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        schema["required"]["latent_image"] = ("LATENT", {"tooltip": "Final-resolution empty H3 video + empty/locked audio, or JR Sequential's 12-token hard-prefix AV latent. Both stages preserve the prefix mask."})
        schema["optional"] = {"vae": ("VAE", {"tooltip": "Same H3 VIDEO VAE for legacy keyframe re-encoding; not needed when all keyframes carry Director pre-encoded masters."})}
        return schema

    def sample(self, model, positive, noise, sampler, sigmas, latent_image,
               transition_step=3, lowres_scale=0.5, transition_seed_offset=1, aggressive_memory_cleanup=False, vae=None):
        return sample_h3_progressive(
            model=model, positive=positive, noise=noise, sampler=sampler, sigmas=sigmas,
            latent_image=latent_image, transition_step=transition_step, lowres_scale=lowres_scale,
            transition_seed_offset=transition_seed_offset, aggressive_memory_cleanup=aggressive_memory_cleanup,
            _guided=True, vae=vae,
        )
