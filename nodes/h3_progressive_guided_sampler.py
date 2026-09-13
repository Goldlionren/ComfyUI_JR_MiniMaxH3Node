"""Reference/keyframe/audio-driven variant, retaining the strict T2VA node."""

from ..utils.h3_progressive_sampler import sample_h3_progressive
from .h3_progressive_sampler import JR_H3_ProgressiveSampler


class JR_H3_ProgressiveGuidedSampler(JR_H3_ProgressiveSampler):
    DESCRIPTION = (
        "Experimental progressive H3 Euler with independent references, first/last keyframes and locked audio. "
        "Connect existing H3 CONDITIONING and final-resolution AV LATENT. For keyframes, connect the same H3 video "
        "VAE to re-encode low-stage guides. Audio drive uses JR Audio Driven Latent Builder. "
        "Batch 1, empty target video, full schedule; no partial masks or hard-prefix continuation. "
        "Scale 1 bypasses guide re-encoding and the resolution transition for A/B testing."
    )

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        schema["required"]["latent_image"] = ("LATENT", {"tooltip": "Final-resolution empty H3 video + empty or fully locked audio."})
        schema["optional"] = {"vae": ("VAE", {"tooltip": "Same H3 VIDEO VAE used for keyframes; required for their low-stage re-encoding."})}
        return schema

    def sample(self, model, positive, noise, sampler, sigmas, latent_image,
               transition_step=3, lowres_scale=0.5, transition_seed_offset=1, aggressive_memory_cleanup=False, vae=None):
        return sample_h3_progressive(
            model=model, positive=positive, noise=noise, sampler=sampler, sigmas=sigmas,
            latent_image=latent_image, transition_step=transition_step, lowres_scale=lowres_scale,
            transition_seed_offset=transition_seed_offset, aggressive_memory_cleanup=aggressive_memory_cleanup,
            _guided=True, vae=vae,
        )
