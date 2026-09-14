"""Additive public streaming node; no production node interfaces are changed."""

from ..utils.h3_stream_cache import LAYER_POLICIES, RETENTIONS
from ..utils.h3_streaming_sampler import MODES, sample_streaming


class JR_H3_StreamingSampler:
    CATEGORY = "JR MiniMax H3/Experimental"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("output", "status")
    EXPERIMENTAL = True
    DESCRIPTION = "5s/10s/15s AV refinement with bounded micro-phases. Keep pass 1 full-length. Not the Hard AV Prefix sampler."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "positive": ("CONDITIONING",), "vae": ("VAE",), "noise": ("NOISE",),
            "sampler": ("SAMPLER",), "sigmas": ("SIGMAS",), "latent_image": ("LATENT",),
            "stream_plan": ("JR_H3_STREAM_PLAN",),
            "streaming_mode": (list(MODES), {"default": "Geometry Only"}),
            "retention": (list(RETENTIONS), {"default": "sink_plus_recent_2"}),
            "layer_policy": (list(LAYER_POLICIES), {"default": "every_4",
                "tooltip": "Only active in Sparse KV. Streaming Attention always caches all layers."}),
            "custom_layers": ("STRING", {"default": ""}),
            "cache_device": (["cpu", "cuda"], {"default": "cpu"}),
            "max_kv_mib": ("INT", {"default": 8192, "min": 64, "max": 262144}),
            "audio_reset_interval_requests": ("INT", {"default": 1, "min": 1, "max": 1000,
                                                       "tooltip": "Reserved for future continuation; each current execution starts with empty KV."}),
        }}

    def sample(self, **kwargs):
        return sample_streaming(**kwargs)
