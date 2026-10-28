"""ComfyUI entry points for Open-H3-Regenerate adapters and video refinement."""
from __future__ import annotations

from ..utils.h3_neural_latent_upscaler import list_h3_upscaler_models
from ..utils.h3_regenerate import ATTENTION_MODES, regenerate
from ..utils.h3_regenerate_lora import patch_regenerate_bundle


class JR_H3_RegenerateLoRA:
    CATEGORY = "JR MiniMax H3/Loader"
    FUNCTION = "load"
    RETURN_TYPES = ("MODEL",)
    DESCRIPTION = (
        "Loads the converted Open-H3-Regenerate Refiner+Turbo+Sharp adapter including all Sharp curve branches. "
        "Use a full BF16 H3 model. Connect BEFORE Unified acceleration; do not add Turbo/Sharp twice."
    )

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1., "min": 0., "max": 2., "step": .05}),
        }}

    def load(self, model, lora_name, strength):
        import comfy.utils
        import folder_paths
        if lora_name not in folder_paths.get_filename_list("loras"):
            raise ValueError("Choose a converted Regenerate bundle from models/loras")
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        raw, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        if metadata.get("format") != "jr-h3-regenerate-v1":
            raise ValueError("Run tools/convert_h3_regenerate_lora.py first; this is not a converted Regenerate bundle")
        return (patch_regenerate_bundle(model, raw, strength),)


class JR_H3_Regenerate:
    CATEGORY = "JR MiniMax H3/Sampling"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("h3_av_latent", "status")
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Video-only H3 regeneration: native VIDEO + H3 VAEs or an existing H3 AV LATENT. "
        "Use MODEL from Regenerate LoRA then optional Unified. Inherit preserves Unified's Sage/FFN. "
        "Audio is locked throughout sampling and the exact source audio latent is returned. "
        "VIDEO is sampled at 24 fps and end-padded to H3's frame grid; trim on decode if needed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",),
                "source_mode": (["H3 latent", "video"],),
                "sigma_schedule": (["low_denoise_shift6", "release_original"], {"default": "low_denoise_shift6",
                    "tooltip": "Internal exact schedules. Low denoise uses 0.15–0.35. Original release fixes its 2/3-step sigmas and ignores denoise."}),
                "denoise": ("FLOAT", {"default": .30, "min": .15, "max": .35, "step": .01,
                                      "tooltip": "Starting sigma, not a fraction of discarded sampler steps."}),
                "steps": ("INT", {"default": 3, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 1766580932, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"default": "Faithful reconstruction of the input video; preserve motion, composition, colors and fine details.",
                                      "multiline": True, "dynamicPrompts": True}),
                "attention": (list(ATTENTION_MODES), {"default": "inherit",
                    "tooltip": "For upstream Unified select inherit. cuda++ is the installed KJ SageAttention 2++ path."}),
                "resize_mode": (["keep", "scale", "megapixels"], {"default": "scale"}),
                "scale": ("FLOAT", {"default": 2., "min": 1., "max": 4., "step": .05}),
                "target_megapixels": ("FLOAT", {"default": 3.6864, "min": .01, "max": 16., "step": .01}),
                "upscaler_model": (list_h3_upscaler_models(), {"default": "auto"}),
            },
            "optional": {
                "latent": ("LATENT",), "video": ("VIDEO",), "vae": ("VAE",), "audio_vae": ("VAE",),
            },
        }

    def sample(self, **kwargs):
        return regenerate(**kwargs)
