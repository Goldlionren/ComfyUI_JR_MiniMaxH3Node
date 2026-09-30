"""Explicit experimental H3→LTX transfer; old H3 workflows are unchanged."""
from __future__ import annotations

from ..utils import h3_ltx_bridge as bridge


def adapter_files():
    from pathlib import Path

    import folder_paths
    if "h3_ltx_adapters" not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path("h3_ltx_adapters", str(Path(folder_paths.models_dir)/"h3_ltx_adapters"))
    return [name for name in folder_paths.get_filename_list("h3_ltx_adapters") if name.replace('\\', '/').endswith('model.safetensors')]


class JR_H3ToLTXLatentAdapter:
    CATEGORY = "JR MiniMax H3/LTX Bridge"
    FUNCTION = "convert"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("ltx_video_latent", "status")
    EXPERIMENTAL = True
    DESCRIPTION = "Convert completed, optionally upscaled H3 video LATENT to LTX-2.5. Split AV first. Native normalized input; preserves duration via internal LTX padding."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_latent": ("LATENT",), "adapter_name": (adapter_files(),)}}

    def convert(self, video_latent, adapter_name):
        import folder_paths
        bridge.video_tensor(video_latent, 24)
        adapter_files()
        path = folder_paths.get_full_path_or_raise("h3_ltx_adapters", adapter_name)
        adapter = bridge.load_adapter(path)
        result, meta = bridge.convert_video(video_latent, adapter)
        return result, f"H3→LTX: {meta['width']}×{meta['height']}, {meta['source_frames']} frames at 24 fps; internal padding to {meta['padded_frames']} frames."


class JR_H3LTXRefineSetup:
    CATEGORY = "JR MiniMax H3/LTX Bridge"
    FUNCTION = "prepare"
    RETURN_TYPES = ("LATENT", "SIGMAS", "AUDIO", "STRING")
    RETURN_NAMES = ("ltx_av_latent", "sigmas", "original_h3_audio", "status")
    EXPERIMENTAL = True
    DESCRIPTION = "Adjust LTX refinement steps and native ComfyUI denoise. Connect the same LTX MODEL used by BasicGuider. Keeps original H3 PCM for final mux."

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {"required": {"ltx_video_latent": ("LATENT",), "original_h3_audio": ("AUDIO",), "ltx_audio_vae": ("VAE",)},
                "optional": {
                    "steps": ("INT", {"default": 3, "min": 1, "max": 100, "tooltip": "LTX refinement updates; independent of H3 Stage1 steps."}),
                    "denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Native BasicScheduler denoise. Try 0.15–0.25 to retain the draft. 0 skips refinement."}),
                    "scheduler": (list(comfy.samplers.SCHEDULER_NAMES) + ["sol_h3_original"], {"default": "simple", "tooltip": "Native scheduler; sol_h3_original requires steps=3 and denoise=1 and reproduces the old fixed sigmas."}),
                    "ltx_model": ("MODEL", {"tooltip": "Connect the same LTX model as BasicGuider; required for native schedules."}),
                }}

    def prepare(self, ltx_video_latent, original_h3_audio, ltx_audio_vae,
                steps=3, denoise=1.0, scheduler="sol_h3_original", ltx_model=None):
        # Omitted fields in old API prompts retain the exact original recipe.
        return bridge.prepare_refine(ltx_video_latent, original_h3_audio, ltx_audio_vae,
                                     steps, denoise, scheduler, ltx_model)


class JR_H3LTXFinishMedia:
    CATEGORY = "JR MiniMax H3/LTX Bridge"
    FUNCTION = "finish"
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "original_h3_audio", "frame_rate", "status")
    EXPERIMENTAL = True
    DESCRIPTION = "After LTX Conv VAE decoding, remove internal temporal padding and restore original H3 audio. Connect the adapter's plain video LATENT for its timeline."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "original_h3_audio": ("AUDIO",), "bridge_latent": ("LATENT",)}}

    def finish(self, images, original_h3_audio, bridge_latent):
        return bridge.finish_media(images, original_h3_audio, bridge_latent)


class JR_LTXBridgeTextEncoderLoader:
    CATEGORY = "JR MiniMax H3/LTX Bridge"
    FUNCTION = "load"
    RETURN_TYPES = ("CLIP",)
    EXPERIMENTAL = True
    DESCRIPTION = "Native ComfyUI LTX text encoder with connectors read from a diffusion_models checkpoint. Select the same LTX-2.5 transformer used for Stage2."

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "text_encoder": (folder_paths.get_filename_list("text_encoders"),),
            "ltx_transformer": (folder_paths.get_filename_list("diffusion_models"),),
        }}

    def load(self, text_encoder, ltx_transformer):
        import comfy.sd
        import folder_paths
        paths = [folder_paths.get_full_path_or_raise("text_encoders", text_encoder),
                 folder_paths.get_full_path_or_raise("diffusion_models", ltx_transformer)]
        clip = comfy.sd.load_clip(ckpt_paths=paths, clip_type=comfy.sd.CLIPType.LTXV,
                                 embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (clip,)
