"""Unified Director PIPE consumer for the official MiniMax H3 conditioning nodes."""

from __future__ import annotations

from ..utils.h3_directed_conditioning import normalize_native_output, prepare_directed_inputs


class JR_H3_DirectedVideoConditioning:
    CATEGORY = "JR MiniMax H3/Generation"
    FUNCTION = "condition"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    DESCRIPTION = (
        "Consumes the immutable Director PIP and delegates to ComfyUI's current native "
        "MiniMax H3 Image-to-Video or Reference-to-Video conditioning implementation."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "pipe": ("JR_H3_DIRECTOR_PIPE",),
                "mode_override": (
                    ["Auto", "Image to Video", "Reference to Video"],
                    {"default": "Auto"},
                ),
                "dimension_source": (
                    ["Prefer Pipe", "Prefer Node"],
                    {"default": "Prefer Pipe"},
                ),
                "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": {"audio_vae": ("VAE",)},
        }

    def condition(
        self,
        clip,
        vae,
        pipe,
        mode_override="Auto",
        dimension_source="Prefer Pipe",
        width=1344,
        height=768,
        length=124,
        ref_image_size="match",
        audio_vae=None,
    ):
        if ref_image_size not in {"match", "max"}:
            raise ValueError("ref_image_size must be 'match' or 'max'.")
        try:
            from comfy_extras import nodes_minimax_h3 as native
        except ImportError:
            raise RuntimeError(
                "JR MiniMax H3 Directed Video Conditioning requires a current ComfyUI build "
                "with comfy_extras.nodes_minimax_h3."
            ) from None
        native_classes = {
            "Image to Video": getattr(native, "MiniMaxH3ImageToVideo", None),
            "Reference to Video": getattr(native, "MiniMaxH3ReferenceToVideo", None),
        }
        if (
            any(not callable(getattr(node_class, "execute", None)) for node_class in native_classes.values())
            or not callable(getattr(native, "adapt_canvas", None))
        ):
            raise RuntimeError(
                "The installed ComfyUI MiniMax H3 conditioning API is incompatible; "
                "update ComfyUI before using JR H3 Directed Video Conditioning."
            )
        prepared = prepare_directed_inputs(
            pipe,
            mode_override=mode_override,
            dimension_source=dimension_source,
            width=width,
            height=height,
            length=length,
            native_module=native,
        )
        if prepared.mode == "Image to Video":
            result = native_classes["Image to Video"].execute(
                clip=clip,
                vae=vae,
                prompt=prepared.prompt,
                width=prepared.width,
                height=prepared.height,
                length=prepared.length,
                first_frame=prepared.first_frame,
                last_frame=prepared.last_frame,
            )
        else:
            if prepared.ref_audios and audio_vae is None:
                raise ValueError(
                    "audio_vae is required when the Director PIP contains Reference or Driving Audio."
                )
            result = native_classes["Reference to Video"].execute(
                clip=clip,
                vae=vae,
                audio_vae=audio_vae,
                prompt=prepared.prompt,
                width=prepared.width,
                height=prepared.height,
                length=prepared.length,
                ref_image_size=ref_image_size,
                ref_images=dict(prepared.ref_images),
                ref_videos=dict(prepared.ref_videos),
                ref_video_audios={},
                ref_audios=dict(prepared.ref_audios),
            )
        return normalize_native_output(result)


__all__ = ["JR_H3_DirectedVideoConditioning"]
