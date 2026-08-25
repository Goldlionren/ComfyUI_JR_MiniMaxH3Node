"""Standard-item Builder and Unpack nodes for JR_H3_DIRECTOR_PIPE."""

from __future__ import annotations

from ..utils.director_pipe_io import build_pipe_from_standard_inputs, unpack_director_pipe


class JR_H3_DirectorPipeBuilder:
    CATEGORY = "JR MiniMax H3/Director"
    FUNCTION = "build"
    RETURN_TYPES = ("JR_H3_DIRECTOR_PIPE",)
    RETURN_NAMES = ("pip",)
    DESCRIPTION = (
        "Builds an immutable Director PIPE from standard ComfyUI STRING, IMAGE, VIDEO and AUDIO values. "
        "Runtime tensors/media stay out of workflow JSON."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.1, "max": 3600.0, "step": 0.1},
                ),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01},
                ),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "reference_video": ("VIDEO",),
                "reference_audio": ("AUDIO",),
                "driving_audio": ("AUDIO",),
            },
        }

    def build(
        self,
        prompt,
        duration_seconds=10.0,
        fps=24.0,
        first_frame=None,
        last_frame=None,
        reference_images=None,
        reference_video=None,
        reference_audio=None,
        driving_audio=None,
    ):
        pipe = build_pipe_from_standard_inputs(
            prompt=prompt,
            duration_seconds=duration_seconds,
            fps=fps,
            first_frame=first_frame,
            last_frame=last_frame,
            reference_images=reference_images,
            reference_video=reference_video,
            reference_audio=reference_audio,
            driving_audio=driving_audio,
        )
        return (pipe,)


class JR_H3_DirectorPipeUnpack:
    CATEGORY = "JR MiniMax H3/Director"
    FUNCTION = "unpack"
    RETURN_TYPES = (
        "JR_H3_DIRECTOR_PIPE",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "FLOAT",
        "FLOAT",
        "INT",
        "INT",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "VIDEO",
        "AUDIO",
        "AUDIO",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "pip",
        "prompt",
        "director_prompt",
        "optimized_prompt",
        "reviewed_prompt",
        "duration_seconds",
        "fps",
        "width",
        "height",
        "first_frame",
        "last_frame",
        "reference_image",
        "reference_video",
        "reference_audio",
        "driving_audio",
        "registry_json",
        "status",
    )
    DESCRIPTION = (
        "Passes the immutable Director PIPE through and exposes prompt stages, timeline metadata and "
        "index-selected standard IMAGE, VIDEO and AUDIO values without mutating the input PIPE."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pip": ("JR_H3_DIRECTOR_PIPE",),
                "reference_image_index": (
                    "INT",
                    {"default": 1, "min": 1, "max": 9, "step": 1},
                ),
                "reference_video_index": (
                    "INT",
                    {"default": 1, "min": 1, "max": 3, "step": 1},
                ),
                "reference_audio_index": (
                    "INT",
                    {"default": 1, "min": 1, "max": 3, "step": 1},
                ),
                "driving_audio_index": (
                    "INT",
                    {"default": 1, "min": 1, "max": 3, "step": 1},
                ),
            }
        }

    def unpack(
        self,
        pip,
        reference_image_index=1,
        reference_video_index=1,
        reference_audio_index=1,
        driving_audio_index=1,
    ):
        unpacked = unpack_director_pipe(
            pip,
            reference_image_index=reference_image_index,
            reference_video_index=reference_video_index,
            reference_audio_index=reference_audio_index,
            driving_audio_index=driving_audio_index,
        )
        return (
            unpacked.pipe,
            unpacked.prompt,
            unpacked.director_prompt,
            unpacked.optimized_prompt,
            unpacked.reviewed_prompt,
            unpacked.duration_seconds,
            unpacked.fps,
            unpacked.width,
            unpacked.height,
            unpacked.first_frame,
            unpacked.last_frame,
            unpacked.reference_image,
            unpacked.reference_video,
            unpacked.reference_audio,
            unpacked.driving_audio,
            unpacked.registry_json,
            unpacked.status,
        )


__all__ = ["JR_H3_DirectorPipeBuilder", "JR_H3_DirectorPipeUnpack"]
