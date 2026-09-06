"""Standard-item Builder and Unpack nodes for JR_H3_DIRECTOR_PIPE."""

from __future__ import annotations

from comfy_api.latest import io

from ..utils.director_pipe_io import build_pipe_from_standard_inputs, unpack_director_pipe


class JR_H3_DirectorPipeBuilder(io.ComfyNode):
    CATEGORY = "JR MiniMax H3/Director"
    RETURN_TYPES = ("JR_H3_DIRECTOR_PIPE",)
    RETURN_NAMES = ("pip",)
    DESCRIPTION = (
        "Builds an immutable Director PIPE from standard ComfyUI STRING, IMAGE, VIDEO and AUDIO values. "
        "Runtime tensors/media stay out of workflow JSON."
    )

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JR_H3_DirectorPipeBuilder",
            display_name="JR MiniMax H3 Director PIPE Builder",
            category=cls.CATEGORY,
            description=cls.DESCRIPTION,
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.Float.Input("duration_seconds", default=10.0, min=0.1, max=3600.0, step=0.1),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.01),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Image.Input("reference_images", optional=True),
                io.Video.Input("reference_video", optional=True),
                io.Audio.Input("reference_audio", optional=True),
                io.Audio.Input("driving_audio", optional=True),
                io.Autogrow.Input("reference_videos", optional=True,
                    template=io.Autogrow.TemplateNames(io.Video.Input("video"),
                        names=["reference_video_2", "reference_video_3"], min=0)),
                io.Autogrow.Input("reference_audios", optional=True,
                    template=io.Autogrow.TemplateNames(io.Audio.Input("audio"),
                        names=["reference_audio_2", "reference_audio_3"], min=0)),
            ],
            outputs=[io.Custom("JR_H3_DIRECTOR_PIPE").Output(display_name="pip")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*cls.build(**kwargs))

    @classmethod
    def build(
        cls,
        prompt,
        duration_seconds=10.0,
        fps=24.0,
        first_frame=None,
        last_frame=None,
        reference_images=None,
        reference_video=None,
        reference_audio=None,
        driving_audio=None,
        reference_videos=None,
        reference_audios=None,
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
            reference_videos=tuple((reference_videos or {}).get(f"reference_video_{i}") for i in (2, 3)),
            reference_audios=tuple((reference_audios or {}).get(f"reference_audio_{i}") for i in (2, 3)),
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
