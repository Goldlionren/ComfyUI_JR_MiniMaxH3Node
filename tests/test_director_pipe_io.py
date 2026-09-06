from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.nodes.director_pipe_io import (
    JR_H3_DirectorPipeBuilder,
    JR_H3_DirectorPipeUnpack,
)
from ComfyUI_JR_MiniMaxH3Node.utils.director_pipe import validate_director_pipe
from ComfyUI_JR_MiniMaxH3Node.utils.director_pipe_io import (
    build_pipe_from_standard_inputs,
    unpack_director_pipe,
)
from ComfyUI_JR_MiniMaxH3Node.utils.h3_directed_conditioning import prepare_directed_inputs


class FakeVideo:
    def __init__(self, frames=None, *, width=96, height=64, duration=1.0, fps=24):
        self.frames = frames if frames is not None else torch.zeros(5, height, width, 3)
        self.width = width
        self.height = height
        self.duration = duration
        self.fps = Fraction(fps, 1)
        self.trim_calls = []

    def get_dimensions(self):
        return self.width, self.height

    def get_duration(self):
        return self.duration

    def get_frame_rate(self):
        return self.fps

    def get_components(self):
        return SimpleNamespace(images=self.frames, frame_rate=self.fps)

    def as_trimmed(self, start_time=0, duration=0, strict_duration=True):
        self.trim_calls.append((start_time, duration, strict_duration))
        return self


class FakeNative:
    nodes = SimpleNamespace(MAX_RESOLUTION=16384)

    @staticmethod
    def adapt_canvas(width, height):
        return width, height


def _audio(samples=24_000, sample_rate=24_000, channels=1):
    return {
        "waveform": torch.zeros(1, channels, samples),
        "sample_rate": sample_rate,
    }


def test_node_contracts_are_compact_and_standard_typed():
    builder = JR_H3_DirectorPipeBuilder.INPUT_TYPES()
    assert list(builder["required"]) == ["prompt", "duration_seconds", "fps"]
    assert list(builder["optional"]) == [
        "first_frame",
        "last_frame",
        "reference_images",
        "reference_video",
        "reference_audio",
        "driving_audio",
        "reference_videos",
        "reference_audios",
    ]
    assert JR_H3_DirectorPipeBuilder.RETURN_TYPES == ("JR_H3_DIRECTOR_PIPE",)
    assert JR_H3_DirectorPipeBuilder.RETURN_NAMES == ("pip",)

    unpack_inputs = JR_H3_DirectorPipeUnpack.INPUT_TYPES()["required"]
    assert list(unpack_inputs) == [
        "pip",
        "reference_image_index",
        "reference_video_index",
        "reference_audio_index",
        "driving_audio_index",
    ]
    assert JR_H3_DirectorPipeUnpack.RETURN_NAMES == (
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
    assert len(JR_H3_DirectorPipeUnpack.RETURN_TYPES) == len(JR_H3_DirectorPipeUnpack.RETURN_NAMES)


def test_builder_and_unpack_round_trip_standard_values_without_mutation():
    first = torch.zeros(1, 8, 12, 3)
    last = torch.ones(1, 8, 12, 3)
    references = torch.stack((
        torch.full((8, 12, 3), 0.25),
        torch.full((8, 12, 3), 0.75),
    ))
    video = FakeVideo(width=96, height=64, duration=2.0)
    reference_audio = _audio(channels=1)
    driving_audio = _audio(samples=48_000, channels=2)

    pipe = build_pipe_from_standard_inputs(
        prompt="Exact final prompt",
        duration_seconds=2.0,
        fps=24.0,
        first_frame=first,
        last_frame=last,
        reference_images=references,
        reference_video=video,
        reference_audio=reference_audio,
        driving_audio=driving_audio,
    )
    assert validate_director_pipe(pipe) is pipe
    assert pipe.final_prompt() == "Exact final prompt"
    assert pipe.optimized_prompt == "Exact final prompt"
    assert pipe.reviewed_prompt == ""
    assert "Exact final prompt" in pipe.compiled_director_prompt
    assert [(item.family, item.role) for item in pipe.reference_registry] == [
        ("Picture", "first_frame"),
        ("Picture", "last_frame"),
        ("Picture", "reference_image"),
        ("Picture", "reference_image"),
        ("Video", "reference_video"),
        ("Audio", "reference_audio"),
        ("Audio", "driving_audio"),
    ]

    unpacked = unpack_director_pipe(
        pipe,
        reference_image_index=2,
        reference_video_index=1,
        reference_audio_index=1,
        driving_audio_index=1,
    )
    assert unpacked.pipe is pipe
    assert unpacked.prompt == "Exact final prompt"
    assert unpacked.duration_seconds == 2.0
    assert unpacked.fps == 24.0
    assert (unpacked.width, unpacked.height) == (12, 8)
    assert unpacked.first_frame is first
    assert unpacked.last_frame is last
    assert torch.equal(unpacked.reference_image, references[1:2])
    assert unpacked.reference_video is video
    assert unpacked.reference_audio["sample_rate"] == reference_audio["sample_rate"]
    assert unpacked.reference_audio["waveform"].shape[1] == 2
    assert reference_audio["waveform"].shape[1] == 1
    assert unpacked.driving_audio["sample_rate"] == driving_audio["sample_rate"]
    assert unpacked.driving_audio["waveform"] is driving_audio["waveform"]
    assert "waveform" not in unpacked.registry_json
    assert "Pictures=2 (selected 2: yes)" in unpacked.status
    assert pipe.optimized_prompt == "Exact final prompt"


def test_builder_node_and_unpack_node_return_declared_arity():
    pipe, = JR_H3_DirectorPipeBuilder().build(
        "Prompt",
        duration_seconds=1.0,
        fps=24.0,
        first_frame=torch.zeros(1, 32, 32, 3),
    )
    output = JR_H3_DirectorPipeUnpack().unpack(pipe, 1, 1, 1, 1)
    assert len(output) == len(JR_H3_DirectorPipeUnpack.RETURN_TYPES)
    assert output[0] is pipe
    assert output[1] == "Prompt"
    assert output[9] is pipe.runtime_media[0].payload
    assert output[11] is output[12] is output[13] is output[14] is None


def test_unpack_out_of_range_selection_is_none_and_pipe_remains_available():
    pipe = build_pipe_from_standard_inputs(
        prompt="Prompt",
        duration_seconds=1.0,
        fps=24.0,
        reference_images=torch.zeros(1, 32, 32, 3),
    )
    unpacked = unpack_director_pipe(
        pipe,
        reference_image_index=9,
        reference_video_index=3,
        reference_audio_index=3,
        driving_audio_index=3,
    )
    assert unpacked.pipe is pipe
    assert unpacked.reference_image is None
    assert unpacked.reference_video is None
    assert unpacked.reference_audio is None
    assert unpacked.driving_audio is None
    assert "selected 9: none" in unpacked.status


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt": ""}, "prompt must not be empty"),
        ({"first_frame": torch.zeros(2, 8, 8, 3)}, "exactly one IMAGE"),
        ({"reference_images": torch.zeros(10, 8, 8, 3)}, "at most 9"),
        ({"reference_audio": {"waveform": torch.zeros(2, 1, 8), "sample_rate": 8}}, "shaped"),
        ({"reference_video": object()}, "standard ComfyUI VIDEO"),
    ],
)
def test_builder_rejects_invalid_standard_inputs(kwargs, message):
    values = {"prompt": "Prompt", "duration_seconds": 1.0, "fps": 24.0}
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        build_pipe_from_standard_inputs(**values)


def test_builder_rejects_anchor_plus_reference_over_native_picture_limit():
    with pytest.raises(ValueError, match="native 9-Picture limit"):
        build_pipe_from_standard_inputs(
            prompt="Prompt",
            duration_seconds=1.0,
            fps=24.0,
            first_frame=torch.zeros(1, 8, 8, 3),
            reference_images=torch.zeros(9, 8, 8, 3),
        )


def test_standard_video_builder_pipe_is_consumed_by_directed_conditioning():
    video = FakeVideo(width=96, height=64, duration=1.0)
    pipe = build_pipe_from_standard_inputs(
        prompt="Prompt",
        duration_seconds=1.0,
        fps=24.0,
        reference_video=video,
    )
    prepared = prepare_directed_inputs(
        pipe,
        mode_override="Auto",
        dimension_source="Prefer Pipe",
        width=64,
        height=64,
        length=24,
        native_module=FakeNative,
    )
    assert prepared.mode == "Reference to Video"
    assert prepared.prompt == "Prompt"
    assert (prepared.width, prepared.height) == (96, 64)
    assert prepared.ref_videos[0][1] is video.frames
    assert video.trim_calls == [(0.0, 1.0, False)]


def test_multiple_references_preserve_order_payloads_and_native_routing():
    videos = [FakeVideo() for _ in range(3)]
    audios = [_audio(channels=2) for _ in range(3)]
    output = JR_H3_DirectorPipeBuilder.execute(
        prompt="Prompt", duration_seconds=1.0, reference_video=videos[0], reference_audio=audios[0],
        reference_videos={"reference_video_3": videos[2], "reference_video_2": videos[1]},
        reference_audios={"reference_audio_3": audios[2], "reference_audio_2": audios[1]},
    )
    pipe = output.result[0]
    validate_director_pipe(pipe)
    prepared = prepare_directed_inputs(
        pipe, mode_override="Auto", dimension_source="Prefer Pipe", width=64, height=64,
        length=24, native_module=FakeNative,
    )
    assert len(prepared.ref_videos) == len(prepared.ref_audios) == 3
    for i in range(3):
        unpacked = unpack_director_pipe(pipe, reference_video_index=i + 1, reference_audio_index=i + 1,
                                       reference_image_index=1, driving_audio_index=1)
        assert unpacked.reference_video is videos[i]
        assert unpacked.reference_audio["waveform"] is audios[i]["waveform"]
        assert prepared.ref_videos[i][1] is videos[i].frames
    assert [r.label for r in pipe.reference_registry if r.family == "Video"] == ["<Video 1>", "<Video 2>", "<Video 3>"]


def test_autogrow_schema_and_sparse_reference_slots():
    schema = JR_H3_DirectorPipeBuilder.INPUT_TYPES()
    assert JR_H3_DirectorPipeBuilder.FUNCTION == "EXECUTE_NORMALIZED"
    for kind in ("video", "audio"):
        template = schema["optional"][f"reference_{kind}s"][1]["template"]
        assert template["min"] == 0
        assert template["names"] == [f"reference_{kind}_2", f"reference_{kind}_3"]
    video = FakeVideo()
    pipe, = JR_H3_DirectorPipeBuilder.build(prompt="Prompt", reference_videos={"reference_video_3": video})
    assert unpack_director_pipe(pipe, reference_image_index=1, reference_video_index=1,
                                reference_audio_index=1, driving_audio_index=1).reference_video is video


@pytest.mark.parametrize("kwargs", [
    {"reference_videos": {"reference_video_2": object()}},
    {"reference_audios": {"reference_audio_3": {"waveform": None, "sample_rate": 24000}}},
])
def test_additional_reference_inputs_are_validated(kwargs):
    with pytest.raises(ValueError):
        JR_H3_DirectorPipeBuilder.build(prompt="Prompt", **kwargs)


def test_native_execution_expands_autogrow_and_preserves_legacy_prompt_inputs():
    import asyncio

    from comfy_api.latest import _io
    from execution import _async_map_node_over_list

    video = FakeVideo()
    for live in (
        {"prompt": "Prompt", "reference_video": video},
        {"prompt": "Prompt", "reference_videos.reference_video_2": video},
        {"prompt": "Prompt"},
    ):
        _, _, v3_data = _io.get_finalized_class_inputs(JR_H3_DirectorPipeBuilder.INPUT_TYPES(), live)
        v3_data["hidden_inputs"] = {}
        result = asyncio.run(_async_map_node_over_list(
            "test", "1", JR_H3_DirectorPipeBuilder, {k: [v] for k, v in live.items()},
            JR_H3_DirectorPipeBuilder.FUNCTION, v3_data=v3_data,
        ))
        pipe = result[0].result[0]
        assert validate_director_pipe(pipe) is pipe
        assert len(pipe.runtime_media) == (0 if len(live) == 1 else 1)
