from __future__ import annotations

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_split_av_latent import JR_H3_SplitAVLatent
from ComfyUI_JR_MiniMaxH3Node.utils.h3_av_latent_builder import build_h3_av_latent
from ComfyUI_JR_MiniMaxH3Node.utils.h3_av_latent_split import (
    ERROR_PREFIX,
    H3AVLatentSplitError,
    split_h3_av_latent,
)


def _video(*, batch=1, channels=24, temporal=37, height=2, width=3, dtype=torch.float32):
    return torch.zeros((batch, channels, temporal, height, width), dtype=dtype)


def _audio(*, batch=1, channels=32, streams=2, temporal=207, dtype=torch.float32):
    return torch.zeros((batch, channels, streams, temporal), dtype=dtype)


def _av(*items):
    return {"samples": NestedTensor(items)}


def _assert_split_error(av_latent, expected):
    with pytest.raises(H3AVLatentSplitError) as exc:
        split_h3_av_latent(av_latent)
    message = str(exc.value)
    assert message.startswith(ERROR_PREFIX)
    assert expected in message


def test_node_schema_and_success_preserve_exact_tensor_objects():
    node = JR_H3_SplitAVLatent()
    assert node.INPUT_TYPES() == {"required": {"av_latent": ("LATENT",)}}
    assert node.RETURN_TYPES == ("LATENT", "LATENT")
    assert node.RETURN_NAMES == ("video_latent", "audio_latent")
    assert node.CATEGORY == "JR MiniMax H3/Latent"

    video = _video(dtype=torch.float16)
    audio = _audio(dtype=torch.float16)
    video_out, audio_out = node.split(_av(video, audio))

    assert video_out == {"samples": video}
    assert audio_out == {"samples": audio}
    assert video_out["samples"] is video
    assert audio_out["samples"] is audio
    assert video_out["samples"].data_ptr() == video.data_ptr()
    assert audio_out["samples"].data_ptr() == audio.data_ptr()
    assert video_out["samples"].shape == video.shape
    assert audio_out["samples"].shape == audio.shape
    assert video_out["samples"].dtype == video.dtype
    assert audio_out["samples"].dtype == audio.dtype
    assert video_out["samples"].device == video.device
    assert audio_out["samples"].device == audio.device


class _FakeNestedTensor:
    def unbind(self):
        return (_video(), _audio())


@pytest.mark.parametrize(
    ("latent", "expected"),
    [
        (None, "must be a LATENT dictionary"),
        ({}, "missing required 'samples'"),
        ({"samples": _video()}, "official comfy.nested_tensor.NestedTensor"),
        ({"samples": _FakeNestedTensor()}, "official comfy.nested_tensor.NestedTensor"),
    ],
)
def test_container_and_exact_official_nested_tensor_validation(latent, expected):
    _assert_split_error(latent, expected)


@pytest.mark.parametrize("items", [(_video(),), (_video(), _audio(), _audio())])
def test_wrong_stream_count_is_rejected(items):
    _assert_split_error(_av(*items), "exactly 2 streams")


def test_non_tensor_component_is_rejected():
    _assert_split_error(_av(_video(), object()), "must both be torch.Tensor")


@pytest.mark.parametrize(
    ("video", "expected"),
    [
        (torch.zeros((1, 24, 37, 2)), "[B,24,T,H,W]"),
        (_video(channels=16), "[B,24,T,H,W]"),
        (_video(temporal=0), "must all be greater than zero"),
        (_video(dtype=torch.int32), "floating-point tensor"),
    ],
)
def test_bad_video_shapes_and_storage_are_rejected(video, expected):
    _assert_split_error(_av(video, _audio()), expected)


@pytest.mark.parametrize(
    ("audio", "expected"),
    [
        (torch.zeros((1, 32, 207)), "[B,32,2,T]"),
        (_audio(channels=16), "[B,32,2,T]"),
        (_audio(streams=1), "[B,32,2,T]"),
        (_audio(temporal=0), "must both be greater than zero"),
        (_audio(dtype=torch.int32), "floating-point tensor"),
    ],
)
def test_bad_audio_shapes_and_storage_are_rejected(audio, expected):
    _assert_split_error(_av(_video(), audio), expected)


def test_batch_mismatch_is_rejected():
    _assert_split_error(_av(_video(batch=2), _audio(batch=1)), "video/audio batch mismatch")


@pytest.mark.parametrize(
    ("stream", "bad_value"),
    [("video", float("nan")), ("video", float("inf")), ("audio", float("nan")), ("audio", float("-inf"))],
)
def test_non_finite_stream_values_are_rejected(stream, bad_value):
    video = _video()
    audio = _audio()
    if stream == "video":
        video[0, 0, 0, 0, 0] = bad_value
    else:
        audio[0, 0, 0, 0] = bad_value
    _assert_split_error(_av(video, audio), "contains NaN or Inf")


def test_builder_to_split_roundtrip_and_save_latent_compatibility():
    video = torch.randn_like(_video())
    audio = torch.randn_like(_audio())
    av_latent, _status = build_h3_av_latent({"samples": video}, {"samples": audio})

    video_out, audio_out = split_h3_av_latent(av_latent)

    assert video_out["samples"] is video
    assert audio_out["samples"] is audio
    assert torch.equal(video_out["samples"], video)
    assert torch.equal(audio_out["samples"], audio)
    # This is the data operation used by ComfyUI's native Save Latent node.
    assert torch.equal(video_out["samples"].contiguous(), video.contiguous())
    assert torch.equal(audio_out["samples"].contiguous(), audio.contiguous())


def test_non_contiguous_streams_are_not_eagerly_made_contiguous():
    video = torch.zeros((1, 24, 37, 2, 6))[..., ::2]
    audio = torch.zeros((1, 32, 2, 414))[..., ::2]
    assert not video.is_contiguous()
    assert not audio.is_contiguous()

    video_out, audio_out = split_h3_av_latent(_av(video, audio))

    assert video_out["samples"] is video
    assert audio_out["samples"] is audio
    assert not video_out["samples"].is_contiguous()
    assert not audio_out["samples"].is_contiguous()
