import pytest
import torch
from comfy.nested_tensor import NestedTensor
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_audio_driven_latent_builder import (
    JR_H3_AudioDrivenLatentBuilder,
)
from ComfyUI_JR_MiniMaxH3Node.utils.h3_audio_driven_latent_builder import (
    ERROR_PREFIX,
    H3AudioDrivenLatentBuilderError,
    build_h3_audio_driven_latent,
)

VIDEO_T = 22
AUDIO_T = 122


def _video(batch=1, *, dtype=torch.float32):
    return torch.full((batch, 24, VIDEO_T, 3, 4), 0.25, dtype=dtype)


def _audio(batch=1, time=AUDIO_T, *, dtype=torch.float32):
    values = torch.arange(batch * 32 * 2 * time, dtype=torch.float32).reshape(batch, 32, 2, time)
    return values.to(dtype=dtype) / max(1, values.numel())


def _av_latent(batch=1, *, dtype=torch.float32, with_mask=True):
    video = _video(batch, dtype=dtype)
    audio = torch.zeros((batch, 32, 2, AUDIO_T), dtype=dtype)
    latent = {
        "samples": NestedTensor((video, audio)),
        "batch_index": list(range(batch)),
        "custom": {"preserve": True},
    }
    if with_mask:
        video_mask = torch.linspace(0, 1, VIDEO_T, dtype=dtype).reshape(1, 1, VIDEO_T, 1, 1)
        audio_mask = torch.ones((1, 1, 2, AUDIO_T), dtype=dtype)
        latent["noise_mask"] = NestedTensor((video_mask, audio_mask))
    return latent


def _output_streams(output):
    video, audio = output["samples"].unbind()
    video_mask, audio_mask = output["noise_mask"].unbind()
    return video, audio, video_mask, audio_mask


def _assert_error(av_latent, audio_drive_latent, expected):
    with pytest.raises(H3AudioDrivenLatentBuilderError) as exc:
        build_h3_audio_driven_latent(av_latent, audio_drive_latent)
    message = str(exc.value)
    assert message.startswith(ERROR_PREFIX)
    assert expected in message


def test_node_contract_and_declared_output_arity():
    inputs = JR_H3_AudioDrivenLatentBuilder.INPUT_TYPES()
    assert list(inputs["required"]) == ["av_latent", "audio_drive_latent"]
    assert JR_H3_AudioDrivenLatentBuilder.CATEGORY == "JR MiniMax H3/Latent"
    assert JR_H3_AudioDrivenLatentBuilder.RETURN_TYPES == ("LATENT", "STRING")
    assert JR_H3_AudioDrivenLatentBuilder.RETURN_NAMES == ("audio_driven_av_latent", "status")

    result = JR_H3_AudioDrivenLatentBuilder().build(_av_latent(), {"samples": _audio()})
    assert len(result) == 2


def test_equal_length_replaces_audio_locks_it_and_preserves_video_mask_and_metadata():
    av_latent = _av_latent()
    audio_drive = {"samples": _audio(), "source": "keep-input-unchanged"}
    original_samples = av_latent["samples"]
    original_noise_mask = av_latent["noise_mask"]
    original_video, original_template = original_samples.unbind()
    original_video_mask, original_audio_mask = original_noise_mask.unbind()

    output, status = build_h3_audio_driven_latent(av_latent, audio_drive)
    video, audio, video_mask, audio_mask = _output_streams(output)

    assert output is not av_latent
    assert output["custom"] is av_latent["custom"]
    assert output["batch_index"] is av_latent["batch_index"]
    assert video is original_video
    assert audio is audio_drive["samples"]
    assert video_mask is original_video_mask
    assert torch.count_nonzero(audio_mask).item() == 0
    assert audio_mask.shape == audio.shape
    assert audio_mask.dtype == audio.dtype
    assert audio_mask.device == audio.device
    assert av_latent["samples"] is original_samples
    assert av_latent["noise_mask"] is original_noise_mask
    assert original_samples.unbind()[1] is original_template
    assert original_noise_mask.unbind()[1] is original_audio_mask
    assert audio_drive["samples"] is audio
    assert "Status: READY" in status
    assert "Time Fit: unchanged" in status
    assert "Batch Fit: unchanged" in status
    assert "Noise Mask: preserved" in status
    assert "Audio: LOCKED" in status


def test_missing_noise_mask_generates_video_ones_and_audio_zeros():
    av_latent = _av_latent(with_mask=False)
    output, status = build_h3_audio_driven_latent(av_latent, {"samples": _audio()})
    video, _, video_mask, audio_mask = _output_streams(output)

    assert torch.equal(video_mask, torch.ones_like(video))
    assert torch.count_nonzero(audio_mask).item() == 0
    assert "Noise Mask: generated ones_like(video)" in status
    assert "noise_mask" not in av_latent


def test_long_audio_is_trimmed_on_final_dimension():
    drive = _audio(time=AUDIO_T + 7)
    output, status = build_h3_audio_driven_latent(_av_latent(), {"samples": drive})
    audio = output["samples"].unbind()[1]

    assert audio.shape == (1, 32, 2, AUDIO_T)
    assert torch.equal(audio, drive[..., :AUDIO_T])
    assert f"Time Fit: trimmed {AUDIO_T + 7} -> {AUDIO_T}" in status


def test_short_audio_is_zero_padded_on_final_dimension():
    input_t = AUDIO_T - 9
    drive = _audio(time=input_t)
    output, status = build_h3_audio_driven_latent(_av_latent(), {"samples": drive})
    audio = output["samples"].unbind()[1]

    assert audio.shape == (1, 32, 2, AUDIO_T)
    assert torch.equal(audio[..., :input_t], drive)
    assert torch.count_nonzero(audio[..., input_t:]).item() == 0
    assert f"Time Fit: padded {input_t} -> {AUDIO_T}" in status


def test_audio_batch_one_is_replicated_to_av_batch_without_reordering():
    drive = _audio(batch=1)
    output, status = build_h3_audio_driven_latent(_av_latent(batch=4), {"samples": drive})
    audio = output["samples"].unbind()[1]

    assert audio.shape == (4, 32, 2, AUDIO_T)
    assert all(torch.equal(audio[index:index + 1], drive) for index in range(4))
    assert "Batch Fit: replicated 1 -> 4" in status


def test_matching_multi_batch_is_preserved_without_copy():
    drive = _audio(batch=3)
    output, status = build_h3_audio_driven_latent(_av_latent(batch=3), {"samples": drive})
    audio = output["samples"].unbind()[1]

    assert audio is drive
    assert "Batch Fit: unchanged" in status


def test_audio_is_cast_to_template_dtype_without_changing_video():
    av_latent = _av_latent(dtype=torch.float16)
    video = av_latent["samples"].unbind()[0]
    drive = _audio(dtype=torch.float32)
    output, status = build_h3_audio_driven_latent(av_latent, {"samples": drive})
    out_video, out_audio = output["samples"].unbind()

    assert out_video is video
    assert out_audio.dtype == torch.float16
    assert drive.dtype == torch.float32
    assert "Dtype Fit: torch.float32 -> torch.float16" in status


def test_output_samples_and_masks_follow_native_sampler_pack_contract():
    from comfy import sampler_helpers, utils

    output, _ = build_h3_audio_driven_latent(_av_latent(batch=2), {"samples": _audio()})
    streams = output["samples"].unbind()
    masks = output["noise_mask"].unbind()
    prepared_masks = [
        sampler_helpers.prepare_mask(mask, stream.shape, stream.device)
        for mask, stream in zip(masks, streams, strict=True)
    ]
    packed_samples, latent_shapes = utils.pack_latents(streams)
    packed_masks, mask_shapes = utils.pack_latents(prepared_masks)

    assert packed_samples.shape == packed_masks.shape
    assert latent_shapes == mask_shapes
    assert torch.count_nonzero(prepared_masks[1]).item() == 0


@pytest.mark.parametrize(
    ("av_latent", "expected"),
    [
        ({}, "missing required 'samples'"),
        ({"samples": torch.zeros(1)}, "valid MiniMax H3 joint AV NestedTensor"),
        ({"samples": NestedTensor((_video(),))}, "exactly 2 streams"),
        ({"samples": NestedTensor((_video(), object()))}, "must both be torch.Tensor"),
        (
            {"samples": NestedTensor((torch.zeros(1, 23, VIDEO_T, 3, 4), torch.zeros(1, 32, 2, AUDIO_T)))},
            "[B,24,T,H,W]",
        ),
        (
            {"samples": NestedTensor((_video(), torch.zeros(1, 31, 2, AUDIO_T)))},
            "[B,32,2,T]",
        ),
        (
            {"samples": NestedTensor((_video(batch=2), torch.zeros(1, 32, 2, AUDIO_T)))},
            "batch mismatch",
        ),
        (
            {"samples": NestedTensor((torch.zeros(1, 24, 21, 3, 4), torch.zeros(1, 32, 2, AUDIO_T)))},
            "invalid H3 temporal grid",
        ),
    ],
)
def test_invalid_av_latent_fails_closed(av_latent, expected):
    _assert_error(av_latent, {"samples": _audio()}, expected)


@pytest.mark.parametrize(
    ("audio_drive_latent", "expected"),
    [
        ({}, "missing required 'samples'"),
        ({"samples": object()}, "must be a torch.Tensor"),
        ({"samples": torch.zeros(1, 32, AUDIO_T)}, "[B,32,2,T]"),
        ({"samples": torch.zeros(1, 31, 2, AUDIO_T)}, "[B,32,2,T]"),
        ({"samples": torch.zeros(3, 32, 2, AUDIO_T)}, "batch size 3 cannot be matched"),
    ],
)
def test_invalid_audio_drive_latent_fails_closed(audio_drive_latent, expected):
    _assert_error(_av_latent(batch=4), audio_drive_latent, expected)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_audio_drive_rejects_nan_and_inf(bad_value):
    drive = _audio()
    drive[0, 0, 0, 0] = bad_value
    _assert_error(_av_latent(), {"samples": drive}, "contains NaN or Inf")


@pytest.mark.parametrize(
    ("noise_mask", "expected"),
    [
        (torch.ones(1), "official two-stream NestedTensor"),
        (NestedTensor((torch.ones(1, 1, VIDEO_T, 1, 1),)), "exactly 2 streams"),
        (
            NestedTensor((torch.ones(1, 1, VIDEO_T, 1), torch.ones(1, 1, 2, AUDIO_T))),
            "video noise mask must have rank 5",
        ),
        (
            NestedTensor((torch.ones(3, 1, VIDEO_T, 1, 1), torch.ones(1, 1, 2, AUDIO_T))),
            "batch size 3 cannot be matched",
        ),
    ],
)
def test_invalid_incoming_noise_mask_fails_closed(noise_mask, expected):
    av_latent = _av_latent(batch=2, with_mask=False)
    av_latent["noise_mask"] = noise_mask
    _assert_error(av_latent, {"samples": _audio()}, expected)


def test_av_template_and_video_nonfinite_values_are_rejected():
    av_latent = _av_latent()
    av_latent["samples"].unbind()[0][0, 0, 0, 0, 0] = float("nan")
    _assert_error(av_latent, {"samples": _audio()}, "AV video stream contains NaN or Inf")

    av_latent = _av_latent()
    av_latent["samples"].unbind()[1][0, 0, 0, 0] = float("inf")
    _assert_error(av_latent, {"samples": _audio()}, "AV template audio stream contains NaN or Inf")
