"""Tail audio allocation, strict AV assembly and real native single-frame sampling."""

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from comfy.samplers import ksampler
from comfy_extras.nodes_custom_sampler import BasicGuider, Noise_RandomNoise, SamplerCustomAdvanced
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_av_latent_builder import JR_MiniMaxH3AVLatentBuilder
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_empty_audio_latent_for_tail import JR_H3_EmptyAudioLatentForTail
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_split_av_latent import JR_H3_SplitAVLatent
from ComfyUI_JR_MiniMaxH3Node.utils.h3_av_latent_builder import H3AVLatentBuilderError, build_h3_av_latent
from ComfyUI_JR_MiniMaxH3Node.utils.h3_keyframe_latent import validate_keyframe_latent
from ComfyUI_JR_MiniMaxH3Node.utils.h3_neural_latent_upscaler import upscale_h3_video_to_size
from ComfyUI_JR_MiniMaxH3Node.utils.h3_tail_frame_latent import extract_h3_tail_frame_latent
from ComfyUI_JR_MiniMaxH3Node.utils.h3_temporal_transport import apply_temporal_transport
from test_h3_progressive_sampler import nearest, tiny_patcher
from test_h3_tail_frame_latent import VAE, video


def tail(dtype=torch.float32, device="cpu"):
    return {"samples": torch.linspace(-.5, .5, 24 * 4 * 6, dtype=dtype, device=device).reshape(1, 24, 1, 4, 6)}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_empty_audio_matches_tail_and_builder_keeps_streams(dtype):
    source = tail(dtype)
    saved = source["samples"].clone()
    audio, status = JR_H3_EmptyAudioLatentForTail().create(source)
    assert set(audio) == {"samples"}
    assert audio["samples"].shape == (1, 32, 2, 2)
    assert audio["samples"].dtype == dtype and audio["samples"].device == saved.device
    assert not audio["samples"].count_nonzero() and "2 audio ticks" in status
    av, report = JR_MiniMaxH3AVLatentBuilder().build(source, audio)
    assert isinstance(av["samples"], NestedTensor)
    v, a = av["samples"].unbind()
    assert v is source["samples"] and a is audio["samples"]
    assert "experimental single-frame" in report and "1 frames @ 24 fps" in report
    assert torch.equal(source["samples"], saved)
    again, _ = JR_H3_EmptyAudioLatentForTail().create(source)
    audio["samples"].fill_(1)
    assert not again["samples"].count_nonzero()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_audio_allocates_directly_on_source_device():
    source = tail(torch.float16, "cuda")
    audio, _ = JR_H3_EmptyAudioLatentForTail().create(source)
    av, _ = build_h3_av_latent(source, audio)
    assert all(z.device == source["samples"].device for z in av["samples"].unbind())


@pytest.mark.parametrize("ticks", [1, 3, 8, 207])
def test_single_frame_does_not_inherit_long_video_tick_tolerance(ticks):
    with pytest.raises(H3AVLatentBuilderError, match="expected audio latent T: 2 ± 0"):
        build_h3_av_latent(tail(), {"samples": torch.zeros(1, 32, 2, ticks)})


@pytest.mark.parametrize("tokens,ticks", [(2, 8), (7, 37), (37, 207), (72, 405), (107, 603)])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_existing_video_timelines_unchanged(tokens, ticks, offset):
    v = {"samples": torch.zeros(1, 24, tokens, 2, 2)}
    a = {"samples": torch.zeros(1, 32, 2, ticks + offset)}
    av, _ = build_h3_av_latent(v, a)
    assert av["samples"].unbind()[0] is v["samples"]


@pytest.mark.parametrize("value", [None, {}, {"samples": torch.zeros(1, 24, 3, 4, 6)},
    {"samples": torch.zeros(2, 24, 1, 4, 6)}, {"samples": torch.zeros(1, 32, 2, 2)},
    {"samples": torch.zeros(1, 24, 1, 4, 6, dtype=torch.int64)},
    {"samples": torch.full((1, 24, 1, 4, 6), float("nan"))},
    {"samples": torch.zeros(1, 24, 1, 4, 6), "noise_mask": torch.ones(1)},
    {"samples": NestedTensor((torch.zeros(1, 24, 1, 4, 6), torch.zeros(1, 32, 2, 2)))}])
def test_only_clean_h3_video_is_accepted(value):
    with pytest.raises(ValueError):
        JR_H3_EmptyAudioLatentForTail().create(value)


@pytest.mark.parametrize("tokens,frames,ticks", [(1, 1, 2), (2, 5, 8), (7, 22, 37),
    (12, 39, 65), (37, 124, 207), (52, 175, 292), (72, 243, 405), (107, 362, 603)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_audio_automatically_matches_native_video_grid(tokens, frames, ticks, dtype):
    source = video(tokens, dtype)
    before = source["samples"].clone()
    audio, status = JR_H3_EmptyAudioLatentForTail().create(source)
    assert audio["samples"].shape == (1, 32, 2, ticks)
    assert audio["samples"].dtype == dtype and audio["samples"].device == before.device
    assert not audio["samples"].count_nonzero()
    assert f"{frames} frames @ 24 fps -> {ticks} audio ticks" in status
    av, report = build_h3_av_latent(source, audio)
    assert av["samples"].unbind()[0] is source["samples"]
    assert "audio delta=+0" in report and torch.equal(source["samples"], before)


@pytest.mark.parametrize("with_tst", [False, True])
def test_tail_context_native_second_pass_and_last_image(monkeypatch, with_tst):
    model = tiny_patcher(monkeypatch)
    if with_tst:
        model = apply_temporal_transport(model, strength=.2)
    _, image, _, context = extract_h3_tail_frame_latent(video(52), VAE())
    assert torch.equal(VAE().decode(context["samples"])[-1:], image)
    upscaled = {"samples": upscale_h3_video_to_size(
        context["samples"], 4, 4,
        neural_runner=lambda z, plan: nearest(z, plan.output_h, plan.output_w),
    )}
    audio, _ = JR_H3_EmptyAudioLatentForTail().create(upscaled)
    assert audio["samples"].shape[-1] == 37
    av, _ = build_h3_av_latent(upscaled, audio)
    before = [z.clone() for z in av["samples"].unbind()]
    outputs = []
    for _ in range(2):
        guider = BasicGuider.execute(model, [[torch.zeros(1, 2, 128), {}]])[0]
        with torch.inference_mode():
            out = SamplerCustomAdvanced.execute(
                Noise_RandomNoise(123), guider, ksampler("euler"), torch.tensor([.3, .15, 0.]), av,
            )[0]
        outputs.append(out)
    for a, b, source, saved in zip(outputs[0]["samples"].unbind(), outputs[1]["samples"].unbind(),
                                    av["samples"].unbind(), before):
        assert torch.isfinite(a).all() and torch.equal(a, b) and torch.equal(source, saved)
    sampled_video, sampled_audio = JR_H3_SplitAVLatent().split(outputs[0])
    assert sampled_video["samples"].shape == (1, 24, 7, 4, 4)
    assert sampled_audio["samples"].shape[-1] == 37
    pixels = VAE().decode(sampled_video["samples"])
    assert pixels.shape == (22, 64, 64, 3)
    assert pixels[-1:].shape == (1, 64, 64, 3)
    with pytest.raises(ValueError, match="single-frame"):
        validate_keyframe_latent(sampled_video)


@pytest.mark.parametrize("with_tst", [False, True])
def test_upscale_assemble_native_second_pass_split_decode_and_reuse(monkeypatch, with_tst):
    model = tiny_patcher(monkeypatch)
    if with_tst:
        model = apply_temporal_transport(model, strength=.2)
    original = tail()
    upscaled = {"samples": upscale_h3_video_to_size(
        original["samples"], 8, 12,
        neural_runner=lambda z, plan: nearest(z, plan.output_h, plan.output_w),
    )}
    audio, _ = JR_H3_EmptyAudioLatentForTail().create(upscaled)
    av, _ = build_h3_av_latent(upscaled, audio)
    before = [z.clone() for z in av["samples"].unbind()]
    positive = [[torch.zeros(1, 2, 128), {}]]
    outputs = []
    for _ in range(2):
        guider = BasicGuider.execute(model, positive)[0]
        with torch.inference_mode():
            out = SamplerCustomAdvanced.execute(
                Noise_RandomNoise(123), guider, ksampler("euler"), torch.tensor([.3, .15, 0.]), av,
            )[0]
        outputs.append(out)
    for a, b, source, saved in zip(outputs[0]["samples"].unbind(), outputs[1]["samples"].unbind(),
                                    av["samples"].unbind(), before):
        assert torch.isfinite(a).all() and torch.equal(a, b)
        assert torch.equal(source, saved)
    sampled_video, sampled_audio = JR_H3_SplitAVLatent().split(outputs[0])
    assert sampled_audio["samples"].shape == (1, 32, 2, 2)
    assert validate_keyframe_latent(sampled_video).shape == (1, 24, 1, 8, 12)
    # The image VAE sees one frame, not a padded 5-frame clip. No re-encode is needed.
    pixels = VAE().decode(sampled_video["samples"])
    assert pixels.shape == (1, 128, 192, 3)


def test_schema_has_no_manual_duration_or_seed():
    schema = JR_H3_EmptyAudioLatentForTail.INPUT_TYPES()
    assert set(schema["required"]) == {"video_latent"}
    assert JR_H3_EmptyAudioLatentForTail.RETURN_TYPES == ("LATENT", "STRING")
