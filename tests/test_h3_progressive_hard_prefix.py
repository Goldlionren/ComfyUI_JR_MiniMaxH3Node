"""Native small-H3 regressions: progressive resolution + disk-backed MV overlap.

Only neural super-resolution and media codecs are replaced. These tests validate
sampling/masks/continuation contracts, not production checkpoint image quality.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_progressive_guided_sampler import JR_H3_ProgressiveGuidedSampler
from ComfyUI_JR_MiniMaxH3Node.utils import h3_progressive_sampler as progressive
from ComfyUI_JR_MiniMaxH3Node.utils.h3_sequential_audio import (
    CHUNK_PRESETS,
    HARD_CONTEXT_LATENT_STEPS,
    apply_continuation_guide,
    checkpoint_sampled_latent,
    load_manifest,
)
from test_h3_progressive_sampler import inputs, nearest, tiny_patcher
from test_h3_sequential_audio_hard_prefix import _audio_frames, _av_latent, _commit, _fake_video_io, _prepare


def prefix_inputs(**changes):
    video = torch.zeros(1, 24, 17, 8, 12)
    video[:, :, :12] = torch.linspace(-.7, .8, 24 * 12 * 8 * 12).reshape(1, 24, 12, 8, 12)
    audio = torch.linspace(-.5, .5, 32 * 2 * 93).reshape(1, 32, 2, 93)
    vm = torch.ones_like(video)
    vm[:, :, :12] = 0
    value = {"samples": NestedTensor((video, audio)), "noise_mask": NestedTensor((vm, torch.zeros_like(audio))),
             "batch_index": [0], "metadata": {"marker": "preserved"}}
    return inputs(latent_image=value, **changes)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [.6, 1.])
def test_native_prefix_every_step_and_original_high_anchor(monkeypatch, dtype, scale):
    from comfy.ldm.minimax.model import VISUAL_COND_TIMESTEP
    from comfy.utils import unpack_latents

    model = tiny_patcher(monkeypatch)
    args = prefix_inputs(model=model, lowres_scale=scale)
    value = args["latent_image"]
    value["samples"] = value["samples"].to(dtype=dtype)
    value["noise_mask"] = value["noise_mask"].to(dtype=dtype)
    video, audio = [z.clone() for z in value["samples"].unbind()]
    before_masks = [z.clone() for z in value["noise_mask"].unbind()]
    monkeypatch.setattr(progressive, "upscale_h3_video_to_size", nearest)
    native_stage = progressive._run_stage
    native_inpaint = model.model.scale_latent_inpaint
    stages, injected = [], []

    def record_stage(*a, **kw):
        v, aud = a[3].unbind()
        expected = video[:, :, :12].float()
        if v.shape[-2:] != video.shape[-2:]:
            expected = F.interpolate(expected[0].movedim(1, 0), size=v.shape[-2:], mode="area")
            expected = expected.movedim(0, 1).unsqueeze(0).to(dtype).float()
        assert torch.equal(v[:, :, :12].float(), expected)
        assert torch.equal(aud.float(), audio.float())
        vm, am = kw["denoise_mask"].unbind()
        assert torch.count_nonzero(vm[:, :, :12]) == 0
        assert (vm[:, :, 12:] == 1).all() and (am == 0).all()
        stages.append(tuple(v.shape))
        return native_stage(*a, **kw)

    def record_inpaint(**kw):
        result = native_inpaint(**kw)
        shapes = model.model.latent_shapes
        clean_v = unpack_latents(kw["latent_image"], shapes)[0]
        noise_v = unpack_latents(kw["noise"], shapes)[0]
        actual = unpack_latents(result, shapes)[0][:, :, :12]
        expected = (VISUAL_COND_TIMESTEP * clean_v + (1 - VISUAL_COND_TIMESTEP) * noise_v)[:, :, :12]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        injected.append(actual.clone())
        return result

    monkeypatch.setattr(progressive, "_run_stage", record_stage)
    monkeypatch.setattr(model.model, "scale_latent_inpaint", record_inpaint)
    out, status = JR_H3_ProgressiveGuidedSampler().sample(**args)
    repeated, _ = JR_H3_ProgressiveGuidedSampler().sample(**args)
    assert len(stages) == (2 if scale == 1 else 4)
    assert len(injected) == 8  # Four actual denoiser/inpaint steps per run.
    ov, oa = out["samples"].unbind()
    assert torch.equal(ov[:, :, :12], video[:, :, :12])
    assert torch.equal(oa, audio) and ov.dtype == oa.dtype == dtype
    assert torch.count_nonzero(ov[:, :, 12:]) > 0
    assert out["noise_mask"] is value["noise_mask"] and out["metadata"] is value["metadata"]
    assert out["batch_index"] is value["batch_index"]
    for a, b in zip(out["samples"].unbind(), repeated["samples"].unbind()):
        assert torch.equal(a, b) and torch.isfinite(a).all()
    for a, b in zip(value["samples"].unbind(), (video, audio)):
        assert torch.equal(a, b)
    for a, b in zip(value["noise_mask"].unbind(), before_masks):
        assert torch.equal(a, b)
    assert "12 latent tokens / 39 frames" in status


def test_identity_matches_native_generation_suffix(monkeypatch):
    args = prefix_inputs(model=tiny_patcher(monkeypatch), lowres_scale=1.)
    value = args["latent_image"]
    native = progressive._run_stage(args["model"], args["positive"], args["noise"].generate_noise(value),
                                    value["samples"], args["sampler"], args["sigmas"], args["noise"].seed,
                                    lambda *a: None, denoise_mask=value["noise_mask"])
    out, _ = JR_H3_ProgressiveGuidedSampler().sample(**args)
    assert torch.equal(out["samples"].unbind()[0][:, :, 12:], native.unbind()[0][:, :, 12:])


def test_prefix_changes_actually_condition_generated_suffix(monkeypatch):
    args = prefix_inputs(model=tiny_patcher(monkeypatch), lowres_scale=.6)
    monkeypatch.setattr(progressive, "upscale_h3_video_to_size", nearest)
    out, _ = JR_H3_ProgressiveGuidedSampler().sample(**args)
    args["latent_image"]["samples"].unbind()[0][:, :, :12].neg_()
    changed, _ = JR_H3_ProgressiveGuidedSampler().sample(**args)
    assert not torch.equal(out["samples"].unbind()[0][:, :, 12:], changed["samples"].unbind()[0][:, :, 12:])


@pytest.mark.parametrize("damage", ["no_mask", "all_one", "short_prefix", "long_prefix", "spatial", "soft",
                                    "suffix_nonempty", "unlocked_audio", "partial_audio", "no_suffix", "nan"])
def test_prefix_contract_rejected_before_sampling_or_vae(monkeypatch, damage):
    args = prefix_inputs()
    value = args["latent_image"]
    v, a = value["samples"].unbind()
    vm, am = value["noise_mask"].unbind()
    if damage == "no_mask":
        value.pop("noise_mask")
    elif damage == "all_one":
        vm.fill_(1)
    elif damage == "short_prefix":
        vm[:, :, 11] = 1
    elif damage == "long_prefix":
        vm[:, :, 12] = 0
    elif damage == "spatial":
        vm[:, :, 0, 0, 0] = 1
    elif damage == "soft":
        vm[:, :, :12] = .5
    elif damage == "suffix_nonempty":
        v[:, :, 12] = .1
    elif damage == "unlocked_audio":
        am.fill_(1)
        a.zero_()
    elif damage == "partial_audio":
        am[..., 0] = 1
    elif damage == "no_suffix":
        value["samples"] = NestedTensor((v[:, :, :12], a[..., :65]))
        value["noise_mask"] = NestedTensor((vm[:, :, :12], am[..., :65]))
    else:
        vm[:, :, 0] = float("nan")
    args["positive"][0][1]["minimax_keyframes"] = [dict(resolved_frame_index=0, latent=torch.zeros(1, 24, 1, 8, 12))]

    def forbidden(*a, **kw):
        pytest.fail("Invalid prefix reached VAE or sampler")

    from types import SimpleNamespace
    args["vae"] = SimpleNamespace(decode=forbidden, encode=forbidden)
    monkeypatch.setattr(progressive, "_run_stage", forbidden)
    with pytest.raises(ValueError):
        JR_H3_ProgressiveGuidedSampler().sample(**args)


@pytest.mark.parametrize("with_tst", [False, True])
@pytest.mark.parametrize("disabled_noise", [False, True])
def test_three_disk_backed_chunks_preserve_audio_timeline_and_overlap(tmp_path, monkeypatch, with_tst, disabled_noise):
    from ComfyUI_JR_MiniMaxH3Node.utils.h3_temporal_transport import apply_temporal_transport

    model = tiny_patcher(monkeypatch)
    if with_tst:
        model = apply_temporal_transport(model, strength=.2)
    monkeypatch.setattr(progressive, "upscale_h3_video_to_size", nearest)
    captured = []
    _fake_video_io(monkeypatch, tmp_path, captured)
    preset = CHUNK_PRESETS[2]  # User's 8 s / 192 frames / 57 tokens / 320 ticks.
    source = _audio_frames(480)
    previous_video = None
    for index in range(3):
        template = _av_latent(preset=preset, height=4, width=4)
        av, context, *_ = _prepare(tmp_path, source, av_latent=template, preset=preset)
        assert context.chunk_index == index and context.frame_start == index * 153
        assert context.generated_frames == 192
        assert context.trim_head_frames == (39 if index else 0)
        positive = [[torch.zeros(1, 2, 128), {
            "minimax_refs": [dict(kind="image", latent_h=2, latent_w=2, latent=torch.ones(1, 24, 1, 2, 2) * .2)]
        }]]
        cond, guided, _ = apply_continuation_guide(positive=positive, latent=av, context=context, vae=object())
        if previous_video is not None:
            assert torch.equal(guided["samples"].unbind()[0][:, :, :12], previous_video[:, :, -12:])
        noise = Noise_EmptyNoise() if disabled_noise else Noise_RandomNoise(context.seed)
        out, status = JR_H3_ProgressiveGuidedSampler().sample(**inputs(
            model=model, positive=cond, latent_image=guided, noise=noise, lowres_scale=.5))
        video, audio = out["samples"].unbind()
        assert video.shape == (1, 24, 57, 4, 4) and audio.shape == (1, 32, 2, 320)
        assert torch.equal(audio, av["samples"].unbind()[1])
        if previous_video is not None:
            assert torch.equal(video[:, :, :HARD_CONTEXT_LATENT_STEPS], previous_video[:, :, -12:])
            assert "hard prefix LOCKED" in status
        checkpoint_sampled_latent(out, context)
        pixels = torch.arange(192).view(192, 1, 1, 1).expand(192, 2, 3, 3).float()
        _commit(context, pixels)
        assert captured[-1][0, 0, 0, 0] == (39 if index else 0)
        previous_video = video.clone()
    manifest = load_manifest(Path(context.job_dir))
    assert len(manifest["segments"]) == 3 and manifest["status"] == "complete"
