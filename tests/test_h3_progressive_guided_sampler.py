"""Guided geometry, native packed-H3 execution and audio-lock regression tests."""

import pytest
import torch
import torch.nn.functional as F
from comfy.nested_tensor import NestedTensor
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_progressive_guided_sampler import JR_H3_ProgressiveGuidedSampler
from ComfyUI_JR_MiniMaxH3Node.utils import h3_progressive_sampler as progressive
from ComfyUI_JR_MiniMaxH3Node.utils.h3_audio_driven_latent_builder import build_h3_audio_driven_latent
from ComfyUI_JR_MiniMaxH3Node.utils.h3_progressive_guidance import prepare_guidance
from test_h3_progressive_sampler import inputs, nearest, tiny_patcher


class GeometryVAE:
    """Only replaces VAE compute; native model, guides, masks and sampling stay real."""
    latent_channels = 24

    def __init__(self):
        self.calls = []

    def spacial_compression_decode(self):
        return 16

    def decode(self, z):
        self.calls.append(tuple(z.shape))
        return F.interpolate(z[0, :3].movedim(0, 1), scale_factor=16, mode="nearest").movedim(1, -1).unsqueeze(0)

    def encode(self, pixels):
        z = F.interpolate(pixels.movedim(-1, 1), scale_factor=1 / 16, mode="area")
        return z.movedim(0, 1).unsqueeze(0).repeat(1, 8, 1, 1, 1)


def guided_inputs(mode="all", **kwargs):
    args = inputs(**kwargs)
    metadata = args["positive"][0][1]
    metadata["marker"] = object()
    if mode in ("refs", "all"):
        # Independent grid intentionally differs from BOTH target stage grids.
        metadata["minimax_refs"] = [dict(kind="image", latent_h=6, latent_w=8,
                                         latent=torch.full((1, 24, 1, 6, 8), .2))]
    if mode in ("first", "firstlast", "all"):
        metadata["minimax_keyframes"] = [dict(resolved_frame_index=0, latent=torch.full((1, 24, 1, 8, 12), .3))]
        if mode != "first":
            metadata["minimax_keyframes"].append(dict(resolved_frame_index=4, latent=torch.full((1, 24, 1, 8, 12), .7)))
    if mode in ("audio", "all"):
        args["latent_image"], _ = build_h3_audio_driven_latent(
            args["latent_image"], {"samples": torch.linspace(-1, 1, 512).reshape(1, 32, 2, 8)})
    args["vae"] = GeometryVAE()
    return args


@pytest.mark.parametrize("mode", ["refs", "first", "firstlast", "audio", "all"])
@pytest.mark.parametrize("scale", [.5, 1.])
def test_real_native_guides_locks_reproducibility_and_stage_geometry(monkeypatch, mode, scale):
    patcher = tiny_patcher(monkeypatch)
    monkeypatch.setattr(progressive, "upscale_h3_video_to_size", nearest)
    args = guided_inputs(mode, model=patcher, lowres_scale=scale)
    original = args["latent_image"]
    original_audio = original["samples"].unbind()[1].clone()
    stage = progressive._run_stage
    seen = []

    def record(*stage_args, **kwargs):
        positive, latent = stage_args[1], stage_args[3]
        seen.append((positive, latent, kwargs["denoise_mask"]))
        return stage(*stage_args, **kwargs)

    monkeypatch.setattr(progressive, "_run_stage", record)
    output, status = JR_H3_ProgressiveGuidedSampler().sample(**args)
    repeated, _ = JR_H3_ProgressiveGuidedSampler().sample(**args)
    assert len(seen) == (4 if scale < 1 else 2)
    for result, repeat in zip(output["samples"].unbind(), repeated["samples"].unbind()):
        assert bool(torch.isfinite(result).all())
        torch.testing.assert_close(result, repeat, rtol=0, atol=0)
    assert output["metadata"] is original["metadata"]
    for cond, samples, mask in seen:
        if mode in ("audio", "all"):
            assert torch.equal(samples.unbind()[1], original_audio)
            assert torch.equal(mask.unbind()[0], torch.ones_like(samples.unbind()[0]))
            assert torch.count_nonzero(mask.unbind()[1]) == 0
            assert torch.equal(output["samples"].unbind()[1], original_audio)
            assert output["noise_mask"] is original["noise_mask"]
        else:
            assert mask is None
        for kf in cond[0][1].get("minimax_keyframes", []):
            assert kf["latent"].shape[-2:] == samples.unbind()[0].shape[-2:]
        if mode in ("refs", "all"):
            assert cond[0][1]["minimax_refs"] is args["positive"][0][1]["minimax_refs"]
    if scale == 1:
        assert not args["vae"].calls
    else:
        assert seen[1][0] is args["positive"]  # Original exact high conditioning.
    for kf in args["positive"][0][1].get("minimax_keyframes", []):
        assert kf["latent"].shape[-2:] == (8, 12)
    assert torch.equal(original["samples"].unbind()[1], original_audio)
    assert "Guides:" in status


def prepare(args):
    video, audio = args["latent_image"]["samples"].unbind()
    plan = progressive.plan_progressive(args["sigmas"], 2, args["lowres_scale"], 8, 12)
    return prepare_guidance(args["positive"], args["latent_image"], video, audio, plan, args.get("vae"))


def test_keyframes_require_video_vae_only_for_transition():
    args = guided_inputs("firstlast")
    args.pop("vae")
    with pytest.raises(ValueError, match="same H3 VIDEO VAE"):
        prepare(args)
    args["lowres_scale"] = 1
    assert prepare(args).low_positive is args["positive"]
    args = guided_inputs("refs")
    args.pop("vae")
    assert prepare(args).low_positive is args["positive"]


@pytest.mark.parametrize("bad", ["partial_audio", "soft_audio", "video_lock", "flat_mask", "shape", "nan", "nonempty_unlocked"])
def test_masks_fail_closed_before_vae_work(bad):
    args = guided_inputs()
    value = args["latent_image"]
    vm, am = value["noise_mask"].unbind()
    if bad == "partial_audio":
        am[..., 0] = 1
    elif bad == "soft_audio":
        am.fill_(.5)
    elif bad == "video_lock":
        vm[..., 0, 0] = 0
    elif bad == "flat_mask":
        value["noise_mask"] = vm
    elif bad == "shape":
        value["noise_mask"] = NestedTensor((vm[..., :1], am))
    elif bad == "nan":
        am.fill_(float("nan"))
    else:
        value.pop("noise_mask")
    with pytest.raises(ValueError):
        prepare(args)
    assert not args["vae"].calls


@pytest.mark.parametrize("bad", ["kf_shape", "kf_time", "kf_nan", "ref_metadata", "ref_kind", "ref_audio"])
def test_bad_guide_contracts_fail_before_vae_work(bad):
    args = guided_inputs()
    md = args["positive"][0][1]
    if bad == "kf_shape":
        md["minimax_keyframes"][0]["latent"] = torch.zeros(1, 24, 1, 4, 6)
    elif bad == "kf_time":
        md["minimax_keyframes"][0]["resolved_frame_index"] = 5
    elif bad == "kf_nan":
        md["minimax_keyframes"][0]["latent"].fill_(float("nan"))
    elif bad == "ref_metadata":
        md["minimax_refs"][0]["latent_h"] = 8
    elif bad == "ref_kind":
        md["minimax_refs"][0]["kind"] = "unknown"
    else:
        md["minimax_refs"].append(dict(kind="audio", ref_audio_t=3, audio_latent=torch.zeros(1, 32, 2, 2)))
    with pytest.raises(ValueError):
        prepare(args)
    assert not args["vae"].calls


def test_audio_reference_and_keyframe_audio_unchanged():
    args = guided_inputs("refs")
    audio = torch.zeros(1, 32, 2, 2)
    md = args["positive"][0][1]
    md["minimax_refs"].append(dict(kind="audio", ref_audio_t=2, audio_latent=audio))
    md["minimax_keyframes"] = [dict(resolved_frame_index=0, audio_latent=audio)]
    assert prepare(args).low_positive is args["positive"]
    assert not args["vae"].calls


def test_bad_vae_temporal_output_rejected():
    args = guided_inputs("first")
    args["vae"].encode = lambda pixels: torch.zeros(1, 24, 2, 4, 6)
    with pytest.raises(ValueError, match="temporal length"):
        prepare(args)


def test_guided_schema_preserves_base_and_registration():
    import ComfyUI_JR_MiniMaxH3Node as package
    from ComfyUI_JR_MiniMaxH3Node.nodes.h3_progressive_sampler import JR_H3_ProgressiveSampler

    assert package.NODE_CLASS_MAPPINGS["JR_H3_ProgressiveGuidedSampler"] is JR_H3_ProgressiveGuidedSampler
    assert JR_H3_ProgressiveGuidedSampler.INPUT_TYPES()["optional"]["vae"][0] == "VAE"
    assert "optional" not in JR_H3_ProgressiveSampler.INPUT_TYPES()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_locked_audio_dtype_and_exact_values_survive_boundary(monkeypatch, dtype):
    monkeypatch.setattr(progressive, "upscale_h3_video_to_size", nearest)
    args = guided_inputs("audio", model=tiny_patcher(monkeypatch))
    value = args["latent_image"]
    value["samples"] = value["samples"].to(dtype=dtype)
    value["noise_mask"] = value["noise_mask"].to(dtype=dtype)
    result, _ = JR_H3_ProgressiveGuidedSampler().sample(**args)
    assert all(t.dtype == dtype for t in result["samples"].unbind())
    assert torch.equal(result["samples"].unbind()[1], value["samples"].unbind()[1])


def test_native_identity_baseline_and_audio_affects_video(monkeypatch):
    args = guided_inputs("all", model=tiny_patcher(monkeypatch), lowres_scale=1.)
    value = args["latent_image"]
    native = progressive._run_stage(args["model"], args["positive"], args["noise"].generate_noise(value),
                                    value["samples"], args["sampler"], args["sigmas"], args["noise"].seed,
                                    lambda *a: None, denoise_mask=value["noise_mask"])
    result, _ = JR_H3_ProgressiveGuidedSampler().sample(**args)
    assert torch.equal(result["samples"].unbind()[0], native.unbind()[0])
    value["samples"] = NestedTensor((value["samples"].unbind()[0], -value["samples"].unbind()[1]))
    different, _ = JR_H3_ProgressiveGuidedSampler().sample(**args)
    assert not torch.equal(result["samples"].unbind()[0], different["samples"].unbind()[0])


def test_guided_workflows_match_generators_and_link_contracts(monkeypatch):
    import json
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project / "tools"))
    from build_progressive_guided_examples import build_guided

    for mode in ("Ref2VA", "FirstLast", "AudioDrive"):
        workflow = json.loads((project / f"examples/JR_H3_Progressive_{mode}_Experimental.json").read_text(encoding="utf-8"))
        assert workflow == build_guided(mode)
        nodes = {n["id"]: n for n in workflow["nodes"]}
        assert len({v[0] for v in workflow["links"]}) == len(workflow["links"])
        for link_id, source, slot, target, target_slot, kind in workflow["links"]:
            assert link_id in nodes[source]["outputs"][slot]["links"]
            assert nodes[target]["inputs"][target_slot]["link"] == link_id
            accepted = nodes[target]["inputs"][target_slot]["type"].split(",")
            assert kind in accepted or "*" in accepted
        assert nodes[133]["widgets_values"] == [5.0]
        assert nodes[140]["widgets_values"][0] is False
        assert all("widgets_values_named" not in n for n in nodes.values())
        models = [v[1:3] for v in workflow["links"] if (v[3], v[4]) in ((124, 0), (125, 0))]
        assert models == [[140, 0], [140, 0]]
