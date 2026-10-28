"""Native numerical regressions for adapter conversion and audio-locked regeneration."""
from types import SimpleNamespace

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from ComfyUI_JR_MiniMaxH3Node.tools.convert_h3_regenerate_lora import convert_linear_pairs
from ComfyUI_JR_MiniMaxH3Node.utils import h3_regenerate as regen
from ComfyUI_JR_MiniMaxH3Node.utils.h3_regenerate_lora import CurveContext, CurveProjection, patch_regenerate_bundle
from test_h3_progressive_sampler import tiny_patcher


def test_qkv_and_swiglu_conversion_preserves_linear_deltas():
    torch.manual_seed(25)
    raw, expected = {}, []
    for part, rank in zip(("q", "k", "v"), (2, 3, 4)):
        a, b = torch.randn(rank, 7), torch.randn(5, rank)
        root = f"transformer.transformer_blocks.0.attn.to_{part}"
        raw[root + ".lora_A.weight"], raw[root + ".lora_B.weight"] = a, b
        expected.append(b @ a)
    a, b = torch.randn(3, 7), torch.randn(10, 3)
    root = "transformer.transformer_blocks.0.ff.net.0.proj"
    raw[root + ".lora_A.weight"], raw[root + ".lora_B.weight"] = a, b
    converted = convert_linear_pairs(raw)
    for root, reference in (
        ("diffusion_model.blocks.0.attn.qkv_proj", torch.cat(expected)),
        ("diffusion_model.blocks.0.mlp.fc1", torch.cat((b @ a).chunk(2)[::-1])),
    ):
        torch.testing.assert_close(converted[root + ".lora_up.weight"] @ converted[root + ".lora_down.weight"],
                                   reference, rtol=1e-6, atol=1e-6)


def test_curve_interpolation_and_exception_cleanup():
    table = torch.linspace(-1, 1, 1025 * 8).reshape(1025, 8)
    context = CurveContext(table)
    a, b = torch.randn(4, 8), torch.randn(12, 4)
    projection = CurveProjection(lambda x: x, b @ a, context, .8)
    def run():
        t = torch.tensor([0., .15, .3, 1.])
        context.capture(lambda t: t, t)
        coords = context.coords.clone()
        expected = (coords @ a.T) @ b.T * .8
        torch.testing.assert_close(projection(torch.zeros(4, 12)), expected, atol=1e-5, rtol=1e-5)
        raise RuntimeError("intentional")
    with pytest.raises(RuntimeError, match="intentional"):
        context.forward(run)
    assert context.coords is None


@pytest.mark.parametrize("denoise", [.15, .3, .35])
@pytest.mark.parametrize("steps", [1, 2, 3, 7])
def test_exact_schedule(denoise, steps):
    sigmas = regen.regenerate_sigmas(denoise, steps)
    assert len(sigmas) == steps + 1 and sigmas[-1] == 0
    assert sigmas[0].item() == pytest.approx(denoise)
    assert (sigmas[:-1] > sigmas[1:]).all()
    raw = sigmas / (6 - 5 * sigmas)
    torch.testing.assert_close(raw, torch.linspace(raw[0], 0, steps + 1))


@pytest.mark.parametrize("denoise,steps", [(.14, 3), (.36, 3), (float("nan"), 3), (.3, 0), (.3, 1.5)])
def test_invalid_schedule(denoise, steps):
    with pytest.raises(ValueError):
        regen.regenerate_sigmas(denoise, steps)


def small_bundle(model):
    dit = model.get_model_object("diffusion_model")
    result = {
        "jr_h3_curve.table": torch.randn(1025, 8) * .01,
        "jr_h3_curve.blocks.0.adaln_proj.linear.weight":
            torch.randn(dit.blocks[0].adaln_proj.linear.weight.shape[0], 8) * .01,
    }
    for target in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
        rows, columns = dit.blocks[0].get_submodule(target).weight.shape
        root = "diffusion_model.blocks.0." + target
        result[root + ".lora_down.weight"] = torch.randn(2, columns) * .01
        result[root + ".lora_up.weight"] = torch.randn(rows, 2) * .01
    return result


def inputs(model, dtype=torch.float32):
    return dict(
        model=model,
        clip=SimpleNamespace(tokenize=lambda prompt: prompt,
                             encode_from_tokens_scheduled=lambda tokens: [[torch.zeros(1, 2, 128), {}]]),
        source_mode="H3 latent", denoise=.3, steps=3, seed=92, prompt="test",
        attention="inherit", resize_mode="keep", scale=2., target_megapixels=3.6864, upscaler_model="auto",
        latent={"samples": NestedTensor((torch.randn(1, 24, 2, 4, 4).to(dtype),
                                        torch.randn(1, 32, 2, 8).to(dtype))),
                "metadata": {"retained": True}},
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_real_h3_audio_locked_each_step_and_patcher_isolation(monkeypatch, dtype):
    base = tiny_patcher(monkeypatch)
    bundle = small_bundle(base)
    model = patch_regenerate_bundle(base, bundle)
    assert not base.patches
    assert "jr_h3_regenerate_lora" not in base.model_options
    args = inputs(model, dtype)
    source_video, source_audio = args["latent"]["samples"].unbind()
    before_video, before_audio = source_video.clone(), source_audio.clone()
    captured, ffn_calls = [], []
    dit = base.get_model_object("diffusion_model")
    original_forward = dit.blocks[0].mlp.forward
    # Same ModelPatcher method route used by installed KJ FFN/Unified.
    def chunked_ffn(x):
        ffn_calls.append(x.shape[0])
        return torch.cat([original_forward(z) for z in x.split(17)], dim=0)
    model.add_object_patch("diffusion_model.blocks.0.mlp.forward", chunked_ffn)
    handle = dit.audio_patch_proj.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().cpu().clone()))
    result, status, bypass = regen.regenerate(**args)
    repeated, _, _ = regen.regenerate(**args)
    handle.remove()
    assert len(captured) == 6 and ffn_calls
    # _forward sees normalized clean audio, not a denoised/re-noised audio stream.
    for actual in captured:
        torch.testing.assert_close(actual, captured[0], rtol=1e-5, atol=1e-5)
    assert result["samples"].unbind()[1] is source_audio
    assert torch.equal(source_audio, before_audio) and torch.equal(source_video, before_video)
    assert torch.equal(result["samples"].unbind()[0], repeated["samples"].unbind()[0])
    assert not torch.equal(result["samples"].unbind()[0], before_video)
    assert result["metadata"] is args["latent"]["metadata"]
    assert "never passed to sampler" in status and bypass is None
    context = model.object_patches["diffusion_model.time_embedder.forward"].context
    assert context.coords is None
    model.unpatch_model()
    assert dit.blocks[0].mlp.forward == original_forward


def test_sage_selection_is_explicit_and_preserves_inherit(monkeypatch):
    marker = object()
    model = SimpleNamespace(model_options={"transformer_options": {"optimized_attention_override": marker}})
    assert regen.select_attention(model, "inherit") is model
    with pytest.raises(ValueError, match="inherit"):
        regen.select_attention(model, "sageattn_qk_int8_pv_fp8_cuda++")
    calls = []
    monkeypatch.setattr(regen.acceleration, "apply_sage", lambda m, **kw: calls.append(kw) or m)
    model.model_options["transformer_options"] = {}
    regen.select_attention(model, "sageattn_qk_int8_pv_fp8_cuda++")
    assert calls == [{"sage_attention": "sageattn_qk_int8_pv_fp8_cuda++", "allow_compile": False}]


def test_original_release_schedules_are_not_generic_truncation():
    torch.testing.assert_close(regen.regenerate_sigmas(.3, 3, "release_original"),
                               torch.tensor([.909375, .8571428571428571, .6666666666666666, 0.]))
    torch.testing.assert_close(regen.regenerate_sigmas(.3, 2, "release_original"),
                               torch.tensor([.7, .49411764705882355, 0.]))
    with pytest.raises(ValueError, match="exactly 2 or 3"):
        regen.regenerate_sigmas(.3, 4, "release_original")


@pytest.mark.parametrize("missing", ["linear", "curve", "table"])
def test_incomplete_bundles_fail_before_patching(monkeypatch, missing):
    model = tiny_patcher(monkeypatch)
    raw = small_bundle(model)
    if missing == "linear":
        raw.pop("diffusion_model.blocks.0.mlp.fc1.lora_down.weight")
        raw.pop("diffusion_model.blocks.0.mlp.fc1.lora_up.weight")
    elif missing == "curve":
        raw.pop("jr_h3_curve.blocks.0.adaln_proj.linear.weight")
    else:
        raw.pop("jr_h3_curve.table")
    before = dict(model.object_patches)
    with pytest.raises(ValueError):
        patch_regenerate_bundle(model, raw)
    assert model.object_patches == before and not model.patches
