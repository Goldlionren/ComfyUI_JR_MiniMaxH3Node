"""TST algebra, backend ownership, full-schedule and native AV contracts."""

import pytest
import torch
from comfy.ldm.minimax.model import PackedLayout
from comfy.ldm.modules.attention import AttentionTensorContainer, attention_pytorch
from ComfyUI_JR_MiniMaxH3Node.utils import h3_temporal_transport as tst
from test_h3_progressive_guided_sampler import guided_inputs
from test_h3_progressive_sampler import inputs, nearest, progressive, tiny_patcher


def test_entropy_extremes_and_nan():
    torch.testing.assert_close(tst.spectral_tension(torch.eye(4)[None]), torch.tensor([-1.]))
    torch.testing.assert_close(tst.spectral_tension(torch.full((1, 4, 4), .25)), torch.tensor([1.]), atol=1e-6, rtol=0)
    assert tst.spectral_tension(torch.ones(1, 1, 1)).item() == 0
    with pytest.raises(RuntimeError, match="Non-finite"):
        tst.spectral_tension(torch.full((1, 2, 2), float("nan")))


@pytest.mark.parametrize("strength", [-1, 1.01, float("nan"), float("inf"), True])
def test_strength_guards(strength):
    with pytest.raises(ValueError, match="tst_strength"):
        tst.validate_strength(strength)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_q_only_non_mutation_and_head_chunk_invariance(dtype):
    torch.manual_seed(41)
    q, k = [torch.randn(1, 4, 15, 8).to(dtype) for _ in range(2)]
    original_q, original_k = q.clone(), k.clone()
    args = dict(start=3, stop=15, frames=3, strength=.2, weight=.8)
    output = tst.transform_video_q(q, k, **args)
    chunked = torch.cat([tst.transform_video_q(q[:, h:h+1], k[:, h:h+1], **args) for h in range(4)], dim=1)
    torch.testing.assert_close(output, chunked, atol=0, rtol=0)
    assert torch.equal(q, original_q) and torch.equal(k, original_k)
    assert torch.equal(output[:, :, :3], q[:, :, :3])
    assert not torch.equal(output[:, :, 3:], q[:, :, 3:])
    assert output.dtype == dtype and output.device == q.device
    assert tst.transform_video_q(q, k, **(args | {"strength": 0})) is q
    assert tst.transform_video_q(q, k, **(args | {"weight": 0})) is q
    assert tst.transform_video_q(q, k, **(args | {"frames": 1})) is q


def test_schedule_is_global_and_layer_weights_use_block_id():
    schedule = (1., .8, .5, .2, 0.)
    assert [tst.schedule_progress(schedule, s) for s in schedule[:-1]] == [0, 1/3, 2/3, 1]
    assert tst.schedule_progress(schedule[2:], .5) == 0  # the bug avoided by passing full schedule
    assert tst.layer_step_weight(3, 4, 0) == 1
    assert tst.layer_step_weight(3, 4, 1) == 0
    assert tst.layer_step_weight(0, 4, 0) < tst.layer_step_weight(3, 4, 0)
    with pytest.raises(RuntimeError, match="block_index"):
        tst.layer_step_weight(None, 4, 0)


def dispatch_context(dispatcher):
    layout = PackedLayout(2, 2, 4, 4, 2)
    return {"optimized_attention_override": dispatcher, "minimax_h3_layout": layout,
            "block_index": 0, tst.CONTEXT_KEY: {"signature": layout.signature, "progress": 0., "seen": set()}}


@pytest.mark.parametrize("containers", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_composite_delegate_once_with_original_fallback_and_containers(containers, fallback):
    calls = []
    def sage(func, q, k, v, heads, **kwargs):
        calls.append("sage")
        return func(q, k, v, heads, **kwargs)
    def sol(func, q, k, v, heads, **kwargs):
        calls.append("sol")
        return sage(func, q, k, v, heads, **kwargs) if fallback else func(q, k, v, heads, **kwargs)
    dispatcher = tst.TemporalTransport(sol, .1, 1)
    options = dispatch_context(dispatcher)
    tensors = [torch.randn(1, 2, options["minimax_h3_layout"].seq_len, 8) for _ in range(3)]
    saved = [t.clone() for t in tensors]
    args = [AttentionTensorContainer(t) for t in tensors] if containers else tensors
    result = attention_pytorch(*args, 2, skip_reshape=True, transformer_options=options)
    assert result.shape == (1, tensors[0].shape[2], 16)
    assert calls == (["sol", "sage"] if fallback else ["sol"])
    assert options[tst.CONTEXT_KEY]["seen"] == {0}
    assert all(torch.equal(a, b) for a, b in zip(tensors, saved))
    if containers:
        assert all(c.tensor is None for c in args)


def test_mask_missing_context_and_layout_fail_closed():
    dispatcher = tst.TemporalTransport(None, .1, 1)
    options = dispatch_context(dispatcher)
    q = torch.randn(1, 2, options["minimax_h3_layout"].seq_len, 8)
    with pytest.raises(RuntimeError, match="unmasked"):
        dispatcher(None, q, q, q, 2, mask=torch.ones(1), skip_reshape=True, transformer_options=options)
    with pytest.raises(RuntimeError, match="context missing"):
        dispatcher(None, q, q, q, 2)
    options[tst.CONTEXT_KEY]["signature"] = (1,)
    with pytest.raises(RuntimeError, match="PackedLayout"):
        dispatcher(None, q, q, q, 2, skip_reshape=True, transformer_options=options)


def test_forward_cancellation_leaves_no_context():
    dispatcher = tst.TemporalTransport(None, .1, 1)
    options = {"optimized_attention_override": dispatcher, "sample_sigmas": torch.tensor([1., .5, 0.])}
    seen = []
    def execute(x, timestep, context, transformer_options, **kwargs):
        seen.append(transformer_options[tst.CONTEXT_KEY])
        raise RuntimeError("cancel")
    with pytest.raises(RuntimeError, match="cancel"):
        dispatcher.forward(execute, [torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 2)],
                           torch.tensor([1000.]), torch.zeros(1, 2, 128), options)
    assert seen == [{}] and tst.CONTEXT_KEY not in options


@pytest.mark.parametrize("mode", ["t2va", "refs", "firstlast", "audio", "all"])
@pytest.mark.parametrize("scale", [.6, 1.])
def test_native_guided_reproducibility_and_global_progress(monkeypatch, mode, scale):
    original = tiny_patcher(monkeypatch)
    patcher = tst.apply_temporal_transport(original, strength=.1)
    assert tst.apply_temporal_transport(original, strength=0) is original
    assert tst.CONFIG_KEY not in original.model_options["transformer_options"]
    monkeypatch.setattr(progressive, "upscale_h3_video_to_size", nearest)
    args = inputs(model=patcher, lowres_scale=scale) if mode == "t2va" else guided_inputs(mode, model=patcher, lowres_scale=scale)
    progress = []
    transform = tst.transform_video_q
    def record(q, k, **kwargs):
        progress.append(kwargs["weight"])
        return transform(q, k, **kwargs)
    monkeypatch.setattr(tst, "transform_video_q", record)
    output, _ = progressive.sample_h3_progressive(**args, _guided=mode != "t2va")
    repeated, _ = progressive.sample_h3_progressive(**args, _guided=mode != "t2va")
    assert progress == pytest.approx([1., .75, .25, 0.] * 2)
    for a, b in zip(output["samples"].unbind(), repeated["samples"].unbind()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.isfinite(a).all()
    if mode in ("audio", "all"):
        assert torch.equal(output["samples"].unbind()[1], args["latent_image"]["samples"].unbind()[1])
    assert tst.SCHEDULE_KEY not in patcher.model_options["transformer_options"]


def test_foreign_patch_and_duplicate_guards(monkeypatch):
    model = tiny_patcher(monkeypatch)
    patched = tst.apply_temporal_transport(model, strength=.1)
    with pytest.raises(RuntimeError, match="already installed"):
        tst.apply_temporal_transport(patched, strength=.2)
    model.model_options["transformer_options"]["sol_morton"] = True
    with pytest.raises(RuntimeError, match="Morton"):
        tst.apply_temporal_transport(model, strength=.1)
    model.model_options["transformer_options"].clear()
    model.object_patches["diffusion_model.blocks.0.attn.forward"] = lambda: None
    with pytest.raises(RuntimeError, match="Unsupported attention"):
        tst.apply_temporal_transport(model, strength=.1)


def test_unified_appends_tst_after_sol_and_defaults_preserve_widgets(monkeypatch):
    from ComfyUI_JR_MiniMaxH3Node.nodes.h3_unified_acceleration import JR_H3_UnifiedAcceleration
    from test_h3_unified_acceleration import install_spies, required_values
    calls = install_spies(monkeypatch)
    def apply(model, **kwargs):
        calls.append(("tst", model, kwargs, model))
        return model
    monkeypatch.setattr(tst, "apply_temporal_transport", apply)
    JR_H3_UnifiedAcceleration().patch(**required_values(), enable_tst=True, tst_strength=.1)
    assert [c[0] for c in calls] == ["sage", "low", "ffn", "sol", "tst"]
    inputs = JR_H3_UnifiedAcceleration.INPUT_TYPES()
    assert list(inputs["optional"]) == ["tau_profile", "enable_tst", "tst_strength"]
    assert inputs["optional"]["enable_tst"][1]["default"] is False
    for key in ("morton", "allow_compile"):
        with pytest.raises(ValueError, match="disable Morton"):
            JR_H3_UnifiedAcceleration().patch(**required_values(**{key: True}), enable_tst=True)


@pytest.mark.parametrize("tst_first", [False, True])
def test_cache_wrapper_orders_and_configuration_isolation(monkeypatch, tst_first):
    from dataclasses import replace

    from ComfyUI_JR_MiniMaxH3Node.utils.h3_cache_config import build_preset_config
    from ComfyUI_JR_MiniMaxH3Node.utils.h3_cache_runtime import H3AdaptiveCacheRuntime

    patcher = tiny_patcher(monkeypatch)
    config = replace(build_preset_config("visual_fast"), front_blocks=0, back_blocks=0, warmup_steps=0,
                     start_percent=0., end_percent=1., video_threshold=1., audio_threshold=1., cache_device="CPU")
    runtime = H3AdaptiveCacheRuntime(config, 1, audio_required=True)
    if tst_first:
        patcher = tst.apply_temporal_transport(patcher, strength=.1)
    patcher.add_wrapper_with_key("diffusion_model", "jr_h3_adaptive_cache", runtime.diffusion_wrapper)
    patcher.set_attachments("jr_h3_adaptive_cache", runtime)
    if not tst_first:
        patcher = tst.apply_temporal_transport(patcher, strength=.1)
    monkeypatch.setattr(progressive, "upscale_h3_video_to_size", nearest)
    result, _ = progressive.sample_h3_progressive(**inputs(model=patcher, lowres_scale=.6))
    assert all(torch.isfinite(t).all() for t in result["samples"].unbind())
    assert runtime._sample_signature is None  # progressive finally resets
    video, audio = inputs()["latent_image"]["samples"].unbind()
    context = torch.zeros(1, 2, 128)
    signature = runtime._make_signature(video, audio, context, {"jr_h3_tst_signature": (.1,)}, patcher.model)
    other = runtime._make_signature(video, audio, context, {"jr_h3_tst_signature": (.2,)}, patcher.model)
    assert signature != other
