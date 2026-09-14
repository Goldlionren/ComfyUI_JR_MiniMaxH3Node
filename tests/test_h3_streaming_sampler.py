import copy
from types import SimpleNamespace

import pytest
import torch
from comfy.ldm.minimax.model import PackedLayout
from comfy.nested_tensor import NestedTensor
from comfy.samplers import CFGGuider, ksampler
from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
from ComfyUI_JR_MiniMaxH3Node.utils.h3_stream_attention import StreamingRuntime, phase_layout
from ComfyUI_JR_MiniMaxH3Node.utils.h3_stream_cache import JR_H3_CleanAVKVCache
from ComfyUI_JR_MiniMaxH3Node.utils.h3_stream_plan import PRESETS, canonical_plan
from ComfyUI_JR_MiniMaxH3Node.utils.h3_streaming_sampler import sample_streaming
from ComfyUI_JR_MiniMaxH3Node.utils.h3_temporal_transport import apply_temporal_transport
from test_h3_progressive_sampler import tiny_patcher


def setup(monkeypatch, mode="Streaming Attention", layers=2):
    model = tiny_patcher(monkeypatch)
    block = model.model.diffusion_model.blocks[0]
    model.model.diffusion_model.blocks = torch.nn.ModuleList([copy.deepcopy(block) for _ in range(layers)])
    vae = SimpleNamespace(latent_channels=24, spacial_compression_decode=lambda: 16, decode=lambda x: x)
    return dict(model=model, positive=[[torch.zeros(1, 2, 128), {}]], vae=vae,
                noise=Noise_RandomNoise(123), sampler=ksampler("euler"), sigmas=torch.tensor([1., .6, .2, 0.]),
                latent_image={"samples": NestedTensor((torch.zeros(1, 24, 37, 2, 2), torch.zeros(1, 32, 2, 207))),
                              "metadata": {"keep": True}}, stream_plan=canonical_plan(), streaming_mode=mode)


@pytest.mark.parametrize("preset", PRESETS)
def test_phase_positions_are_full_timeline_slices(preset):
    plan = canonical_plan(preset)
    video_t, audio_t = plan.video_latent_count, plan.audio_latent_count
    payload = {"keyframes": [{"resolved_frame_index": plan.native_frame_count - 1, "latent": torch.zeros(1, 24, 1, 4, 4)}]}
    full = PackedLayout(2, video_t, 4, 4, audio_t, keyframes=payload["keyframes"])
    fv = full.position_ids[-video_t*4:].reshape(video_t, 4, 3)
    aa, ab, _ = next(s for s in full.segments if s[2] == "audio")
    fa = full.position_ids[aa:ab].reshape(2, audio_t, 3)
    for p in plan.phases:
        local = phase_layout(plan, p, 2, 4, 4, payload)
        la, lb, _ = next(s for s in local.segments if s[2] == "audio")
        assert torch.equal(local.position_ids[la:lb], fa[:, p.audio_latent_start:p.audio_latent_stop].reshape(-1, 3))
        assert torch.equal(local.position_ids[lb:], fv[p.video_latent_start:p.video_latent_stop].reshape(-1, 3))
        assert torch.equal(local.position_ids[:la], full.position_ids[:aa])
    # The cycle after latent 12 differs from local reset, even with an offset.
    p = plan.phases[1]
    local = phase_layout(plan, p, 2, 4, 4, payload)
    fresh = PackedLayout(2, 10, 4, 4, p.audio_latent_count)
    assert not torch.equal(local.position_ids[-40:, 0].diff(), fresh.position_ids[-40:, 0].diff())


@pytest.mark.parametrize("mode", ["Micro Chunk", "Clean Commit", "Clean KV", "Streaming Attention", "Sparse KV"])
def test_native_modes_and_no_leak(monkeypatch, mode):
    args = setup(monkeypatch, mode)
    model = args["model"]
    before_opts = copy.deepcopy(model.model_options)
    before_wrappers = copy.deepcopy(model.wrappers)
    output, status = sample_streaming(**args)
    assert output["metadata"] is args["latent_image"]["metadata"]
    assert [tuple(t.shape) for t in output["samples"].unbind()] == [(1, 24, 37, 2, 2), (1, 32, 2, 207)]
    assert all(torch.isfinite(t).all() for t in output["samples"].unbind())
    assert "Denoise forwards: 12" in status
    assert f"clean forwards: {0 if mode == 'Micro Chunk' else 4}" in status
    assert model.model_options == before_opts and model.wrappers == before_wrappers
    assert model.model.latent_shapes is None


def test_clean_commit_does_not_change_output(monkeypatch):
    args = setup(monkeypatch, "Micro Chunk")
    plain, _ = sample_streaming(**args)
    args["streaming_mode"] = "Clean KV"
    committed, _ = sample_streaming(**args)
    for a, b in zip(plain["samples"].unbind(), committed["samples"].unbind()):
        assert torch.equal(a, b)


def test_geometry_only_matches_ordinary(monkeypatch):
    args = setup(monkeypatch, "Geometry Only")
    output, _ = sample_streaming(**args)
    guider = CFGGuider(args["model"])
    guider.set_conds(args["positive"], [])
    expected = guider.sample(args["noise"].generate_noise(args["latent_image"]), args["latent_image"]["samples"],
                             args["sampler"], args["sigmas"], seed=123)
    assert all(torch.equal(a, b) for a, b in zip(output["samples"].unbind(), expected.unbind()))


def test_geometry_only_returns_samples_on_native_intermediate_device(monkeypatch):
    import comfy.model_management

    args = setup(monkeypatch, "Geometry Only")
    sampled = args["latent_image"]["samples"]
    transferred = NestedTensor(tuple(t.clone() for t in sampled.unbind()))
    moves = []

    def move(self, device):
        assert self is sampled
        moves.append(device)
        return transferred

    monkeypatch.setattr(CFGGuider, "sample", lambda *a, **kw: sampled)
    monkeypatch.setattr(comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(NestedTensor, "to", move)
    output, _ = sample_streaming(**args)
    assert moves == [torch.device("cpu")]
    assert output["samples"] is transferred
    assert output["metadata"] is args["latent_image"]["metadata"]


def test_tst_repeat_and_locked_audio_with_anchor(monkeypatch):
    args = setup(monkeypatch)
    args["model"] = apply_temporal_transport(args["model"], strength=.2)
    v, a = args["latent_image"]["samples"].unbind()
    a.fill_(.125)
    args["latent_image"]["noise_mask"] = NestedTensor((torch.ones(1, 1, 37, 1, 1), torch.zeros(1, 1, 2, 207)))
    args["positive"][0][1]["minimax_keyframes"] = [{"resolved_frame_index": 123, "latent": torch.zeros(1, 24, 1, 2, 2)}]
    args["positive"][0][1]["minimax_refs"] = [dict(kind="image", latent_h=4, latent_w=6,
                                                latent=torch.full((1, 24, 1, 4, 6), .2))]
    first, status = sample_streaming(**args)
    second, _ = sample_streaming(**args)
    assert "TST: 0.2" in status
    for x, y in zip(first["samples"].unbind(), second["samples"].unbind()):
        assert torch.equal(x, y)
    assert torch.equal(first["samples"].unbind()[1], a)


def test_full_50_layer_reference_and_sparse(monkeypatch):
    args = setup(monkeypatch, layers=50)
    _, status = sample_streaming(**args)
    assert "Cached layers: 50/50" in status
    args["streaming_mode"] = "Sparse KV"
    _, sparse = sample_streaming(**args)
    assert "Cached layers: 13/50" in sparse and "74.0%" in sparse


def test_failure_clears_runtime_and_next_job(monkeypatch):
    args = setup(monkeypatch)
    original = JR_H3_CleanAVKVCache.stage
    captures = []
    def fail(self, *a, **kw):
        captures.append(self)
        original(self, *a, **kw)
        raise RuntimeError("injected after staging")
    monkeypatch.setattr(JR_H3_CleanAVKVCache, "stage", fail)
    with pytest.raises(RuntimeError, match="injected"):
        sample_streaming(**args)
    assert captures and all(c.nbytes == 0 and c.staged_bytes == 0 and not c.active for c in captures)
    assert args["model"].model.latent_shapes is None
    monkeypatch.setattr(JR_H3_CleanAVKVCache, "stage", original)
    sample_streaming(**args)


def test_unknown_cache_wrapper_rejected_before_execution(monkeypatch):
    args = setup(monkeypatch)
    args["model"].add_wrapper_with_key("diffusion_model", "unknown_cache", lambda *a: None)
    with pytest.raises(ValueError, match="wrapper"):
        sample_streaming(**args)


def test_conditioning_isolation_and_real_history_effect():
    plan = canonical_plan()
    cache = JR_H3_CleanAVKVCache((0,), 1)
    runtime = StreamingRuntime(plan, cache, 1, 1, None, use_history=True, collect=True, clean_enabled=True)
    runtime.phase = plan.phases[0]
    runtime.layout = phase_layout(plan, runtime.phase, 2, 2, 2, {})
    runtime.active = True
    s = runtime.layout.seq_len
    q = torch.zeros(1, 1, s, 4)
    k = q.clone()
    v = torch.ones_like(q)
    def backend(q, k, v, heads, **kw):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)
    options = {"block_index": 0}
    baseline = runtime(backend, q, k, v, 1, skip_reshape=True, skip_output_reshape=True, transformer_options=options)
    cache.begin_clean_commit(0)
    x = torch.full((1, 1, 2, 4), 10.)
    cache.stage(0, 0, x * 0, x, x * 0, x)
    cache.commit()
    runtime.head_counts = {}
    changed = runtime(backend, q, k, v, 1, skip_reshape=True, skip_output_reshape=True, transformer_options=options)
    assert torch.equal(changed[:, :, :2], baseline[:, :, :2])
    assert bool((changed[:, :, 2:] > baseline[:, :, 2:]).all())
    runtime.clear()


@pytest.mark.parametrize("change,message", [
    ({"stream_plan": None}, "stream plan"),
    ({"streaming_mode": "bad"}, "mode"),
    ({"sigmas": torch.tensor([1., .5, .1])}, "sigmas"),
    ({"sigmas": torch.tensor([1., float("nan"), 0.])}, "sigmas"),
    ({"max_kv_mib": 0}, "max_kv"),
    ({"layer_policy": "custom", "custom_layers": "99", "streaming_mode": "Sparse KV"}, "indices"),
    ({"vae": object()}, "VAE"),
])
def test_runtime_input_guards(monkeypatch, change, message):
    args = setup(monkeypatch)
    args.update(change)
    with pytest.raises(ValueError, match=message):
        sample_streaming(**args)


def test_dtype_input_mask_and_sparse_bytes(monkeypatch):
    args = setup(monkeypatch, "Sparse KV", layers=5)
    v, a = args["latent_image"]["samples"].unbind()
    args["latent_image"]["samples"] = NestedTensor((v.bfloat16(), a.bfloat16()))
    output, status = sample_streaming(**args)
    assert all(t.dtype == torch.bfloat16 for t in output["samples"].unbind())
    assert "Cached layers: 2/5" in status and "60.0%" in status
    args["latent_image"]["noise_mask"] = NestedTensor((torch.full_like(v, .5), torch.ones_like(a)))
    with pytest.raises(ValueError, match="binary"):
        sample_streaming(**args)


def long_inputs(monkeypatch, preset):
    args = setup(monkeypatch, "Sparse KV")
    plan = canonical_plan(preset)
    args["stream_plan"] = plan
    args["sigmas"] = torch.tensor([.3, .2, .1, 0.])  # refinement, not a fresh full-noise generation
    args["layer_policy"] = "every_2"
    v = torch.full((1, 24, plan.video_latent_count, 2, 2), .25)
    a = torch.full((1, 32, 2, plan.audio_latent_count), .125)
    vm = torch.ones(1, 1, plan.video_latent_count, 1, 1)
    vm[:, :, -1] = 0
    args["latent_image"] = {"samples": NestedTensor((v, a)), "metadata": {"keep": True},
        "noise_mask": NestedTensor((vm, torch.zeros(1, 1, 2, plan.audio_latent_count)))}
    args["positive"][0][1]["minimax_keyframes"] = [
        {"resolved_frame_index": plan.native_frame_count - 1, "latent": v[:, :, -1:].clone()}]
    args["model"] = apply_temporal_transport(args["model"], strength=.2)
    return args


@pytest.mark.parametrize("preset", PRESETS[1:])
@pytest.mark.parametrize("retention", ["previous_only", "sink_plus_recent_1", "sink_plus_recent_2"])
def test_long_refinement_preserves_schedule_masks_noise_and_bounded_history(monkeypatch, preset, retention):
    from comfy.samplers import KSAMPLER

    args = long_inputs(monkeypatch, preset)
    args["retention"] = retention
    p = args["stream_plan"]
    original_trim = JR_H3_CleanAVKVCache.trim
    original_sample = KSAMPLER.sample
    original_noise = Noise_RandomNoise.generate_noise
    snapshots, schedules, noise_calls = [], [], []

    def trim(cache):
        original_trim(cache)
        snapshots.append((cache, cache.metrics()))

    def sample(self, model_wrap, sigmas, *a, **kw):
        schedules.append(sigmas.clone())
        return original_sample(self, model_wrap, sigmas, *a, **kw)

    def noise(self, latent):
        noise_calls.append(latent)
        return original_noise(self, latent)

    monkeypatch.setattr(JR_H3_CleanAVKVCache, "trim", trim)
    monkeypatch.setattr(KSAMPLER, "sample", sample)
    monkeypatch.setattr(Noise_RandomNoise, "generate_noise", noise)
    before = tuple(t.clone() for t in args["latent_image"]["samples"].unbind())
    first, status = sample_streaming(**args)
    second, _ = sample_streaming(**args)
    assert len(noise_calls) == 2 and all(x is args["latent_image"] for x in noise_calls)
    assert len(schedules) == 2 * len(p.phases)
    assert all(torch.equal(s, args["sigmas"]) for s in schedules)
    for actual, repeated, source, original in zip(first["samples"].unbind(), second["samples"].unbind(),
                                                args["latent_image"]["samples"].unbind(), before):
        assert torch.equal(actual, repeated) and torch.equal(source, original)
        assert actual.shape == source.shape and actual.dtype == source.dtype
        assert torch.isfinite(actual).all()
    assert torch.equal(first["samples"].unbind()[1], before[1])
    assert torch.equal(first["samples"].unbind()[0][:, :, -1], before[0][:, :, -1])
    assert first["metadata"] is args["latent_image"]["metadata"]
    assert first["noise_mask"] is args["latent_image"]["noise_mask"]
    recent = 2 if retention == "sink_plus_recent_2" else 1
    sink = 0 if retention == "previous_only" else p.phases[0].video_latent_count
    max_rows = max(s.video_latent_count + 2 * s.audio_latent_count for s in p.phases)
    assert all(m["history_tokens"] <= sink + recent * max_rows for _, m in snapshots)
    # Window boundary must retain prior history, rather than resetting to a new request.
    assert len(snapshots) == 2 * len(p.phases)
    assert all(m["history_tokens"] > 0 for _, m in snapshots)
    assert all(c.nbytes == c.staged_bytes == 0 and not c.active for c, _ in snapshots)
    assert args["model"].model.latent_shapes is None
    assert f"Denoise forwards: {len(p.phases) * 3}; clean forwards: {len(p.phases)}" in status
    assert "SIGMAS: 0.3, 0.2, 0.1, 0; unchanged" in status


def test_long_timeline_mismatch_is_actionable(monkeypatch):
    args = setup(monkeypatch)
    args["stream_plan"] = canonical_plan(PRESETS[1])
    with pytest.raises(ValueError, match="72 video latents / 405 audio latents.*Match the full first-pass duration"):
        sample_streaming(**args)


def test_failure_in_second_window_cleans_up_and_can_retry(monkeypatch):
    args = long_inputs(monkeypatch, PRESETS[1])
    original = JR_H3_CleanAVKVCache.stage
    captured = []

    def fail(cache, *a, **kw):
        original(cache, *a, **kw)
        captured.append(cache)
        if cache._active == 4:
            raise RuntimeError("injected second-window failure")

    monkeypatch.setattr(JR_H3_CleanAVKVCache, "stage", fail)
    with pytest.raises(RuntimeError, match="second-window"):
        sample_streaming(**args)
    assert captured and all(c.nbytes == c.staged_bytes == 0 and not c.active for c in captured)
    assert args["model"].model.latent_shapes is None
    monkeypatch.setattr(JR_H3_CleanAVKVCache, "stage", original)
    sample_streaming(**args)
