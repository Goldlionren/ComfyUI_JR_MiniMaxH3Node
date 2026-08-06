from dataclasses import replace

import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_adaptive_cache import JR_H3_AdaptiveCache, detect_cache_conflict
from ComfyUI_JR_MiniMaxH3Node.utils.h3_cache_config import build_preset_config
from ComfyUI_JR_MiniMaxH3Node.utils.h3_cache_runtime import H3AdaptiveCacheRuntime


class MiniMaxH3Model:
    def __init__(self):
        self.blocks = [object() for _ in range(50)]


MiniMaxH3Model.__module__ = "comfy.ldm.minimax.model"


class FakePatcher:
    def __init__(self, diffusion=None):
        self.diffusion = diffusion or MiniMaxH3Model()
        self.model_options = {"transformer_options": {}}
        self.wrappers = {}
        self.callbacks = {}
        self.attachments = {}

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diffusion

    def clone(self):
        clone = FakePatcher(self.diffusion)
        clone.model_options = {"transformer_options": {}}
        return clone

    def set_attachments(self, key, value):
        self.attachments[key] = value

    def get_attachment(self, key):
        return self.attachments.get(key)

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.setdefault(kind, {})[key] = [wrapper]

    def add_callback_with_key(self, kind, key, callback):
        self.callbacks.setdefault(kind, {})[key] = [callback]

    def set_model_patch_replace(self, patch, name, block_name, index):
        target = self.model_options["transformer_options"].setdefault("patches_replace", {}).setdefault(name, {})
        target[(block_name, index)] = patch


def _node_args(**changes):
    values = dict(model=FakePatcher(), mode="Auto", quality_level="Balanced", audio_content="Auto", profile_hint="",
                  start_percent=0.1, end_percent=0.9, warmup_steps=2, front_blocks=1, back_blocks=2,
                  video_threshold=0.02, audio_threshold=0.012, fast_path_threshold=0.008,
                  probe_path_threshold=0.035, max_full_step_hits=1, max_block_hits=2,
                  video_metric_stride=12, audio_metric_stride=6, cache_device="CPU", gpu_reserve_mb=2048,
                  strict_model_check=True, verbose=False, cache_config=None)
    values.update(changes)
    return values


def test_off_returns_original_model_and_manual_auto_selection():
    model = FakePatcher()
    output = JR_H3_AdaptiveCache().apply_cache(**_node_args(model=model, mode="Off"))
    assert output[0] is model and output[1] == "off"
    output = JR_H3_AdaptiveCache().apply_cache(**_node_args(audio_content="Speech"))
    assert output[1] == "dialogue_safe"


def test_router_config_overrides_every_manual_widget():
    config = build_preset_config("action_safe", "Conservative", source="router", cache_device="CPU")
    output = JR_H3_AdaptiveCache().apply_cache(**_node_args(
        mode="Off", quality_level="Custom", front_blocks=48, back_blocks=48,
        video_threshold=0.99, cache_config=config))
    assert output[1] == "action_safe"
    assert "Configuration source: Router" in output[2] and "Manual widget values ignored" in output[2]


def test_invalid_config_schema_and_non_h3_rejected():
    config = build_preset_config("balanced")
    object.__setattr__(config, "schema_version", 99)
    with pytest.raises(ValueError, match="schema_version"):
        JR_H3_AdaptiveCache().apply_cache(**_node_args(cache_config=config))
    wrong = FakePatcher(type("Other", (), {"blocks": [1] * 50})())
    with pytest.raises(RuntimeError, match="requires native"):
        JR_H3_AdaptiveCache().apply_cache(**_node_args(model=wrong))
    assert JR_H3_AdaptiveCache().apply_cache(**_node_args(model=wrong, strict_model_check=False))[1] == "off"


def test_conflict_detection_and_patch_registration():
    model = FakePatcher()
    model.wrappers = {"diffusion_model": {"TeaCache": [object()]}}
    assert "tea" in detect_cache_conflict(model)
    clean = FakePatcher()
    patched, profile, _ = JR_H3_AdaptiveCache().apply_cache(**_node_args(model=clean, mode="Dialogue Safe"))
    assert profile == "dialogue_safe"
    assert "jr_h3_adaptive_cache" in patched.wrappers["diffusion_model"]
    assert len(patched.model_options["transformer_options"]["patches_replace"]["dit"]) == 50


class Layout:
    signature = (1, 1, 1, 1, 1)
    segments = [(0, 1, "text"), (1, 5, "audio"), (5, 9, "video")]


class MockExecutor:
    class_obj = object()

    def __init__(self, runtime):
        self.runtime = runtime
        self.calls = []

    def __call__(self, x, timestep, context, transformer_options, minimax_payload=None, **kwargs):
        h = torch.cat([torch.zeros(1, 4), x[1].reshape(4, 1).expand(4, 4), x[0].reshape(4, 1).expand(4, 4)])
        for index in range(50):
            def original(args, i=index):
                self.calls.append(i)
                return {"img": args["img"] + 0.001 * (i + 1)}
            h = self.runtime.block_wrapper(index)({"img": h}, {"original_block": original})["img"]
        return [x[0] + h[5:9, 0].mean(), x[1] + h[1:5, 0].mean()]


def _runtime(profile, **changes):
    config = build_preset_config(profile, cache_device="CPU", audio_content="Auto")
    overrides = {"warmup_steps": 0, "start_percent": 0.0, "end_percent": 1.0}
    overrides.update(changes)
    config = replace(config, **overrides)
    return H3AdaptiveCacheRuntime(config, 50, audio_required=True)


def _run(runtime, executor, value, timestep, context):
    executor.calls.clear()
    video = torch.full((1, 1, 1, 2, 2), value)
    audio = torch.full((1, 1, 2, 2), value)
    payload = {"layout": Layout(), "seed": 123}
    runtime.diffusion_wrapper(executor, [video, audio], torch.tensor([timestep]), context, {}, payload)
    return list(executor.calls)


def test_mock_h3_off_and_visual_fast_paths():
    off = _runtime("off")
    ex = MockExecutor(off); context = torch.ones(1, 1, 4)
    assert len(_run(off, ex, 1.0, 900, context)) == 50
    visual = _runtime("visual_fast")
    ex = MockExecutor(visual)
    assert len(_run(visual, ex, 1.0, 900, context)) == 50
    assert _run(visual, ex, 1.0, 800, context) == []


@pytest.mark.parametrize(("profile", "expected"), [("dialogue_safe", [0, 48, 49]), ("action_safe", [0, 1, 48, 49])])
def test_mock_h3_safe_block_paths(profile, expected):
    runtime = _runtime(profile)
    executor = MockExecutor(runtime); context = torch.ones(1, 1, 4)
    assert len(_run(runtime, executor, 1.0, 900, context)) == 50
    assert _run(runtime, executor, 1.0, 800, context) == expected


def test_balanced_fast_probe_and_full_paths():
    context = torch.ones(1, 1, 4)
    fast = _runtime("balanced", fast_path_threshold=0.02, probe_path_threshold=0.1)
    ex = MockExecutor(fast)
    _run(fast, ex, 1.0, 900, context)
    assert _run(fast, ex, 1.0, 800, context) == []

    probe = _runtime("balanced", fast_path_threshold=0.001, probe_path_threshold=0.1,
                     video_threshold=0.1, audio_threshold=0.1, max_full_step_hits=0)
    ex = MockExecutor(probe)
    _run(probe, ex, 1.0, 900, context)
    assert len(_run(probe, ex, 1.01, 800, context)) == 50
    assert _run(probe, ex, 1.02, 700, context) == [0, 48, 49]

    full = _runtime("balanced", fast_path_threshold=0.001, probe_path_threshold=0.01)
    ex = MockExecutor(full)
    _run(full, ex, 1.0, 900, context)
    assert len(_run(full, ex, 2.0, 800, context)) == 50


def test_audio_and_video_can_independently_veto_block_cache():
    context = torch.ones(1, 1, 4)
    runtime = _runtime("dialogue_safe", video_threshold=0.05, audio_threshold=0.05)
    executor = MockExecutor(runtime)
    _run(runtime, executor, 1.0, 900, context)
    _run(runtime, executor, 1.0, 800, context)
    executor.calls.clear()
    video = torch.ones(1, 1, 1, 2, 2)
    audio = torch.full((1, 1, 2, 2), 3.0)
    runtime.diffusion_wrapper(executor, [video, audio], torch.tensor([700]), context, {}, {"layout": Layout(), "seed": 123})
    assert len(executor.calls) == 50 and runtime.stats.audio_veto_count > 0

    runtime = _runtime("dialogue_safe", video_threshold=0.05, audio_threshold=0.05)
    executor = MockExecutor(runtime)
    _run(runtime, executor, 1.0, 900, context)
    _run(runtime, executor, 1.0, 800, context)
    executor.calls.clear()
    runtime.diffusion_wrapper(executor, [torch.full_like(video, 3.0), torch.ones_like(audio)], torch.tensor([700]),
                              context, {}, {"layout": Layout(), "seed": 123})
    assert len(executor.calls) == 50 and runtime.stats.video_veto_count > 0


def test_state_resets_on_shape_dtype_audio_presence_and_timestep_restart():
    runtime = _runtime("visual_fast")
    executor = MockExecutor(runtime); context = torch.ones(1, 1, 4)
    _run(runtime, executor, 1.0, 900, context)
    before = runtime.stats.cache_resets
    # Timestep increase is a new sampling run even with identical shapes.
    _run(runtime, executor, 1.0, 950, context)
    assert runtime.stats.cache_resets > before


def test_warmup_window_and_consecutive_hit_limit_force_full_refresh():
    context = torch.ones(1, 1, 4)
    warm = _runtime("visual_fast", warmup_steps=2)
    executor = MockExecutor(warm)
    assert len(_run(warm, executor, 1.0, 900, context)) == 50
    assert len(_run(warm, executor, 1.0, 800, context)) == 50
    assert _run(warm, executor, 1.0, 700, context) == []

    window = _runtime("visual_fast", start_percent=0.5, end_percent=0.8)
    executor = MockExecutor(window)
    assert len(_run(window, executor, 1.0, 900, context)) == 50
    assert len(_run(window, executor, 1.0, 800, context)) == 50
    assert _run(window, executor, 1.0, 400, context) == []

    limited = _runtime("visual_fast", max_full_step_hits=1)
    executor = MockExecutor(limited)
    _run(limited, executor, 1.0, 900, context)
    assert _run(limited, executor, 1.0, 800, context) == []
    assert len(_run(limited, executor, 1.0, 700, context)) == 50
    assert limited.stats.forced_refresh_count == 1


class PassExecutor:
    class_obj = object()

    def __call__(self, x, *_args, **_kwargs):
        return [x[0].clone(), x[1].clone()]


def test_shape_dtype_batch_and_audio_layout_changes_reset_state():
    runtime = _runtime("visual_fast")
    executor = PassExecutor()
    context = torch.ones(1, 1, 4)
    payload = {"layout": Layout(), "seed": 1}
    video = torch.ones(1, 1, 1, 2, 2)
    audio = torch.ones(1, 1, 2, 2)
    runtime.diffusion_wrapper(executor, [video, audio], torch.tensor([900]), context, {}, payload)
    reset_count = runtime.stats.cache_resets
    runtime.diffusion_wrapper(executor, [torch.ones(1, 1, 1, 2, 3), audio], torch.tensor([800]), context, {}, payload)
    assert runtime.stats.cache_resets > reset_count
    reset_count = runtime.stats.cache_resets
    runtime.diffusion_wrapper(executor, [video.double(), audio.double()], torch.tensor([700]), context, {}, payload)
    assert runtime.stats.cache_resets > reset_count
    reset_count = runtime.stats.cache_resets
    runtime.diffusion_wrapper(executor, [video.expand(2, -1, -1, -1, -1), audio.expand(2, -1, -1, -1)],
                              torch.tensor([600]), context, {}, payload)
    assert runtime.stats.cache_resets > reset_count
    reset_count = runtime.stats.cache_resets
    runtime.diffusion_wrapper(executor, [video, torch.ones(1, 1, 2, 3)], torch.tensor([500]), context, {}, payload)
    assert runtime.stats.cache_resets > reset_count


def test_equivalent_recreated_context_and_reference_storage_do_not_reset_state():
    runtime = _runtime("visual_fast")
    executor = PassExecutor()
    video = torch.ones(1, 1, 1, 2, 2)
    audio = torch.ones(1, 1, 2, 2)
    context = torch.ones(1, 1, 4)
    reference = torch.ones(1, 2, 2)
    payload = {
        "layout": Layout(),
        "seed": 7,
        "refs": [{"kind": "image", "latent_h": 2, "latent_w": 2, "latent": reference}],
    }
    runtime.diffusion_wrapper(executor, [video, audio], torch.tensor([900]), context, {}, payload)
    reset_count = runtime.stats.cache_resets
    recreated_payload = {
        "layout": Layout(),
        "seed": 7,
        "refs": [{"kind": "image", "latent_h": 2, "latent_w": 2, "latent": reference.clone()}],
    }
    runtime.diffusion_wrapper(
        executor, [video.clone(), audio.clone()], torch.tensor([800]), context.clone(), {}, recreated_payload
    )
    assert runtime.stats.cache_resets == reset_count
    assert runtime.stats.full_step_cache_hits == 1


def test_reference_structure_seed_and_layout_segments_still_invalidate_state():
    runtime = _runtime("visual_fast")
    video = torch.ones(1, 1, 1, 2, 2)
    audio = torch.ones(1, 1, 2, 2)
    context = torch.ones(1, 1, 4)
    model_obj = object()
    reference = torch.ones(1, 2, 2)
    base = {
        "layout": Layout(),
        "seed": 7,
        "refs": [{"kind": "image", "latent_h": 2, "latent_w": 2, "latent": reference}],
    }
    runtime._begin_forward(video, audio, torch.tensor([900]), context, base, model_obj)
    assert runtime.stats.cache_resets == 0

    changed_seed = dict(base, seed=8)
    runtime._begin_forward(video, audio, torch.tensor([800]), context, changed_seed, model_obj)
    assert runtime.stats.cache_resets == 1

    changed_ref = dict(base, seed=8, refs=[{
        "kind": "image", "latent_h": 2, "latent_w": 3, "latent": torch.ones(1, 2, 3)
    }])
    runtime._begin_forward(video, audio, torch.tensor([700]), context, changed_ref, model_obj)
    assert runtime.stats.cache_resets == 2

    class ChangedLayout(Layout):
        segments = [(0, 1, "text"), (1, 4, "audio"), (4, 9, "video")]

    changed_layout = dict(changed_ref, layout=ChangedLayout())
    runtime._begin_forward(video, audio, torch.tensor([600]), context, changed_layout, model_obj)
    assert runtime.stats.cache_resets == 3


def test_cleanup_clears_per_sampling_statistics_and_state():
    runtime = _runtime("visual_fast")
    executor = PassExecutor()
    context = torch.ones(1, 1, 4)
    video = torch.ones(1, 1, 1, 2, 2)
    audio = torch.ones(1, 1, 2, 2)
    runtime.diffusion_wrapper(
        executor, [video, audio], torch.tensor([900]), context, {}, {"layout": Layout(), "seed": 1}
    )
    assert runtime.stats.total_steps == 1
    runtime.cleanup()
    assert runtime.stats.total_steps == 0
    assert runtime.stats.cache_resets == 0
    assert runtime._sample_signature is None


def test_forward_exception_clears_pending_cache_state():
    class BrokenExecutor:
        class_obj = object()
        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("mock failure")

    runtime = _runtime("visual_fast")
    with pytest.raises(RuntimeError, match="mock failure"):
        runtime.diffusion_wrapper(BrokenExecutor(), [torch.ones(1, 1, 1, 2, 2), torch.ones(1, 1, 2, 2)],
                                  torch.tensor([900]), torch.ones(1, 1, 4), {}, {"layout": Layout(), "seed": 1})
    assert runtime._sample_signature is None
