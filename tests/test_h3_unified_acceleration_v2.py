import copy
import types

import pytest
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_unified_acceleration import JR_H3_UnifiedAcceleration
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_unified_acceleration_v2 import JR_H3_UnifiedAccelerationV2
from ComfyUI_JR_MiniMaxH3Node.utils import h3_sparse_backend as sparse


class Model:
    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.object_patches = {}
        self.callbacks = {}
        self.wrappers = {}
        self.diffusion = types.SimpleNamespace(
            rope_freqs=True, _forward=lambda: None,
            blocks=[types.SimpleNamespace(attn=types.SimpleNamespace(qkv_proj=True),
                                          mlp=types.SimpleNamespace(fc1=True, fc2=True)) for _ in range(50)],
        )

    def get_model_object(self, name):
        return self.diffusion

    def clone(self):
        result = copy.copy(self)
        result.model_options = copy.deepcopy(self.model_options)
        result.object_patches = dict(self.object_patches)
        result.callbacks = copy.deepcopy(self.callbacks)
        return result

    def add_callback_with_key(self, kind, key, callback):
        self.callbacks.setdefault(kind, {}).setdefault(key, []).append(callback)


def disabled_passes(**kwargs):
    return dict(sage_attention="disabled", enable_low_vram_attention=False, enable_low_vram_ffn=False, **kwargs)


@pytest.fixture
def core(monkeypatch):
    calls = []
    monkeypatch.setattr(sparse, "check_core_support", lambda *a: None)

    def apply(model, parameters):
        calls.append(parameters)
        return model.clone()

    monkeypatch.setattr(sparse, "apply_core", apply)
    return calls


@pytest.mark.parametrize("text,expected", [("-1", "49"), ("-3--1", "47,48,49"),
                                          ("1,0,1,47-49", "0,1,47,48,49"), ("4-2", "2,3,4"),
                                          ("-999,999", ""), ("", "")])
def test_legacy_dense_semantics(text, expected):
    assert sparse.normalize_dense_blocks(text, 50) == expected


def test_invalid_blocks_do_not_silently_enable_sparse():
    with pytest.raises(ValueError, match="dense_blocks"):
        sparse.normalize_dense_blocks("bad", 50)


def test_old_schema_unchanged():
    old = copy.deepcopy(JR_H3_UnifiedAcceleration.INPUT_TYPES())
    new = JR_H3_UnifiedAccelerationV2.INPUT_TYPES()
    assert JR_H3_UnifiedAcceleration.INPUT_TYPES() == old
    assert old["required"]["min_tokens"][1]["default"] == 4096
    assert "sparse_backend" not in old["required"]
    assert new["required"]["min_tokens"][1]["default"] == 12288


def test_core_no_legacy_dependency_and_explicit_values(core):
    source = Model()
    result = JR_H3_UnifiedAccelerationV2().patch(source, sparse_backend="core", dense_blocks="-1", **disabled_passes())[0]
    assert source.model_options == {"transformer_options": {}}
    assert source.callbacks == {}
    assert core[0]["selection"] == {"selection": "sol-attn", "tau": 1.3}
    assert core[0]["extra_tokens"] == 256
    assert core[0]["min_tokens"] == 12288
    assert core[0]["dense_blocks"] == "49"
    assert "int8_qk" not in core[0]
    assert result.model_options["transformer_options"][sparse.CONFIG_KEY][1] == "core"


@pytest.mark.parametrize("field,value", [("tau_profile", "0-3=2"), ("morton", True), ("int8_pv", False),
                                         ("int8_qk", False), ("use_tma", True), ("allow_compile", True),
                                         ("enable_tst", True)])
def test_core_rejects_semantic_changes_before_patch(core, field, value):
    with pytest.raises(ValueError, match="legacy|TST"):
        JR_H3_UnifiedAccelerationV2().patch(Model(), sparse_backend="core", **disabled_passes(**{field: value}))
    assert not core


def test_tst_zero_is_noop(core):
    JR_H3_UnifiedAccelerationV2().patch(Model(), sparse_backend="core", enable_tst=True, tst_strength=0, **disabled_passes())
    assert len(core) == 1


@pytest.mark.parametrize("requested", ["core", "auto", "disabled", "legacy_kijai"])
def test_existing_tst_rejected_even_current_switch_off(core, requested):
    model = Model()
    model.model_options["transformer_options"]["jr_h3_tst_config"] = ("pooled-q-v1", .1, 50)
    with pytest.raises(ValueError, match="already has TST"):
        JR_H3_UnifiedAccelerationV2().patch(model, sparse_backend=requested, **disabled_passes())


def test_block_replacement_never_overwritten(core):
    model = Model()
    original = object()
    model.model_options["transformer_options"]["patches_replace"] = {"dit": {("double_block", 0): original}}
    with pytest.raises(ValueError, match="block replacement"):
        JR_H3_UnifiedAccelerationV2().patch(model, sparse_backend="core", **disabled_passes())
    assert model.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)] is original


def test_lora_projection_patches_preserved(core):
    source = Model()
    adaln = object()
    projection = object()
    source.object_patches = {"diffusion_model.blocks.0.adaln_proj.forward": adaln,
                             "diffusion_model.blocks.0.attn.qkv_proj.forward": projection}
    result = JR_H3_UnifiedAccelerationV2().patch(source, sparse_backend="core", **disabled_passes())[0]
    assert result.object_patches == source.object_patches


def test_duplicate_rejected(core):
    node = JR_H3_UnifiedAccelerationV2()
    result = node.patch(Model(), sparse_backend="core", **disabled_passes())[0]
    with pytest.raises(ValueError, match="already installed"):
        node.patch(result, sparse_backend="core", **disabled_passes())


def test_downstream_override_rejected_at_sampling(core):
    result = JR_H3_UnifiedAccelerationV2().patch(Model(), sparse_backend="core", **disabled_passes())[0]
    result.model_options["transformer_options"]["optimized_attention_override"] = object()
    callback = result.callbacks["on_pre_run"][sparse.CONFIG_KEY][0]
    with pytest.raises(RuntimeError, match="downstream"):
        callback(result)


def test_cached_model_reports_each_sampling(core, caplog):
    result = JR_H3_UnifiedAccelerationV2().patch(Model(), sparse_backend="core", **disabled_passes())[0]
    callback = result.callbacks["on_pre_run"][sparse.CONFIG_KEY][0]
    with caplog.at_level("INFO"):
        callback(result)
        callback(result)
    assert sum("requested=core resolved=core" in r.message for r in caplog.records) == 2


def test_enable_false_is_identity_even_invalid_input():
    model = object()
    assert JR_H3_UnifiedAccelerationV2().patch(model, enable=False)[0] is model


@pytest.mark.parametrize("backend,flag", [("disabled", True), ("core", False)])
def test_disabled_does_not_probe_dependencies(monkeypatch, backend, flag):
    monkeypatch.setattr(sparse, "check_core_support", lambda *a: pytest.fail("unexpected probe"))
    result = JR_H3_UnifiedAccelerationV2().patch(Model(), sparse_backend=backend, enable_sol_attn=flag, **disabled_passes())[0]
    assert result.model_options["transformer_options"][sparse.CONFIG_KEY][1] == "disabled"


def test_auto_only_catches_known_unavailable(monkeypatch):
    p = {name: value[1]["default"] for name, value in JR_H3_UnifiedAcceleration.INPUT_TYPES()["required"].items() if name != "model"}
    p.update(enable_tst=False, tst_strength=.1, tau_profile=None)
    def unavailable(*a):
        raise sparse.CoreUnavailable("missing kernel")
    monkeypatch.setattr(sparse, "check_core_support", unavailable)
    assert sparse.resolve_backend(Model(), "auto", p, 256) == ("legacy_kijai", "missing kernel")
    with pytest.raises(RuntimeError, match="requested core"):
        sparse.resolve_backend(Model(), "core", p, 256)
    def bug(*a):
        raise RuntimeError("CUDA illegal memory access")
    monkeypatch.setattr(sparse, "check_core_support", bug)
    with pytest.raises(RuntimeError, match="CUDA"):
        sparse.resolve_backend(Model(), "auto", p, 256)
    p["tau_profile"] = "0=2"
    assert sparse.resolve_backend(Model(), "auto", p, 256)[0] == "legacy_kijai"


def test_core_runtime_error_is_not_retried(core, monkeypatch):
    def fail(*a):
        raise RuntimeError("out of memory")
    monkeypatch.setattr(sparse, "apply_core", fail)
    with pytest.raises(RuntimeError, match="out of memory"):
        JR_H3_UnifiedAccelerationV2().patch(Model(), sparse_backend="auto", **disabled_passes())


def test_old_lowvram_forward_rejected():
    model = Model()
    model.object_patches["diffusion_model.blocks.0.forward"] = lambda x: x
    with pytest.raises(sparse.adapters.H3AccelerationCompatibilityError, match="attention="):
        sparse.check_core_block_patches(model)


@pytest.mark.parametrize("changes", [{"tau": float("nan")}, {"start_percent": .9, "end_percent": .2},
                                      {"extra_tokens": 32}, {"min_tokens": -1}])
def test_bad_values_rejected(core, changes):
    with pytest.raises(ValueError):
        JR_H3_UnifiedAccelerationV2().patch(Model(), sparse_backend="core", **disabled_passes(**changes))
    assert not core
