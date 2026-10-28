import copy
import importlib
import types

import pytest
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_veda_attention import JR_H3_VedaAttention
from ComfyUI_JR_MiniMaxH3Node.utils import h3_veda_attention as veda
from test_h3_progressive_sampler import tiny_patcher


class Model:
    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.object_patches = {}
        self.callbacks = {}
        self.wrappers = {}
        self.diffusion = types.SimpleNamespace(
            rope_freqs=True, _forward=lambda: None,
            blocks=[types.SimpleNamespace(
                attn=types.SimpleNamespace(qkv_proj=True),
                mlp=types.SimpleNamespace(fc1=True, fc2=True),
            ) for _ in range(50)],
        )

    def get_model_object(self, name):
        return self.diffusion

    def clone(self):
        result = copy.copy(self)
        result.model_options = copy.deepcopy(self.model_options)
        result.object_patches = dict(self.object_patches)
        result.callbacks = copy.deepcopy(self.callbacks)
        result.wrappers = copy.deepcopy(self.wrappers)
        return result

    def add_callback_with_key(self, kind, key, callback):
        self.callbacks.setdefault(kind, {}).setdefault(key, []).append(callback)


def no_memory(**kwargs):
    return dict(enable_low_vram_attention=False, enable_low_vram_ffn=False, **kwargs)


@pytest.fixture
def vendor(monkeypatch):
    calls = []

    class OfficialVeda(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="VedaSparseAttention", inputs=[], outputs=[io.Model.Output()])

        @classmethod
        def execute(cls, model, predictor, generated_sparsity="90%", reference_sparsity="90%",
                    full_attention_layers="", full_attention_steps="", verbose=False):
            calls.append(dict(
                model=model, predictor=predictor, generated=generated_sparsity,
                reference=reference_sparsity, layers=full_attention_layers,
                steps=full_attention_steps, verbose=verbose, unique_id=cls.hidden.unique_id,
            ))
            result = model.clone()
            result.model_options["transformer_options"]["optimized_attention_override"] = lambda *a: None
            result.add_callback_with_key("on_cleanup", "official_veda", lambda *a: None)
            return io.NodeOutput(result)

    monkeypatch.setattr(veda.adapters, "_runtime_node_registry", lambda: {"VedaSparseAttention": OfficialVeda})
    return calls, OfficialVeda


def test_bypass_before_all_validation(monkeypatch):
    monkeypatch.setattr(veda, "_veda_handler", lambda *a: pytest.fail("dependency resolved"))
    model = object()
    assert JR_H3_VedaAttention().patch(model, enable=False, head_chunks=-1, sage_attention="invalid") == (
        model, "VEDA bypassed",
    )


def test_missing_upstream_fails_before_any_pass(monkeypatch):
    monkeypatch.setattr(veda.adapters, "_runtime_node_registry", lambda: {})
    monkeypatch.setattr(veda.adapters, "apply_h3_low_vram_attention", lambda *a, **k: pytest.fail("pass ran"))
    with pytest.raises(RuntimeError, match="official Veda-on-ComfyUI"):
        veda.apply(Model())


def test_real_v3_clone_preserves_hidden_and_input_branch(vendor):
    calls, registered = vendor
    source = Model()
    source.object_patches["diffusion_model.blocks.0.attn.qkv_proj.forward"] = object()
    result, status = veda.apply(
        source, unique_id="jr-42", generated_sparsity="80%", reference_sparsity="0%",
        full_attention_layers="0,47-49", full_attention_steps="0", verbose=True, **no_memory(),
    )
    assert result is not source
    assert source.model_options == {"transformer_options": {}}
    assert source.callbacks == {}
    assert result.object_patches == source.object_patches
    assert calls[0]["unique_id"] == "jr-42"
    assert calls[0]["generated"] == "80%"
    assert calls[0]["reference"] == "0%"
    assert calls[0]["layers"] == "0,47-49"
    assert calls[0]["steps"] == "0"
    assert calls[0]["verbose"] is True
    assert registered.hidden is None
    assert "configured" in status
    assert "official_veda" in result.callbacks["on_cleanup"]


def test_hidden_state_is_independent_between_nodes(vendor):
    calls, registered = vendor
    veda.apply(Model(), unique_id="first", **no_memory())
    veda.apply(Model(), unique_id="second", **no_memory())
    assert [call["unique_id"] for call in calls] == ["first", "second"]
    assert registered.hidden is None


@pytest.mark.parametrize("marker", ["jr_h3_unified_v2", "sol_compose", "sol_morton", "jr_h3_tst_config"])
def test_sol_core_tst_rejected_without_touching_source(monkeypatch, marker):
    source = Model()
    original = object()
    source.model_options["transformer_options"][marker] = original
    monkeypatch.setattr(veda, "_veda_handler", lambda *a: pytest.fail("upstream resolved"))
    with pytest.raises(ValueError, match="Sol-H3/Core/TST"):
        veda.apply(source, **no_memory())
    assert source.model_options["transformer_options"][marker] is original
    assert source.callbacks == {}


def test_existing_override_rejected(vendor):
    source = Model()
    source.model_options["transformer_options"]["optimized_attention_override"] = object()
    with pytest.raises(ValueError, match="already has an attention override"):
        veda.apply(source, **no_memory())


@pytest.mark.parametrize("kind", ["block", "object", "core_callback", "stream", "tst"])
def test_incompatible_stack_rejected(vendor, kind):
    source = Model()
    if kind == "block":
        source.model_options["transformer_options"]["patches_replace"] = {"dit": {(0, 0): object()}}
    elif kind == "object":
        source.object_patches["diffusion_model.blocks.0.attn.forward"] = object()
    elif kind == "core_callback":
        source.callbacks["on_prepare_state"] = {"block_sparse_attention": []}
    else:
        key = "jr_h3_streaming" if kind == "stream" else "jr_h3_temporal_transport"
        source.wrappers["diffusion_model"] = {key: []}
    with pytest.raises(ValueError):
        veda.apply(source, **no_memory())


def test_dense_memory_veda_order_and_compile_off(vendor, monkeypatch):
    calls, _ = vendor
    passes = []

    def run(name):
        def patch(model, **kwargs):
            passes.append((name, kwargs))
            model.model_options["transformer_options"][name] = True
            return model
        return patch

    monkeypatch.setattr(veda.adapters, "apply_sage", run("sage"))
    monkeypatch.setattr(veda.adapters, "apply_h3_low_vram_attention", run("low_vram"))
    monkeypatch.setattr(veda.adapters, "apply_h3_chunk_ffn", run("ffn"))
    source = Model()
    veda.apply(source, sage_attention="auto", head_chunks=7, ffn_chunks=3)
    assert [name for name, _ in passes] == ["sage", "low_vram", "ffn"]
    assert passes[0][1]["allow_compile"] is False
    assert passes[1][1] == {"head_chunks": 7}
    assert passes[2][1]["chunks"] == 3
    assert calls[0]["model"].model_options["transformer_options"]["ffn"] is True
    assert source.model_options == {"transformer_options": {}}


@pytest.mark.parametrize("field,value", [
    ("head_chunks", 0), ("head_chunks", True), ("ffn_chunks", 65),
    ("ffn_seq_threshold", 255), ("sage_attention", "unknown"),
])
def test_invalid_profile_rejected(vendor, field, value):
    with pytest.raises(ValueError, match=field.replace("sage_attention", "Sage")):
        veda.apply(Model(), **no_memory(**{field: value}))


def test_duplicate_and_downstream_override_rejected(vendor):
    result, _ = veda.apply(Model(), **no_memory())
    with pytest.raises(ValueError, match="already installed"):
        veda.apply(result, **no_memory())
    check = result.callbacks["on_pre_run"][veda.CONFIG_KEY][0]
    check(result)
    check(result)  # cached MODEL can be used for another request
    result.model_options["transformer_options"]["optimized_attention_override"] = object()
    with pytest.raises(RuntimeError, match="downstream"):
        check(result)


def test_downstream_core_block_rejected(vendor):
    result, _ = veda.apply(Model(), **no_memory())
    result.model_options["transformer_options"]["patches_replace"] = {"dit": {(0, 0): object()}}
    with pytest.raises(RuntimeError, match="downstream"):
        result.callbacks["on_pre_run"][veda.CONFIG_KEY][0](result)


@pytest.mark.parametrize("shift_first", [False, True])
def test_native_sigma_shift_preserves_veda_attention_stack(vendor, monkeypatch, shift_first):
    source = tiny_patcher(monkeypatch)
    source.get_model_object("model_sampling").set_noise_scale(0.75)
    source_object_patches = dict(source.object_patches)
    model = source
    if shift_first:
        model = MiniMaxH3SigmaShift.execute(model, shift_video=6.0, shift_audio=3.0).result[0]
    patched, _ = veda.apply(model, **no_memory())
    shifted = patched if shift_first else MiniMaxH3SigmaShift.execute(
        patched, shift_video=6.0, shift_audio=3.0,
    ).result[0]
    override = patched.model_options["transformer_options"]["optimized_attention_override"]
    shifted.pre_run()
    shifted.pre_run()  # native cached branch may run again
    assert shifted.model_options["transformer_options"]["optimized_attention_override"] is override
    sampling = shifted.get_model_object("model_sampling")
    assert sampling.shift == 6.0
    assert sampling.audio_shift == 3.0
    assert sampling.noise_scale == 0.75
    assert veda.CONFIG_KEY not in source.model_options["transformer_options"]
    assert source.object_patches == source_object_patches


@pytest.mark.parametrize("mutation", ["override", "forward"])
def test_native_sigma_shift_still_rejects_late_attention_changes(vendor, monkeypatch, mutation):
    patched, _ = veda.apply(tiny_patcher(monkeypatch), **no_memory())
    shifted = MiniMaxH3SigmaShift.execute(patched, shift_video=6.0, shift_audio=3.0).result[0]
    if mutation == "override":
        shifted.model_options["transformer_options"]["optimized_attention_override"] = lambda *args: None
    else:
        shifted.add_object_patch("diffusion_model.blocks.0.attn.forward", lambda *args: None)
    with pytest.raises(RuntimeError, match="downstream"):
        shifted.pre_run()


def test_r2va_metadata_failure_is_actionable(monkeypatch):
    def execute(**kwargs):
        raise ValueError("incomplete metadata ('keep_ratio')")
    monkeypatch.setattr(veda, "_veda_handler", lambda *a: execute)
    with pytest.raises(RuntimeError, match="fixed-tile R2VA"):
        veda.apply(Model(), **no_memory())


def test_upstream_error_not_hidden_as_success(monkeypatch):
    def execute(**kwargs):
        raise ValueError("missing predictor")
    monkeypatch.setattr(veda, "_veda_handler", lambda *a: execute)
    with pytest.raises(ValueError, match="missing predictor"):
        veda.apply(Model(), **no_memory())


def test_schema_works_without_veda_or_its_models(monkeypatch):
    monkeypatch.setattr(veda, "predictor_names", lambda: [veda.DEFAULT_PREDICTOR])
    schema = JR_H3_VedaAttention.INPUT_TYPES()
    assert schema["required"]["predictor"][0] == [veda.DEFAULT_PREDICTOR]
    assert schema["hidden"] == {"unique_id": "UNIQUE_ID"}
    assert schema["required"]["sage_attention"][1]["default"] == "disabled"
    assert JR_H3_VedaAttention.RETURN_TYPES == ("MODEL", "STRING")


def test_import_does_not_resolve_gpu_or_vendor(monkeypatch):
    monkeypatch.setattr(veda.adapters, "_runtime_node_registry", lambda: pytest.fail("registry imported"))
    importlib.reload(veda)


def test_upstream_noop_is_not_reported_as_configured(monkeypatch):
    monkeypatch.setattr(veda, "_veda_handler", lambda *a: lambda **kw: kw["model"])
    with pytest.raises(RuntimeError, match="without a usable attention override"):
        veda.apply(Model(), **no_memory())
