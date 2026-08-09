import types

import pytest
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_unified_acceleration import JR_H3_UnifiedAcceleration
from ComfyUI_JR_MiniMaxH3Node.utils import h3_acceleration_adapters as adapters


class FakeH3Model:
    def __init__(self, label="model"):
        self.label = label
        block = types.SimpleNamespace(
            attn=types.SimpleNamespace(qkv_proj=object()),
            mlp=types.SimpleNamespace(fc1=object(), fc2=object()),
        )
        self.diffusion = types.SimpleNamespace(
            rope_freqs=object(),
            _forward=lambda: None,
            blocks=[block],
        )

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.diffusion


def required_values(**overrides):
    values = {
        "model": FakeH3Model(),
        "enable": True,
        "sage_attention": "sageattn_qk_int8_pv_fp8_cuda++",
        "allow_compile": False,
        "enable_low_vram_attention": True,
        "head_chunks": 4,
        "enable_low_vram_ffn": True,
        "ffn_chunks": 4,
        "ffn_seq_threshold": 4096,
        "enable_sol_attn": True,
        "tau": 1.3,
        "start_percent": 0.2,
        "end_percent": 0.9,
        "min_tokens": 4096,
        "int8_qk": True,
        "int8_pv": True,
        "sink_conditioning": "exact_kv_and_rows",
        "morton": False,
        "morton_curve": "2d_frame",
        "verbose": False,
        "use_tma": False,
        "dense_blocks": "",
        "tau_profile": None,
    }
    values.update(overrides)
    return values


def install_spies(monkeypatch):
    calls = []

    def stage(name):
        def apply(model, **kwargs):
            result = FakeH3Model(name)
            calls.append((name, model, kwargs, result))
            return result
        return apply

    monkeypatch.setattr(adapters, "apply_sage", stage("sage"))
    monkeypatch.setattr(adapters, "apply_h3_low_vram_attention", stage("low"))
    monkeypatch.setattr(adapters, "apply_h3_chunk_ffn", stage("ffn"))
    monkeypatch.setattr(adapters, "apply_sol_attn", stage("sol"))
    return calls


def test_node_definition_and_validated_defaults():
    inputs = JR_H3_UnifiedAcceleration.INPUT_TYPES()
    required = inputs["required"]
    assert JR_H3_UnifiedAcceleration.CATEGORY == "JR MiniMax H3/Optimization"
    assert JR_H3_UnifiedAcceleration.RETURN_TYPES == ("MODEL",)
    assert JR_H3_UnifiedAcceleration.FUNCTION == "patch"
    assert required["model"] == ("MODEL",)
    assert required["sage_attention"][0] == list(adapters.SAGE_ATTENTION_MODES)
    assert required["sage_attention"][1]["default"] == "sageattn_qk_int8_pv_fp8_cuda++"
    assert required["head_chunks"][1] == {"default": 4, "min": 1, "max": 56, "step": 1}
    assert required["ffn_chunks"][1]["default"] == 4
    assert required["ffn_seq_threshold"][1]["default"] == 4096
    assert required["tau"][1]["default"] == 1.3
    assert required["sink_conditioning"][1]["default"] == "exact_kv_and_rows"
    assert required["morton_curve"][1]["default"] == "2d_frame"
    assert inputs["optional"]["tau_profile"] == ("STRING", {"forceInput": True})


def test_complete_patch_order_and_parameter_forwarding(monkeypatch):
    calls = install_spies(monkeypatch)
    original = FakeH3Model("original")
    values = required_values(
        model=original,
        allow_compile=True,
        head_chunks=56,
        ffn_chunks=64,
        ffn_seq_threshold=262144,
        tau=4.0,
        start_percent=0.0,
        end_percent=1.0,
        min_tokens=1048576,
        int8_qk=False,
        int8_pv=False,
        sink_conditioning="off",
        morton=True,
        morton_curve="3d",
        verbose=True,
        use_tma=True,
        dense_blocks="0-2,-1",
        tau_profile="0-30=2.0\n39-42=0.9",
    )
    output = JR_H3_UnifiedAcceleration().patch(**values)[0]
    assert [call[0] for call in calls] == ["sage", "low", "ffn", "sol"]
    assert calls[0][1] is original
    assert calls[1][1] is calls[0][3]
    assert calls[2][1] is calls[1][3]
    assert calls[3][1] is calls[2][3]
    assert output is calls[3][3]
    assert calls[0][2] == {
        "sage_attention": "sageattn_qk_int8_pv_fp8_cuda++",
        "allow_compile": True,
    }
    assert calls[1][2] == {"head_chunks": 56}
    assert calls[2][2] == {"chunks": 64, "seq_threshold": 262144}
    assert calls[3][2] == {
        "tau": 4.0, "start_percent": 0.0, "end_percent": 1.0,
        "min_tokens": 1048576, "int8_qk": False, "int8_pv": False,
        "sink_conditioning": "off", "morton": True, "morton_curve": "3d",
        "verbose": True, "use_tma": True, "dense_blocks": "0-2,-1",
        "tau_profile": "0-30=2.0\n39-42=0.9",
    }


def test_global_disable_returns_original_without_validation_or_adapters(monkeypatch):
    original = object()
    monkeypatch.setattr(adapters, "ensure_minimax_h3_model", lambda model: pytest.fail("validation called"))
    for name in ("apply_sage", "apply_h3_low_vram_attention", "apply_h3_chunk_ffn", "apply_sol_attn"):
        monkeypatch.setattr(adapters, name, lambda *args, **kwargs: pytest.fail("adapter called"))
    assert JR_H3_UnifiedAcceleration().patch(**required_values(model=original, enable=False))[0] is original


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"sage_attention": "disabled"}, ["low", "ffn", "sol"]),
        ({"enable_low_vram_attention": False, "head_chunks": 1}, ["sage", "ffn", "sol"]),
        ({"enable_low_vram_ffn": False, "ffn_chunks": 1}, ["sage", "low", "sol"]),
        ({"enable_sol_attn": False}, ["sage", "low", "ffn"]),
        (
            {
                "sage_attention": "disabled", "enable_low_vram_attention": False,
                "enable_low_vram_ffn": False, "enable_sol_attn": False,
            },
            [],
        ),
    ],
)
def test_component_switches_are_true_bypasses(monkeypatch, overrides, expected):
    calls = install_spies(monkeypatch)
    original = FakeH3Model("original")
    output = JR_H3_UnifiedAcceleration().patch(**required_values(model=original, **overrides))[0]
    assert [call[0] for call in calls] == expected
    assert output is (calls[-1][3] if calls else original)


@pytest.mark.parametrize("tau_profile", [None, "", "0-30=2.0\n39-42=0.9"])
def test_tau_profile_is_forwarded_without_normalization(monkeypatch, tau_profile):
    calls = install_spies(monkeypatch)
    JR_H3_UnifiedAcceleration().patch(**required_values(tau_profile=tau_profile))
    assert calls[-1][0] == "sol"
    assert calls[-1][2]["tau_profile"] is tau_profile


def test_non_h3_fails_before_any_patch(monkeypatch):
    calls = install_spies(monkeypatch)

    class NonH3:
        def get_model_object(self, name):
            return types.SimpleNamespace(blocks=[])

    with pytest.raises(RuntimeError, match="MiniMax H3 models only"):
        JR_H3_UnifiedAcceleration().patch(**required_values(model=NonH3()))
    assert calls == []


def test_upstream_clone_and_composition_markers_survive_full_chain(monkeypatch):
    class NodeOutput:
        def __init__(self, value):
            self.args = (value,)

        @property
        def result(self):
            return self.args

    class Model(FakeH3Model):
        def __init__(self, label="model"):
            super().__init__(label)
            self.model_options = {"transformer_options": {}}
            self.object_patches = {}

        def clone(self):
            clone = Model(self.label + "-clone")
            clone.model_options = {
                "transformer_options": dict(self.model_options["transformer_options"]),
            }
            clone.object_patches = dict(self.object_patches)
            return clone

    class Sage:
        FUNCTION = "patch"

        def patch(self, model, sage_attention, allow_compile=False):
            clone = model.clone()
            clone.model_options["transformer_options"]["optimized_attention_override"] = "sage"
            return (clone,)

    class Low:
        @classmethod
        def execute(cls, model, head_chunks):
            clone = model.clone()
            clone.model_options["transformer_options"]["sol_take_forward"] = "low-forward"
            clone.object_patches["diffusion_model.blocks.0.attn.forward"] = "low-attn"
            return NodeOutput(clone)

    class FFN:
        @classmethod
        def execute(cls, model, chunks, seq_threshold):
            clone = model.clone()
            clone.object_patches["diffusion_model.blocks.0.mlp.forward"] = (chunks, seq_threshold)
            return NodeOutput(clone)

    class Sol:
        @classmethod
        def execute(cls, model, **kwargs):
            clone = model.clone()
            options = clone.model_options["transformer_options"]
            options["sol_previous"] = options.get("optimized_attention_override")
            options["optimized_attention_override"] = "sol"
            return NodeOutput(clone)

    monkeypatch.setattr(adapters, "_runtime_node_registry", lambda: {
        adapters.KJ_SAGE_NODE_ID: Sage,
        adapters.KJ_LOW_VRAM_NODE_ID: Low,
        adapters.KJ_FFN_NODE_ID: FFN,
        adapters.SOL_NODE_ID: Sol,
    })
    original = Model("original")
    result = JR_H3_UnifiedAcceleration().patch(**required_values(model=original))[0]
    options = result.model_options["transformer_options"]

    assert original.model_options == {"transformer_options": {}}
    assert original.object_patches == {}
    assert options["sol_previous"] == "sage"
    assert options["optimized_attention_override"] == "sol"
    assert options["sol_take_forward"] == "low-forward"
    assert result.object_patches["diffusion_model.blocks.0.attn.forward"] == "low-attn"
    assert result.object_patches["diffusion_model.blocks.0.mlp.forward"] == (4, 4096)
