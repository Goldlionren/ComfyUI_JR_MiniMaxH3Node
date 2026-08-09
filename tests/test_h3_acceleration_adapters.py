import sys
import types

import pytest
from ComfyUI_JR_MiniMaxH3Node.utils import h3_acceleration_adapters as adapters


class FakeNodeOutput:
    def __init__(self, *args):
        self.args = args

    @property
    def result(self):
        return self.args or None


@pytest.mark.parametrize("style", ["direct", "tuple", "node_output"])
def test_normalize_model_output(style):
    model = object()
    value = {"direct": model, "tuple": (model,), "node_output": FakeNodeOutput(model)}[style]
    assert adapters.normalize_model_output(value, "upstream") is model


@pytest.mark.parametrize("value", [None, (), (object(), object()), FakeNodeOutput(), {}, {"model": object()}])
def test_normalize_rejects_malformed_outputs(value):
    with pytest.raises(adapters.H3AccelerationCompatibilityError, match="upstream"):
        adapters.normalize_model_output(value, "upstream")


def test_upstream_adapters_call_real_api_styles_by_keyword(monkeypatch):
    calls = []
    outputs = [object(), object(), object(), object()]

    class Sage:
        FUNCTION = "patch"

        def patch(self, model, sage_attention, allow_compile=False):
            calls.append(("sage", model, sage_attention, allow_compile))
            return (outputs[0],)

    class Low:
        @classmethod
        def execute(cls, model, head_chunks):
            calls.append(("low", model, head_chunks))
            return FakeNodeOutput(outputs[1])

    class FFN:
        @classmethod
        def execute(cls, model, chunks, seq_threshold):
            calls.append(("ffn", model, chunks, seq_threshold))
            return FakeNodeOutput(outputs[2])

    class Sol:
        @classmethod
        def execute(
            cls, model, tau, start_percent, end_percent, min_tokens, int8_qk,
            sink_conditioning, morton, morton_curve, dense_blocks, verbose,
            tau_profile=None, use_tma=False, int8_pv=True,
        ):
            calls.append((
                "sol", model, tau, start_percent, end_percent, min_tokens, int8_qk,
                sink_conditioning, morton, morton_curve, dense_blocks, verbose,
                tau_profile, use_tma, int8_pv,
            ))
            return FakeNodeOutput(outputs[3])

    monkeypatch.setattr(adapters, "_runtime_node_registry", lambda: {
        adapters.KJ_SAGE_NODE_ID: Sage,
        adapters.KJ_LOW_VRAM_NODE_ID: Low,
        adapters.KJ_FFN_NODE_ID: FFN,
        adapters.SOL_NODE_ID: Sol,
    })

    original = object()
    assert adapters.apply_sage(original, sage_attention="auto", allow_compile=True) is outputs[0]
    assert adapters.apply_h3_low_vram_attention(outputs[0], head_chunks=56) is outputs[1]
    assert adapters.apply_h3_chunk_ffn(outputs[1], chunks=64, seq_threshold=262144) is outputs[2]
    profile = "0-30=2.0\n39-42=0.9"
    assert adapters.apply_sol_attn(
        outputs[2], tau=4.0, start_percent=0.0, end_percent=1.0,
        min_tokens=1048576, int8_qk=False, int8_pv=False,
        sink_conditioning="off", morton=True, morton_curve="3d",
        verbose=True, use_tma=True, dense_blocks="0-2,-1", tau_profile=profile,
    ) is outputs[3]

    assert calls == [
        ("sage", original, "auto", True),
        ("low", outputs[0], 56),
        ("ffn", outputs[1], 64, 262144),
        (
            "sol", outputs[2], 4.0, 0.0, 1.0, 1048576, False, "off", True,
            "3d", "0-2,-1", True, profile, True, False,
        ),
    ]


@pytest.mark.parametrize("tau_profile", [None, "", "0-30=2.0\n39-42=0.9"])
def test_tau_profile_semantics_are_preserved(monkeypatch, tau_profile):
    seen = []

    class Sol:
        @classmethod
        def execute(cls, **kwargs):
            seen.append(kwargs["tau_profile"])
            return FakeNodeOutput(kwargs["model"])

    monkeypatch.setattr(adapters, "_runtime_node_registry", lambda: {adapters.SOL_NODE_ID: Sol})
    model = object()
    result = adapters.apply_sol_attn(model, tau_profile=tau_profile)
    assert result is model
    assert seen == [tau_profile]


def test_missing_dependency_and_api_drift_errors_are_clear(monkeypatch):
    monkeypatch.setattr(adapters, "_runtime_node_registry", lambda: {})
    with pytest.raises(adapters.H3AccelerationCompatibilityError, match="MiniMaxLowVRAMAttention"):
        adapters.apply_h3_low_vram_attention(object(), head_chunks=4)

    class Drifted:
        @classmethod
        def renamed(cls, model):
            return model

    monkeypatch.setattr(
        adapters,
        "_runtime_node_registry",
        lambda: {adapters.KJ_FFN_NODE_ID: Drifted},
    )
    with pytest.raises(adapters.H3AccelerationCompatibilityError, match="API drift"):
        adapters.apply_h3_chunk_ffn(object(), chunks=4, seq_threshold=4096)


def test_incompatible_signature_is_reported(monkeypatch):
    class OldLow:
        @classmethod
        def execute(cls, model):
            return FakeNodeOutput(model)

    monkeypatch.setattr(
        adapters,
        "_runtime_node_registry",
        lambda: {adapters.KJ_LOW_VRAM_NODE_ID: OldLow},
    )
    with pytest.raises(adapters.H3AccelerationCompatibilityError, match="call signature"):
        adapters.apply_h3_low_vram_attention(object(), head_chunks=4)


def test_runtime_import_failure_is_not_silently_ignored(monkeypatch):
    class Sage:
        FUNCTION = "patch"

        def patch(self, model, sage_attention, allow_compile=False):
            raise ModuleNotFoundError("sageattention")

    monkeypatch.setattr(
        adapters,
        "_runtime_node_registry",
        lambda: {adapters.KJ_SAGE_NODE_ID: Sage},
    )
    with pytest.raises(RuntimeError, match="Sage Attention.*runtime dependency"):
        adapters.apply_sage(object(), sage_attention="auto", allow_compile=False)


def test_non_h3_model_has_clear_error():
    class Model:
        def get_model_object(self, name):
            assert name == "diffusion_model"
            return types.SimpleNamespace(blocks=[])

    with pytest.raises(RuntimeError, match="MiniMax H3 models only"):
        adapters.ensure_minimax_h3_model(Model())


def test_adapter_module_import_does_not_load_gpu_dependencies():
    assert "sageattention" not in sys.modules
    assert "triton" not in sys.modules
