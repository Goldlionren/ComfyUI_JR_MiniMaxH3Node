from __future__ import annotations

from pathlib import Path

import pytest
from ComfyUI_JR_MiniMaxH3Node.utils.h3_hybrid_plan import build_hybrid_plan
from ComfyUI_JR_MiniMaxH3Node.utils.h3_hybrid_tensor_family import H3HybridCompatibilityError

MODEL_ROOT = Path(r"F:\ComfyUI-aki-v3\ComfyUI\models\diffusion_models")
REAL_PAIRS = (
    ("bf16", "minimax_h3_fl2va_bf16.safetensors", "minimax_h3_ref2va_bf16.safetensors", 50, 13_010_457_600),
    ("int8", "minimax_h3_fl2va_int8_convrot.safetensors", "minimax_h3_ref2va_int8_convrot.safetensors", 100, 6_517_326_575),
    (
        "pruned_int8",
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        50,
        43_545_600,
    ),
)


@pytest.mark.parametrize(("kind", "fl_name", "ref_name", "tensor_count", "selected_bytes"), REAL_PAIRS)
def test_real_h3_headers_and_recommended_accounting(kind, fl_name, ref_name, tensor_count, selected_bytes):
    fl = MODEL_ROOT / fl_name
    ref = MODEL_ROOT / ref_name
    if not fl.is_file() or not ref.is_file():
        pytest.skip(f"Local {kind} H3 pair is not installed")
    plan = build_hybrid_plan(fl, ref)
    assert plan.selected_blocks == tuple(range(25, 50))
    assert len(plan.selected_families) == 25
    assert len(plan.selected_keys) == tensor_count
    assert plan.selected_bytes == selected_bytes
    assert plan.selected_bytes < ref.stat().st_size


def test_real_cross_quant_family_is_rejected():
    fl = MODEL_ROOT / "minimax_h3_fl2va_int8_convrot.safetensors"
    ref = MODEL_ROOT / "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    if not fl.is_file() or not ref.is_file():
        pytest.skip("Local int8/pruned H3 checkpoints are not installed")
    with pytest.raises(H3HybridCompatibilityError, match="mismatch"):
        build_hybrid_plan(fl, ref)
