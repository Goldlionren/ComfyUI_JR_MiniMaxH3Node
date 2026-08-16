from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.utils.h3_hybrid_plan import build_hybrid_plan
from ComfyUI_JR_MiniMaxH3Node.utils.h3_hybrid_tensor_family import H3HybridCompatibilityError
from safetensors.torch import save_file


def _state(*, quantized: bool = False, extra: str | None = None) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for index in range(50):
        stem = f"blocks.{index}.adaln_proj.linear"
        if quantized:
            state[f"{stem}.weight"] = torch.full((2, 2), index % 100, dtype=torch.int8)
            state[f"{stem}.weight_scale"] = torch.ones((2, 1), dtype=torch.float32)
            state[f"{stem}.comfy_quant"] = torch.tensor([1, 2, 3], dtype=torch.uint8)
        else:
            state[f"{stem}.weight"] = torch.full((2, 2), float(index), dtype=torch.float16)
        state[f"{stem}.bias"] = torch.full((2,), float(index), dtype=torch.float16)
    state["final_layer.adaln_proj.linear.weight"] = torch.ones((2, 2), dtype=torch.float16)
    state["final_layer.adaln_proj.linear.bias"] = torch.ones((2,), dtype=torch.float16)
    state["unrelated.weight"] = torch.ones((1,), dtype=torch.float16)
    if extra:
        state[extra] = torch.ones((1,), dtype=torch.float16)
    return state


def _pair(tmp_path: Path, *, quantized: bool = False) -> tuple[Path, Path]:
    fl = tmp_path / "FL 模型.safetensors"
    ref = tmp_path / "REF 模型.safetensors"
    save_file(_state(quantized=quantized, extra="fl_only.weight"), fl, metadata={"source": "FL"})
    save_file(_state(quantized=quantized, extra="ref_only.weight"), ref, metadata={"source": "REF"})
    return fl, ref


def test_recommended_and_all_profiles_are_deterministic(tmp_path):
    fl, ref = _pair(tmp_path)
    recommended = build_hybrid_plan(fl, ref)
    assert recommended.selected_blocks == tuple(range(25, 50))
    assert len(recommended.selected_families) == 25
    assert len(recommended.selected_keys) == 50
    assert recommended.final_adaln_source == "FL"
    assert recommended.selected_keys == tuple(sorted(recommended.selected_keys))
    assert recommended.fingerprint == build_hybrid_plan(fl, ref).fingerprint

    all_blocks = build_hybrid_plan(fl, ref, profile="All Block AdaLN")
    assert all_blocks.selected_blocks == tuple(range(50))
    assert len(all_blocks.selected_families) == 50
    assert all_blocks.final_adaln_source == "FL"

    with_final = build_hybrid_plan(fl, ref, profile="All Block AdaLN + Final")
    assert len(with_final.selected_families) == 51
    assert with_final.final_adaln_source == "REF"


def test_custom_range_and_invalid_range(tmp_path):
    fl, ref = _pair(tmp_path)
    plan = build_hybrid_plan(
        fl,
        ref,
        profile="Custom Range",
        block_range_start=7,
        block_range_end=9,
        final_adaln_from_ref=True,
    )
    assert plan.selected_blocks == (7, 8, 9)
    assert plan.final_adaln_source == "REF"
    with pytest.raises(H3HybridCompatibilityError, match="Invalid block range"):
        build_hybrid_plan(fl, ref, profile="Custom Range", block_range_start=10, block_range_end=9)


def test_advanced_custom_ref_and_family_level_fl_override(tmp_path):
    fl, ref = _pair(tmp_path)
    plan = build_hybrid_plan(
        fl,
        ref,
        profile="Advanced Custom",
        custom_ref="blocks.48., blocks.49.",
        custom_fl="blocks.48.adaln_proj.linear.bias",
    )
    assert plan.selected_blocks == (49,)
    assert all(key.startswith("blocks.49.") for key in plan.selected_keys)


def test_unrelated_global_key_mismatch_is_allowed(tmp_path):
    fl, ref = _pair(tmp_path)
    plan = build_hybrid_plan(fl, ref)
    assert "fl_only.weight" not in plan.selected_keys
    assert "ref_only.weight" not in plan.selected_keys


def test_selected_family_shape_dtype_and_quant_mismatch_fail_closed(tmp_path):
    fl = tmp_path / "fl.safetensors"
    ref = tmp_path / "ref.safetensors"
    fl_state = _state(quantized=True)
    ref_state = _state(quantized=True)
    del ref_state["blocks.25.adaln_proj.linear.weight_scale"]
    save_file(fl_state, fl)
    save_file(ref_state, ref)
    with pytest.raises(H3HybridCompatibilityError, match="Quant family mismatch"):
        build_hybrid_plan(fl, ref)

    ref_state = _state(quantized=True)
    ref_state["blocks.25.adaln_proj.linear.weight"] = torch.ones((3, 2), dtype=torch.int8)
    save_file(ref_state, ref)
    with pytest.raises(H3HybridCompatibilityError, match="shape mismatch"):
        build_hybrid_plan(fl, ref)

    ref_state = _state(quantized=True)
    ref_state["blocks.25.adaln_proj.linear.bias"] = torch.ones((2,), dtype=torch.float32)
    save_file(ref_state, ref)
    with pytest.raises(H3HybridCompatibilityError, match="dtype mismatch"):
        build_hybrid_plan(fl, ref)


def test_quant_siblings_co_travel(tmp_path):
    fl, ref = _pair(tmp_path, quantized=True)
    plan = build_hybrid_plan(
        fl,
        ref,
        profile="Advanced Custom",
        custom_ref="blocks.49.adaln_proj.linear.weight",
    )
    assert plan.selected_keys == (
        "blocks.49.adaln_proj.linear.bias",
        "blocks.49.adaln_proj.linear.comfy_quant",
        "blocks.49.adaln_proj.linear.weight",
        "blocks.49.adaln_proj.linear.weight_scale",
    )
