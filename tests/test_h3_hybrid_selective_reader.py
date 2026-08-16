from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.utils.h3_hybrid_plan import build_hybrid_plan
from ComfyUI_JR_MiniMaxH3Node.utils.h3_hybrid_selective_reader import read_selected_ref_tensors
from ComfyUI_JR_MiniMaxH3Node.utils.h3_hybrid_tensor_family import H3HybridCompatibilityError
from safetensors.torch import save_file


def _checkpoint(path: Path, value: float) -> None:
    state = {}
    for index in range(50):
        stem = f"blocks.{index}.adaln_proj.linear"
        state[f"{stem}.weight"] = torch.full((2, 2), value + index, dtype=torch.float16)
        state[f"{stem}.bias"] = torch.full((2,), value + index, dtype=torch.float16)
        state[f"blocks.{index}.attn.weight"] = torch.full((1,), value, dtype=torch.float16)
    state["final_layer.adaln_proj.linear.weight"] = torch.ones((2, 2), dtype=torch.float16)
    state["final_layer.adaln_proj.linear.bias"] = torch.ones((2,), dtype=torch.float16)
    save_file(state, path)


class TrackingHandle:
    def __init__(self, tensors):
        self.tensors = tensors
        self.reads = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, key):
        self.reads.append(key)
        return self.tensors[key]


def test_reader_calls_get_tensor_for_selected_keys_only_and_owns_storage(tmp_path):
    fl = tmp_path / "fl.safetensors"
    ref = tmp_path / "ref.safetensors"
    _checkpoint(fl, 0)
    _checkpoint(ref, 100)
    plan = build_hybrid_plan(fl, ref)
    source = {
        spec.key: torch.ones(spec.shape, dtype=torch.float16)
        for spec in plan.selected_tensors
    }
    source["blocks.0.attn.weight"] = torch.zeros((1,), dtype=torch.float32)
    handle = TrackingHandle(source)

    owned = read_selected_ref_tensors(plan, opener=lambda _path: handle)

    assert handle.closed
    assert handle.reads == list(plan.selected_keys)
    assert "blocks.0.attn.weight" not in handle.reads
    assert set(owned) == set(plan.selected_keys)
    first = plan.selected_keys[0]
    assert owned[first].data_ptr() != source[first].data_ptr()
    source[first].fill_(9)
    assert torch.all(owned[first] == 1)


def test_reader_rejects_ref_replacement_after_header_plan(tmp_path):
    fl = tmp_path / "fl.safetensors"
    ref = tmp_path / "ref.safetensors"
    _checkpoint(fl, 0)
    _checkpoint(ref, 100)
    plan = build_hybrid_plan(fl, ref)
    with ref.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(H3HybridCompatibilityError, match="changed after HybridPlan"):
        read_selected_ref_tensors(plan)
