from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.nodes import h3_hybrid_loader as loader
from safetensors.torch import save_file


def _state(value: float):
    state = {}
    for index in range(50):
        stem = f"blocks.{index}.adaln_proj.linear"
        state[f"{stem}.weight"] = torch.full((2, 2), value + index, dtype=torch.float16)
        state[f"{stem}.bias"] = torch.full((2,), value + index, dtype=torch.float16)
    state["final_layer.adaln_proj.linear.weight"] = torch.full((2, 2), value, dtype=torch.float16)
    state["final_layer.adaln_proj.linear.bias"] = torch.full((2,), value, dtype=torch.float16)
    state["unrelated.weight"] = torch.full((2, 2), value, dtype=torch.float16)
    return state


def test_pure_modes_resolve_and_load_only_selected_checkpoint(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "folder_paths.get_full_path_or_raise",
        lambda category, name: calls.append((category, name)) or f"X:/{name}",
    )
    monkeypatch.setattr(loader, "_stock_load", lambda path, dtype: (path, dtype))
    node = loader.JR_H3_HybridLoader()

    assert node.load_model("fl.safetensors", "ref.safetensors", profile="Pure FL") == (
        ("X:/fl.safetensors", "default"),
    )
    assert calls == [("diffusion_models", "fl.safetensors")]
    calls.clear()
    assert node.load_model("fl.safetensors", "ref.safetensors", profile="Pure REF") == (
        ("X:/ref.safetensors", "default"),
    )
    assert calls == [("diffusion_models", "ref.safetensors")]


def test_hybrid_preserves_native_fl_storage_metadata_and_cached_factory(tmp_path, monkeypatch):
    import comfy.sd
    import comfy.utils

    fl_path = tmp_path / "fl.safetensors"
    ref_path = tmp_path / "ref.safetensors"
    fl_state = _state(0)
    ref_state = _state(100)
    save_file(fl_state, fl_path, metadata={"origin": "FL"})
    save_file(ref_state, ref_path, metadata={"origin": "REF"})
    native_fl = {key: value.clone() for key, value in fl_state.items()}
    unselected = native_fl["unrelated.weight"]
    captured = {}
    fake_model = SimpleNamespace(cached_patcher_init=None, attachments={})

    def fake_native_load(path, return_metadata=False):
        assert path == str(fl_path)
        assert return_metadata is True
        return native_fl, {"origin": "FL"}

    def fake_construct(sd, *, model_options, metadata, disable_dynamic):
        captured.update(sd=sd, model_options=model_options, metadata=metadata, disable_dynamic=disable_dynamic)
        return fake_model

    monkeypatch.setattr(comfy.utils, "load_torch_file", fake_native_load)
    monkeypatch.setattr(comfy.sd, "load_diffusion_model_state_dict", fake_construct)

    model = loader.load_h3_hybrid_model(
        str(fl_path),
        str(ref_path),
        "Recommended",
        "default",
        25,
        49,
        False,
        "",
        "",
    )

    assert model is fake_model
    assert captured["sd"] is native_fl
    assert captured["sd"]["unrelated.weight"] is unselected
    assert captured["metadata"] == {"origin": "FL"}
    assert torch.all(captured["sd"]["blocks.25.adaln_proj.linear.weight"] == 125)
    assert torch.all(captured["sd"]["blocks.24.adaln_proj.linear.weight"] == 24)
    factory, args = model.cached_patcher_init
    assert factory is loader.load_h3_hybrid_model
    assert args == (
        str(fl_path), str(ref_path), "Recommended", "default", 25, 49, False, "", ""
    )
    inspect.signature(factory).bind(*args, disable_dynamic=True)
    assert model.attachments["jr_h3_hybrid_plan"].fingerprint


def test_same_checkpoint_hybrid_short_circuits_stock_loader(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(
        loader,
        "_stock_load",
        lambda path, dtype, disable_dynamic=False: calls.append((path, dtype, disable_dynamic)) or sentinel,
    )
    result = loader.load_h3_hybrid_model("same.safetensors", "same.safetensors", "Recommended", "default", 25, 49, False, "", "")
    assert result is sentinel
    assert calls == [("same.safetensors", "default", False)]


def test_unknown_profile_fails_before_checkpoint_resolution(monkeypatch):
    monkeypatch.setattr(
        "folder_paths.get_full_path_or_raise",
        lambda *args: pytest.fail("unknown profile must fail before resolving checkpoints"),
    )
    with pytest.raises(loader.H3HybridCompatibilityError, match="Unknown Hybrid Loader profile"):
        loader.JR_H3_HybridLoader().load_model(
            "fl.safetensors",
            "ref.safetensors",
            profile="not-a-profile",
        )


def test_input_and_output_contract(monkeypatch):
    monkeypatch.setattr(
        "folder_paths.get_filename_list",
        lambda category: ["model_fl2va.safetensors", "model_ref2va.safetensors"],
    )
    spec = loader.JR_H3_HybridLoader.INPUT_TYPES()
    assert list(spec["required"])[0:4] == ["fl_model_name", "ref_model_name", "profile", "weight_dtype"]
    assert spec["required"]["fl_model_name"][1]["default"] == "model_fl2va.safetensors"
    assert spec["required"]["ref_model_name"][1]["default"] == "model_ref2va.safetensors"
    assert spec["required"]["profile"][1]["default"] == "Recommended"
    assert loader.JR_H3_HybridLoader.RETURN_TYPES == ("MODEL",)
    assert loader.JR_H3_HybridLoader.CATEGORY == "JR MiniMax H3/Loaders"
