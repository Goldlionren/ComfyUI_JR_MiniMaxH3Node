"""Real local H3 hybrid loader and process-memory smoke.

This tool never downloads models. It is intentionally opt-in through explicit
checkpoint paths and reports observations as JSON without tensor contents.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import types
from pathlib import Path


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--fl", required=True, type=Path)
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--profile", default="Recommended")
    parser.add_argument("--construct-model", action="store_true")
    parser.add_argument("--enable-aimdo", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.comfy_root))
    sys.path.insert(0, str(args.project_root.parent))

    # Import the focused modules without executing the plugin root registration;
    # PromptServer.instance only exists inside a running ComfyUI server.
    package = types.ModuleType("ComfyUI_JR_MiniMaxH3Node")
    package.__path__ = [str(args.project_root)]
    sys.modules[package.__name__] = package

    from ComfyUI_JR_MiniMaxH3Node.nodes import h3_hybrid_loader as loader
    from ComfyUI_JR_MiniMaxH3Node.utils.h3_hybrid_plan import build_hybrid_plan

    plan = build_hybrid_plan(args.fl, args.ref, profile=args.profile)
    observations: dict[str, object] = {
        "profile": plan.profile,
        "fingerprint": plan.fingerprint,
        "fl_bytes": args.fl.stat().st_size,
        "ref_bytes": args.ref.stat().st_size,
        "selected_families": len(plan.selected_families),
        "selected_tensors": len(plan.selected_keys),
        "selected_ref_bytes": plan.selected_bytes,
        "selected_ref_ratio": plan.selected_bytes / args.ref.stat().st_size,
        "rss_before": _rss_bytes(),
        "model_constructed": False,
    }
    if not args.construct_model:
        print(json.dumps(observations, indent=2, sort_keys=True))
        return

    import comfy.memory_management
    import comfy.model_management
    import comfy.model_patcher
    import comfy.sd
    import comfy.utils

    if args.enable_aimdo:
        import comfy_aimdo.control

        comfy_aimdo.control.init()
        devices = comfy.model_management.get_all_torch_devices()
        try:
            initialized = comfy_aimdo.control.init_devices((device.index, 0) for device in devices)
        except TypeError:
            initialized = comfy_aimdo.control.init_devices(device.index for device in devices)
        if not initialized:
            raise RuntimeError("comfy-aimdo did not initialize for the local devices")
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True

    observations["aimdo_enabled"] = bool(comfy.memory_management.aimdo_enabled)
    original_fl_load = comfy.utils.load_torch_file
    original_reader = loader.read_selected_ref_tensors
    original_construct = comfy.sd.load_diffusion_model_state_dict

    def measured_fl_load(*values, **options):
        result = original_fl_load(*values, **options)
        observations["rss_after_fl_native_state"] = _rss_bytes()
        state = result[0] if isinstance(result, tuple) else result
        probe = state.get("blocks.0.attn.qkv_proj.weight")
        if probe is not None:
            storage = probe.untyped_storage()
            observations["fl_tensor_file_slice"] = hasattr(storage, "_comfy_tensor_file_slice")
            observations["fl_tensor_mmap_refs"] = hasattr(storage, "_comfy_tensor_mmap_refs")
        return result

    def measured_ref_read(*values, **options):
        result = original_reader(*values, **options)
        observations["rss_after_ref_selected_copy"] = _rss_bytes()
        return result

    def measured_construct(*values, **options):
        result = original_construct(*values, **options)
        observations["rss_after_model_constructed"] = _rss_bytes()
        return result

    comfy.utils.load_torch_file = measured_fl_load
    loader.read_selected_ref_tensors = measured_ref_read
    comfy.sd.load_diffusion_model_state_dict = measured_construct
    try:
        model = loader.load_h3_hybrid_model(
            str(args.fl),
            str(args.ref),
            args.profile,
            "default",
            25,
            49,
            False,
            "",
            "",
        )
        observations["model_constructed"] = True
        observations["patcher_type"] = f"{type(model).__module__}.{type(model).__name__}"
        observations["model_type"] = f"{type(model.model).__module__}.{type(model.model).__name__}"
        observations["cached_patcher_init"] = model.cached_patcher_init is not None
        observations["dynamic"] = bool(model.is_dynamic())
        del model
        gc.collect()
        observations["rss_after_cleanup"] = _rss_bytes()
    finally:
        comfy.utils.load_torch_file = original_fl_load
        loader.read_selected_ref_tensors = original_reader
        comfy.sd.load_diffusion_model_state_dict = original_construct
    print(json.dumps(observations, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
