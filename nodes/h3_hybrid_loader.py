"""Memory-efficient FL-base / selective-REF MiniMax H3 model loader."""

from __future__ import annotations

import logging
from typing import Any

from ..utils.h3_hybrid_plan import PROFILES, build_hybrid_plan, validate_plan_file_identity
from ..utils.h3_hybrid_selective_reader import read_selected_ref_tensors
from ..utils.h3_hybrid_tensor_family import H3HybridCompatibilityError

LOGGER = logging.getLogger(__name__)
WEIGHT_DTYPES = ("default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2")


def _preferred_model(models: list[str], marker: str) -> str | None:
    return next((name for name in models if marker in name.lower()), models[0] if models else None)


def _model_options(weight_dtype: str) -> dict[str, Any]:
    if weight_dtype not in WEIGHT_DTYPES:
        raise H3HybridCompatibilityError(f"Unsupported weight_dtype {weight_dtype!r}.")
    if weight_dtype == "default":
        return {}
    import torch

    options: dict[str, Any] = {}
    if weight_dtype in {"fp8_e4m3fn", "fp8_e4m3fn_fast"}:
        options["dtype"] = torch.float8_e4m3fn
        if weight_dtype == "fp8_e4m3fn_fast":
            options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        options["dtype"] = torch.float8_e5m2
    return options


def _stock_load(path: str, weight_dtype: str, disable_dynamic: bool = False):
    import comfy.sd

    return comfy.sd.load_diffusion_model(
        path,
        model_options=_model_options(weight_dtype),
        disable_dynamic=disable_dynamic,
    )


def load_h3_hybrid_model(
    fl_path: str,
    ref_path: str,
    profile: str,
    weight_dtype: str,
    block_range_start: int,
    block_range_end: int,
    final_adaln_from_ref: bool,
    custom_ref: str,
    custom_fl: str,
    disable_dynamic: bool = False,
):
    """Load one stock MODEL from an FL-native state dict plus selected REF families."""

    if profile not in PROFILES:
        raise H3HybridCompatibilityError(f"Unknown Hybrid Loader profile {profile!r}.")
    if profile == "Pure FL":
        return _stock_load(fl_path, weight_dtype, disable_dynamic=disable_dynamic)
    if profile == "Pure REF":
        return _stock_load(ref_path, weight_dtype, disable_dynamic=disable_dynamic)
    if fl_path == ref_path:
        LOGGER.warning("JR H3 Hybrid Loader: FL and REF resolve to the same checkpoint; using stock FL load.")
        return _stock_load(fl_path, weight_dtype, disable_dynamic=disable_dynamic)

    import comfy.sd
    import comfy.utils

    plan = build_hybrid_plan(
        fl_path,
        ref_path,
        profile=profile,
        weight_dtype=weight_dtype,
        block_range_start=block_range_start,
        block_range_end=block_range_end,
        final_adaln_from_ref=final_adaln_from_ref,
        custom_ref=custom_ref,
        custom_fl=custom_fl,
    )
    LOGGER.info(
        "JR H3 Hybrid Loader plan=%s profile=%s blocks=%s families=%d tensors=%d REF=%.2f MiB",
        plan.fingerprint,
        plan.profile,
        (
            f"{plan.selected_blocks[0]}-{plan.selected_blocks[-1]}"
            if plan.selected_blocks
            else "custom"
        ),
        len(plan.selected_families),
        len(plan.selected_keys),
        plan.selected_bytes / 1024**2,
    )
    for warning in plan.warnings:
        LOGGER.warning("JR H3 Hybrid Loader plan=%s: %s", plan.fingerprint, warning)

    validate_plan_file_identity(plan)
    fl_state, fl_metadata = comfy.utils.load_torch_file(fl_path, return_metadata=True)
    selected_ref = read_selected_ref_tensors(plan)
    for key, tensor in selected_ref.items():
        if key not in fl_state:
            raise H3HybridCompatibilityError(f"Selected FL tensor disappeared during native load: {key}.")
        fl_state[key] = tensor
    del selected_ref

    model = comfy.sd.load_diffusion_model_state_dict(
        fl_state,
        model_options=_model_options(weight_dtype),
        metadata=fl_metadata,
        disable_dynamic=disable_dynamic,
    )
    if model is None:
        raise H3HybridCompatibilityError(
            f"ComfyUI model detection failed for hybrid profile {profile!r} "
            f"using {plan.fl_path.name!r} and {plan.ref_path.name!r}."
        )
    model.cached_patcher_init = (
        load_h3_hybrid_model,
        (
            fl_path,
            ref_path,
            profile,
            weight_dtype,
            block_range_start,
            block_range_end,
            final_adaln_from_ref,
            custom_ref,
            custom_fl,
        ),
    )
    attachments = getattr(model, "attachments", None)
    if isinstance(attachments, dict):
        attachments["jr_h3_hybrid_plan"] = plan
    return model


class JR_H3_HybridLoader:
    CATEGORY = "JR MiniMax H3/Loaders"
    FUNCTION = "load_model"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = (
        "Loads FL2VA as the only full native base and selectively overlays compatible "
        "REF2VA AdaLN tensor families before stock ComfyUI model construction."
    )

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        models = folder_paths.get_filename_list("diffusion_models")
        fl_default = _preferred_model(models, "fl2va")
        ref_default = _preferred_model(models, "ref2va")
        fl_options: tuple[Any, ...] = (models, {"default": fl_default}) if fl_default else (models,)
        ref_options: tuple[Any, ...] = (models, {"default": ref_default}) if ref_default else (models,)
        return {
            "required": {
                "fl_model_name": fl_options,
                "ref_model_name": ref_options,
                "profile": (list(PROFILES), {"default": "Recommended"}),
                "weight_dtype": (list(WEIGHT_DTYPES), {"default": "default", "advanced": True}),
                "block_range_start": ("INT", {"default": 25, "min": 0, "max": 49, "step": 1, "advanced": True}),
                "block_range_end": ("INT", {"default": 49, "min": 0, "max": 49, "step": 1, "advanced": True}),
                "final_adaln_from_ref": ("BOOLEAN", {"default": False, "advanced": True}),
                "custom_ref": (
                    "STRING",
                    {"default": "", "multiline": True, "advanced": True, "dynamicPrompts": False},
                ),
                "custom_fl": (
                    "STRING",
                    {"default": "", "multiline": True, "advanced": True, "dynamicPrompts": False},
                ),
            }
        }

    def load_model(
        self,
        fl_model_name: str,
        ref_model_name: str,
        profile: str = "Recommended",
        weight_dtype: str = "default",
        block_range_start: int = 25,
        block_range_end: int = 49,
        final_adaln_from_ref: bool = False,
        custom_ref: str = "",
        custom_fl: str = "",
    ):
        import folder_paths

        if profile not in PROFILES:
            raise H3HybridCompatibilityError(f"Unknown Hybrid Loader profile {profile!r}.")
        if profile == "Pure FL":
            fl_path = folder_paths.get_full_path_or_raise("diffusion_models", fl_model_name)
            return (_stock_load(fl_path, weight_dtype),)
        if profile == "Pure REF":
            ref_path = folder_paths.get_full_path_or_raise("diffusion_models", ref_model_name)
            return (_stock_load(ref_path, weight_dtype),)
        fl_path = folder_paths.get_full_path_or_raise("diffusion_models", fl_model_name)
        ref_path = folder_paths.get_full_path_or_raise("diffusion_models", ref_model_name)
        model = load_h3_hybrid_model(
            fl_path,
            ref_path,
            profile,
            weight_dtype,
            block_range_start,
            block_range_end,
            final_adaln_from_ref,
            custom_ref,
            custom_fl,
        )
        return (model,)
