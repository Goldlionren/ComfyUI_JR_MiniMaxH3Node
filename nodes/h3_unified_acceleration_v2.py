"""Opt-in Core sparse orchestration; the original Unified node stays unchanged."""

from __future__ import annotations

import copy
import inspect

from ..utils import h3_sparse_backend as sparse
from .h3_unified_acceleration import JR_H3_UnifiedAcceleration


class JR_H3_UnifiedAccelerationV2(JR_H3_UnifiedAcceleration):
    DESCRIPTION = (
        "Experimental H3 sparse backend selection. Start from a MODEL without attention patches. "
        "Legacy is the conservative default; select core to evaluate the native chunked producer. "
        "Auto may choose legacy before sampling, never after a CUDA/runtime failure. "
        "Core currently excludes TST, tau_profile, custom legacy quantization and allow_compile."
    )

    @classmethod
    def INPUT_TYPES(cls):
        schema = copy.deepcopy(super().INPUT_TYPES())
        schema["required"]["sparse_backend"] = (list(sparse.BACKENDS), {"default": "legacy_kijai"})
        schema["required"]["extra_tokens"] = ("INT", {"default": 256, "min": 0, "max": 256, "step": 64})
        schema["required"]["min_tokens"][1]["default"] = 12288
        return schema

    def patch(self, model, sparse_backend="legacy_kijai", extra_tokens=256, **kwargs):
        # Bind the stable legacy contract, including defaults, without duplicating its signature.
        bound = inspect.signature(JR_H3_UnifiedAcceleration.patch).bind(self, model, **kwargs)
        bound.apply_defaults()
        p = dict(bound.arguments)
        p.pop("self")
        p.pop("model")
        if not p["enable"]:
            return (model,)
        if "min_tokens" not in kwargs:
            p["min_tokens"] = 12288
        sparse.adapters.ensure_minimax_h3_model(model)
        sparse.validate_parameters(p, extra_tokens)
        sparse.validate_clean_input(model)
        p["dense_blocks"] = sparse.normalize_dense_blocks(p["dense_blocks"], len(model.get_model_object("diffusion_model").blocks))
        backend, reason = sparse.resolve_backend(model, sparse_backend, p, extra_tokens)
        sparse.preflight_stack(model, p, backend)
        # Never attach a marker/callback to the caller's MODEL, even when all passes bypass.
        patched = model.clone()
        legacy = dict(p)
        legacy["enable_sol_attn"] = backend == "legacy_kijai"
        patched = super().patch(patched, **legacy)[0]
        if backend == "core":
            sparse.check_core_block_patches(patched)
            patched = sparse.apply_core(patched, sparse.core_parameters(p, extra_tokens))
        sparse.attach_report(patched, sparse_backend, backend, reason, p, extra_tokens)
        return (patched,)
