"""Independent VEDA orchestration. No changes to the Sol-H3 stack."""
from __future__ import annotations

import importlib
import inspect
import logging
import re

from . import h3_acceleration_adapters as adapters

LOGGER = logging.getLogger(__name__)
NODE_ID = "VedaSparseAttention"
CONFIG_KEY = "jr_h3_veda_attention"
DEFAULT_PREDICTOR = "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"
DEPENDENCY = "https://github.com/veda-sparse/Veda-on-ComfyUI"


def predictor_names():
    """Read an already-registered model folder; never register/download anything."""
    try:
        folders = importlib.import_module("folder_paths")
    except ModuleNotFoundError as exc:
        if exc.name != "folder_paths":
            raise
        return [DEFAULT_PREDICTOR]
    if "veda" not in folders.folder_names_and_paths:
        return [DEFAULT_PREDICTOR]
    names = [name for name in folders.get_filename_list("veda")
             if name.lower().endswith(".safetensors")]
    return sorted(set(names) | {DEFAULT_PREDICTOR})


def validate_clean_input(model):
    options = model.model_options.get("transformer_options", {})
    if CONFIG_KEY in options:
        raise ValueError("JR H3 VEDA: already installed; use one VEDA node per MODEL branch.")
    if any(key in options for key in (
        "jr_h3_unified_v2", "sol_compose", "sol_morton", "jr_h3_tst_config",
    )):
        raise ValueError(
            "JR H3 VEDA: Sol-H3/Core/TST is already installed. "
            "Use a separate MODEL branch from the loader/LoRA; keep the Sol-H3 branch intact."
        )
    if options.get("optimized_attention_override") is not None:
        raise ValueError(
            "JR H3 VEDA: input already has an attention override. Start from a clean "
            "loader/LoRA MODEL and select dense Sage on this node."
        )
    if options.get("patches_replace", {}).get("dit"):
        raise ValueError("JR H3 VEDA: input has DiT block replacements; use a separate clean MODEL branch.")
    for key in getattr(model, "object_patches", {}):
        if (re.fullmatch(r"diffusion_model\.blocks\.\d+\.(?:(?:attn|mlp)\.)?forward", key)
                or key in ("diffusion_model.forward", "diffusion_model._forward")):
            raise ValueError(f"JR H3 VEDA: existing forward patch {key}; use a clean MODEL branch.")
    for groups in getattr(model, "callbacks", {}).values():
        if "block_sparse_attention" in groups:
            raise ValueError("JR H3 VEDA: Core sparse callbacks are already installed.")
    for groups in getattr(model, "wrappers", {}).values():
        if any(key in groups for key in (
            "jr_h3_temporal_transport", "jr_h3_streaming", "jr_h3_adaptive_cache",
        )):
            raise ValueError("JR H3 VEDA: TST/Streaming/Cache wrapper is unsupported in this first version.")


def _veda_handler(unique_id, kwargs):
    node = adapters._runtime_node_registry().get(NODE_ID)
    if node is None:
        raise RuntimeError(
            f"JR H3 VEDA requires the official Veda-on-ComfyUI node ({NODE_ID}). "
            f"Install veda-sparse-attention, restart ComfyUI, and put the predictor in models/veda. {DEPENDENCY}"
        )
    prepare = getattr(node, "PREPARE_CLASS_CLONE", None)
    if not isinstance(node, type) or not callable(prepare):
        raise RuntimeError("JR H3 VEDA: unsupported upstream V3 API; update Veda-on-ComfyUI and ComfyUI >= 0.38.")
    # ComfyUI's own clone API isolates hidden state; never mutate the registered class.
    bound_node = prepare({"hidden_inputs": {"UNIQUE_ID": unique_id}})
    handler = getattr(bound_node, "execute", None)
    if not callable(handler):
        raise RuntimeError("JR H3 VEDA: the upstream execute method is unavailable.")
    try:
        inspect.signature(handler).bind(**kwargs)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("JR H3 VEDA: upstream input signature changed; update the JR VEDA adapter.") from exc
    return handler


def _attach_guard(model, record):
    options = model.model_options.setdefault("transformer_options", {})
    expected_override = options.get("optimized_attention_override")
    if not callable(expected_override):
        raise RuntimeError("JR H3 VEDA: upstream returned a MODEL without a usable attention override.")
    options[CONFIG_KEY] = record
    expected_blocks = dict(options.get("patches_replace", {}).get("dit", {}))
    expected_objects = dict(getattr(model, "object_patches", {}))

    def check(patcher):
        current = patcher.model_options.get("transformer_options", {})
        if (current.get("optimized_attention_override") is not expected_override
                or current.get("patches_replace", {}).get("dit", {}) != expected_blocks
                or getattr(patcher, "object_patches", {}) != expected_objects
                or current.get("jr_h3_tst_config")
                or current.get("jr_h3_unified_v2") or current.get("sol_compose")):
            raise RuntimeError(
                "JR H3 VEDA: a downstream node changed the attention stack. "
                "Keep VEDA last before the guider/sampler and keep Sol-H3 on a separate branch."
            )
        LOGGER.info("JR H3 VEDA: configured predictor=%s generated=%s reference=%s; "
                    "runtime sparse/dense statistics are reported by Veda", *record)

    model.add_callback_with_key("on_pre_run", CONFIG_KEY, check)
    check(model)


def apply(model, *, enable=True, predictor=DEFAULT_PREDICTOR,
          generated_sparsity="90%", reference_sparsity="90%",
          full_attention_layers="", full_attention_steps="", verbose=False,
          sage_attention="disabled", enable_low_vram_attention=True, head_chunks=4,
          enable_low_vram_ffn=True, ffn_chunks=4, ffn_seq_threshold=4096,
          unique_id=None):
    if not enable:
        return model, "VEDA bypassed"
    if sage_attention not in adapters.SAGE_ATTENTION_MODES:
        raise ValueError("JR H3 VEDA: invalid dense Sage mode.")
    for name, value, low, high in (
        ("head_chunks", head_chunks, 1, 56),
        ("ffn_chunks", ffn_chunks, 1, 64),
        ("ffn_seq_threshold", ffn_seq_threshold, 256, 262144),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"JR H3 VEDA: {name} must be an integer in [{low}, {high}].")
    adapters.ensure_minimax_h3_model(model)
    validate_clean_input(model)
    kwargs = dict(model=model, predictor=predictor,
                  generated_sparsity=generated_sparsity, reference_sparsity=reference_sparsity,
                  full_attention_layers=full_attention_layers, full_attention_steps=full_attention_steps,
                  verbose=verbose)
    # Resolve and bind VEDA before any memory/dense pass is applied.
    handler = _veda_handler(unique_id, kwargs)
    patched = model.clone()
    if sage_attention != "disabled":
        patched = adapters.apply_sage(patched, sage_attention=sage_attention, allow_compile=False)
    if enable_low_vram_attention:
        patched = adapters.apply_h3_low_vram_attention(patched, head_chunks=head_chunks)
    if enable_low_vram_ffn:
        patched = adapters.apply_h3_chunk_ffn(patched, chunks=ffn_chunks, seq_threshold=ffn_seq_threshold)
    kwargs["model"] = patched
    try:
        patched = adapters.normalize_model_output(handler(**kwargs), NODE_ID)
    except Exception as exc:
        if "keep_ratio" in str(exc):
            raise RuntimeError(
                "JR H3 VEDA: this predictor uses metadata the installed VEDA loader cannot read "
                "(keep_ratio is missing). The new fixed-tile R2VA predictor requires an upstream "
                "loader update; use the supported T2VA predictor meanwhile. No budget was substituted."
            ) from exc
        raise
    _attach_guard(patched, (predictor, generated_sparsity, reference_sparsity))
    return patched, (
        f"VEDA configured: {predictor}; generated={generated_sparsity}, reference={reference_sparsity}. "
        "This is configuration status; inspect the node's runtime progress/verbose output for sparse calls or fallback."
    )
