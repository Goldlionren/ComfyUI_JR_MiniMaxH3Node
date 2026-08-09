"""Runtime adapters for the upstream MiniMax H3 acceleration nodes.

This module deliberately imports neither ComfyUI nor any GPU dependency at
module import time.  Installed custom-node classes are resolved from ComfyUI's
runtime registry only when a corresponding acceleration pass is enabled.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any

SAGE_ATTENTION_MODES = (
    "disabled",
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
)

KJ_SAGE_NODE_ID = "PathchSageAttentionKJ"
KJ_LOW_VRAM_NODE_ID = "MiniMaxLowVRAMAttention"
KJ_FFN_NODE_ID = "MiniMaxChunkFeedForward"
SOL_NODE_ID = "SolAttnPatch"


class H3AccelerationCompatibilityError(RuntimeError):
    """An installed upstream node is missing or has an unsupported API."""


def _runtime_node_registry() -> Mapping[str, type]:
    try:
        comfy_nodes = importlib.import_module("nodes")
    except Exception as exc:
        raise H3AccelerationCompatibilityError(
            "JR H3 Unified Acceleration could not access ComfyUI's runtime node registry. "
            "Run this node inside a fully started ComfyUI instance."
        ) from exc

    registry = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(registry, Mapping):
        raise H3AccelerationCompatibilityError(
            "JR H3 Unified Acceleration found an incompatible ComfyUI node registry "
            "(NODE_CLASS_MAPPINGS is unavailable)."
        )
    return registry


def _resolve_node_class(node_id: str, dependency: str) -> type:
    node_class = _runtime_node_registry().get(node_id)
    if node_class is None:
        raise H3AccelerationCompatibilityError(
            f"JR H3 Unified Acceleration requires upstream node '{node_id}' from {dependency}, "
            "but it is not registered. Install/update that custom node and restart ComfyUI."
        )
    if not isinstance(node_class, type):
        raise H3AccelerationCompatibilityError(
            f"JR H3 Unified Acceleration found incompatible registration for upstream node "
            f"'{node_id}' from {dependency}: expected a class."
        )
    return node_class


def _resolve_handler(node_class: type, node_id: str, preferred: str) -> Callable[..., Any]:
    if preferred == "execute":
        execute = getattr(node_class, "execute", None)
        if callable(execute):
            return execute

    function_name = getattr(node_class, "FUNCTION", None)
    if isinstance(function_name, str):
        try:
            instance = node_class()
            function = getattr(instance, function_name)
        except Exception as exc:
            raise H3AccelerationCompatibilityError(
                f"JR H3 Unified Acceleration could not construct upstream node '{node_id}' "
                f"or resolve its FUNCTION '{function_name}'."
            ) from exc
        if callable(function):
            return function

    fallback = getattr(node_class, preferred, None)
    if callable(fallback):
        try:
            return fallback if inspect.ismethod(fallback) else getattr(node_class(), preferred)
        except Exception as exc:
            raise H3AccelerationCompatibilityError(
                f"JR H3 Unified Acceleration could not bind upstream method "
                f"'{node_id}.{preferred}'."
            ) from exc

    raise H3AccelerationCompatibilityError(
        f"JR H3 Unified Acceleration detected upstream API drift for '{node_id}': "
        f"expected callable '{preferred}' or a valid FUNCTION attribute."
    )


def _validate_call_signature(handler: Callable[..., Any], node_id: str, kwargs: Mapping[str, Any]) -> None:
    try:
        inspect.signature(handler).bind(**kwargs)
    except (TypeError, ValueError) as exc:
        raise H3AccelerationCompatibilityError(
            f"JR H3 Unified Acceleration detected an incompatible '{node_id}' call signature. "
            f"Expected keyword inputs: {', '.join(kwargs)}."
        ) from exc


def normalize_model_output(value: Any, node_id: str) -> Any:
    """Normalize MODEL, ``(MODEL,)`` and ``io.NodeOutput(MODEL)``."""
    if value is None:
        raise H3AccelerationCompatibilityError(
            f"JR H3 Unified Acceleration received no MODEL from upstream node '{node_id}'."
        )

    if isinstance(value, (tuple, list)):
        values = tuple(value)
    elif hasattr(value, "args") and hasattr(value, "result"):
        values = tuple(value.args)
    elif isinstance(value, Mapping):
        raise H3AccelerationCompatibilityError(
            f"JR H3 Unified Acceleration received an incompatible mapping output from '{node_id}'; "
            "expected a MODEL, one-item tuple, or NodeOutput."
        )
    else:
        return value

    if len(values) != 1 or values[0] is None:
        raise H3AccelerationCompatibilityError(
            f"JR H3 Unified Acceleration received an incompatible output from '{node_id}': "
            "expected exactly one MODEL."
        )
    return values[0]


def _invoke(
    *,
    node_id: str,
    dependency: str,
    preferred: str,
    layer: str,
    kwargs: Mapping[str, Any],
) -> Any:
    node_class = _resolve_node_class(node_id, dependency)
    handler = _resolve_handler(node_class, node_id, preferred)
    _validate_call_signature(handler, node_id, kwargs)
    try:
        result = handler(**kwargs)
    except H3AccelerationCompatibilityError:
        raise
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            f"JR H3 Unified Acceleration: {layer} is enabled, but its runtime dependency "
            f"could not be imported: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"JR H3 Unified Acceleration: {layer} failed through upstream node "
            f"'{node_id}': {exc}"
        ) from exc
    return normalize_model_output(result, node_id)


def ensure_minimax_h3_model(model: Any) -> None:
    """Fail clearly before upstream code can produce obscure block errors."""
    get_model_object = getattr(model, "get_model_object", None)
    if not callable(get_model_object):
        raise RuntimeError(
            "JR H3 Unified Acceleration requires a ComfyUI MODEL containing a MiniMax H3 diffusion model."
        )
    try:
        diffusion_model = get_model_object("diffusion_model")
    except Exception as exc:
        raise RuntimeError(
            "JR H3 Unified Acceleration could not inspect the MODEL diffusion_model; "
            "a MiniMax H3 MODEL is required."
        ) from exc

    blocks = getattr(diffusion_model, "blocks", None)
    first_block = blocks[0] if blocks is not None and len(blocks) > 0 else None
    attention = getattr(first_block, "attn", None)
    mlp = getattr(first_block, "mlp", None)
    is_h3 = (
        hasattr(diffusion_model, "rope_freqs")
        and hasattr(diffusion_model, "_forward")
        and first_block is not None
        and hasattr(attention, "qkv_proj")
        and hasattr(mlp, "fc1")
        and hasattr(mlp, "fc2")
    )
    if not is_h3:
        raise RuntimeError(
            "JR H3 Unified Acceleration supports MiniMax H3 models only; the connected MODEL "
            "does not expose the expected H3 blocks/attention/FFN structure."
        )


def apply_sage(model: Any, *, sage_attention: str, allow_compile: bool) -> Any:
    if sage_attention not in SAGE_ATTENTION_MODES or sage_attention == "disabled":
        raise ValueError(f"Invalid enabled SageAttention mode: {sage_attention!r}")
    return _invoke(
        node_id=KJ_SAGE_NODE_ID,
        dependency="ComfyUI-KJNodes",
        preferred="patch",
        layer=f"Sage Attention ({sage_attention})",
        kwargs={"model": model, "sage_attention": sage_attention, "allow_compile": allow_compile},
    )


def apply_h3_low_vram_attention(model: Any, *, head_chunks: int) -> Any:
    return _invoke(
        node_id=KJ_LOW_VRAM_NODE_ID,
        dependency="ComfyUI-KJNodes",
        preferred="execute",
        layer="MiniMax H3 Low VRAM Attention",
        kwargs={"model": model, "head_chunks": head_chunks},
    )


def apply_h3_chunk_ffn(model: Any, *, chunks: int, seq_threshold: int) -> Any:
    return _invoke(
        node_id=KJ_FFN_NODE_ID,
        dependency="ComfyUI-KJNodes",
        preferred="execute",
        layer="MiniMax H3 Chunk FeedForward",
        kwargs={"model": model, "chunks": chunks, "seq_threshold": seq_threshold},
    )


def apply_sol_attn(model: Any, **parameters: Any) -> Any:
    return _invoke(
        node_id=SOL_NODE_ID,
        dependency="ComfyUI-SolAttn_triton",
        preferred="execute",
        layer="Sol-Attn",
        kwargs={"model": model, **parameters},
    )
