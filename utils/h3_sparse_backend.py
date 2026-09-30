"""Sparse backend policy. No ComfyUI/CUDA imports until execution."""

from __future__ import annotations

import importlib
import inspect
import logging
import math
import re

from . import h3_acceleration_adapters as adapters

LOGGER = logging.getLogger(__name__)
CORE_ID = "BlockSparseAttention"
CONFIG_KEY = "jr_h3_unified_v2"
BACKENDS = ("legacy_kijai", "core", "auto", "disabled")


class CoreUnavailable(RuntimeError):
    """Known missing capability; AUTO alone may choose legacy before sampling."""


def normalize_dense_blocks(spec: str, count: int) -> str:
    """Preserve legacy negative indices, clipping and reversed ranges; reject garbage."""
    blocks = set()
    for part in "".join(str(spec).split()).split(","):
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part)
        if match is None:
            raise ValueError(f"JR H3 v2: invalid dense_blocks entry {part!r}.")
        first = int(match[1])
        last = first if match[2] is None else int(match[2])
        first = first if first >= 0 else count + first
        last = last if last >= 0 else count + last
        lo, hi = sorted((first, last))
        blocks.update(range(max(0, lo), min(count - 1, hi) + 1))
    return ",".join(map(str, sorted(blocks)))


def validate_parameters(p, extra_tokens):
    for name, lo, hi in (("tau", 0, 4), ("start_percent", 0, 1),
                         ("end_percent", 0, 1), ("tst_strength", 0, 1)):
        value = p[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"JR H3 v2: {name} must be finite and in [{lo}, {hi}].")
    if p["start_percent"] > p["end_percent"]:
        raise ValueError("JR H3 v2: start_percent must not exceed end_percent.")
    for name, lo, hi in (("min_tokens", 0, 1 << 20), ("head_chunks", 1, 56),
                         ("ffn_chunks", 1, 64), ("ffn_seq_threshold", 256, 262144)):
        if type(p[name]) is not int or not lo <= p[name] <= hi:
            raise ValueError(f"JR H3 v2: {name} must be an integer in [{lo}, {hi}].")
    if type(extra_tokens) is not int or extra_tokens not in (0, 64, 128, 192, 256):
        raise ValueError("JR H3 v2: extra_tokens must be 0, 64, 128, 192 or 256.")
    if p["sink_conditioning"] not in ("off", "exact_kv", "exact_kv_and_rows"):
        raise ValueError("JR H3 v2: invalid sink_conditioning.")
    if p["sage_attention"] not in adapters.SAGE_ATTENTION_MODES:
        raise ValueError("JR H3 v2: invalid sage_attention.")


def validate_clean_input(model):
    """The v2 node owns its whole attention stack; never silently replace an owner."""
    options = model.model_options.get("transformer_options", {})
    if options.get(CONFIG_KEY):
        raise ValueError("JR H3 v2: already installed; use one Unified node per MODEL branch.")
    if options.get("jr_h3_tst_config") or any(
        "jr_h3_temporal_transport" in keyed for keyed in getattr(model, "wrappers", {}).values()
    ):
        raise ValueError("JR H3 v2: input MODEL already has TST; start from the unpatched MODEL and enable TST here.")
    if options.get("optimized_attention_override") is not None:
        raise ValueError("JR H3 v2: input MODEL already has an attention override; start before the other attention/Unified node.")
    if options.get("patches_replace", {}).get("dit"):
        raise ValueError("JR H3 v2: input MODEL has a DiT block replacement; refusing to overwrite it.")
    if any("block_sparse_attention" in keyed for keyed in getattr(model, "callbacks", {}).values()):
        raise ValueError("JR H3 v2: input MODEL already has Core sparse callbacks.")
    for key in getattr(model, "object_patches", {}):
        if re.fullmatch(r"diffusion_model\.blocks\.\d+\.(?:(?:attn|mlp)\.)?forward", key):
            raise ValueError(f"JR H3 v2: input MODEL has an existing block/attention/FFN patch: {key}.")


def core_parameters(p, extra_tokens):
    return {
        "selection": {"selection": "sol-attn", "tau": p["tau"]},
        **{key: p[key] for key in ("start_percent", "end_percent", "min_tokens", "dense_blocks", "sink_conditioning", "verbose")},
        "extra_tokens": extra_tokens,
    }


def validate_core_api(node_class, kwargs):
    handler = adapters._resolve_handler(node_class, CORE_ID, "execute")
    adapters._validate_call_signature(handler, CORE_ID, kwargs)
    try:
        schema = node_class.define_schema()
        inputs = {entry.id: entry for entry in schema.inputs}
        choice = next(option for option in inputs["selection"].options if option.key == "sol-attn")
        if not set(kwargs).issubset(inputs) or "tau" not in {entry.id for entry in choice.inputs}:
            raise ValueError("required inputs or sol-attn/tau missing")
    except (AttributeError, KeyError, StopIteration, TypeError, ValueError) as exc:
        raise adapters.H3AccelerationCompatibilityError("JR H3 v2: Core sparse schema drift; update the JR adapter.") from exc
    return handler


def check_core_support(model, kwargs):
    node_class = adapters._runtime_node_registry().get(CORE_ID)
    if node_class is None:
        raise CoreUnavailable("Core BlockSparseAttention is not registered")
    validate_core_api(node_class, {"model": model, **kwargs})
    try:
        ck = importlib.import_module("comfy_kitchen")
    except ModuleNotFoundError as exc:
        if exc.name == "comfy_kitchen":
            raise CoreUnavailable("comfy-kitchen is not installed") from exc
        raise
    for name in ("sol_attn_is_available", "sol_attn_chunked"):
        if not callable(getattr(ck, name, None)):
            raise CoreUnavailable(f"comfy-kitchen lacks {name}")
    native = importlib.import_module("comfy.ldm.minimax.model")
    diffusion = model.get_model_object("diffusion_model")
    if not isinstance(diffusion, native.MiniMaxH3Model):
        raise CoreUnavailable("MODEL does not use the native H3 class required by the Core producer")
    if any("attention" not in inspect.signature(block.forward).parameters for block in diffusion.blocks):
        raise CoreUnavailable("H3 block forward lacks the Core attention injection interface")
    mm = importlib.import_module("comfy.model_management")
    device = getattr(model, "load_device", None)
    if device is None:
        device = mm.get_torch_device()
    if not ck.sol_attn_is_available(device):
        raise CoreUnavailable(f"no compiled Sol kernel available on {device}")


def resolve_backend(model, requested, p, extra_tokens):
    if requested not in BACKENDS:
        raise ValueError(f"JR H3 v2: unknown sparse_backend {requested!r}.")
    if not p["enable_sol_attn"] or requested == "disabled":
        return "disabled", "sparse disabled by user"
    if requested == "legacy_kijai":
        return requested, "explicit legacy selection"
    constraints = []
    if p["enable_tst"] and p["tst_strength"] > 0:
        constraints.append("active TST bypassed by Core producer")
    if p["tau_profile"] and str(p["tau_profile"]).strip():
        constraints.append("per-layer tau_profile requires legacy")
    if p["morton"] or p["use_tma"] or not p["int8_qk"] or not p["int8_pv"]:
        constraints.append("explicit legacy quantization/Morton/TMA settings require legacy")
    if p["allow_compile"]:
        constraints.append("Core + allow_compile is not validated in v2")
    if constraints:
        reason = "; ".join(constraints)
        if requested == "core":
            raise ValueError(f"JR H3 v2: {reason}; use legacy_kijai or remove the incompatible settings.")
        return "legacy_kijai", reason
    try:
        check_core_support(model, core_parameters(p, extra_tokens))
    except CoreUnavailable as exc:
        if requested == "core":
            raise RuntimeError(f"JR H3 v2: requested core is unavailable: {exc}") from exc
        return "legacy_kijai", str(exc)
    return "core", "Core API and compiled kernel available; actual path determined per call"


def preflight_stack(model, p, backend):
    """Bind every enabled call before applying any patch."""
    calls = []
    if p["sage_attention"] != "disabled":
        calls.append((adapters.KJ_SAGE_NODE_ID, "patch", {"sage_attention": p["sage_attention"], "allow_compile": p["allow_compile"]}))
    if p["enable_low_vram_attention"]:
        calls.append((adapters.KJ_LOW_VRAM_NODE_ID, "execute", {"head_chunks": p["head_chunks"]}))
    if p["enable_low_vram_ffn"]:
        calls.append((adapters.KJ_FFN_NODE_ID, "execute", {"chunks": p["ffn_chunks"], "seq_threshold": p["ffn_seq_threshold"]}))
    if backend == "legacy_kijai":
        keys = ("tau", "start_percent", "end_percent", "min_tokens", "int8_qk", "int8_pv",
                "sink_conditioning", "morton", "morton_curve", "verbose", "use_tma", "dense_blocks", "tau_profile")
        calls.append((adapters.SOL_NODE_ID, "execute", {key: p[key] for key in keys}))
    for node_id, preferred, kwargs in calls:
        cls = adapters._resolve_node_class(node_id, "ComfyUI-SolAttn_triton" if node_id == adapters.SOL_NODE_ID else "ComfyUI-KJNodes")
        handler = adapters._resolve_handler(cls, node_id, preferred)
        adapters._validate_call_signature(handler, node_id, {"model": model, **kwargs})


def check_core_block_patches(model):
    """KJ older forwards cannot accept the Core attention injection argument."""
    for key, forward in model.object_patches.items():
        if re.fullmatch(r"diffusion_model\.blocks\.\d+\.forward", key):
            if "attention" not in inspect.signature(forward).parameters:
                raise adapters.H3AccelerationCompatibilityError(
                    f"JR H3 v2: {key} lacks attention= support; update KJNodes or disable LowVRAM attention."
                )


def apply_core(model, parameters):
    return adapters._invoke(node_id=CORE_ID, dependency="ComfyUI Core + comfy-kitchen", preferred="execute",
                            layer="Core Sol-Attn", kwargs={"model": model, **parameters})


def attach_report(model, requested, backend, reason, p, extra_tokens):
    # Immutable configuration, no shared counters. Core owns its per-sampling state.
    record = (requested, backend, reason, p["tau"], p["start_percent"], p["end_percent"],
              p["min_tokens"], extra_tokens if backend == "core" else None, p["dense_blocks"])
    model.model_options.setdefault("transformer_options", {})[CONFIG_KEY] = record
    expected_override = model.model_options["transformer_options"].get("optimized_attention_override")
    expected_blocks = dict(model.model_options["transformer_options"].get("patches_replace", {}).get("dit", {}))
    expected_objects = {key: value for key, value in model.object_patches.items() if key.startswith("diffusion_model.blocks.")}

    def report(_patcher):
        options = _patcher.model_options.get("transformer_options", {})
        blocks = options.get("patches_replace", {}).get("dit", {})
        objects = {key: value for key, value in _patcher.object_patches.items() if key.startswith("diffusion_model.blocks.")}
        if (options.get("optimized_attention_override") is not expected_override or blocks != expected_blocks
                or objects != expected_objects or (backend == "core" and options.get("jr_h3_tst_config"))):
            raise RuntimeError("JR H3 v2: a downstream node changed the attention stack; keep Unified v2 as the final attention patch.")
        LOGGER.info("JR H3 Unified v2: requested=%s resolved=%s reason=%s tau=%s range=%s-%s min_tokens=%s extra_tokens=%s dense_blocks=%s; actual_path=per-call (enable verbose for upstream trace)", *record)

    model.add_callback_with_key("on_pre_run", CONFIG_KEY, report)
    report(model)
