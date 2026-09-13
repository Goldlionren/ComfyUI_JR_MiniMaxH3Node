"""Experimental H3 Q pre-transform, independently implemented from TST equations.

Tang et al., Temporal State Transport, arXiv:2609.08505v1, equations 1-7.
H3 adaptation: spatially mean-pool target-video Q/K per latent frame/head.
This proxy is NOT the full AV attention matrix. No SelfLift code is vendored.
The immutable dispatcher delegates to the existing backend exactly once.
"""

from __future__ import annotations

import math

import torch

CONFIG_KEY = "jr_h3_tst_config"
SCHEDULE_KEY = "jr_h3_tst_schedule"
CONTEXT_KEY = "jr_h3_tst_forward"
WRAPPER_KEY = "jr_h3_temporal_transport"


def _error(message):
    return RuntimeError("JR H3 TST: " + message)


def validate_strength(strength):
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("JR H3 TST: tst_strength must be finite and in [0,1].")


def schedule_progress(schedule, sigma):
    """Nearest evaluation on the full schedule; no per-head/per-block counters."""
    if isinstance(schedule, torch.Tensor):
        schedule = schedule.detach().float().cpu().tolist()
    if not isinstance(schedule, (tuple, list)) or len(schedule) < 2:
        raise _error("A native sample_sigmas schedule is required.")
    if (any(not math.isfinite(s) or s < 0 for s in schedule) or
            any(a <= b for a, b in zip(schedule, schedule[1:])) or not math.isfinite(sigma)):
        raise _error("Expected finite, strictly decreasing sampling sigmas.")
    step = min(range(len(schedule) - 1), key=lambda i: abs(schedule[i] - sigma))
    return step / max(1, len(schedule) - 2)


def layer_step_weight(block_index, layers, progress):
    if type(block_index) is not int or not 0 <= block_index < layers:
        raise _error("Missing or invalid native block_index.")
    depth = (block_index + 1) / layers
    return (0.5 - 0.5 * math.cos(math.pi * depth)) * (0.5 + 0.5 * math.cos(math.pi * progress))


def spectral_tension(operator):
    """Normalized row entropy minus Gram spectral entropy, one value per head."""
    frames = operator.shape[-1]
    if frames < 2:
        return operator.new_zeros(operator.shape[:-2])
    a = operator.float()
    if not bool(torch.isfinite(a).all()):
        raise _error("Non-finite temporal statistics; no correction was applied.")
    row_entropy = -(a * a.clamp_min(1e-12).log()).sum(-1).mean(-1)
    gram = a @ a.transpose(-1, -2)
    density = gram / gram.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None].clamp_min(1e-12)
    eigenvalues = torch.linalg.eigvalsh(density).clamp_min(0)
    eigenvalues = eigenvalues / eigenvalues.sum(-1, keepdim=True).clamp_min(1e-12)
    entropy = -(eigenvalues * eigenvalues.clamp_min(1e-12).log()).sum(-1)
    return ((row_entropy - entropy) / math.log(frames)).clamp(-1, 1)


def transform_video_q(q, k, *, start, stop, frames, strength, weight, scale=None):
    """Non-mutating Q-only transform; never allocate token-by-token attention."""
    if strength == 0 or weight == 0 or frames < 2:
        return q
    if (q.ndim != 4 or q.shape != k.shape or q.shape[0] != 1 or
            q.device != k.device or q.dtype != k.dtype or not q.is_floating_point() or
            not 0 <= start < stop <= q.shape[2] or (stop - start) % frames):
        raise _error("Unsupported Q/K shape or target-video span.")
    rows = (stop - start) // frames
    shape = (q.shape[1], frames, rows, q.shape[-1])
    # Accumulate in fp32 without casting the full spatial Q/K buffers to fp32.
    pooled_q = q[0, :, start:stop].reshape(shape).mean(-2, dtype=torch.float32)
    pooled_k = k[0, :, start:stop].reshape(shape).mean(-2, dtype=torch.float32)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    operator = ((pooled_q @ pooled_k.transpose(-1, -2)) * scale).softmax(-1)
    gamma = (spectral_tension(operator) * (strength * weight)).exp().to(q.dtype)
    # Q can share a qkv allocation or a caller's cache: do not mutate that owner.
    result = q.clone()
    result[:, :, start:stop].mul_(gamma[None, :, None, None])
    return result


class TemporalTransport:
    """Immutable model configuration; all changing state lives in one forward."""

    def __init__(self, previous, strength, layers):
        self.previous = previous
        self.strength = strength
        self.layers = layers

    def __call__(self, func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        options = kwargs.get("transformer_options", {})
        context = options.get(CONTEXT_KEY)
        if context is None:
            raise _error("Forward context missing. Keep Unified as the final attention patch node.")
        if mask is not None or not skip_reshape:
            raise _error("The initial TST experiment requires unmasked native H3 head-major attention.")
        layout = options.get("minimax_h3_layout")
        signature = getattr(layout, "signature", None)
        spans = [s for s in getattr(layout, "segments", ()) if s[2] == "video"]
        if (signature != context["signature"] or len(spans) != 1 or
                getattr(layout, "seq_len", None) != q.shape[2]):
            raise _error("PackedLayout does not match the active target video.")
        start, stop, _ = spans[0]
        frames, height, width = signature[1:4]
        if stop - start != frames * ((height + 1) // 2) * ((width + 1) // 2):
            raise _error("Unsupported target-video patch grid.")
        block = options.get("block_index")
        weight = layer_step_weight(block, self.layers, context["progress"])
        context["seen"].add(block)
        q = transform_video_q(q, k, start=start, stop=stop, frames=frames,
                              strength=self.strength, weight=weight, scale=kwargs.get("scale"))
        target = func if self.previous is None else lambda *a, **kw: self.previous(func, *a, **kw)
        return target(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                      skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

    def forward(self, executor, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
        options = dict(transformer_options or {})
        if options.get("optimized_attention_override") is not self:
            raise _error("Another node replaced the composed attention backend. Put Unified last.")
        if options.get("sol_morton"):
            raise _error("Disable Sol Morton token reordering for this experiment.")
        video = x[0]
        if video.shape[0] != 1:
            raise _error("Only batch-1 H3 is supported.")
        sigma = float(timestep.flatten()[0]) / 1000.0
        progress = schedule_progress(options.get(SCHEDULE_KEY, options.get("sample_sigmas")), sigma)
        # Core pads spatial dimensions to its (1,2,2) patch grid before layout.
        signature = (context.shape[1], video.shape[2], (video.shape[3] + 1) // 2 * 2,
                     (video.shape[4] + 1) // 2 * 2, x[1].shape[-1])
        state = {"signature": signature, "progress": progress, "seen": set()}
        options[CONTEXT_KEY] = state
        try:
            result = executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
            # Cache can legitimately bypass blocks; without cache every block must reach us.
            if not (options.get("jr_h3_tst_cache_active") or state.get("cache_active")) and state["seen"] != set(range(self.layers)):
                raise _error("An attention patch bypassed TST; remove unsupported forward patches.")
            return result
        finally:
            state.clear()


def apply_temporal_transport(model, *, strength):
    validate_strength(strength)
    if strength == 0:
        return model
    from .h3_acceleration_adapters import ensure_minimax_h3_model

    ensure_minimax_h3_model(model)
    options = model.model_options.get("transformer_options", {})
    if options.get(CONFIG_KEY):
        raise _error("TST is already installed; use one Unified node on this MODEL path.")
    if options.get("sol_morton"):
        raise _error("Disable Morton before enabling TST.")
    diffusion = model.get_model_object("diffusion_model")
    if tuple(diffusion.patch_size) != (1, 2, 2):
        raise _error("Only the native H3 (1,2,2) patch grid is supported.")
    for key, forward in model.object_patches.items():
        if ".attn.forward" in key and not getattr(forward, "_uses_optimized_attention", False):
            raise _error(f"Unsupported attention forward patch: {key}.")
    patched = model.clone()
    dispatcher = TemporalTransport(options.get("optimized_attention_override"), float(strength), len(diffusion.blocks))
    target = patched.model_options.setdefault("transformer_options", {})
    target[CONFIG_KEY] = ("pooled-q-v1", float(strength), len(diffusion.blocks))
    target["optimized_attention_override"] = dispatcher
    patched.add_wrapper_with_key("diffusion_model", WRAPPER_KEY, dispatcher.forward)
    return patched
