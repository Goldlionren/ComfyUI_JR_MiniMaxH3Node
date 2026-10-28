"""Native ModelPatcher integration for the converted Regenerate bundle."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .h3_acceleration_adapters import ensure_minimax_h3_model

MARKER = "jr_h3_regenerate_lora"


class CurveContext:
    """One patch's transient coordinates, bounded to a diffusion forward."""

    def __init__(self, table):
        self.table = table.float().cpu()
        self.coords = None

    def capture(self, original, t):
        table = self.table.to(device=t.device)
        pos = t.float().clamp(0, 1) * (len(table) - 1)
        lo = pos.floor().long().clamp(max=len(table) - 2)
        self.coords = torch.lerp(table[lo], table[lo + 1], (pos - lo).unsqueeze(-1))
        return original(t)

    def forward(self, executor, *args, **kwargs):
        previous = self.coords
        self.coords = None
        try:
            return executor(*args, **kwargs)
        finally:
            self.coords = previous


class CaptureTime:
    def __init__(self, original, context):
        self.original, self.context = original, context

    def __call__(self, t):
        return self.context.capture(self.original, t)


class CurveProjection:
    def __init__(self, original, delta, context, strength):
        self.original, self.delta = original, delta.float().cpu()
        self.context, self.strength = context, strength

    def __call__(self, x):
        out = self.original(x)
        coords = self.context.coords
        if coords is None or coords.shape[0] != x.shape[0]:
            raise RuntimeError("JR H3 Regenerate: missing Sharp timestep coordinates")
        delta = F.linear(coords, self.delta.to(device=coords.device))
        return out + delta.to(out.dtype) * self.strength


def patch_regenerate_bundle(model, raw, strength=1.0):
    import comfy.lora

    ensure_minimax_h3_model(model)
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 2:
        raise ValueError("Regenerate LoRA strength must be between 0 and 2")
    if MARKER in model.model_options:
        raise ValueError("Regenerate LoRA already loaded; do not stack this merged adapter twice")
    dit = model.get_model_object("diffusion_model")
    if dit.use_adaln_curves:
        raise ValueError("Regenerate requires the full BF16 time-embedding model, not a pruned curve/INT8 model")
    if any("on_pre_run" == kind and any("unified" in str(k) or "sparse" in str(k) for k in keys)
           for kind, keys in model.callbacks.items()):
        raise ValueError("Place Regenerate LoRA BEFORE Unified acceleration")
    table = raw.get("jr_h3_curve.table")
    if table is None or tuple(table.shape) != (1025, 8):
        raise ValueError("Not a complete converted Regenerate bundle: missing Sharp curve table")
    curve_keys = {f"jr_h3_curve.blocks.{i}.adaln_proj.linear.weight" for i in range(len(dit.blocks))}
    actual = {k for k in raw if k.startswith("jr_h3_curve.") and k != "jr_h3_curve.table"}
    if actual != curve_keys:
        raise ValueError("Regenerate curve branches do not match all H3 blocks")
    linear = {k: v for k, v in raw.items() if not k.startswith("jr_h3_curve.")}
    roots = {k[:-len(".lora_down.weight")] for k in linear if k.endswith(".lora_down.weight")}
    if not roots or set(linear) != {r + s for r in roots for s in (".lora_down.weight", ".lora_up.weight")}:
        raise ValueError("Regenerate bundle must contain complete native LoRA pairs")
    stems = [f"blocks.{i}" for i in range(len(dit.blocks))]
    stems += [f"token_refiner.blocks.{i}" for i in range(len(dit.token_refiner.blocks))]
    expected_roots = {"diffusion_model." + stem + "." + target for stem in stems
                      for target in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")}
    if roots != expected_roots:
        raise ValueError("Incomplete Regenerate adapter: every DiT and token-refiner projection is required")
    key_map = comfy.lora.model_lora_keys_unet(model.model)
    for root in roots:
        if root not in key_map:
            raise ValueError(f"Unknown Regenerate model target: {root}")
        target = model.get_model_object(key_map[root])
        a, b = linear[root + ".lora_down.weight"], linear[root + ".lora_up.weight"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or target.shape != (b.shape[0], a.shape[1]):
            raise ValueError(f"Regenerate LoRA shape mismatch: {root}")
    for i, block in enumerate(dit.blocks):
        delta = raw[f"jr_h3_curve.blocks.{i}.adaln_proj.linear.weight"]
        if delta.shape != (block.adaln_proj.linear.weight.shape[0], 8):
            raise ValueError(f"Sharp curve shape mismatch at block {i}")
    if not strength:
        return model.clone()
    patches = comfy.lora.load_lora(linear, key_map)
    if len(patches) != len(roots):
        raise ValueError("Some Regenerate linear pairs could not be loaded")
    patched = model.clone()
    accepted = patched.add_patches(patches, strength)
    if len(accepted) != len(patches):
        raise ValueError("Some Regenerate model patches were rejected")
    context = CurveContext(table)
    time_key = "diffusion_model.time_embedder.forward"
    if time_key in patched.object_patches:
        raise ValueError("An existing time-embedding patch conflicts with Sharp curves")
    patched.add_object_patch(time_key, CaptureTime(model.get_model_object(time_key), context))
    for i, block in enumerate(dit.blocks):
        key = f"diffusion_model.blocks.{i}.adaln_proj.linear.forward"
        if key in patched.object_patches:
            raise ValueError(f"An existing AdaLN patch conflicts with Sharp: {key}")
        delta = raw[f"jr_h3_curve.blocks.{i}.adaln_proj.linear.weight"]
        patched.add_object_patch(key, CurveProjection(model.get_model_object(key), delta, context, strength))
    patched.add_wrapper_with_key("diffusion_model", MARKER, context.forward)
    patched.model_options[MARKER] = {"strength": strength, "linear_pairs": len(roots), "curves": len(curve_keys)}
    return patched
