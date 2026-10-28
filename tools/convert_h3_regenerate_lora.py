"""Convert the published Diffusers adapter to native H3 linear names.

No base model is rewritten. Q/K/V deltas are combined by exact rank
concatenation; SwiGLU output halves are swapped back to ComfyUI order.
Sharp's separate time-curve branch remains explicitly represented.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOTS = {
    "proj_in": "video_patch_proj", "audio_proj_in": "audio_patch_proj",
    "context_embedder": "condition_proj",
    "time_embedder.linear_1": "time_embedder.proj_in",
    "time_embedder.linear_2": "time_embedder.proj_out",
    "norm_out.linear": "final_layer.adaln_proj.linear",
    "proj_out": "final_layer.video_out", "audio_proj_out": "final_layer.audio_out",
}


def native_name(name):
    for source, target in ROOTS.items():
        if name == source:
            return target
    name = name.replace("transformer_blocks.", "blocks.").replace(
        "token_refiner.refiner_blocks.", "token_refiner.blocks.")
    return name.replace(".attn.to_out.0", ".attn.out_proj").replace(
        ".ff.net.0.proj", ".mlp.fc1").replace(".ff.net.2", ".mlp.fc2")


def convert_linear_pairs(raw):
    roots = sorted(k[:-len(".lora_A.weight")] for k in raw if k.endswith(".lora_A.weight"))
    if set(raw) != {r + s for r in roots for s in (".lora_A.weight", ".lora_B.weight")}:
        raise ValueError("Expected only complete LoRA A/B pairs")
    result, qkv = {}, {}
    for root in roots:
        if not root.startswith("transformer."):
            raise ValueError(f"Unexpected root: {root}")
        name = native_name(root[len("transformer."):])
        a, b = raw[root + ".lora_A.weight"], raw[root + ".lora_B.weight"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"Invalid pair: {root}")
        if ".attn.to_" in name:
            stem, part = name.rsplit(".to_", 1)
            if part not in ("q", "k", "v"):
                raise ValueError(f"Unknown attention projection: {name}")
            qkv.setdefault(stem, {})[part] = (a, b)
            continue
        if name.endswith(".mlp.fc1"):
            if b.shape[0] % 2:
                raise ValueError("SwiGLU projection must have even rows")
            b = torch.cat(b.chunk(2, dim=0)[::-1], dim=0)
        prefix = "diffusion_model." + name
        result[prefix + ".lora_down.weight"] = a.contiguous()
        result[prefix + ".lora_up.weight"] = b.contiguous()
    for stem, parts in qkv.items():
        if set(parts) != {"q", "k", "v"}:
            raise ValueError(f"Incomplete QKV: {stem}")
        pairs = [parts[p] for p in ("q", "k", "v")]
        prefix = "diffusion_model." + stem + ".qkv_proj"
        result[prefix + ".lora_down.weight"] = torch.cat([a for a, _ in pairs], dim=0)
        result[prefix + ".lora_up.weight"] = torch.block_diag(*[b for _, b in pairs])
    return result


def convert_bundle(lora_path, curve_path, table_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = load_file(str(lora_path))
    result = convert_linear_pairs(raw)
    del raw
    curves = load_file(str(curve_path))
    roots = sorted(k[:-len(".curve_A.weight")] for k in curves if k.endswith(".curve_A.weight"))
    if len(roots) != 50 or len(curves) != 100:
        raise ValueError("Expected all 50 Sharp curve pairs")
    for root in roots:
        a, b = curves[root + ".curve_A.weight"].float(), curves[root + ".curve_B.weight"].float()
        name = native_name(root.removeprefix("transformer."))
        if a.shape[1] != 8 or a.shape[0] != b.shape[1] or not name.endswith(".adaln_proj.linear"):
            raise ValueError(f"Invalid curve branch: {root}")
        # A rank-64 factorization with only 8 input coordinates can be folded
        # to a small [out,8] matrix without discarding any branch.
        result["jr_h3_curve." + name + ".weight"] = (b.double() @ a.double()).float().contiguous()
    table = torch.frombuffer(bytearray(Path(table_path).read_bytes()), dtype=torch.float32).clone()
    result["jr_h3_curve.table"] = table.reshape(1025, 8)
    metadata = {
        "format": "jr-h3-regenerate-v1", "base": "MiniMax-H3 full time-embedding BF16",
        "linear_scale": "alpha equals rank; Refiner+Turbo+Sharp strengths already folded",
        "curve": "50 Sharp branches; requires JR H3 Regenerate LoRA loader for full effect",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        save_file(result, str(temporary), metadata=metadata)
        temporary.rename(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"output": str(output), "tensors": len(result), "bytes": output.stat().st_size, **metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora", required=True, type=Path)
    parser.add_argument("--curves", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    print(json.dumps(convert_bundle(args.lora, args.curves, args.table, args.output), indent=2))


if __name__ == "__main__":
    sys.exit(main())
