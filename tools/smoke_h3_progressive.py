"""Opt-in local smoke: tiny native H3 + installed learned upscaler on CUDA.

Does not load the full H3 checkpoint or submit a production queue job. This is
runtime/weight compatibility evidence, not a video quality benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--video-vae", type=Path, help="Optional installed H3 VAE: smoke all three guided modes together.")
    parser.add_argument("--tst-strength", type=float, default=0.0)
    parser.add_argument("--unified-plugins-root", type=Path, help="Opt-in: use installed KJNodes and Sol in this separate process.")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project.parent))
    sys.path.insert(0, str(args.comfy_root))

    import torch
    from aiohttp import web
    from comfy.ldm.minimax import model as h3_model
    from comfy.ldm.modules.attention import attention_pytorch
    from comfy.model_base import MiniMaxH3
    from comfy.model_patcher import CoreModelPatcher
    from comfy.nested_tensor import NestedTensor
    from comfy.samplers import ksampler
    from comfy.supported_models import MiniMaxH3 as H3Config
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

    import nodes as comfy_nodes
    import server

    server.PromptServer.instance = types.SimpleNamespace(
        routes=web.RouteTableDef(), node_replace_manager=types.SimpleNamespace(register=lambda value: None),
    )
    # Exercise the production path-based loader, not just package-import classes.
    sampler_path = str((args.comfy_root / "comfy_extras/nodes_custom_sampler.py").resolve())
    if not asyncio.run(comfy_nodes.load_custom_node(sampler_path, module_parent="comfy_extras")):
        raise RuntimeError("Could not load native custom-sampler nodes through ComfyUI's loader.")
    runtime_noise = comfy_nodes.NODE_CLASS_MAPPINGS["RandomNoise"].execute(123).result[0]
    if type(runtime_noise) is Noise_RandomNoise:
        raise AssertionError("Smoke did not exercise the distinct production NOISE class identity")

    progressive = importlib.import_module(f"{project.name}.utils.h3_progressive_sampler")
    if not torch.cuda.is_available():
        raise RuntimeError("This opt-in smoke requires CUDA for the installed neural upscaler.")
    h3_model.optimized_attention = attention_pytorch
    torch.manual_seed(42)
    hidden = 512 if args.unified_plugins_root else 128
    compute_dtype = torch.bfloat16 if args.unified_plugins_root else torch.float32
    config = H3Config(dict(hidden_size=hidden, num_layers=2, token_refiner_num_layers=0,
                           num_attention_heads=hidden // 128, attention_head_dim=128, ffn_hidden_size=hidden * 2,
                           text_dim=hidden, timestep_input_dim=32, time_embed_hidden_size=128,
                           time_embed_dim=64, dtype=compute_dtype))
    base = MiniMaxH3(config, device=torch.device("cpu"))
    with torch.no_grad():
        for parameter in base.diffusion_model.parameters():
            parameter.uniform_(-.02, .02)
        base.diffusion_model.rope.inv_freq.copy_(torch.linspace(.01, 1, 16))
    patcher = CoreModelPatcher(base, load_device=torch.device("cuda"), offload_device=torch.device("cpu"))
    sol_stats = None
    if args.unified_plugins_root:
        def load_module(name, path, package=False):
            spec = importlib.util.spec_from_file_location(name, path,
                    submodule_search_locations=[str(path.parent)] if package else None)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        root = args.unified_plugins_root
        kj = load_module("jr_smoke_kj_h3", root / "comfyui-KJNodes/nodes/minimax_nodes.py")
        sage = load_module("jr_smoke_kj_sage", root / "comfyui-KJNodes/nodes/model_optimization_nodes.py")
        sol = load_module("jr_smoke_sol", root / "ComfyUI-SolAttn_triton/__init__.py", package=True)
        comfy_nodes.NODE_CLASS_MAPPINGS.update(PathchSageAttentionKJ=sage.PathchSageAttentionKJ,
            MiniMaxLowVRAMAttention=kj.MiniMaxLowVRAMAttention, MiniMaxChunkFeedForward=kj.MiniMaxChunkFeedForward,
            SolAttnPatch=sol.SolAttnPatch)
        unified = importlib.import_module(f"{project.name}.nodes.h3_unified_acceleration")
        patcher = unified.JR_H3_UnifiedAcceleration().patch(patcher, enable_tst=args.tst_strength > 0,
                                                          tst_strength=args.tst_strength, ffn_seq_threshold=256)[0]
        # Force one real sparse case and one dense/Sage fallback, beyond the tiny
        # model's token count. This is an integration check, not a speed claim.
        tst = importlib.import_module(f"{project.name}.utils.h3_temporal_transport")
        options = dict(patcher.model_options["transformer_options"])
        layout = h3_model.PackedLayout(2, 4, 64, 64, 8)
        options.update(minimax_h3_layout=layout, block_index=1, sol_block=1)
        options[tst.CONTEXT_KEY] = dict(signature=layout.signature, progress=.5, seen=set())
        qkv = [torch.randn(1, 4, layout.seq_len, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        middle_sigma = patcher.get_model_object("model_sampling").percent_to_sigma(.5)
        for sigma in (1., middle_sigma):
            options["sigmas"] = torch.tensor([sigma])
            out = attention_pytorch(*qkv, 4, skip_reshape=True, transformer_options=options)
            if not torch.isfinite(out).all():
                raise AssertionError("Unified/TST attention produced non-finite values")
        if sol._stats["sparse"] < 1 or sol._stats["outside_range"] < 1 or sol._stats["errors"]:
            raise AssertionError(f"Did not validate error-free sparse and fallback paths: {sol._stats}")
        del qkv, out
        sol_stats = sol._stats
    elif args.tst_strength > 0:
        tst = importlib.import_module(f"{project.name}.utils.h3_temporal_transport")
        patcher = tst.apply_temporal_transport(patcher, strength=args.tst_strength)
    latent = {"samples": NestedTensor((torch.zeros(1, 24, 2, 8, 12), torch.zeros(1, 32, 2, 8)))}
    kwargs = dict(model=patcher, positive=[[torch.zeros(1, 2, hidden, dtype=compute_dtype), {}]],
                  noise=runtime_noise, sampler=ksampler("euler"),
                  sigmas=torch.tensor([1., .8, .5, .2, 0.]), latent_image=latent,
                  transition_step=2, lowres_scale=.5)
    if args.video_vae:
        import comfy.sd
        import comfy.utils

        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(args.video_vae)))
        pixels = torch.linspace(0, 1, 128 * 192 * 3).reshape(1, 128, 192, 3)
        first = vae.encode(pixels)
        last = vae.encode(1 - pixels)
        drive = importlib.import_module(f"{project.name}.utils.h3_audio_driven_latent_builder")
        kwargs["latent_image"], _ = drive.build_h3_audio_driven_latent(
            latent, {"samples": torch.linspace(-1, 1, 512).reshape(1, 32, 2, 8)})
        kwargs["positive"][0][1].update(
            minimax_keyframes=[dict(resolved_frame_index=0, latent=first), dict(resolved_frame_index=4, latent=last)],
            minimax_refs=[dict(kind="image", latent_h=8, latent_w=12, latent=first)],
        )
        kwargs.update(_guided=True, vae=vae)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result, status = progressive.sample_h3_progressive(**kwargs)
    repeated, _ = progressive.sample_h3_progressive(**kwargs)
    torch.cuda.synchronize()
    diffs = []
    for actual, second in zip(result["samples"].unbind(), repeated["samples"].unbind()):
        if not torch.isfinite(actual).all():
            raise AssertionError("Non-finite learned-upscaler output")
        torch.testing.assert_close(actual, second, rtol=0, atol=0)
        diffs.append(float((actual - second).abs().max()))
    if args.video_vae:
        assert torch.equal(result["samples"].unbind()[1], kwargs["latent_image"]["samples"].unbind()[1])
    print(json.dumps({"result": "PASS", "device": torch.cuda.get_device_name(),
                      "model": "random tiny native H3", "upscaler": "installed learned checkpoint",
                      "noise_class": f"{type(runtime_noise).__module__}.{type(runtime_noise).__qualname__}",
                      "guided_video_vae": str(args.video_vae) if args.video_vae else None,
                      "tst_strength": args.tst_strength, "sol_stats": sol_stats,
                      "shapes": [list(t.shape) for t in result["samples"].unbind()],
                      "repeat_max_abs_diff": diffs, "two_runs_seconds": time.perf_counter() - started,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "status": status}, indent=2))


if __name__ == "__main__":
    main()
