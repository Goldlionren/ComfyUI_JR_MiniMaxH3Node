"""Opt-in separate-process RTX test with tiny native H3 and installed Unified.

No model downloads, production queue submissions, server starts or engine builds.
This validates runtime integration, not full-checkpoint generation quality.
"""

import argparse
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--unified-plugins-root", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--preset", default="TaoMate 5s Canonical", help="Planner preset; 10s/15s remain full single timelines")
    parser.add_argument("--report", type=Path, help="Optional local JSON verification report")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root.parent))
    sys.path.insert(0, str(args.comfy_root))

    import torch
    from aiohttp import web
    from comfy.ldm.minimax import model as h3
    from comfy.ldm.modules.attention import attention_pytorch
    from comfy.model_base import MiniMaxH3
    from comfy.model_patcher import CoreModelPatcher
    from comfy.nested_tensor import NestedTensor
    from comfy.samplers import ksampler
    from comfy.supported_models import MiniMaxH3 as H3Config
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

    import nodes
    import server

    server.PromptServer.instance = types.SimpleNamespace(routes=web.RouteTableDef(),
        node_replace_manager=types.SimpleNamespace(register=lambda value: None))
    if not torch.cuda.is_available():
        raise RuntimeError("RTX smoke requires CUDA")
    h3.optimized_attention = attention_pytorch

    def load_module(name, path, package=False):
        spec = importlib.util.spec_from_file_location(name, path,
            submodule_search_locations=[str(path.parent)] if package else None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    plugins = args.unified_plugins_root
    kj = load_module("jr_stream_kj", plugins / "comfyui-KJNodes/nodes/minimax_nodes.py")
    sage = load_module("jr_stream_sage", plugins / "comfyui-KJNodes/nodes/model_optimization_nodes.py")
    sol = load_module("jr_stream_sol", plugins / "ComfyUI-SolAttn_triton/__init__.py", True)
    nodes.NODE_CLASS_MAPPINGS.update(PathchSageAttentionKJ=sage.PathchSageAttentionKJ,
        MiniMaxLowVRAMAttention=kj.MiniMaxLowVRAMAttention, MiniMaxChunkFeedForward=kj.MiniMaxChunkFeedForward,
        SolAttnPatch=sol.SolAttnPatch)
    unified = importlib.import_module(f"{root.name}.nodes.h3_unified_acceleration")
    streaming = importlib.import_module(f"{root.name}.utils.h3_streaming_sampler")
    plan = importlib.import_module(f"{root.name}.utils.h3_stream_plan").canonical_plan(args.preset)
    torch.manual_seed(42)
    config = H3Config(dict(hidden_size=512, num_layers=args.layers, token_refiner_num_layers=0,
        num_attention_heads=4, attention_head_dim=128, ffn_hidden_size=1024, text_dim=512,
        timestep_input_dim=32, time_embed_hidden_size=128, time_embed_dim=64, dtype=torch.bfloat16))
    base = MiniMaxH3(config, device=torch.device("cpu"))
    with torch.no_grad():
        for param in base.diffusion_model.parameters():
            param.uniform_(-.02, .02)
        base.diffusion_model.rope.inv_freq.copy_(torch.linspace(.01, 1, 16))
    original = CoreModelPatcher(base, load_device=torch.device("cuda"), offload_device=torch.device("cpu"))
    vae = types.SimpleNamespace(latent_channels=24, spacial_compression_decode=lambda: 16, decode=lambda x: x)
    reports = []
    for strength, mode, storage in ((0., "Streaming Attention", "cpu"), (.2, "Streaming Attention", "cpu"),
                                   (.2, "Sparse KV", "cuda")):
        patcher = unified.JR_H3_UnifiedAcceleration().patch(original, enable_tst=strength > 0, tst_strength=strength,
                                                           head_chunks=4, ffn_seq_threshold=256)[0]
        video_t, audio_t = plan.video_latent_count, plan.audio_latent_count
        audio = torch.full((1, 32, 2, audio_t), .125)
        latent = {"samples": NestedTensor((torch.zeros(1, 24, video_t, 8, 8), audio)),
                  "noise_mask": NestedTensor((torch.ones(1, 1, video_t, 1, 1), torch.zeros(1, 1, 2, audio_t)))}
        inputs = dict(model=patcher, positive=[[torch.zeros(1, 2, 512), {}]], vae=vae, noise=Noise_RandomNoise(123),
                      sampler=ksampler("euler"), sigmas=torch.tensor([1., .6, .2, 0.]), latent_image=latent,
                      stream_plan=plan, streaming_mode=mode, cache_device=storage)
        first, status = streaming.sample_streaming(**inputs)
        second, _ = streaming.sample_streaming(**inputs)
        diffs = [float((a-b).abs().max()) for a, b in zip(first["samples"].unbind(), second["samples"].unbind())]
        if diffs != [0., 0.] or not torch.equal(first["samples"].unbind()[1], audio):
            raise AssertionError(f"Repeatability/locked audio failed: {diffs}")
        reports.append(dict(mode=mode, tst=strength, storage=storage, repeat_max_difference=diffs, status=status))
    report = dict(device=torch.cuda.get_device_name(), tiny_layers=args.layers, preset=args.preset, reports=reports,
                  sol_stats=sol.sol_attn_stats())
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
