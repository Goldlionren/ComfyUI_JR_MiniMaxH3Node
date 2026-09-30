"""Opt-in native H3 path checks on a small randomly initialized model, not quality benchmarks."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("JR_H3_GPU_INTEGRATION") != "1", reason="opt-in GPU integration")


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native(monkeypatch):
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.model_patcher import ModelPatcher
    from comfy_extras import nodes_sparse_attention as core

    import nodes

    kj_root = Path(os.environ["JR_H3_KJ_ROOT"])
    kj = load_file("jr_test_kj_minimax", kj_root / "nodes/minimax_nodes.py")
    sage = load_file("jr_test_kj_optimization", kj_root / "nodes/model_optimization_nodes.py")
    for node_id, cls in {"BlockSparseAttention": core.BlockSparseAttention,
                         "MiniMaxLowVRAMAttention": kj.MiniMaxLowVRAMAttention,
                         "MiniMaxChunkFeedForward": kj.MiniMaxChunkFeedForward,
                         "PathchSageAttentionKJ": sage.PathchSageAttentionKJ}.items():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node_id, cls)
    torch.manual_seed(123)
    diffusion = MiniMaxH3Model(hidden_size=256, num_layers=2, token_refiner_num_layers=0,
                              num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=512,
                              text_dim=256, time_embed_hidden_size=256, time_embed_dim=128,
                              dtype=torch.bfloat16, device="cuda", operations=comfy.ops.manual_cast)
    diffusion.requires_grad_(False)
    with torch.no_grad():
        for name, parameter in diffusion.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(0, .02)
        diffusion.rope.inv_freq.copy_(torch.exp(-torch.arange(16, device="cuda") / 16 * 9.21))
    container = torch.nn.Module()
    container.diffusion_model = diffusion
    container.model_sampling = SimpleNamespace(percent_to_sigma=lambda percent: 1 - percent)
    patcher = ModelPatcher(container, torch.device("cuda:0"), torch.device("cpu"))
    x = [torch.randn(1, 24, 4, 64, 64, device="cuda", dtype=torch.bfloat16),
         torch.randn(1, 32, 2, 32, device="cuda", dtype=torch.bfloat16)]
    context = torch.randn(1, 64, 256, device="cuda", dtype=torch.bfloat16)
    return patcher, diffusion, x, context, core


@pytest.mark.parametrize("mode", ["chunked", "min_tokens", "dense_blocks", "sigma_window"])
def test_real_h3_producer_and_sage_fallback(native, monkeypatch, mode):
    import comfy_kitchen as ck
    import sageattention
    from ComfyUI_JR_MiniMaxH3Node.nodes.h3_unified_acceleration_v2 import JR_H3_UnifiedAccelerationV2

    source, diffusion, x, context, core = native
    count = {"producer": 0, "generic": 0, "sage": 0}
    sizes = []
    original_chunked, original_generic = ck.sol_attn_chunked, ck.sol_attn
    original_sage = sageattention.sageattn_qk_int8_pv_fp8_cuda

    def sage(*args, **kwargs):
        count["sage"] += 1
        return original_sage(*args, **kwargs)

    monkeypatch.setattr(sageattention, "sageattn_qk_int8_pv_fp8_cuda", sage)

    def chunked(*args, **kwargs):
        count["producer"] += 1
        return original_chunked(*args, **kwargs)

    def generic(*args, **kwargs):
        count["generic"] += 1
        return original_generic(*args, **kwargs)

    monkeypatch.setattr(ck, "sol_attn_chunked", chunked)
    monkeypatch.setattr(ck, "sol_attn", generic)
    hook = diffusion.blocks[0].attn.qkv_proj.register_forward_pre_hook(lambda module, args: sizes.append(args[0].shape[0]))
    p = JR_H3_UnifiedAccelerationV2().patch(
        source, sparse_backend="core", sage_attention="sageattn_qk_int8_pv_fp8_cuda++",
        head_chunks=2, ffn_chunks=2, ffn_seq_threshold=256, min_tokens=100000 if mode == "min_tokens" else 0,
        dense_blocks="0,-1" if mode == "dense_blocks" else "", verbose=True, extra_tokens=128,
    )[0]
    assert source.object_patches == {}
    assert source.model_options["transformer_options"] == {}
    p.patch_model(load_weights=False)
    try:
        outputs = []
        for run in range(2):
            p.pre_run()
            options = dict(p.model_options["transformer_options"])
            sigma = .99 if mode == "sigma_window" else .5
            options.update(sigmas=torch.tensor([sigma], device="cuda"), uuids=("positive",))
            # Exercise the real lifecycle callback before each forward.
            p.prepare_state(torch.tensor([sigma], device="cuda"), {"transformer_options": options})
            with torch.inference_mode():
                outputs.append(diffusion(x, torch.tensor([sigma * 1000], device="cuda"), context, options))
            assert all(torch.isfinite(t).all() for t in outputs[-1])
            for callback in p.get_all_callbacks("on_cleanup"):
                callback(p)
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(outputs[0], outputs[1]))
        assert count["generic"] == 0
        if mode == "chunked":
            assert count["producer"] == 4
            assert count["sage"] == 0
            assert sizes == [4096, 128, 4096, 128] * 2  # bootstrap repeats after cleanup
        else:
            assert count["producer"] == 0
            assert count["sage"] == 8  # two head groups, two blocks, two requests
            assert sizes == [4224, 4224]
    finally:
        hook.remove()
        p.unpatch_model(unpatch_weights=False)
