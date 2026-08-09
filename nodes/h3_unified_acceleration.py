"""Unified orchestration node for the verified MiniMax H3 patch stack."""

from __future__ import annotations

import logging
from typing import Any

from ..utils import h3_acceleration_adapters as adapters

LOGGER = logging.getLogger(__name__)


class JR_H3_UnifiedAcceleration:
    """Apply upstream Sage, H3 memory patches, and Sol-Attn in a fixed order."""

    CATEGORY = "JR MiniMax H3/Optimization"
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = (
        "Composes the installed KJNodes Sage/Low-VRAM/FFN patches and Sol-Attn "
        "for MiniMax H3. Upstream implementations remain external dependencies."
    )
    EXPERIMENTAL = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
                "sage_attention": (
                    list(adapters.SAGE_ATTENTION_MODES),
                    {"default": "sageattn_qk_int8_pv_fp8_cuda++"},
                ),
                "allow_compile": ("BOOLEAN", {"default": False}),
                "enable_low_vram_attention": ("BOOLEAN", {"default": True}),
                "head_chunks": ("INT", {"default": 4, "min": 1, "max": 56, "step": 1}),
                "enable_low_vram_ffn": ("BOOLEAN", {"default": True}),
                "ffn_chunks": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "ffn_seq_threshold": (
                    "INT",
                    {"default": 4096, "min": 256, "max": 262144, "step": 256},
                ),
                "enable_sol_attn": ("BOOLEAN", {"default": True}),
                "tau": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 4.0, "step": 0.05}),
                "start_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_tokens": ("INT", {"default": 4096, "min": 0, "max": 1 << 20, "step": 512}),
                "int8_qk": ("BOOLEAN", {"default": True}),
                "int8_pv": ("BOOLEAN", {"default": True}),
                "sink_conditioning": (
                    ["exact_kv", "exact_kv_and_rows", "off"],
                    {"default": "exact_kv_and_rows"},
                ),
                "morton": ("BOOLEAN", {"default": False}),
                "morton_curve": (["3d", "2d_frame"], {"default": "2d_frame"}),
                "verbose": ("BOOLEAN", {"default": False}),
                "use_tma": ("BOOLEAN", {"default": False}),
                "dense_blocks": ("STRING", {"default": ""}),
            },
            "optional": {
                "tau_profile": ("STRING", {"forceInput": True}),
            },
        }

    def patch(
        self,
        model: Any,
        enable: bool = True,
        sage_attention: str = "sageattn_qk_int8_pv_fp8_cuda++",
        allow_compile: bool = False,
        enable_low_vram_attention: bool = True,
        head_chunks: int = 4,
        enable_low_vram_ffn: bool = True,
        ffn_chunks: int = 4,
        ffn_seq_threshold: int = 4096,
        enable_sol_attn: bool = True,
        tau: float = 1.3,
        start_percent: float = 0.2,
        end_percent: float = 0.9,
        min_tokens: int = 4096,
        int8_qk: bool = True,
        int8_pv: bool = True,
        sink_conditioning: str = "exact_kv_and_rows",
        morton: bool = False,
        morton_curve: str = "2d_frame",
        verbose: bool = False,
        use_tma: bool = False,
        dense_blocks: str = "",
        tau_profile: str | None = None,
    ):
        if not enable:
            return (model,)

        adapters.ensure_minimax_h3_model(model)
        patched = model

        # The order is a compatibility contract: Sol must capture Sage as its
        # previous dense backend and compose with the KJ low-VRAM hand-off.
        if sage_attention != "disabled":
            patched = adapters.apply_sage(
                patched,
                sage_attention=sage_attention,
                allow_compile=allow_compile,
            )
        if enable_low_vram_attention:
            patched = adapters.apply_h3_low_vram_attention(patched, head_chunks=head_chunks)
        if enable_low_vram_ffn:
            patched = adapters.apply_h3_chunk_ffn(
                patched,
                chunks=ffn_chunks,
                seq_threshold=ffn_seq_threshold,
            )
        if enable_sol_attn:
            patched = adapters.apply_sol_attn(
                patched,
                tau=tau,
                start_percent=start_percent,
                end_percent=end_percent,
                min_tokens=min_tokens,
                int8_qk=int8_qk,
                int8_pv=int8_pv,
                sink_conditioning=sink_conditioning,
                morton=morton,
                morton_curve=morton_curve,
                verbose=verbose,
                use_tma=use_tma,
                dense_blocks=dense_blocks,
                tau_profile=tau_profile,
            )

        LOGGER.info(
            "JR H3 Unified Acceleration: Sage=%s LowVRAM=%s FFN=%s Sol=%s",
            sage_attention,
            f"enabled(head_chunks={head_chunks})" if enable_low_vram_attention else "disabled",
            (
                f"enabled(chunks={ffn_chunks},threshold={ffn_seq_threshold})"
                if enable_low_vram_ffn
                else "disabled"
            ),
            (
                f"enabled(tau={tau:.2f},range={start_percent:.2f}-{end_percent:.2f},"
                f"int8_qk={str(int8_qk).lower()},int8_pv={str(int8_pv).lower()})"
                if enable_sol_attn
                else "disabled"
            ),
        )
        return (patched,)
