"""Optional VEDA node, isolated from both existing Unified nodes."""
from __future__ import annotations

from ..utils import h3_veda_attention as veda
from ..utils.h3_acceleration_adapters import SAGE_ATTENTION_MODES


class JR_H3_VedaAttention:
    CATEGORY = "JR MiniMax H3/Optimization"
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Independent VEDA sparse attention for native MiniMax H3. Use a clean MODEL branch "
        "after the loader/LoRA, last before sampling. Requires official Veda-on-ComfyUI. "
        "Choose this branch or Sol-H3; existing Unified/Sol nodes are unchanged."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
                "predictor": (veda.predictor_names(), {"default": veda.DEFAULT_PREDICTOR}),
                "generated_sparsity": ("STRING", {"default": "90%", "tooltip": "90% skips 90% of visual key tiles. An integer such as 32 keeps that many tiles."}),
                "reference_sparsity": ("STRING", {"default": "90%", "tooltip": "0% keeps visual references dense. Text/audio stay dense in VEDA."}),
                "full_attention_layers": ("STRING", {"default": "", "tooltip": "0-based full-attention layers, e.g. 0,1,47-49."}),
                "full_attention_steps": ("STRING", {"default": "", "tooltip": "0-based full-attention steps, e.g. 0."}),
                "verbose": ("BOOLEAN", {"default": False}),
                "sage_attention": (list(SAGE_ATTENTION_MODES), {"default": "disabled", "tooltip": "Dense fallback only. Disabled uses ComfyUI's configured default attention."}),
                "enable_low_vram_attention": ("BOOLEAN", {"default": True}),
                "head_chunks": ("INT", {"default": 4, "min": 1, "max": 56}),
                "enable_low_vram_ffn": ("BOOLEAN", {"default": True}),
                "ffn_chunks": ("INT", {"default": 4, "min": 1, "max": 64}),
                "ffn_seq_threshold": ("INT", {"default": 4096, "min": 256, "max": 262144, "step": 256}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def patch(self, model, enable=True, predictor=veda.DEFAULT_PREDICTOR,
              generated_sparsity="90%", reference_sparsity="90%",
              full_attention_layers="", full_attention_steps="", verbose=False,
              sage_attention="disabled", enable_low_vram_attention=True, head_chunks=4,
              enable_low_vram_ffn=True, ffn_chunks=4, ffn_seq_threshold=4096,
              unique_id=None):
        return veda.apply(
            model, enable=enable, predictor=predictor,
            generated_sparsity=generated_sparsity, reference_sparsity=reference_sparsity,
            full_attention_layers=full_attention_layers, full_attention_steps=full_attention_steps,
            verbose=verbose, sage_attention=sage_attention,
            enable_low_vram_attention=enable_low_vram_attention, head_chunks=head_chunks,
            enable_low_vram_ffn=enable_low_vram_ffn, ffn_chunks=ffn_chunks,
            ffn_seq_threshold=ffn_seq_threshold, unique_id=unique_id,
        )
