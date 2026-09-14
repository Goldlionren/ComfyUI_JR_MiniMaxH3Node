"""Geometry-only planner; no model loading or CUDA execution."""

from ..utils.h3_stream_plan import PRESET, PRESETS, canonical_plan


class JR_H3_TaoMateChunkPlanner:
    CATEGORY = "JR MiniMax H3/Experimental"
    FUNCTION = "plan"
    RETURN_TYPES = ("JR_H3_STREAM_PLAN", "STRING", "INT", "INT")
    RETURN_NAMES = ("stream_plan", "status", "native_frames", "audio_ticks")
    EXPERIMENTAL = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"preset": (list(PRESETS), {"default": PRESET,
            "tooltip": "Full output timeline: 124 / 243 / 362 frames at 24fps. Keep pass 1 full-length; only pass 2 streams."})}}

    def plan(self, preset=PRESET):
        plan = canonical_plan(preset)
        return plan, plan.status(), plan.native_frame_count, plan.audio_latent_count
