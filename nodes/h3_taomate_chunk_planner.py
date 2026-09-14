"""Geometry-only planner; no model loading or CUDA execution."""

from ..utils.h3_stream_plan import PRESET, canonical_plan


class JR_H3_TaoMateChunkPlanner:
    CATEGORY = "JR MiniMax H3/Experimental"
    FUNCTION = "plan"
    RETURN_TYPES = ("JR_H3_STREAM_PLAN", "STRING")
    RETURN_NAMES = ("stream_plan", "status")
    EXPERIMENTAL = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"preset": ([PRESET], {"default": PRESET})}}

    def plan(self, preset=PRESET):
        plan = canonical_plan(preset)
        return plan, plan.status()
