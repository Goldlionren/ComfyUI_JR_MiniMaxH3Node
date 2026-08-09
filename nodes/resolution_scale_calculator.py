"""Deterministic area-preserving resolution calculator."""

import math

_ASPECTS = {"Source": None, "1:1": (1, 1), "2:3": (2, 3), "3:2": (3, 2), "16:9": (16, 9), "9:16": (9, 16), "Custom": None}


def calculate_resolution(source_width, source_height, target_megapixels, divisor, aspect="Source", custom_width=16, custom_height=9):
    if int(source_width) <= 0 or int(source_height) <= 0:
        raise ValueError("Source width and height must be positive.")
    if not 0.001 <= float(target_megapixels) <= 256.0:
        raise ValueError("target_megapixels must be between 0.001 and 256.")
    if int(divisor) not in (8, 16, 32):
        raise ValueError("divisor must be 8, 16, or 32.")
    if aspect == "Source":
        ratio = float(source_width) / float(source_height)
    elif aspect == "Custom":
        if int(custom_width) <= 0 or int(custom_height) <= 0:
            raise ValueError("Custom aspect dimensions must be positive.")
        ratio = float(custom_width) / float(custom_height)
    else:
        pair = _ASPECTS.get(aspect)
        if pair is None:
            raise ValueError(f"Unknown aspect ratio: {aspect}")
        ratio = pair[0] / pair[1]
    pixels = float(target_megapixels) * 1_000_000.0
    raw_w, raw_h = math.sqrt(pixels * ratio), math.sqrt(pixels / ratio)
    d = int(divisor)
    width = max(d, int(round(raw_w / d)) * d)
    height = max(d, int(round(raw_h / d)) * d)
    scale = math.sqrt((width * height) / (int(source_width) * int(source_height)))
    return width, height, float(scale), float(width * height / 1_000_000.0)


class JR_H3_ResolutionScaleCalculator:
    CATEGORY = "JR MiniMax H3/Scaling"
    FUNCTION = "calculate"
    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("width", "height", "scale", "actual_megapixels")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_width": ("INT", {"default": 768, "min": 1, "max": 16384}),
            "source_height": ("INT", {"default": 1152, "min": 1, "max": 16384}),
            "target_megapixels": ("FLOAT", {"default": 0.88, "min": 0.001, "max": 256.0, "step": 0.01}),
            "aspect": (list(_ASPECTS), {"default": "Source"}),
            "custom_aspect_width": ("INT", {"default": 16, "min": 1, "max": 8192}),
            "custom_aspect_height": ("INT", {"default": 9, "min": 1, "max": 8192}),
            "divisor": (["8", "16", "32"], {"default": "32"}),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, divisor):
        """Accept string combo values and numeric values saved by older workflows."""
        try:
            normalized = int(divisor)
        except (TypeError, ValueError):
            return "divisor must be 8, 16, or 32."
        if normalized not in (8, 16, 32):
            return "divisor must be 8, 16, or 32."
        return True

    def calculate(self, source_width, source_height, target_megapixels, aspect, custom_aspect_width, custom_aspect_height, divisor):
        return calculate_resolution(source_width, source_height, target_megapixels, divisor, aspect, custom_aspect_width, custom_aspect_height)
