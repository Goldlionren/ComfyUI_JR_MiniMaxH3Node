"""MiniMax H3-specific model patch using official wrapper and block replacement APIs."""

from __future__ import annotations

from ..utils.h3_cache_config import H3CacheConfig, build_custom_config, build_preset_config, select_manual_profile
from ..utils.h3_cache_runtime import H3AdaptiveCacheRuntime

_ATTACHMENT_KEY = "jr_h3_adaptive_cache"
_CONFLICT_MARKERS = ("easycache", "teacache", "first_block", "firstblock", "cachedit", "cache_dit", "blockcache")


def _model_info(model):
    diffusion = model.get_model_object("diffusion_model")
    module = diffusion.__class__.__module__
    name = diffusion.__class__.__name__
    blocks = getattr(diffusion, "blocks", None)
    count = len(blocks) if blocks is not None else 0
    return diffusion, module, name, count


def _dict_keys_recursive(value):
    if not isinstance(value, dict):
        return []
    keys = []
    for key, child in value.items():
        keys.append(str(key).lower())
        keys.extend(_dict_keys_recursive(child))
    return keys


def detect_cache_conflict(model) -> str | None:
    if hasattr(model, "get_attachment") and model.get_attachment(_ATTACHMENT_KEY) is not None:
        return "a second JR H3 Adaptive Cache"
    keys = _dict_keys_recursive(getattr(model, "model_options", {}))
    keys.extend(str(key).lower() for groups in getattr(model, "wrappers", {}).values() for key in groups)
    for key in keys:
        if any(marker in key for marker in _CONFLICT_MARKERS):
            return key
    patches = getattr(model, "model_options", {}).get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
    if patches:
        return "an existing DiT block replacement"
    return None


class JR_H3_AdaptiveCache:
    CATEGORY = "JR MiniMax H3/Cache"
    FUNCTION = "apply_cache"
    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "selected_profile", "status")
    DESCRIPTION = "Scene-aware dual-stream full-step and block-probe cache for the native 50-block MiniMax H3 DiT."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["Auto", "Visual Fast", "Dialogue Safe", "Action Safe", "Balanced", "Off"], {"default": "Auto"}),
                "quality_level": (["Conservative", "Balanced", "Aggressive", "Custom"], {"default": "Balanced"}),
                "audio_content": (["Auto", "None", "Speech", "Singing", "Music", "Ambient"], {"default": "Auto"}),
                "profile_hint": ("STRING", {"default": ""}),
                "start_percent": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.99, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.90, "min": 0.01, "max": 1.0, "step": 0.01}),
                "warmup_steps": ("INT", {"default": 2, "min": 0, "max": 100}),
                "front_blocks": ("INT", {"default": 1, "min": 0, "max": 48}),
                "back_blocks": ("INT", {"default": 2, "min": 0, "max": 48}),
                "video_threshold": ("FLOAT", {"default": 0.020, "min": 0.0, "max": 1.0, "step": 0.001}),
                "audio_threshold": ("FLOAT", {"default": 0.012, "min": 0.0, "max": 1.0, "step": 0.001}),
                "fast_path_threshold": ("FLOAT", {"default": 0.008, "min": 0.0, "max": 1.0, "step": 0.001}),
                "probe_path_threshold": ("FLOAT", {"default": 0.035, "min": 0.0, "max": 1.0, "step": 0.001}),
                "max_full_step_hits": ("INT", {"default": 1, "min": 0, "max": 20}),
                "max_block_hits": ("INT", {"default": 2, "min": 0, "max": 20}),
                "video_metric_stride": ("INT", {"default": 12, "min": 1, "max": 1024}),
                "audio_metric_stride": ("INT", {"default": 6, "min": 1, "max": 1024}),
                "cache_device": (["Auto", "GPU", "CPU"], {"default": "Auto"}),
                "gpu_reserve_mb": ("INT", {"default": 2048, "min": 0, "max": 131072, "step": 128}),
                "strict_model_check": ("BOOLEAN", {"default": True}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
            "optional": {"cache_config": ("JR_H3_CACHE_CONFIG",)},
        }

    def _manual_config(self, mode, quality_level, audio_content, profile_hint, **values):
        profile = select_manual_profile(mode, audio_content, profile_hint)
        if quality_level != "Custom":
            return build_preset_config(profile, quality_level, source="manual", cache_device=values["cache_device"],
                                       gpu_reserve_mb=values["gpu_reserve_mb"], audio_content=audio_content)
        return build_custom_config(profile, source="manual", audio_content=audio_content,
                                   start_percent=values["start_percent"], end_percent=values["end_percent"],
                                   warmup_steps=values["warmup_steps"], front_blocks=values["front_blocks"],
                                   back_blocks=values["back_blocks"], video_threshold=values["video_threshold"],
                                   audio_threshold=values["audio_threshold"], fast_path_threshold=values["fast_path_threshold"],
                                   probe_path_threshold=values["probe_path_threshold"], max_full_step_hits=values["max_full_step_hits"],
                                   max_block_hits=values["max_block_hits"], video_metric_stride=values["video_metric_stride"],
                                   audio_metric_stride=values["audio_metric_stride"], cache_device=values["cache_device"],
                                   gpu_reserve_mb=values["gpu_reserve_mb"])

    def apply_cache(self, model, mode, quality_level, audio_content, profile_hint, start_percent, end_percent,
                    warmup_steps, front_blocks, back_blocks, video_threshold, audio_threshold, fast_path_threshold,
                    probe_path_threshold, max_full_step_hits, max_block_hits, video_metric_stride, audio_metric_stride,
                    cache_device, gpu_reserve_mb, strict_model_check, verbose, cache_config=None):
        manual = locals().copy()
        for key in ("self", "model", "mode", "quality_level", "audio_content", "profile_hint",
                    "strict_model_check", "verbose", "cache_config"):
            manual.pop(key, None)
        if cache_config is not None:
            if not isinstance(cache_config, H3CacheConfig):
                raise ValueError("cache_config is not a valid immutable JR_H3_CACHE_CONFIG object.")
            config = cache_config
            config.__post_init__()
            source_text = "Configuration source: Router\nManual widget values ignored."
        else:
            config = self._manual_config(mode, quality_level, audio_content, profile_hint, **manual)
            source_text = "Configuration source: Manual"
        if config.profile == "off" or mode == "Off" and cache_config is None:
            return model, "off", source_text + "\nCache disabled; MODEL returned unchanged."
        conflict = detect_cache_conflict(model)
        if conflict:
            raise RuntimeError(f"JR H3 Adaptive Cache cannot stack with {conflict}. Remove the other cache patch first.")
        try:
            _diffusion, module, name, block_count = _model_info(model)
        except Exception as error:
            if strict_model_check:
                raise RuntimeError("JR H3 Adaptive Cache could not inspect the input MODEL. Connect the native MiniMax H3 MODEL.") from error
            return model, "off", source_text + "\nUnsupported MODEL; cache safely disabled."
        is_h3 = name == "MiniMaxH3Model" and module.endswith("comfy.ldm.minimax.model") and block_count > 0
        if not is_h3:
            if strict_model_check:
                raise RuntimeError(f"JR H3 Adaptive Cache requires native MiniMaxH3Model; received {module}.{name}.")
            return model, "off", source_text + "\nUnsupported MODEL; cache safely disabled."
        if config.front_blocks + config.back_blocks >= block_count:
            raise ValueError(f"Cache front/back blocks leave no middle range for the detected {block_count}-block H3 model.")
        patched = model.clone()
        runtime = H3AdaptiveCacheRuntime(config, block_count, audio_required=config.audio_content != "None", verbose=verbose)
        patched.set_attachments(_ATTACHMENT_KEY, runtime)
        patched.add_wrapper_with_key("diffusion_model", _ATTACHMENT_KEY, runtime.diffusion_wrapper)
        patched.add_callback_with_key("on_cleanup", _ATTACHMENT_KEY, runtime.cleanup)
        if config.profile in {"dialogue_safe", "action_safe", "balanced"}:
            for index in range(block_count):
                patched.set_model_patch_replace(runtime.block_wrapper(index), "dit", "double_block", index)
        status = (source_text + f"\n{config.profile} | {block_count} blocks | F{runtime.front_blocks}-"
                  f"M{block_count - runtime.front_blocks - runtime.back_blocks}-B{runtime.back_blocks} | "
                  f"{config.cache_device} cache")
        return patched, config.profile, status
