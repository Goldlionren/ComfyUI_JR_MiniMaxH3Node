"""Independent LLM scene classifier and deterministic H3 cache config router."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..utils.h3_cache_config import build_preset_config
from ..utils.openai_compat import discover_model, normalize_api_urls, parse_chat_content, request_chat
from ..utils.safe_logging import safe_error

SYSTEM_PROMPT = """You classify the compute risk of an already-finished MiniMax H3 video-generation prompt.
Never rewrite, expand, correct, translate, or optimize that prompt. Analyze only speech density, lip-sync sensitivity,
body-motion intensity, camera-motion intensity, scene changes, and audio/video timing sensitivity.

Return exactly one JSON object and no Markdown, prefix, suffix, or explanation. Use only these fields and enums:
scene_class: visual | dialogue | action | mixed
speech_intensity: none | low | medium | high
motion_intensity: low | medium | high
camera_motion: low | medium | high
lip_sync_critical: boolean
audio_timing_sensitive: boolean
recommended_profile: visual_fast | dialogue_safe | action_safe | balanced
confidence: number from 0.0 to 1.0
reason: one short sentence

Classification policy:
- landscapes, products, still life, slow motion/camera and no strict speech/lip sync => visual_fast
- dialogue, speech, singing or talking-head with low/medium motion => dialogue_safe
- combat, dance, running, fast turns/camera, large motion/occlusion without strict lip sync => action_safe
- strong speech plus high motion, or several simultaneous high-risk factors => balanced
Do not return thresholds, block counts, cache windows, or hit limits."""

_ENUMS = {
    "scene_class": {"visual", "dialogue", "action", "mixed"},
    "speech_intensity": {"none", "low", "medium", "high"},
    "motion_intensity": {"low", "medium", "high"},
    "camera_motion": {"low", "medium", "high"},
    "recommended_profile": {"visual_fast", "dialogue_safe", "action_safe", "balanced"},
}


@dataclass(frozen=True)
class SceneClassification:
    scene_class: str
    speech_intensity: str
    motion_intensity: str
    camera_motion: str
    lip_sync_critical: bool
    audio_timing_sensitive: bool
    recommended_profile: str
    confidence: float
    reason: str


def parse_classifier_content(content: str) -> SceneClassification:
    text = str(content).strip()
    decoder = json.JSONDecoder()
    parsed = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break
    if parsed is None:
        raise ValueError("Cache classifier response does not contain a JSON object.")
    required = {*_ENUMS, "lip_sync_critical", "audio_timing_sensitive", "confidence", "reason"}
    missing = required.difference(parsed)
    if missing:
        raise ValueError("Cache classifier response is missing required fields.")
    if set(parsed).difference(required):
        raise ValueError("Cache classifier response contains unsupported fields.")
    for field, allowed in _ENUMS.items():
        if parsed[field] not in allowed:
            raise ValueError(f"Cache classifier returned an invalid {field} value.")
    if not isinstance(parsed["lip_sync_critical"], bool) or not isinstance(parsed["audio_timing_sensitive"], bool):
        raise ValueError("Cache classifier boolean fields are invalid.")
    try:
        confidence = max(0.0, min(1.0, float(parsed["confidence"])))
    except (TypeError, ValueError):
        raise ValueError("Cache classifier confidence is invalid.") from None
    reason = " ".join(str(parsed["reason"]).split())[:240]
    if not reason:
        raise ValueError("Cache classifier reason is empty.")
    return SceneClassification(confidence=confidence, reason=reason,
                               **{key: parsed[key] for key in required if key not in {"confidence", "reason"}})


def select_reviewed_profile(result: SceneClassification, *, audio_content: str = "Auto",
                            has_reference_audio: bool = False, has_reference_video: bool = False) -> str:
    speech = result.speech_intensity
    motion = result.motion_intensity
    camera = result.camera_motion
    if result.lip_sync_critical and motion == "high":
        profile = "balanced"
    elif speech == "high" and motion != "high":
        profile = "dialogue_safe"
    elif motion == "high" or camera == "high":
        profile = "action_safe" if speech in {"none", "low"} and not result.lip_sync_critical else "balanced"
    elif speech == "none" and motion == "low" and camera in {"low", "medium"}:
        profile = "visual_fast"
    elif result.lip_sync_critical or speech in {"medium", "high"}:
        profile = "dialogue_safe"
    else:
        profile = "balanced"
    if audio_content in {"Speech", "Singing"} and profile == "visual_fast":
        profile = "dialogue_safe"
    if has_reference_audio and has_reference_video and (speech != "none" or motion != "low"):
        profile = "balanced"
    return profile


def _chat_content(data) -> str:
    """Extract response text without applying Prompt Optimizer cleanup rules."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (TypeError, KeyError, IndexError):
        raise ValueError("Cache classifier response is missing choices[0].message.content.") from None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    raise ValueError("Cache classifier response content is invalid.")


def safe_fallback(fail_mode: str, *, quality_level: str, cache_device: str, gpu_reserve_mb: int,
                  audio_content: str, reason: str):
    if fail_mode == "Stop Workflow":
        raise RuntimeError(reason)
    if fail_mode == "Disable Cache":
        profile, quality, source = "off", quality_level, "router_disabled_cache_fallback"
        analysis = "Classifier unavailable or invalid; cache disabled."
    else:
        profile, quality, source = "balanced", "Conservative", "router_safe_fallback"
        analysis = "Classifier unavailable or invalid; using safe Balanced configuration."
    config = build_preset_config(profile, quality, source=source, cache_device=cache_device,
                                 gpu_reserve_mb=gpu_reserve_mb, audio_content=audio_content,
                                 confidence=0.0, analysis_summary=analysis)
    return config, profile, analysis


class JR_H3_CacheConfigRouter:
    CATEGORY = "JR MiniMax H3/Cache"
    FUNCTION = "route"
    RETURN_TYPES = ("JR_H3_CACHE_CONFIG", "STRING", "STRING")
    RETURN_NAMES = ("cache_config", "selected_profile", "analysis")
    DESCRIPTION = "Classifies a finished H3 prompt, then maps it to versioned local cache presets."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "optimized_prompt": ("STRING", {"multiline": True, "forceInput": True}),
            "enable": ("BOOLEAN", {"default": True}),
            "api_base_url": ("STRING", {"default": "http://127.0.0.1:10000"}),
            "model": ("STRING", {"default": ""}),
            "api_key": ("STRING", {"default": ""}),
            "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_tokens": ("INT", {"default": 256, "min": 64, "max": 2048}),
            "timeout_seconds": ("INT", {"default": 60, "min": 1, "max": 1800}),
            "disable_reasoning": ("BOOLEAN", {"default": True}),
            "quality_level": (["Conservative", "Balanced", "Aggressive"], {"default": "Balanced"}),
            "cache_device": (["Auto", "GPU", "CPU"], {"default": "Auto"}),
            "gpu_reserve_mb": ("INT", {"default": 2048, "min": 0, "max": 131072, "step": 128}),
            "fail_mode": (["Safe Balanced", "Disable Cache", "Stop Workflow"], {"default": "Safe Balanced"}),
            "audio_content": (["Auto", "None", "Speech", "Singing", "Music", "Ambient"], {"default": "Auto"}),
            "has_reference_audio": ("BOOLEAN", {"default": False}),
            "has_reference_video": ("BOOLEAN", {"default": False}),
        }}

    def route(self, optimized_prompt, enable, api_base_url, model, api_key, temperature, top_p, max_tokens,
              timeout_seconds, disable_reasoning, quality_level, cache_device, gpu_reserve_mb, fail_mode,
              audio_content, has_reference_audio, has_reference_video):
        if not enable:
            profile = "off" if fail_mode == "Disable Cache" else "balanced"
            config = build_preset_config(profile, quality_level, source="router_disabled_fallback",
                                         cache_device=cache_device, gpu_reserve_mb=gpu_reserve_mb,
                                         audio_content=audio_content, confidence=0.0,
                                         analysis_summary="Router disabled; local deterministic fallback used.")
            return config, profile, config.analysis_summary
        try:
            models_url, chat_url = normalize_api_urls(api_base_url)
            selected_model = str(model).strip() or discover_model(models_url, timeout_seconds, api_key)
            facts = {"audio_content": audio_content, "has_reference_audio": bool(has_reference_audio),
                     "has_reference_video": bool(has_reference_video)}
            payload = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Workflow facts: " + json.dumps(facts) + "\nFinished H3 prompt:\n" + str(optimized_prompt)},
                ],
                "temperature": float(temperature), "top_p": float(top_p),
                "max_tokens": int(max_tokens), "stream": False,
            }
            if disable_reasoning:
                payload["reasoning_effort"] = "none"
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            response = request_chat(chat_url, payload, timeout_seconds, api_key, disable_reasoning)
            result = parse_classifier_content(_chat_content(response))
            profile = select_reviewed_profile(result, audio_content=audio_content,
                                              has_reference_audio=has_reference_audio,
                                              has_reference_video=has_reference_video)
            analysis = f"{result.reason} Confidence {result.confidence:.2f}; local review selected {profile}."
            config = build_preset_config(profile, quality_level, source="router", cache_device=cache_device,
                                         gpu_reserve_mb=gpu_reserve_mb, audio_content=audio_content,
                                         confidence=result.confidence, analysis_summary=analysis)
            return config, profile, analysis
        except Exception as error:
            message = safe_error(error, api_key)
            prompt_text = str(optimized_prompt)
            if prompt_text:
                message = message.replace(prompt_text, "<prompt-redacted>")
            message = message[:1000]
            return safe_fallback(fail_mode, quality_level=quality_level, cache_device=cache_device,
                                 gpu_reserve_mb=gpu_reserve_mb, audio_content=audio_content, reason=message)


# Assert responsibility separation at import time without touching network or ComfyUI.
assert parse_chat_content is not _chat_content
