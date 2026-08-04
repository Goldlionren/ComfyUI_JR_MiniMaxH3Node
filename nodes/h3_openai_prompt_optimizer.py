"""Clean-room OpenAI-compatible MiniMax H3 prompt optimizer."""

from __future__ import annotations

from ..utils.image_conversion import image_batch_to_jpeg_data_urls
from ..utils.openai_compat import (
    discover_model,
    normalize_api_urls,
    normalize_picture_markers,
    parse_chat_content,
    request_chat,
)
from ..utils.safe_logging import safe_error

_PROFILES = {
    "Standard": "Prioritize subject, setting, action, camera movement, lighting, continuity, and a definite final state.",
    "Cinematic Drama": "Prioritize relationships, concise dialogue only when intended, micro-expressions, emotional turns, performance rhythm, shot-size changes, and the dramatic outcome.",
    "Action": "Build one causal action chain with setup, impact, resistance, counteraction, speed changes, camera direction, and stable screen geography.",
    "Character Consistency": "Lock identity, clothing, hair, face, body, props, left/right placement, and scene continuity throughout the shot.",
}


def _system_prompt(profile: str, duration: int, width: int, height: int) -> str:
    return (
        "Rewrite the supplied idea as one production-ready MiniMax H3 video prompt. "
        "Return only the finished prompt: no title, analysis, notes, summary, quotation, or Markdown fence. "
        "Use explicit chronological order and fit the event density to "
        f"{duration} seconds at {width}x{height}. {_PROFILES[profile]} "
        "Keep the user's identities and intent; do not invent conflicting characters, dialogue, or plot. "
        "Refer to supplied images only as <Picture 1>, <Picture 2>, and so on. End with a visible final state."
    )


class JR_H3_OpenAICompatiblePromptOptimizer:
    CATEGORY = "JR MiniMax H3/Prompt"
    FUNCTION = "optimize"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("optimized_prompt", "original_prompt", "status")
    DESCRIPTION = "Optimizes MiniMax H3 prompts through an OpenAI-compatible chat-completions endpoint."

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "enable": ("BOOLEAN", {"default": True}),
            "api_base_url": ("STRING", {"default": "http://127.0.0.1:10000"}),
            "model": ("STRING", {"default": ""}),
            "prompt_profile": (list(_PROFILES), {"default": "Standard"}),
            "duration_seconds": ("INT", {"default": 10, "min": 1, "max": 60}),
            "target_width": ("INT", {"default": 768, "min": 64, "max": 8192}),
            "target_height": ("INT", {"default": 1152, "min": 64, "max": 8192}),
            "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
            "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_tokens": ("INT", {"default": 1800, "min": 32, "max": 32768}),
            "timeout_seconds": ("INT", {"default": 180, "min": 1, "max": 1800}),
            "image_send_size": ("INT", {"default": 768, "min": 64, "max": 4096}),
            "fail_mode": (["Return Original", "Stop Workflow"], {"default": "Return Original"}),
            "disable_reasoning": ("BOOLEAN", {"default": True}),
        }
        optional = {"api_key": ("STRING", {"default": ""})}
        optional.update({f"ref_image_{index}": ("IMAGE",) for index in range(1, 10)})
        return {"required": required, "optional": optional}

    def optimize(
        self, prompt, enable, api_base_url, model, prompt_profile, duration_seconds,
        target_width, target_height, temperature, top_p, max_tokens, timeout_seconds,
        image_send_size, fail_mode, disable_reasoning, api_key="", **kwargs,
    ):
        original = str(prompt)
        if not enable:
            return original, original, "Disabled: original prompt returned"
        try:
            models_url, chat_url = normalize_api_urls(api_base_url)
            selected_model = model.strip() or discover_model(models_url, timeout_seconds, api_key)
            user_content = [{"type": "text", "text": normalize_picture_markers(original)}]
            image_count = 0
            for index in range(1, 10):
                image = kwargs.get(f"ref_image_{index}")
                if image is None:
                    continue
                for data_url in image_batch_to_jpeg_data_urls(image, image_send_size):
                    image_count += 1
                    user_content.append({"type": "text", "text": f"[Picture {image_count}]"})
                    user_content.append({"type": "image_url", "image_url": {"url": data_url}})
            payload = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": _system_prompt(prompt_profile, duration_seconds, target_width, target_height)},
                    {"role": "user", "content": user_content},
                ],
                "temperature": float(temperature), "top_p": float(top_p),
                "max_tokens": int(max_tokens), "stream": False,
            }
            if disable_reasoning:
                payload["reasoning_effort"] = "none"
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            response = request_chat(chat_url, payload, timeout_seconds, api_key, disable_reasoning)
            optimized = parse_chat_content(response)
            return optimized, original, f"Success: model={selected_model}, images={image_count}"
        except Exception as error:
            message = safe_error(error, api_key)
            if fail_mode == "Stop Workflow":
                raise RuntimeError(message) from error
            return original, original, f"Fallback: {message}"
