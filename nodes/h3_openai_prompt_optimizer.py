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
    "Standard": "补全主体、环境、动作、镜头运动、光线、声音和连续性；表达紧凑、具体，结尾必须形成明确可见的状态。",
    "Cinematic Drama": "强化人物关系、表演节奏、微表情、情绪转折、景别变化、布光、景深、色彩和声音；只有用户明确需要时才加入对白，不得擅自扩写剧情。",
    "Action": "把动作写成连续的因果链，明确起手、发力、移动、接触、受力、减速、反制和结果；保持方位与动作匹配，避免悬浮、瞬移、穿模和无关动作堆砌。",
    "Character Consistency": "优先锁定人物身份、脸部、年龄、体型、发型、服装、配饰、道具、左右站位和场景连续性，防止混脸、串服装和人物突然变化。",
}


def _system_prompt(profile: str, duration: int, width: int, height: int) -> str:
    return f"""你是专门为 MiniMax H3 图生视频设计提示词的中文导演、分镜设计师和摄影指导。
当前优化档位要求：{_PROFILES[profile]}

用户原始需求具有最高优先级。不得添加与用户意图冲突的人物、身份、服装、地点、对白、道具或剧情。参考图只能提供画面中确实可观察的外观、场景、构图、道具、光线和氛围；模糊、遮挡或未提供的信息不得当作事实编造。

参考图按消息中的出现顺序对应 <Picture 1>、<Picture 2>、<Picture 3>……。最终提示词只能使用 H3 原生的 <Picture N> 标签，不得写成 <imageN>、<image N> 或其他别名。不同参考图中的人物身份必须保持独立。

只输出一份完整、连续、可直接提交给 H3 的中文成品提示词。第一字符就开始正文；不要复述用户输入，不要输出标题、分析、解释、思考过程、注意事项、总结、引号、Markdown 代码围栏或 JSON。

成品按以下逻辑组织：参考图映射；总体画面与风格；人物、场景和道具锁定；按时间顺序展开的镜头；摄影、光线、声音和剪辑；负面约束；最终构图与结束状态。

目标视频时长为 {duration} 秒，画布参考为 {width}×{height}。使用【镜头N｜起始秒—结束秒】作为每个镜头的独立标题，并写清景别、机位、主体动作、微表情、运镜、声音与连续关系。时间轴必须从 0.0 秒开始，最后一个有效镜头必须恰好结束于 {duration} 秒，不得超时；每个镜头必须具有实际时长，禁止零时长或短于 0.2 秒的镜头。默认使用硬切，除非用户明确要求，不使用淡入淡出、溶解、叠化或把形态变化写成转场。

局部变化只能作用于用户指定的局部，不得扩展成全身、服装、道具或场景变化。需要持续变化时，应写成画面内连续发生的自然过程。用户输入中的“必须、最后、结尾、保持、不要、禁止、只能”等要求均为硬约束，必须明确落实到相应时间段和最终构图；如果指定最后使用全景或远景，结尾镜头不得用其他景别代替。

始终保持第 0 秒起始构图、人物身份、脸部、服装、道具、左右位置、空间透视、运动方向和光线方向连续，不新增多余人物、物体、肢体或画面文字。结尾必须给出清晰、稳定、可见且符合用户要求的最终状态。"""


def _user_prompt(prompt: str, profile: str, duration: int, width: int, height: int, image_count: int) -> str:
    picture_map = "、".join(f"<Picture {index}>" for index in range(1, image_count + 1)) or "无参考图"
    return (
        "请把下面的创意整理为一份 MiniMax H3 图生视频提示词。\n"
        f"目标时长：{duration} 秒；画布参考：{width}×{height}；优化档位：{profile}。\n"
        f"参考图标签：{picture_map}。\n\n"
        "用户原始创意：\n"
        f"{normalize_picture_markers(prompt)}"
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
            encoded_images = []
            image_count = 0
            for index in range(1, 10):
                image = kwargs.get(f"ref_image_{index}")
                if image is None:
                    continue
                for data_url in image_batch_to_jpeg_data_urls(image, image_send_size):
                    image_count += 1
                    encoded_images.append((image_count, data_url))
            user_content = [{
                "type": "text",
                "text": _user_prompt(original, prompt_profile, duration_seconds, target_width, target_height, image_count),
            }]
            for image_index, data_url in encoded_images:
                user_content.append({"type": "text", "text": f"[Picture {image_index}]"})
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
