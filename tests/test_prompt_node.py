import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_openai_prompt_optimizer import (
    JR_H3_OpenAICompatiblePromptOptimizer,
    _system_prompt,
    _user_prompt,
)


def _args(**overrides):
    values = dict(prompt="rough", enable=False, api_base_url="http://127.0.0.1:9", model="x",
                  prompt_profile="Standard", duration_seconds=10, target_width=768, target_height=1152,
                  temperature=0.6, top_p=0.9, max_tokens=100, timeout_seconds=1, image_send_size=768,
                  fail_mode="Return Original", disable_reasoning=True, api_key="")
    values.update(overrides); return values


def test_disabled_is_three_outputs_and_no_request():
    output = JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args())
    assert output == ("rough", "rough", "Disabled: original prompt returned")


def test_return_original_failure_is_safe_and_three_outputs():
    output = JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args(enable=True, api_key="TOPSECRET"))
    assert len(output) == 3 and output[:2] == ("rough", "rough")
    assert output[2].startswith("Fallback:") and "TOPSECRET" not in output[2]


def test_stop_workflow_raises():
    try:
        JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args(enable=True, fail_mode="Stop Workflow"))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_input_types_have_nine_optional_images():
    optional = JR_H3_OpenAICompatiblePromptOptimizer.INPUT_TYPES()["optional"]
    assert all(f"ref_image_{i}" in optional for i in range(1, 10))


def test_multimodal_structure_has_separators(monkeypatch):
    captured = {}
    monkeypatch.setattr("ComfyUI_JR_MiniMaxH3Node.nodes.h3_openai_prompt_optimizer.request_chat", lambda url, payload, *a: captured.setdefault("payload", payload) or {})
    monkeypatch.setattr("ComfyUI_JR_MiniMaxH3Node.nodes.h3_openai_prompt_optimizer.parse_chat_content", lambda data: "done")
    output = JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args(enable=True, ref_image_1=torch.zeros(2, 8, 8, 3)))
    content = captured["payload"]["messages"][1]["content"]
    assert [item["type"] for item in content] == ["text", "text", "image_url", "text", "image_url"]
    assert "<Picture 1>、<Picture 2>" in content[0]["text"]
    assert output[2].endswith("images=2")


def test_system_prompt_contains_h3_timeline_and_hard_constraints():
    prompt = _system_prompt("Standard", 6, 768, 1152)
    assert "MiniMax H3" in prompt
    assert "【镜头N｜起始秒—结束秒】" in prompt
    assert "0.0 秒" in prompt and "恰好结束于 6 秒" in prompt
    assert "768×1152" in prompt
    assert "<Picture N>" in prompt
    assert "硬约束" in prompt and "最终状态" in prompt


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("Standard", "主体、环境、动作"),
        ("Cinematic Drama", "微表情、情绪转折"),
        ("Action", "起手、发力、移动、接触"),
        ("Character Consistency", "人物身份、脸部、年龄"),
    ],
)
def test_each_profile_contributes_h3_specific_direction(profile, expected):
    assert expected in _system_prompt(profile, 10, 1280, 720)


def test_user_prompt_normalizes_reference_tags_and_supplies_context():
    prompt = _user_prompt("让<image1>向<image 2>转身", "Action", 8, 1280, 720, 2)
    assert "目标时长：8 秒" in prompt
    assert "画布参考：1280×720" in prompt
    assert "优化档位：Action" in prompt
    assert "<Picture 1>、<Picture 2>" in prompt
    assert "让<Picture 1>向<Picture 2>转身" in prompt
    assert "<image" not in prompt
