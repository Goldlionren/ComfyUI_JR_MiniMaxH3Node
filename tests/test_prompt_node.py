import torch
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_openai_prompt_optimizer import JR_H3_OpenAICompatiblePromptOptimizer


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
    assert output[2].endswith("images=2")
