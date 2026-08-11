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
                  fail_mode="Return Original", disable_reasoning=True, h3_input_mode="Auto",
                  reference_instructions="", api_key="")
    values.update(overrides); return values


def test_disabled_is_four_outputs_and_no_request():
    output = JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args())
    assert output == ("rough", "rough", "Disabled: original prompt returned", None)


def test_return_original_failure_is_safe_and_four_outputs():
    output = JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args(enable=True, api_key="TOPSECRET"))
    assert len(output) == 4 and output[:2] == ("rough", "rough")
    assert output[3] is None
    assert output[2].startswith("Fallback:") and "TOPSECRET" not in output[2]


def test_oversized_legacy_text_is_rejected_before_network_request(monkeypatch):
    called = False

    def unexpected_request(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("network request must not run")

    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        unexpected_request,
    )
    original = "x" * (512 * 1024 + 1)
    output = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(enable=True, prompt=original)
    )
    assert output[:2] == (original, original)
    assert output[2] == "Fallback: ValueError: prompt exceeds the 524288-byte limit."
    assert called is False


def test_stop_workflow_raises():
    try:
        JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args(enable=True, fail_mode="Stop Workflow"))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_input_types_have_nine_optional_images():
    inputs = JR_H3_OpenAICompatiblePromptOptimizer.INPUT_TYPES()
    optional = inputs["optional"]
    assert all(f"ref_image_{i}" in optional for i in range(1, 10))
    assert list(inputs["required"])[-2:] == ["h3_input_mode", "reference_instructions"]
    assert list(optional)[-3:] == ["first_frame", "last_frame", "pip"]
    assert inputs["required"]["h3_input_mode"][0] == ["Auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"]


def test_multimodal_structure_has_separators(monkeypatch):
    captured = {}
    response = """subject_definitions:
<Subject 1> is a referenced person defined by <Picture 1> and <Picture 2>.
summary: [reference generation] <Subject 1> performs the requested action.
retention_analysis:
<Picture 1>: fully_preserved - appearance source.
<Picture 2>: fully_preserved - appearance source.
detailed_description: [Shot 1] <Subject 1> remains visible.
overall_soundscape: Quiet room tone.
non_diegetic_music: N/A"""
    def fake_request(url, payload, *args):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": response}}]}

    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        fake_request,
    )
    output = JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args(enable=True, ref_image_1=torch.zeros(2, 8, 8, 3)))
    content = captured["payload"]["messages"][1]["content"]
    assert [item["type"] for item in content] == ["text", "text", "image_url", "text", "image_url"]
    assert "<Picture 1>:" in content[0]["text"] and "<Picture 2>:" in content[0]["text"]
    assert output[2] == "Success: model=x, mode=Ref2VA, repaired=0"


def test_system_prompt_contains_h3_timeline_and_hard_constraints():
    prompt = _system_prompt("Standard", 6, 768, 1152)
    assert "MiniMax H3" in prompt
    assert "[Shot 1] has no timestamp" in prompt
    assert "[Shot N] At MM:SS.mmm," in prompt
    assert "Duration: 6 seconds" in prompt and "768x1152" in prompt
    assert "user's explicit intent has highest priority" in prompt
    assert "must never override field names" in prompt


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("Standard", "natural motion, physical continuity"),
        ("Cinematic Drama", "micro-expressions"),
        ("Action", "weight transfer"),
        ("Character Consistency", "stable identity"),
    ],
)
def test_each_profile_contributes_h3_specific_direction(profile, expected):
    assert expected in _system_prompt(profile, 10, 1280, 720)


def test_user_prompt_normalizes_reference_tags_and_supplies_context():
    prompt = _user_prompt("让<image1>向<image 2>转身", "Action", 8, 1280, 720, 2)
    assert "Target duration: 8 seconds" in prompt
    assert "Canvas reference: 1280x720" in prompt
    assert "<Picture 1>:" in prompt and "<Picture 2>:" in prompt
    assert "让<Picture 1>向<Picture 2>转身" in prompt
    assert "<image" not in prompt
