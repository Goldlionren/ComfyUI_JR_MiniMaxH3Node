import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_openai_prompt_optimizer import (
    JR_H3_OpenAICompatiblePromptOptimizer,
)


def _args(**overrides):
    values = dict(
        prompt="A paper boat moves across a shallow pool.",
        enable=True,
        api_base_url="http://127.0.0.1:10000",
        model="test-model",
        prompt_profile="Standard",
        duration_seconds=5,
        target_width=768,
        target_height=1152,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1800,
        timeout_seconds=2,
        image_send_size=768,
        fail_mode="Stop Workflow",
        disable_reasoning=True,
        h3_input_mode="Auto",
        reference_instructions="",
        api_key="",
    )
    values.update(overrides)
    return values


def _base(body="[Shot 1] A paper boat moves across a shallow pool."):
    return (
        f"integrated_multimodal_description: {body}\n"
        "overall_soundscape: Water moves softly around the paper hull.\n"
        "non_diegetic_music: N/A"
    )


def _install_response(monkeypatch, text, captured):
    def fake_request(url, payload, *args):
        captured["url"] = url
        captured["payload"] = payload
        captured.setdefault("calls", []).append(payload)
        return {"choices": [{"message": {"content": text}}]}

    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        fake_request,
    )


@pytest.mark.parametrize(
    ("expected_mode", "inputs", "response", "expected_labels"),
    [
        ("T2VA", {}, _base(), []),
        (
            "I2VA",
            {"first_frame": torch.zeros(1, 8, 8, 3)},
            "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
            + _base("[Shot 1] The boat starts from <Picture 1> and moves forward."),
            ["[Picture 1]"],
        ),
        (
            "FL2VA",
            {"first_frame": torch.zeros(1, 8, 8, 3), "last_frame": torch.ones(1, 8, 8, 3)},
            "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 5.00-second mark of the target video.\n\n"
            + _base("[Shot 1] The boat moves continuously from <Picture 1> to <Picture 2>."),
            ["[Picture 1]", "[Picture 2]"],
        ),
        (
            "L2VA",
            {"last_frame": torch.ones(1, 8, 8, 3)},
            "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 5.00-second mark of the target video.\n\n"
            + _base("[Shot 1] The boat settles into the final state in <Picture 1>."),
            ["[Picture 1]"],
        ),
        (
            "Ref2VA",
            {"ref_image_1": torch.zeros(1, 8, 8, 3)},
            "subject_definitions:\n<Subject 1> is the paper boat in <Picture 1>.\n"
            "summary: [reference generation] <Subject 1> crosses the pool.\n"
            "retention_analysis:\n<Picture 1>: fully_preserved - the boat appearance is retained.\n"
            "detailed_description: [Shot 1] <Subject 1> crosses the pool.\n"
            "overall_soundscape: Water moves softly.\nnon_diegetic_music: N/A",
            ["[Picture 1]"],
        ),
    ],
)
def test_auto_mode_end_to_end(monkeypatch, expected_mode, inputs, response, expected_labels):
    captured = {}
    _install_response(monkeypatch, response, captured)
    optimized, original, status, output_pipe = JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args(**inputs))
    assert optimized == response
    assert original == _args()["prompt"]
    assert status == f"Success: model=test-model, mode={expected_mode}, repaired=0"
    assert output_pipe is None
    content = captured["payload"]["messages"][1]["content"]
    assert [item["text"] for item in content if item["type"] == "text" and item["text"].startswith("[Picture")] == expected_labels
    assert f"Resolved mode: {expected_mode}" in captured["payload"]["messages"][0]["content"]


def test_reference_instruction_registers_downstream_video_without_upload(monkeypatch):
    response = (
        "subject_definitions:\n<Video 1> is the downstream source video.\n"
        "summary: [video continuation] The target continues <Video 1>.\n"
        "retention_analysis:\n<Video 1>: fully_preserved - its ending state is retained.\n"
        "detailed_description: [Shot 1] The scene continues from <Video 1>.\n"
        "overall_soundscape: Source ambience continues.\nnon_diegetic_music: N/A"
    )
    captured = {}
    _install_response(monkeypatch, response, captured)
    result = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(reference_instructions="<Video 1> supplies the motion and ending state.")
    )
    assert result[0] == response
    assert result[2] == "Success: model=test-model, mode=Ref2VA, repaired=0"


def test_dialogue_must_survive_static_validation(monkeypatch):
    captured = {}
    _install_response(monkeypatch, _base(), captured)
    result = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(prompt='女孩说：“介绍一下MiniMax H3”', fail_mode="Return Original")
    )
    assert result[0] == result[1]
    assert result[2].startswith("Fallback:")
    assert len(captured["calls"]) == 2


def test_one_low_temperature_format_repair_can_recover(monkeypatch):
    original = '女孩说：“介绍一下MiniMax H3”'
    initial = (
        "integrated_multimodal_description: The girl says <d>[Chinese] 介绍一下MiniMax H3</d>\n"
        "overall_soundscape: Quiet room tone.\nnon_diegetic_music: N/A"
    )
    repaired = _base("[Shot 1] The girl says <d>[Chinese] 介绍一下MiniMax H3</d>")
    responses = iter((initial, repaired))
    calls = []
    retry_flags = []

    def fake_request(url, payload, *args):
        calls.append(payload)
        retry_flags.append(args[-1])
        return {"choices": [{"message": {"content": next(responses)}}]}

    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        fake_request,
    )
    result = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(prompt=original, fail_mode="Stop Workflow")
    )
    assert result == (
        repaired,
        original,
        "Success: model=test-model, mode=T2VA, repaired=1",
        None,
    )
    assert len(calls) == 2
    repair = calls[1]
    assert repair["temperature"] == 0.1
    assert repair["top_p"] == 1.0
    assert "reasoning_effort" not in repair
    assert "chat_template_kwargs" not in repair
    assert "Do not rewrite" in repair["messages"][0]["content"]
    assert "介绍一下MiniMax H3" in repair["messages"][1]["content"]
    assert "prompt must contain at least one shot" in repair["messages"][1]["content"]
    assert "__JR_H3_PRESERVED_LITERAL_01__" in repair["messages"][1]["content"]
    assert retry_flags == [True, False]


def test_repair_shields_and_restores_a_literal_with_inserted_whitespace(monkeypatch):
    original = '女孩说：“介绍一下MiniMax H3”'
    initial = _base(
        "[Shot 1] The girl says <d>[Chinese] 介绍一下 MiniMax H3</d>"
    )
    repaired_with_sentinel = _base(
        "[Shot 1] The girl says <d>[Chinese] __JR_H3_PRESERVED_LITERAL_01__</d>"
    )
    calls = []

    def fake_request(url, payload, *args):
        calls.append(payload)
        response = initial if len(calls) == 1 else repaired_with_sentinel
        return {"choices": [{"message": {"content": response}}]}

    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        fake_request,
    )
    optimized, returned_original, status, output_pipe = (
        JR_H3_OpenAICompatiblePromptOptimizer().optimize(
            **_args(prompt=original, fail_mode="Stop Workflow")
        )
    )
    assert returned_original == original
    assert "介绍一下MiniMax H3" in optimized
    assert "介绍一下 MiniMax H3" not in optimized
    assert "__JR_H3_PRESERVED_LITERAL" not in optimized
    assert status == "Success: model=test-model, mode=T2VA, repaired=1"
    assert len(calls) == 2
    assert output_pipe is None
    candidate = calls[1]["messages"][1]["content"].split("Candidate prompt:\n", 1)[1]
    assert "__JR_H3_PRESERVED_LITERAL_01__" in candidate
    assert "介绍一下 MiniMax H3" not in candidate


def test_ref2va_repair_canonicalizes_inline_section_headings(monkeypatch):
    inline = (
        "subject_definitions: <Subject 1> is a woman drinking tea.\n"
        "summary: <Subject 1> introduces MiniMax H3 and then stands.\n"
        "retention_analysis: <Video 1>: fully_preserved - retain composition and motion.\n"
        "detailed_description: [Shot 1] <Subject 1> speaks and then stands.\n"
        "overall_soundscape: Clear speech and quiet room tone.\n"
        "non_diegetic_music: None."
    )
    calls = []

    def fake_request(url, payload, *args):
        calls.append(payload)
        return {"choices": [{"message": {"content": inline}}]}

    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        fake_request,
    )
    optimized, _, status, output_pipe = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(
            h3_input_mode="Ref2VA",
            reference_instructions=(
                "<Video 1> supplies the overall composition and motion reference."
            ),
        )
    )
    assert optimized.startswith("subject_definitions:\n<Subject 1> is")
    assert "summary:\n<Subject 1> introduces" in optimized
    assert "retention_analysis:\n<Video 1>: fully_preserved" in optimized
    assert "detailed_description:\n[Shot 1]" in optimized
    assert status == "Success: model=test-model, mode=Ref2VA, repaired=1"
    assert len(calls) == 2
    assert output_pipe is None


def test_ref2va_repair_canonicalizes_colon_subject_definition(monkeypatch):
    response = (
        "subject_definitions:\n<Subject 1>: a woman drinking tea.\n"
        "summary: <Subject 1> introduces MiniMax H3.\n"
        "retention_analysis:\n<Video 1>: fully_preserved - retain composition.\n"
        "detailed_description: [Shot 1] <Subject 1> speaks to camera.\n"
        "overall_soundscape: Clear speech and quiet room tone.\n"
        "non_diegetic_music: None."
    )
    calls = []

    def fake_request(url, payload, *args):
        calls.append(payload)
        return {"choices": [{"message": {"content": response}}]}

    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        fake_request,
    )
    optimized, _, status, output_pipe = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(
            h3_input_mode="Ref2VA",
            reference_instructions="<Video 1> supplies composition.",
        )
    )
    assert "<Subject 1> is a woman drinking tea." in optimized
    assert "<Subject 1>: a woman" not in optimized
    assert status == "Success: model=test-model, mode=Ref2VA, repaired=1"
    assert len(calls) == 2
    assert output_pipe is None


@pytest.mark.parametrize(
    ("label", "invalid_value", "expected_value"),
    [
        ("<Video 1>", "fully_copy", "fully_preserved"),
        ("<Audio 1>", "fully_preserved", "fully_copy"),
    ],
)
def test_ref2va_repair_canonicalizes_cross_taxonomy_retention_values(
    monkeypatch, label, invalid_value, expected_value
):
    response = (
        "subject_definitions:\n<Subject 1> is a woman drinking tea.\n"
        "summary: <Subject 1> introduces MiniMax H3.\n"
        f"retention_analysis:\n{label}: {invalid_value} - retain the source.\n"
        "detailed_description: [Shot 1] <Subject 1> speaks to camera.\n"
        "overall_soundscape: Clear speech and quiet room tone.\n"
        "non_diegetic_music: None."
    )
    calls = []

    def fake_request(url, payload, *args):
        calls.append(payload)
        return {"choices": [{"message": {"content": response}}]}

    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        fake_request,
    )
    family = label[1:].split(" ", 1)[0]
    optimized, _, status, output_pipe = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(
            h3_input_mode="Ref2VA",
            reference_instructions=f"{label} supplies the {family.lower()} reference.",
        )
    )
    assert f"{label}: {expected_value}" in optimized
    assert f"{label}: {invalid_value}" not in optimized
    assert status == "Success: model=test-model, mode=Ref2VA, repaired=1"
    assert len(calls) == 2
    assert output_pipe is None


@pytest.mark.parametrize("fail_mode", ["Return Original", "Stop Workflow"])
def test_repair_failure_obeys_final_fail_mode_and_never_retries_more_than_once(
    monkeypatch, fail_mode
):
    calls = []

    def invalid_response(url, payload, *args):
        calls.append(payload)
        return {"choices": [{"message": {"content": "not an H3 prompt"}}]}

    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        invalid_response,
    )
    call = lambda: JR_H3_OpenAICompatiblePromptOptimizer().optimize(  # noqa: E731
        **_args(fail_mode=fail_mode)
    )
    if fail_mode == "Return Original":
        result = call()
        assert result[:2] == (_args()["prompt"], _args()["prompt"])
        assert result[2].startswith("Fallback: missing required section")
    else:
        with pytest.raises(ValueError, match="after one format-repair attempt"):
            call()
    assert len(calls) == 2


def test_repair_cannot_drop_a_preserved_literal(monkeypatch):
    original = '女孩说：“介绍一下MiniMax H3”'
    initial = "not an H3 prompt but 介绍一下MiniMax H3 remains"
    repaired_without_literal = _base("[Shot 1] A girl speaks to camera.")
    responses = iter((initial, repaired_without_literal))
    calls = []

    def fake_request(url, payload, *args):
        calls.append(payload)
        return {"choices": [{"message": {"content": next(responses)}}]}

    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        fake_request,
    )
    result = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(prompt=original, fail_mode="Return Original")
    )
    assert result[:2] == (original, original)
    assert "repair changed protected literal sentinel" in result[2]
    assert len(calls) == 2


def test_repair_cannot_duplicate_a_preserved_literal_sentinel(monkeypatch):
    original = '女孩说：“介绍一下MiniMax H3”'
    initial = "not an H3 prompt but 介绍一下MiniMax H3 remains"
    duplicated = _base(
        "[Shot 1] The girl repeats __JR_H3_PRESERVED_LITERAL_01__ and "
        "__JR_H3_PRESERVED_LITERAL_01__."
    )
    responses = iter((initial, duplicated))

    def fake_request(url, payload, *args):
        return {"choices": [{"message": {"content": next(responses)}}]}

    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        fake_request,
    )
    result = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(prompt=original, fail_mode="Return Original")
    )
    assert result[:2] == (original, original)
    assert "repair changed protected literal sentinel" in result[2]


def test_explicit_mode_conflict_obeys_stop_workflow():
    with pytest.raises(RuntimeError, match="I2VA input validation failed"):
        JR_H3_OpenAICompatiblePromptOptimizer().optimize(
            **_args(h3_input_mode="I2VA", last_frame=torch.zeros(1, 8, 8, 3))
        )


def test_anchor_batch_must_be_single_image():
    with pytest.raises(RuntimeError, match="first_frame must contain exactly one IMAGE"):
        JR_H3_OpenAICompatiblePromptOptimizer().optimize(
            **_args(first_frame=torch.zeros(2, 8, 8, 3))
        )
