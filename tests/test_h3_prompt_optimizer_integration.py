import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_openai_prompt_optimizer import (
    JR_H3_OpenAICompatiblePromptOptimizer,
)
from h3_semantic_helpers import base_semantic, ref_semantic


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


def _install_responses(monkeypatch, responses, captured):
    pending = iter(responses if isinstance(responses, (list, tuple)) else (responses,))

    def fake_request(url, payload, *args):
        captured["url"] = url
        captured.setdefault("calls", []).append(payload)
        return {"choices": [{"message": {"content": next(pending)}}]}

    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        fake_request,
    )


@pytest.mark.parametrize(
    ("expected_mode", "inputs", "semantic", "expected_labels"),
    [
        ("T2VA", {}, base_semantic(), []),
        (
            "I2VA",
            {"first_frame": torch.zeros(1, 8, 8, 3)},
            base_semantic(("The boat starts from <Picture 1> and moves forward.",)),
            ["[Picture 1]"],
        ),
        (
            "FL2VA",
            {
                "first_frame": torch.zeros(1, 8, 8, 3),
                "last_frame": torch.ones(1, 8, 8, 3),
            },
            base_semantic(("The boat moves continuously from <Picture 1> to <Picture 2>.",)),
            ["[Picture 1]", "[Picture 2]"],
        ),
        (
            "L2VA",
            {"last_frame": torch.ones(1, 8, 8, 3)},
            base_semantic(("The boat settles into the final state in <Picture 1>.",)),
            ["[Picture 1]"],
        ),
        (
            "Ref2VA",
            {"ref_image_1": torch.zeros(1, 8, 8, 3)},
            ref_semantic(("<Picture 1>",)),
            ["[Picture 1]"],
        ),
    ],
)
def test_auto_mode_semantic_json_to_deterministic_official_prompt(
    monkeypatch, expected_mode, inputs, semantic, expected_labels
):
    captured = {}
    _install_responses(monkeypatch, semantic, captured)
    optimized, original, status, output_pipe = (
        JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args(**inputs))
    )
    assert original == _args()["prompt"]
    assert status == f"Success: model=test-model, mode={expected_mode}, repaired=0"
    assert output_pipe is None
    assert "integrated_multimodal_description:" in optimized or optimized.startswith(
        "subject_definitions:"
    )
    assert not optimized.lstrip().startswith("{")
    payload = captured["calls"][0]
    assert "Return one JSON object containing audiovisual semantics" in payload["messages"][0]["content"]
    assert "Do not output the final H3 prompt" in payload["messages"][0]["content"]
    assert [
        item["text"] for item in payload["messages"][1]["content"] if item["type"] == "text"
    ][1:] == expected_labels


def test_reference_instruction_registers_video_without_binary_upload(monkeypatch):
    captured = {}
    semantic = ref_semantic(("<Video 1>",))
    _install_responses(monkeypatch, semantic, captured)
    optimized, _, status, _ = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(reference_instructions="<Video 1> supplies the motion rhythm.")
    )
    assert "<Video 1>: fully_preserved" in optimized
    assert "mode=Ref2VA" in status
    payload = captured["calls"][0]
    assert all(item["type"] != "image_url" for item in payload["messages"][1]["content"])
    assert "data:video" not in str(payload)


def test_dialogue_is_inserted_byte_exactly_once_with_deterministic_language_and_speaker(monkeypatch):
    literal = "介绍一下 MiniMax H3"
    semantic = base_semantic(
        ("A presenter turns toward the camera.",),
        dialogues=(({
            "literal_index": 1,
            "speaker_key": "presenter",
            "speaker_description": "The presenter",
            "delivery": "says clearly",
        },),),
    )
    captured = {}
    _install_responses(monkeypatch, semantic, captured)
    optimized, _, _, _ = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(prompt=f'女孩说：“{literal}”')
    )
    assert optimized.count(literal) == 1
    assert f"(S1) says clearly: <d>[Chinese] {literal}</d>" in optimized
    assert literal not in optimized.split("overall_soundscape:", 1)[1]


def test_one_low_temperature_structured_repair_can_recover(monkeypatch):
    captured = {}
    _install_responses(monkeypatch, ["not json", base_semantic()], captured)
    result = JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args())
    assert result[2] == "Success: model=test-model, mode=T2VA, repaired=1"
    assert len(captured["calls"]) == 2
    repair = captured["calls"][1]
    assert repair["temperature"] == 0.1
    assert repair["top_p"] == 1.0
    assert "reasoning_effort" not in repair
    assert "constrained semantic JSON repair" in repair["messages"][0]["content"]
    assert "Preserve only candidate semantics supported by the authoritative source request" in repair["messages"][0]["content"]
    assert "Authoritative source request and protected-dialogue mapping" in repair["messages"][1]["content"]
    assert "A paper boat moves across a shallow pool." in repair["messages"][1]["content"]
    assert "Candidate semantic response:\nnot json" in repair["messages"][1]["content"]


@pytest.mark.parametrize("fail_mode", ["Return Original", "Stop Workflow"])
def test_second_invalid_semantic_response_obeys_fail_mode_and_never_retries(
    monkeypatch, fail_mode
):
    captured = {}
    _install_responses(monkeypatch, ["not json", "still not json"], captured)
    call = lambda: JR_H3_OpenAICompatiblePromptOptimizer().optimize(  # noqa: E731
        **_args(fail_mode=fail_mode)
    )
    if fail_mode == "Return Original":
        result = call()
        assert result[:2] == (_args()["prompt"], _args()["prompt"])
        assert result[2].startswith("Fallback: Semantic response is not valid JSON")
    else:
        with pytest.raises(ValueError, match="after one structured repair"):
            call()
    assert len(captured["calls"]) == 2


def test_repair_cannot_drop_protected_dialogue(monkeypatch):
    captured = {}
    _install_responses(monkeypatch, ["not json", base_semantic()], captured)
    original = '女孩说：“介绍一下 MiniMax H3”'
    result = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(prompt=original, fail_mode="Return Original")
    )
    assert result[:2] == (original, original)
    assert "dialogue literal_index values must reference every protected dialogue exactly once" in result[2]
    assert len(captured["calls"]) == 2


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
