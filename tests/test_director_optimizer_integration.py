import json

import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_openai_prompt_optimizer import (
    JR_H3_OpenAICompatiblePromptOptimizer,
)
from ComfyUI_JR_MiniMaxH3Node.utils.director_pipe import RuntimeMedia, build_director_pipe
from ComfyUI_JR_MiniMaxH3Node.utils.director_state import (
    DEFAULT_DIRECTOR_STATE_JSON,
    director_state_from_dict,
)


def _args(**updates):
    values = dict(
        prompt="", enable=True, api_base_url="http://127.0.0.1:10000", model="test-model",
        prompt_profile="Standard", duration_seconds=60, target_width=768, target_height=1152,
        temperature=0.0, top_p=1.0, max_tokens=1800, timeout_seconds=2,
        image_send_size=768, fail_mode="Stop Workflow", disable_reasoning=True,
        h3_input_mode="Auto", reference_instructions="", api_key="",
    )
    values.update(updates)
    return values


def _asset(identifier, kind, filename):
    return {
        "id": identifier, "kind": kind, "filename": filename, "subfolder": "director",
        "type": "input", "display_name": filename, "status": "ready",
        "duration_seconds": 5 if kind != "image" else None,
    }


def _reference_pipe():
    raw = json.loads(DEFAULT_DIRECTOR_STATE_JSON)
    raw["timeline"]["duration_seconds"] = 5
    raw["shots"] = [{"id": "shot-1", "start": 0, "end": 5, "direction": "A boat crosses.", "notes": ""}]
    raw["visual_items"] = [
        {"id": "picture", "kind": "image", "role": "reference_image", "start": 0, "end": 5,
         "source_in": None, "source_out": None, "direction": "Keep the paper texture.", "notes": "",
         "registry_order": 1, "asset": _asset("image-asset", "image", "参考.png")},
        {"id": "video", "kind": "video", "role": "reference_video", "start": 0, "end": 5,
         "source_in": 0, "source_out": 5, "direction": "Use the motion rhythm.", "notes": "",
         "registry_order": 1, "asset": _asset("video-asset", "video", "motion.mp4")},
    ]
    raw["audio_items"] = [
        {"id": "audio", "role": "reference_audio", "start": 0, "end": 5,
         "source_in": 0, "source_out": 5, "direction": "Use water ambience.", "notes": "",
         "registry_order": 1, "asset": _asset("audio-asset", "audio", "water.wav")},
    ]
    state = director_state_from_dict(raw)
    image = torch.zeros(1, 8, 8, 3)
    return build_director_pipe(
        state,
        runtime_resolver=lambda _state: (
            RuntimeMedia("image-asset", "picture", "image", image),
            RuntimeMedia("video-asset", "video", "video", None),
            RuntimeMedia("audio-asset", "audio", "audio", None),
        ),
    )


def _first_frame_pipe():
    raw = json.loads(DEFAULT_DIRECTOR_STATE_JSON)
    raw["timeline"]["duration_seconds"] = 5
    raw["shots"] = [{"id": "shot-1", "start": 0, "end": 5, "direction": "She turns.", "notes": ""}]
    raw["visual_items"] = [{
        "id": "first", "kind": "image", "role": "first_frame", "start": 0, "end": 0,
        "source_in": None, "source_out": None, "direction": "Opening composition.", "notes": "",
        "registry_order": 1, "asset": _asset("first-asset", "image", "first.png"),
    }]
    state = director_state_from_dict(raw)
    return build_director_pipe(
        state,
        runtime_resolver=lambda _state: (RuntimeMedia("first-asset", "first", "image", torch.zeros(1, 8, 8, 3)),),
    )


def _install(monkeypatch, response, captured):
    def fake_request(url, payload, *args):
        captured["payload"] = payload
        captured["calls"] = captured.get("calls", 0) + 1
        return {"choices": [{"message": {"content": response}}]}

    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        fake_request,
    )


def test_pip_is_authoritative_and_video_audio_are_text_only(monkeypatch):
    response = (
        "subject_definitions:\n<Subject 1> is a paper boat from <Picture 1>.\n"
        "summary: [reference generation] <Subject 1> crosses the water.\n"
        "retention_analysis:\n"
        "<Picture 1>: fully_preserved - preserve the paper texture.\n"
        "<Video 1>: partially_preserved - use its motion rhythm.\n"
        "<Audio 1>: fully_copy - use the water ambience.\n"
        "detailed_description: [Shot 1] <Subject 1> crosses the water.\n"
        "overall_soundscape: Water moves softly.\nnon_diegetic_music: N/A"
    )
    pipe = _reference_pipe()
    captured = {}
    _install(monkeypatch, response, captured)
    before = pipe.to_persisted()
    optimized, original, status = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(pip=pipe)
    )
    assert optimized == response
    assert original == pipe.compiled_director_prompt
    assert status == "Success: model=test-model, mode=Ref2VA, repaired=0, source=pip"
    assert pipe.to_persisted() == before
    payload = captured["payload"]
    assert "Target duration: 5 seconds" in payload["messages"][1]["content"][0]["text"]
    assert sum(item["type"] == "image_url" for item in payload["messages"][1]["content"]) == 1
    text = payload["messages"][1]["content"][0]["text"]
    assert "<Video 1>" in text and "<Audio 1>" in text
    assert "data:video" not in str(payload) and "data:audio" not in str(payload)


def test_first_frame_only_pip_routes_to_i2va(monkeypatch):
    response = (
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
        "integrated_multimodal_description: [Shot 1] She turns from <Picture 1>.\n"
        "overall_soundscape: Quiet room tone.\nnon_diegetic_music: N/A"
    )
    pipe = _first_frame_pipe()
    captured = {}
    _install(monkeypatch, response, captured)
    result = JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args(pip=pipe))
    assert result[2] == "Success: model=test-model, mode=I2VA, repaired=0, source=pip"
    assert sum(item["type"] == "image_url" for item in captured["payload"]["messages"][1]["content"]) == 1


@pytest.mark.parametrize(
    "conflict",
    [
        {"prompt": "different prompt"},
        {"reference_instructions": "<Video 9> legacy"},
        {"first_frame": torch.zeros(1, 8, 8, 3)},
        {"ref_image_1": torch.zeros(1, 8, 8, 3)},
    ],
)
def test_pip_legacy_conflicts_fail_before_network(monkeypatch, conflict):
    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        lambda *_args, **_kwargs: pytest.fail("network must not be called for a PIP conflict"),
    )
    with pytest.raises(RuntimeError, match="Director PIP conflict"):
        JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args(pip=_reference_pipe(), **conflict))


def test_pip_return_original_fallback_returns_director_prompt(monkeypatch):
    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        lambda *_args, **_kwargs: pytest.fail("network must not be called"),
    )
    pipe = _reference_pipe()
    result = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(pip=pipe, prompt="conflicting", fail_mode="Return Original")
    )
    assert result[0] == result[1] == pipe.compiled_director_prompt
    assert result[2].startswith("Fallback: ValueError: Director PIP conflict")


def test_disabled_pip_returns_compiled_prompt_without_network(monkeypatch):
    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        lambda *_args, **_kwargs: pytest.fail("disabled optimizer must not call network"),
    )
    pipe = _reference_pipe()
    result = JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args(pip=pipe, enable=False))
    assert result == (pipe.compiled_director_prompt, pipe.compiled_director_prompt, "Disabled: original prompt returned")


def test_pip_input_is_appended_without_changing_legacy_optional_order():
    optional = list(JR_H3_OpenAICompatiblePromptOptimizer.INPUT_TYPES()["optional"])
    assert optional == ["api_key", *[f"ref_image_{index}" for index in range(1, 10)], "first_frame", "last_frame", "pip"]
