import json

import torch
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_openai_prompt_optimizer import (
    JR_H3_OpenAICompatiblePromptOptimizer,
)
from ComfyUI_JR_MiniMaxH3Node.utils.director_pipe import RuntimeMedia, build_director_pipe
from ComfyUI_JR_MiniMaxH3Node.utils.director_state import (
    DEFAULT_DIRECTOR_STATE_JSON,
    director_state_from_dict,
)
from h3_semantic_helpers import base_semantic, ref_semantic

DIALOGUES = (
    "老师好",
    "我这个造型还可以吗？",
    "觉得怎么样？",
    "这个感觉对吗？",
)
SHOT_RANGES = ((0, 2), (2, 5), (5, 7), (7, 10))
SHOT_ACTIONS = ("女孩站起", "女孩跳舞", "女孩转圈", "女孩摆姿势")


def _asset(identifier="picture-asset"):
    return {
        "id": identifier,
        "kind": "image",
        "filename": "reference.png",
        "subfolder": "director-fixtures",
        "type": "input",
        "display_name": "reference.png",
        "status": "ready",
        "duration_seconds": None,
    }


def _four_shot_pipe(role):
    raw = json.loads(DEFAULT_DIRECTOR_STATE_JSON)
    raw["timeline"] = {"duration_seconds": 10, "fps": 24}
    raw["global_direction"] = "Keep the same girl and a continuous performance."
    raw["shots"] = [
        {
            "id": f"shot-{index}",
            "start": start,
            "end": end,
            "direction": f"{action}，女孩说：“{dialogue}”",
            "notes": "",
        }
        for index, ((start, end), action, dialogue) in enumerate(
            zip(SHOT_RANGES, SHOT_ACTIONS, DIALOGUES), 1
        )
    ]
    raw["visual_items"] = [
        {
            "id": "picture",
            "kind": "image",
            "role": role,
            "start": 0,
            "end": 0 if role == "first_frame" else 10,
            "source_in": None,
            "source_out": None,
            "direction": "Preserve the same identity throughout.",
            "notes": "",
            "registry_order": 1,
            "asset": _asset(),
        }
    ]
    raw["audio_items"] = []
    state = director_state_from_dict(raw)
    return build_director_pipe(
        state,
        runtime_resolver=lambda _state: (
            RuntimeMedia("picture-asset", "picture", "image", torch.zeros(1, 8, 8, 3)),
        ),
    )


def _args(pipe):
    return dict(
        prompt="",
        enable=True,
        api_base_url="http://127.0.0.1:10000",
        model="fixture-model",
        prompt_profile="Standard",
        duration_seconds=10,
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
        pip=pipe,
    )


def _dialogue_semantics():
    return tuple(
        ({
            "literal_index": index,
            "speaker_key": "girl",
            "speaker_description": "The same young woman",
            "delivery": "says naturally",
        },)
        for index in range(1, 5)
    )


def _install(monkeypatch, response):
    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        lambda *_args, **_kwargs: {"choices": [{"message": {"content": response}}]},
    )


def test_case_a_pipe_first_frame_preserves_four_shot_timing_and_dialogue(monkeypatch):
    pipe = _four_shot_pipe("first_frame")
    response = base_semantic(
        tuple(f"The girl {action}." for action in SHOT_ACTIONS),
        starts=(0, 2, 5, 7),
        dialogues=_dialogue_semantics(),
    )
    _install(monkeypatch, response)
    optimized, _, status, output_pipe = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(pipe)
    )
    assert "mode=I2VA" in status
    assert "[Shot 1]" in optimized and "[Shot 1] At" not in optimized
    assert "[Shot 2] At 00:02.000," in optimized
    assert "[Shot 3] At 00:05.000," in optimized
    assert "[Shot 4] At 00:07.000," in optimized
    assert output_pipe.optimized_prompt == optimized
    assert output_pipe.timeline is pipe.timeline and output_pipe.shots is pipe.shots


def test_case_b_one_full_timeline_reference_image_remains_one_picture(monkeypatch):
    pipe = _four_shot_pipe("reference_image")
    response = ref_semantic(
        ("<Picture 1>",),
        tuple(f"The same girl {action}." for action in SHOT_ACTIONS),
        starts=(0, 2, 5, 7),
        dialogues=_dialogue_semantics(),
    )
    _install(monkeypatch, response)
    optimized, _, status, output_pipe = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_args(pipe)
    )
    assert "mode=Ref2VA" in status
    assert tuple(record.label for record in output_pipe.reference_registry) == ("<Picture 1>",)
    assert optimized.count("<Picture 1> is") == 1
    assert "<Picture 2>" not in optimized
    assert "<Picture 1>: fully_preserved" in optimized
    assert "[Shot 2] At 00:02.000," in optimized
    assert "[Shot 4] At 00:07.000," in optimized


def test_case_c_four_chinese_literals_stable_speaker_and_never_enter_soundscape(monkeypatch):
    pipe = _four_shot_pipe("first_frame")
    response = base_semantic(
        tuple(f"The girl {action}." for action in SHOT_ACTIONS),
        starts=(0, 2, 5, 7),
        dialogues=_dialogue_semantics(),
        soundscape="Soft footsteps, clothing movement, and quiet room ambience.",
    )
    _install(monkeypatch, response)
    optimized = JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_args(pipe))[0]
    description, soundscape = optimized.split("overall_soundscape:", 1)
    for literal in DIALOGUES:
        assert optimized.count(literal) == 1
        assert f"(S1) says naturally: <d>[Chinese] {literal}</d>" in description
        assert literal not in soundscape
    assert "(S2)" not in optimized
