import json
from dataclasses import replace

import pytest
from ComfyUI_JR_MiniMaxH3Node.utils.director_prompt_compiler import (
    DirectorValidationError,
    build_reference_registry,
    compile_director_prompt,
    validate_director_state,
)
from ComfyUI_JR_MiniMaxH3Node.utils.director_state import (
    DEFAULT_DIRECTOR_STATE_JSON,
    AssetDescriptor,
    AudioState,
    director_state_from_dict,
    director_state_from_json,
)


def _state(**updates):
    raw = json.loads(DEFAULT_DIRECTOR_STATE_JSON)
    raw.update(updates)
    return director_state_from_dict(raw)


def _asset(identifier, kind, name):
    return {
        "id": identifier, "kind": kind, "filename": name, "subfolder": "导演台",
        "type": "input", "display_name": name, "status": "ready",
        "duration_seconds": 10 if kind != "image" else None,
    }


def test_compiler_keeps_overlapping_references_and_stable_registry_order():
    visual = [
        {"id": "image-b", "kind": "image", "role": "reference_image", "start": 0, "end": 10,
         "direction": "B direction", "notes": "B notes", "registry_order": 2,
         "asset": _asset("asset-b", "image", "B 图.png")},
        {"id": "image-a", "kind": "image", "role": "reference_image", "start": 0, "end": 10,
         "direction": "A direction", "notes": "A notes", "registry_order": 1,
         "asset": _asset("asset-a", "image", "A 图.png")},
        {"id": "video-a", "kind": "video", "role": "reference_video", "start": 5, "end": 10,
         "source_in": 1, "source_out": 6, "direction": "Video direction", "notes": "",
         "registry_order": 1, "asset": _asset("asset-v", "video", "参考.mp4")},
    ]
    first = _state(visual_items=visual)
    second = _state(visual_items=list(reversed(visual)))
    registry = build_reference_registry(first)
    assert [entry.label for entry in registry] == ["<Picture 1>", "<Picture 2>", "<Video 1>"]
    prompt = compile_director_prompt(first, registry)
    assert prompt == compile_director_prompt(second)
    assert prompt.count("<Picture 1>") == 2  # registry plus active Shot line
    assert "<Picture 2>" in prompt and "<Video 1>" in prompt
    assert "source_range: 1.0s-6.0s" in prompt


def test_first_frame_is_unique_zero_second_point_anchor():
    item = {
        "id": "first", "kind": "image", "role": "first_frame", "start": 0, "end": 0,
        "direction": "Opening anchor", "notes": "", "registry_order": 99,
        "asset": _asset("asset-first", "image", "首帧.png"),
    }
    state = _state(visual_items=[item])
    assert build_reference_registry(state)[0].label == "<Picture 1>"
    assert "timeline=0.0s point anchor" in compile_director_prompt(state)
    with pytest.raises(DirectorValidationError, match="Only one First Frame"):
        validate_director_state(_state(visual_items=[item, {**item, "id": "first-2", "asset": {**item["asset"], "id": "asset-2"}}]))
    with pytest.raises(DirectorValidationError, match="first Shot start at 0.0"):
        validate_director_state(_state(
            shots=[{"id": "late", "start": 2, "end": 10, "direction": "", "notes": ""}],
            visual_items=[item],
        ))


def test_image_source_range_is_rejected_before_prompt_compilation():
    image = {
        "id": "image-a", "kind": "image", "role": "reference_image", "start": 0, "end": 10,
        "source_in": 1, "source_out": None, "direction": "", "notes": "", "registry_order": 1,
        "asset": _asset("asset-image", "image", "reference.png"),
    }
    with pytest.raises(DirectorValidationError, match="IMAGE assets cannot define a source range"):
        validate_director_state(_state(visual_items=[image]))


def test_shot_overlap_is_rejected_but_touching_is_valid():
    touching = _state(shots=[
        {"id": "s1", "start": 0, "end": 5, "direction": "A", "notes": ""},
        {"id": "s2", "start": 5, "end": 10, "direction": "B", "notes": ""},
    ])
    validate_director_state(touching)
    with pytest.raises(DirectorValidationError, match="overlap"):
        validate_director_state(_state(shots=[
            {"id": "s1", "start": 0, "end": 6, "direction": "A", "notes": ""},
            {"id": "s2", "start": 5, "end": 10, "direction": "B", "notes": ""},
        ]))


def test_reference_audio_overlap_is_legal_and_driving_overlap_is_rejected():
    base = {
        "start": 0, "end": 6, "source_in": 0, "source_out": 6,
        "direction": "", "notes": "", "registry_order": 1,
        "asset": _asset("asset-a", "audio", "A.wav"),
    }
    reference = _state(audio_items=[
        {**base, "id": "a1", "role": "reference_audio"},
        {**base, "id": "a2", "role": "reference_audio", "start": 2, "end": 8,
         "source_in": 1, "source_out": 7, "registry_order": 2,
         "asset": _asset("asset-b", "audio", "B.wav")},
    ])
    validate_director_state(reference)
    raw = json.loads(json.dumps({"schema": reference.schema, "schema_version": 1}))
    assert raw["schema"] == "jr_h3_director_state"
    with pytest.raises(DirectorValidationError, match="Driving Audio"):
        validate_director_state(_state(audio_items=[
            {**base, "id": "a1", "role": "driving_audio"},
            {**base, "id": "a2", "role": "driving_audio", "start": 2, "end": 8,
             "source_in": 1, "source_out": 7, "registry_order": 2,
             "asset": _asset("asset-b", "audio", "B.wav")},
        ]))


def test_compiler_output_has_required_sections_and_exact_times():
    state = _state(
        global_direction="电影感，保持人物一致。",
        shots=[
            {"id": "s1", "start": 0, "end": 3.0, "direction": "Start", "notes": ""},
            {"id": "s2", "start": 3.0, "end": 10, "direction": "Finish", "notes": "continuity"},
        ],
    )
    prompt = compile_director_prompt(state)
    assert prompt.startswith("GLOBAL DIRECTION\n")
    assert "\nREFERENCE MEDIA\n" in prompt
    assert "\nTIMELINE\n" in prompt
    assert "[Shot 2 | 3.0s-10.0s | id=s2]" in prompt
    assert prompt.endswith("final_shot: s2 (3.0s-10.0s)")


def test_direct_dataclass_state_cannot_bypass_parser_contracts():
    state = director_state_from_json(DEFAULT_DIRECTOR_STATE_JSON)
    asset = AssetDescriptor(
        id="audio-asset", kind="audio", filename="audio.wav", subfolder="",
        folder_type="input", display_name="audio.wav",
    )
    invalid = AudioState(
        id="audio-1", role="bogus", start=-0.1, end=1.0, source_in=0.0, source_out=1.0,
        direction="", notes="", registry_order=1, asset=asset,
    )
    with pytest.raises(DirectorValidationError, match="role is unsupported|at least 0.0"):
        validate_director_state(replace(state, audio_items=(invalid,)))
