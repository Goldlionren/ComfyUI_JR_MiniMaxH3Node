import json

import pytest
from ComfyUI_JR_MiniMaxH3Node.utils.director_state import (
    DEFAULT_DIRECTOR_STATE_JSON,
    DirectorStateError,
    director_state_from_dict,
    director_state_from_json,
    director_state_to_dict,
    director_state_to_json,
)


def test_default_state_roundtrip_is_schema_versioned_and_json_only():
    state = director_state_from_json(DEFAULT_DIRECTOR_STATE_JSON)
    assert state.schema == "jr_h3_director_state"
    assert state.schema_version == 1
    assert state.timeline.duration_seconds == 10.0
    assert len(state.shots) == 1
    encoded = director_state_to_json(state)
    assert director_state_from_json(encoded) == state
    assert "base64" not in encoded.lower()
    assert "tensor" not in encoded.lower()


def test_unicode_asset_descriptor_and_direction_roundtrip():
    raw = json.loads(DEFAULT_DIRECTOR_STATE_JSON)
    raw["global_direction"] = "雨夜，保持霓虹方向。"
    raw["visual_items"] = [{
        "id": "visual-1", "kind": "image", "role": "reference_image",
        "start": 0, "end": 10, "direction": "锁定人物衣服。", "notes": "镜头 A",
        "registry_order": 7,
        "asset": {
            "id": "asset-1", "kind": "image", "filename": "参考 图.png",
            "subfolder": "导演台/镜头 一", "type": "input", "display_name": "参考 图.png",
            "mime_type": "image/png", "width": 64, "height": 64, "status": "ready",
        },
    }]
    state = director_state_from_dict(raw)
    restored = director_state_from_json(director_state_to_json(state))
    assert restored == state
    assert restored.visual_items[0].asset.subfolder == "导演台/镜头 一"


@pytest.mark.parametrize(
    ("filename", "subfolder"),
    [
        ("../secret.png", ""),
        ("C:/secret.png", ""),
        ("secret.png", "../outside"),
        ("secret.png", "/absolute"),
        ("folder/secret.png", ""),
    ],
)
def test_asset_descriptor_rejects_unsafe_paths(filename, subfolder):
    raw = json.loads(DEFAULT_DIRECTOR_STATE_JSON)
    raw["visual_items"] = [{
        "id": "visual-1", "kind": "image", "role": "reference_image",
        "start": 0, "end": 10, "direction": "", "notes": "", "registry_order": 1,
        "asset": {
            "id": "asset-1", "kind": "image", "filename": filename,
            "subfolder": subfolder, "type": "input", "display_name": "asset",
        },
    }]
    with pytest.raises(DirectorStateError, match="relative|unsafe|basename"):
        director_state_from_dict(raw)


def test_state_size_and_schema_are_bounded():
    with pytest.raises(DirectorStateError, match="byte limit"):
        director_state_from_json(" " * (512 * 1024 + 1))
    raw = json.loads(DEFAULT_DIRECTOR_STATE_JSON)
    raw["schema_version"] = 99
    with pytest.raises(DirectorStateError, match="schema_version"):
        director_state_from_dict(raw)
    raw["schema_version"] = True
    with pytest.raises(DirectorStateError, match="schema_version"):
        director_state_from_dict(raw)


def test_time_rounding_and_lane_order_match_frontend_contract():
    raw = json.loads(DEFAULT_DIRECTOR_STATE_JSON)
    raw["shots"][0]["end"] = 0.25
    raw["timeline"]["duration_seconds"] = 0.25
    state = director_state_from_dict(raw)
    assert state.timeline.duration_seconds == 0.3
    assert state.shots[0].end == 0.3

    raw = json.loads(DEFAULT_DIRECTOR_STATE_JSON)
    raw["ui"]["lane_order"] = {"visual": ["missing"], "audio": []}
    with pytest.raises(DirectorStateError, match="unknown visual item"):
        director_state_from_dict(raw)


def test_extreme_time_and_deep_json_fail_as_director_state_errors():
    raw = json.loads(DEFAULT_DIRECTOR_STATE_JSON)
    raw["timeline"]["duration_seconds"] = 1e308
    with pytest.raises(DirectorStateError, match="0.1-second precision"):
        director_state_from_dict(raw)

    deeply_nested = "[" * 5000 + "0" + "]" * 5000
    with pytest.raises(DirectorStateError, match="nesting is too deep"):
        director_state_from_json(deeply_nested)

    with pytest.raises(DirectorStateError, match="unsupported JSON value"):
        director_state_from_json("9" * 5000)

    raw = json.loads(DEFAULT_DIRECTOR_STATE_JSON)
    raw["timeline"]["fps"] = 10 ** 1000
    with pytest.raises(DirectorStateError, match="finite number"):
        director_state_from_dict(raw)


def test_serialized_state_does_not_expose_runtime_fields():
    data = director_state_to_dict(director_state_from_json(DEFAULT_DIRECTOR_STATE_JSON))
    assert "runtime_media" not in data
    assert "compiled_director_prompt" not in data
    assert "absolute_path" not in json.dumps(data)
    assert data["ui"]["lane_order"] == {"visual": [], "audio": []}
