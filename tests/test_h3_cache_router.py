import json

import pytest
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_cache_config_router import (
    SYSTEM_PROMPT,
    JR_H3_CacheConfigRouter,
    SceneClassification,
    parse_classifier_content,
    select_reviewed_profile,
)


def _classification(**changes):
    data = dict(scene_class="mixed", speech_intensity="medium", motion_intensity="medium",
                camera_motion="medium", lip_sync_critical=False, audio_timing_sensitive=True,
                recommended_profile="balanced", confidence=0.9, reason="Mixed timing risk.")
    data.update(changes)
    return data


def _args(**changes):
    data = dict(optimized_prompt="FINAL PROMPT <Picture 1>", enable=True, api_base_url="http://local.test/v1",
                model="classifier", api_key="SECRET", temperature=0.0, top_p=1.0, max_tokens=256,
                timeout_seconds=10, disable_reasoning=True, quality_level="Balanced", cache_device="Auto",
                gpu_reserve_mb=2048, fail_mode="Safe Balanced", audio_content="Auto",
                has_reference_audio=False, has_reference_video=False)
    data.update(changes)
    return data


@pytest.mark.parametrize("wrapped", [
    lambda text: text,
    lambda text: "```json\n" + text + "\n```",
    lambda text: "classification follows: " + text + " trailing text",
])
def test_json_parser_accepts_standard_fence_and_prefix(wrapped):
    result = parse_classifier_content(wrapped(json.dumps(_classification())))
    assert result.recommended_profile == "balanced"


def test_parser_rejects_unknown_enum_and_missing_fields_and_clamps_confidence():
    bad = _classification(scene_class="unknown")
    with pytest.raises(ValueError, match="scene_class"):
        parse_classifier_content(json.dumps(bad))
    bad = _classification(); bad.pop("reason")
    with pytest.raises(ValueError, match="missing"):
        parse_classifier_content(json.dumps(bad))
    assert parse_classifier_content(json.dumps(_classification(confidence=8))).confidence == 1.0
    with pytest.raises(ValueError, match="unsupported"):
        parse_classifier_content(json.dumps({**_classification(), "video_threshold": 0.9}))


@pytest.mark.parametrize(("changes", "expected"), [
    ({"speech_intensity": "high", "motion_intensity": "low"}, "dialogue_safe"),
    ({"speech_intensity": "none", "motion_intensity": "high"}, "action_safe"),
    ({"speech_intensity": "none", "motion_intensity": "low", "camera_motion": "low"}, "visual_fast"),
    ({"speech_intensity": "high", "motion_intensity": "high"}, "balanced"),
    ({"lip_sync_critical": True, "motion_intensity": "high"}, "balanced"),
])
def test_local_semantic_review(changes, expected):
    result = SceneClassification(**_classification(**changes))
    assert select_reviewed_profile(result) == expected


def test_speech_auxiliary_prevents_visual_fast():
    result = SceneClassification(**_classification(speech_intensity="none", motion_intensity="low", camera_motion="low"))
    assert select_reviewed_profile(result, audio_content="Speech") == "dialogue_safe"


def test_router_uses_independent_prompt_and_local_numbers(monkeypatch):
    captured = {}
    def request(_url, payload, *_args):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": json.dumps(_classification(
            speech_intensity="high", motion_intensity="low", recommended_profile="visual_fast"))}}]}
    monkeypatch.setattr("ComfyUI_JR_MiniMaxH3Node.nodes.h3_cache_config_router.request_chat", request)
    config, profile, analysis = JR_H3_CacheConfigRouter().route(**_args())
    assert profile == "dialogue_safe"
    assert config.video_threshold > 0 and config.source == "router"
    assert "threshold" not in captured["payload"]["messages"][1]["content"].lower()
    assert captured["payload"]["messages"][0]["content"] == SYSTEM_PROMPT
    assert captured["payload"]["messages"][1]["content"].endswith("FINAL PROMPT <Picture 1>")
    assert "SECRET" not in config.__dict__.values() and "FINAL PROMPT" not in repr(config)
    assert analysis


def test_router_fallback_modes_and_disabled_do_not_call(monkeypatch):
    monkeypatch.setattr("ComfyUI_JR_MiniMaxH3Node.nodes.h3_cache_config_router.request_chat",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("down")))
    config, profile, _ = JR_H3_CacheConfigRouter().route(**_args())
    assert profile == "balanced" and config.quality_level == "Conservative"
    config, profile, _ = JR_H3_CacheConfigRouter().route(**_args(fail_mode="Disable Cache"))
    assert profile == "off"
    with pytest.raises(RuntimeError):
        JR_H3_CacheConfigRouter().route(**_args(fail_mode="Stop Workflow"))
    monkeypatch.setattr("ComfyUI_JR_MiniMaxH3Node.nodes.h3_cache_config_router.request_chat",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not call")))
    config, _, _ = JR_H3_CacheConfigRouter().route(**_args(enable=False))
    assert config.source == "router_disabled_fallback"


def test_prompt_optimizer_source_is_not_modified_by_router():
    import inspect

    from ComfyUI_JR_MiniMaxH3Node.nodes import h3_openai_prompt_optimizer
    assert "SYSTEM_PROMPT" not in inspect.getsource(h3_openai_prompt_optimizer)
    assert "cache_config" not in h3_openai_prompt_optimizer.JR_H3_OpenAICompatiblePromptOptimizer.RETURN_NAMES
