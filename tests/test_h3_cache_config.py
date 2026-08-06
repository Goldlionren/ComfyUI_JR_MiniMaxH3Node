from dataclasses import FrozenInstanceError

import pytest
from ComfyUI_JR_MiniMaxH3Node.utils.h3_cache_config import (
    H3CacheConfig,
    build_preset_config,
    select_manual_profile,
)


@pytest.mark.parametrize(("audio", "profile"), [
    ("Speech", "dialogue_safe"), ("Singing", "dialogue_safe"),
    ("Music", "visual_fast"), ("Ambient", "visual_fast"),
    ("None", "visual_fast"), ("Auto", "balanced"),
])
def test_auto_profile(audio, profile):
    assert select_manual_profile("Auto", audio) == profile


def test_profile_hint_precedes_audio_and_invalid_hint_is_safe():
    assert select_manual_profile("Auto", "Speech", "action_safe") == "action_safe"
    assert select_manual_profile("Auto", "Auto", "nonsense") == "balanced"


def test_config_is_frozen_and_contains_no_secrets_or_prompt():
    config = build_preset_config("balanced", analysis_summary="short classification")
    with pytest.raises(FrozenInstanceError):
        config.profile = "off"
    assert "api_key" not in config.__dict__
    assert "optimized_prompt" not in config.__dict__


def test_quality_levels_are_deterministic_and_local():
    conservative = build_preset_config("balanced", "Conservative")
    balanced_a = build_preset_config("balanced", "Balanced")
    balanced_b = build_preset_config("balanced", "Balanced")
    aggressive = build_preset_config("balanced", "Aggressive")
    assert balanced_a == balanced_b
    assert conservative.video_threshold < balanced_a.video_threshold < aggressive.video_threshold
    assert conservative.max_block_hits <= balanced_a.max_block_hits < aggressive.max_block_hits


def test_calibrated_profiles_preserve_safety_order_and_viable_h3_scale():
    visual = build_preset_config("visual_fast")
    dialogue = build_preset_config("dialogue_safe")
    action = build_preset_config("action_safe")
    balanced = build_preset_config("balanced")
    assert action.video_threshold < dialogue.video_threshold < visual.video_threshold
    assert action.audio_threshold < dialogue.audio_threshold < visual.audio_threshold
    assert action.max_full_step_hits == dialogue.max_full_step_hits == 0
    assert dialogue.video_threshold >= 0.10 and dialogue.audio_threshold >= 0.08
    assert balanced.fast_path_threshold < balanced.probe_path_threshold


def test_invalid_schema_and_ranges_are_rejected():
    values = build_preset_config("balanced").__dict__.copy()
    values["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        H3CacheConfig(**values)
    values = build_preset_config("balanced").__dict__.copy()
    values["video_metric_stride"] = 0
    with pytest.raises(ValueError, match="stride"):
        H3CacheConfig(**values)
