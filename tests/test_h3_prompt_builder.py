import pytest
from ComfyUI_JR_MiniMaxH3Node.utils.h3_prompt_builder import (
    JR_DIRECTOR_PROFILES,
    PromptBuildContext,
    build_system_prompt,
    build_user_prompt,
    extract_preserved_literals,
    extract_protected_dialogues,
    registry_as_text,
)


def _context(**overrides):
    values = dict(
        original_prompt='女孩说：“介绍一下 MiniMax H3”',
        profile="Standard",
        mode="T2VA",
        duration_seconds=10,
        target_width=768,
        target_height=1152,
    )
    values.update(overrides)
    return PromptBuildContext(**values)


@pytest.mark.parametrize("profile", list(JR_DIRECTOR_PROFILES))
def test_system_prompt_requests_semantic_json_and_keeps_profile_subordinate(profile):
    prompt = build_system_prompt(_context(profile=profile))
    assert "Return one JSON object containing audiovisual semantics" in prompt
    assert "Do not output the final H3 prompt" in prompt
    assert "Python formats and validates the official output" in prompt
    assert "JR creative director layer" in prompt
    assert JR_DIRECTOR_PROFILES[profile] in prompt


@pytest.mark.parametrize("mode", ["T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"])
def test_mode_contract_is_semantic_only(mode):
    prompt = build_system_prompt(_context(mode=mode, duration_seconds=8))
    assert f"Resolved mode: {mode}" in prompt
    assert '"shots"' in prompt
    assert "never emit these headings" in prompt
    assert "Shot headers" in prompt and "timestamps" in prompt


def test_ref_contract_contains_both_official_retention_taxonomies():
    prompt = build_system_prompt(_context(mode="Ref2VA"))
    assert "fully_preserved" in prompt and "attribute_transfer" in prompt
    assert "fully_copy" in prompt and "partially_copy" in prompt
    assert "references must contain every allowed label exactly once and in order" in prompt


def test_system_prompt_carries_authoritative_shot_starts_and_reference_order():
    prompt = build_system_prompt(
        _context(
            mode="Ref2VA",
            shot_starts=(0.0, 2.0, 7.5),
            reference_labels=("<Picture 1>", "<Video 1>", "<Audio 1>"),
        )
    )
    assert "Authoritative Director shot starts: 0, 2, 7.5" in prompt
    assert "Allowed reference labels in exact order: <Picture 1>, <Video 1>, <Audio 1>" in prompt


def test_extract_preserved_and_dialogue_literals_keep_unicode_in_order():
    text = '牌子写着“JR-42”，女孩说：“介绍一下 MiniMax H3”。 He says "Keep this exact."'
    assert extract_preserved_literals(text) == (
        "JR-42",
        "介绍一下 MiniMax H3",
        "Keep this exact.",
    )
    assert extract_protected_dialogues(text) == (
        "介绍一下 MiniMax H3",
        "Keep this exact.",
    )


def test_existing_dialogue_block_is_protected():
    text = "(S1) says <d>[Chinese] 请介绍一下</d>"
    assert extract_protected_dialogues(text) == ("请介绍一下",)
    assert extract_preserved_literals(text) == ("请介绍一下",)


def test_user_prompt_carries_original_and_indexed_dialogue_without_requesting_final_text():
    context = _context(protected_dialogues=("介绍一下 MiniMax H3",))
    result = build_user_prompt(context, extract_preserved_literals(context.original_prompt))
    assert context.original_prompt in result
    assert "literal_index=1: 介绍一下 MiniMax H3" in result
    assert "do not copy their text into JSON" in result
    assert "Resolved input mode: T2VA" in result


def test_registry_serialization_is_duck_typed():
    class Entry:
        label = "<Picture 1>"
        source_input = "first_frame"
        role = "first_frame"
        subject_binding = None

    assert registry_as_text([Entry()]) == "<Picture 1>: source_input=first_frame; role=first_frame"
