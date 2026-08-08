import pytest
from ComfyUI_JR_MiniMaxH3Node.utils.h3_prompt_builder import (
    JR_DIRECTOR_PROFILES,
    PromptBuildContext,
    build_system_prompt,
    build_user_prompt,
    extract_preserved_literals,
    registry_as_text,
)


def _context(**overrides):
    values = dict(
        original_prompt='女孩说：“介绍一下MiniMax H3”',
        profile="Standard",
        mode="T2VA",
        duration_seconds=10,
        target_width=768,
        target_height=1152,
    )
    values.update(overrides)
    return PromptBuildContext(**values)


@pytest.mark.parametrize("profile", list(JR_DIRECTOR_PROFILES))
def test_director_profiles_are_separate_from_official_contract(profile):
    prompt = build_system_prompt(_context(profile=profile))
    assert "Official H3 interoperability contract" in prompt
    assert "JR creative director layer" in prompt
    assert prompt.index("Official H3 interoperability contract") < prompt.index("JR creative director layer")
    assert "must never override field names" in prompt


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("T2VA", "Begin directly with integrated_multimodal_description"),
        ("I2VA", "at 0.00 seconds into the target video"),
        ("FL2VA", "Picture 2 (from Shot N)"),
        ("L2VA", "<Picture 1> (from [Shot N])"),
        ("Ref2VA", "subject_definitions: -> summary: -> retention_analysis:"),
    ],
)
def test_mode_contracts(mode, expected):
    assert expected in build_system_prompt(_context(mode=mode, duration_seconds=8))


def test_ref_contract_contains_both_retention_taxonomies():
    prompt = build_system_prompt(_context(mode="Ref2VA"))
    assert "fully_preserved" in prompt and "attribute_transfer" in prompt
    assert "fully_copy" in prompt and "partially_copy" in prompt


@pytest.mark.parametrize(
    ("mode", "first_field", "shot_field"),
    [
        ("T2VA", "integrated_multimodal_description:", "integrated_multimodal_description: [Shot 1]"),
        ("Ref2VA", "subject_definitions:", "detailed_description: [Shot 1]"),
    ],
)
def test_system_prompt_includes_a_minimum_valid_output_skeleton(mode, first_field, shot_field):
    prompt = build_system_prompt(_context(mode=mode))
    skeleton = prompt.split("minimum syntactic skeleton exactly", 1)[1]
    assert first_field in skeleton
    assert shot_field in skeleton
    if mode == "Ref2VA":
        assert "subject_definitions:\n<Subject N> is" in skeleton
        assert "subject_definitions: <Subject N>" not in skeleton


def test_extract_preserved_literals_keeps_multilingual_text_in_order():
    text = '她说：“介绍一下MiniMax H3”。彼は「行きましょう！」と言う. He says "Keep JR-42 unchanged."'
    assert extract_preserved_literals(text) == (
        "介绍一下MiniMax H3",
        "行きましょう！",
        "Keep JR-42 unchanged.",
    )


def test_existing_dialogue_block_is_protected():
    assert extract_preserved_literals("(S1) says <d>[Chinese] 原样保留。</d>") == ("原样保留。",)


def test_user_prompt_carries_original_and_verbatim_contract():
    context = _context()
    result = build_user_prompt(context, extract_preserved_literals(context.original_prompt))
    assert context.original_prompt in result
    assert "介绍一下MiniMax H3" in result
    assert "Resolved input mode: T2VA" in result


def test_registry_serialization_is_duck_typed():
    class Entry:
        label = "<Picture 1>"
        source_input = "first_frame"
        role = "first_frame"
        subject_binding = None

    assert registry_as_text([Entry()]) == "<Picture 1>: source_input=first_frame; role=first_frame"
