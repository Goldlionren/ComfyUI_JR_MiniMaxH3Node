from pathlib import Path

import pytest

from utils.h3_prompt_validator import (
    AUDIO_VALUES,
    REF_MODE,
    VISIBLE_RETENTION_VALUES,
    cleanup_prompt,
    validate_prompt,
)

FIXTURES = Path(__file__).parent / "fixtures" / "h3_prompts"


def _base(body: str = "[Shot 1] A quiet scene.") -> str:
    return (
        f"integrated_multimodal_description: {body}\n"
        "overall_soundscape: A natural room tone.\n"
        "non_diegetic_music: None."
    )


@pytest.mark.parametrize(
    ("mode", "prefix", "labels"),
    [
        ("T2VA", "", []),
        ("I2VA", "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n", ["<Picture 1>"]),
        ("FL2VA", "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 5.00-second mark of the target video.\n\n", ["<Picture 1>", "<Picture 2>"]),
        ("L2VA", "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 5.00-second mark of the target video.\n\n", ["<Picture 1>"]),
    ],
)
def test_minimal_base_modes_validate(mode, prefix, labels):
    result = validate_prompt(prefix + _base(), mode=mode, duration_seconds=5, allowed_labels=labels)
    assert result.valid, result.errors


def test_cleanup_removes_only_transport_wrappers_and_preserves_literals():
    prompt = """\nFinal Answer: ```\n<think>internal reasoning</think>\nintegrated_multimodal_description: [Shot 1] 你好，こんにちは, hello; 名称。\noverall_soundscape: visible text\nnon_diegetic_music: None\n```\n"""
    cleaned = cleanup_prompt(prompt)
    assert "<think>" not in cleaned
    assert "你好，こんにちは, hello; 名称" in cleaned
    assert "<Picture" not in cleaned
    result = validate_prompt(cleaned, mode="T2VA", duration_seconds=5, preserved_literals=["你好", "こんにちは", "hello", "名称"])
    assert result.valid, result.errors


def test_cleanup_does_not_remove_nested_fence():
    prompt = "```\n```\nintegrated_multimodal_description: [Shot 1] x\noverall_soundscape: y\nnon_diegetic_music: z\n```"
    result = validate_prompt(prompt, mode="T2VA", duration_seconds=5)
    assert not result.valid
    assert any("fence" in error for error in result.errors)


def test_ref_fixture_and_registry_resolution():
    prompt = (FIXTURES / "ref2va_valid.txt").read_text(encoding="utf-8")
    result = validate_prompt(
        prompt,
        mode=REF_MODE,
        duration_seconds=5,
        allowed_labels={"Picture": ["1"], "Video": ["1"], "Audio": ["1"]},
    )
    assert result.valid, result.errors


@pytest.mark.parametrize("name", ["t2va_valid", "i2va_valid", "fl2va_valid", "l2va_valid"])
def test_base_fixtures_load(name):
    prompt = (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")
    mode = name.split("_", 1)[0].upper()
    labels = ["<Picture 1>", "<Picture 2>"] if mode == "FL2VA" else (["<Picture 1>"] if mode in {"I2VA", "L2VA"} else [])
    result = validate_prompt(prompt, mode=mode, duration_seconds=5, allowed_labels=labels)
    assert result.valid, result.errors


def test_invalid_fixtures_report_structural_errors():
    missing = validate_prompt((FIXTURES / "invalid_missing_section.txt").read_text(), "T2VA", 5)
    assert not missing.valid
    assert any("missing required section" in error for error in missing.errors)

    timestamp = validate_prompt((FIXTURES / "invalid_timestamp.txt").read_text(), "T2VA", 5)
    assert not timestamp.valid
    assert any("outside duration" in error for error in timestamp.errors)

    reference = validate_prompt(
        (FIXTURES / "invalid_reference.txt").read_text(), "T2VA", 5, allowed_labels=["<Picture 1>"]
    )
    assert not reference.valid
    assert any("unresolved reference" in error for error in reference.errors)


def test_sections_are_exact_and_ordered():
    result = validate_prompt(
        "overall_soundscape: x\nintegrated_multimodal_description: [Shot 1] y\nnon_diegetic_music: z",
        "T2VA",
        5,
    )
    assert not result.valid
    assert any("out of order" in error for error in result.errors)


def test_shots_inline_sequence_and_duration_rules():
    valid = validate_prompt(
        _base("[Shot 1] one. [Shot 2] At 00:00.500, two. [Shot 3] At 00:01.000, three."),
        "T2VA",
        2,
    )
    assert valid.valid, valid.errors

    bad = validate_prompt(
        _base("[Shot 1] one. [Shot 3] At 00:01.000, three. [Shot 2] At 00:01.000, two."),
        "T2VA",
        2,
    )
    assert not bad.valid
    assert any("numbering" in error for error in bad.errors)
    assert any("strictly increasing" in error for error in bad.errors)


def test_alignment_duration_is_not_a_cut_timestamp():
    prefix = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
    result = validate_prompt(
        prefix + _base("the alignment ends at 5.00 seconds; [Shot 1] one."),
        "I2VA",
        5,
        allowed_labels=["<Picture 1>"],
    )
    assert result.valid, result.errors


def test_shot_one_must_not_have_timestamp_and_later_syntax_is_exact():
    first = validate_prompt(_base("[Shot 1] At 00:00.000, one."), "T2VA", 5)
    assert not first.valid
    assert any("shot 1" in error for error in first.errors)

    later = validate_prompt(_base("[Shot 1] one. [Shot 2] at 00:01.000, two."), "T2VA", 5)
    assert not later.valid
    assert any("MM:SS.mmm" in error for error in later.errors)


@pytest.mark.parametrize("tag", ["<Picture 1>", "<Subject 1>", "<Video 1>", "<Audio 1>"])
def test_reference_syntax_families(tag):
    if tag.startswith("<Subject"):
        prompt = (
            "subject_definitions:\n<Subject 1> is a person.\nsummary: x\n"
            "retention_analysis: y\ndetailed_description: [Shot 1] " + tag + "\n"
            "overall_soundscape: z\nnon_diegetic_music: z"
        )
        result = validate_prompt(prompt, "Ref2VA", 5)
        assert result.valid, result.errors
    else:
        result = validate_prompt(_base(f"[Shot 1] use {tag}."), "T2VA", 5, allowed_labels=[tag])
        if tag == "<Picture 1>":
            assert result.valid, result.errors
        else:
            assert not result.valid


def test_malformed_reference_and_unresolved_registry_label():
    malformed = validate_prompt(_base("[Shot 1] use <picture 1>."), "T2VA", 5, allowed_labels=["<Picture 1>"])
    assert not malformed.valid
    assert any("invalid reference syntax" in error for error in malformed.errors)
    unresolved = validate_prompt(_base("[Shot 1] use <Picture 2>."), "T2VA", 5, allowed_labels=["<Picture 1>"])
    assert any("unresolved" in error for error in unresolved.errors)


def test_subject_definitions_must_precede_use_and_duplicates_fail():
    before = (
        "subject_definitions:\n<Subject 1> is a person.\nsummary: <Subject 1>\n"
        "retention_analysis: <Picture 1>: fully_preserved\n"
        "detailed_description: [Shot 1] <Subject 2> appears.\n"
        "overall_soundscape: x\nnon_diegetic_music: y"
    )
    result = validate_prompt(before, "Ref2VA", 5, allowed_labels=["<Picture 1>"])
    assert not result.valid
    assert any("before definition" in error for error in result.errors)

    duplicate = before.replace("<Subject 1> is a person.", "<Subject 1> is a person.\n<Subject 1> is a duplicate.")
    duplicate = duplicate.replace("<Subject 2> appears", "<Subject 1> appears")
    result = validate_prompt(duplicate, "Ref2VA", 5, allowed_labels=["<Picture 1>"])
    assert any("duplicate subject definition" in error for error in result.errors)


@pytest.mark.parametrize("value", VISIBLE_RETENTION_VALUES)
def test_visible_retention_taxonomy(value):
    prompt = (
        "subject_definitions:\n<Subject 1> is a person.\nsummary: x\n"
        f"retention_analysis:\n<Picture 1> (visual): {value} - note\n"
        "detailed_description: [Shot 1] <Subject 1>\n"
        "overall_soundscape: x\nnon_diegetic_music: y"
    )
    result = validate_prompt(prompt, "Ref2VA", 5, allowed_labels=["<Picture 1>"])
    assert result.valid, result.errors


@pytest.mark.parametrize("value", AUDIO_VALUES)
def test_audio_taxonomy(value):
    prompt = (
        "subject_definitions:\n<Subject 1> is a person.\nsummary: x\n"
        f"retention_analysis:\n<Audio 1>: {value} - note\n"
        "detailed_description: [Shot 1] <Subject 1>\n"
        "overall_soundscape: x\nnon_diegetic_music: y"
    )
    result = validate_prompt(prompt, "Ref2VA", 5, allowed_labels=["<Audio 1>"])
    assert result.valid, result.errors


def test_invalid_taxonomy_value_is_descriptive():
    prompt = (
        "subject_definitions:\n<Subject 1> is a person.\nsummary: x\n"
        "retention_analysis:\n<Picture 1>: preserve\n<Audio 1>: copy\n"
        "detailed_description: [Shot 1] <Subject 1>\n"
        "overall_soundscape: x\nnon_diegetic_music: y"
    )
    result = validate_prompt(prompt, "Ref2VA", 5, allowed_labels=["<Picture 1>", "<Audio 1>"])
    assert not result.valid
    assert any("invalid visible retention" in error for error in result.errors)
    assert any("invalid audio" in error for error in result.errors)


def test_preserved_literals_are_checked_exactly():
    result = validate_prompt(_base("[Shot 1] English dialogue; 中文对白; 日本語の台詞; Alice."), "T2VA", 5,
                             preserved_literals=["English dialogue", "中文对白", "日本語の台詞", "Alice"])
    assert result.valid, result.errors
    missing = validate_prompt(_base("[Shot 1] Alice."), "T2VA", 5, preserved_literals=["Bob"])
    assert not missing.valid
    assert any("preserved literal missing" in error for error in missing.errors)
