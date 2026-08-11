import json

import pytest
from ComfyUI_JR_MiniMaxH3Node.utils.h3_official_prompt_formatter import (
    H3OfficialFormatError,
    format_official_prompt,
)
from ComfyUI_JR_MiniMaxH3Node.utils.h3_official_prompt_schema import (
    H3SemanticError,
    parse_semantic_response,
)
from h3_semantic_helpers import base_semantic, ref_semantic


@pytest.mark.parametrize(
    ("mode", "prefix"),
    [
        ("T2VA", "integrated_multimodal_description:"),
        (
            "I2VA",
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.",
        ),
        (
            "FL2VA",
            "How the reference pictures align with the target video — Picture 1 "
            "(from Shot 1) aligns with the 0.00-second mark of the target video; "
            "Picture 2 (from Shot 1) aligns with the 5.00-second mark of the target video.",
        ),
        (
            "L2VA",
            "How the reference pictures align with the target video — <Picture 1> "
            "(from [Shot 1]) aligns with the 5.00-second mark of the target video.",
        ),
    ],
)
def test_base_modes_have_exact_python_owned_preamble_and_section_order(mode, prefix):
    semantic = parse_semantic_response(base_semantic(), mode=mode)
    result = format_official_prompt(semantic, mode=mode, duration_seconds=5)
    assert result.startswith(prefix)
    assert result.index("integrated_multimodal_description:") < result.index(
        "overall_soundscape:"
    ) < result.index("non_diegetic_music:")
    assert result.count("[Shot 1]") >= 1
    assert not result.lstrip().startswith("{")


def test_director_shot_starts_are_authoritative_and_formatted_to_milliseconds():
    semantic = parse_semantic_response(
        base_semantic(("First action.", "Second action."), starts=(0.0, 9.9)),
        mode="T2VA",
        expected_shot_count=2,
    )
    result = format_official_prompt(
        semantic,
        mode="T2VA",
        duration_seconds=10,
        authoritative_shot_starts=(0.0, 2.345),
    )
    assert "[Shot 1] First action." in result
    assert "[Shot 2] At 00:02.345, Second action." in result
    assert "00:09.900" not in result


def test_ref2va_has_exact_six_section_order_registry_order_and_taxonomy():
    labels = ("<Picture 1>", "<Video 1>", "<Audio 1>")
    semantic = parse_semantic_response(
        ref_semantic(labels), mode="Ref2VA", allowed_labels=labels
    )
    result = format_official_prompt(semantic, mode="Ref2VA", duration_seconds=5)
    headings = (
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )
    positions = tuple(result.index(heading) for heading in headings)
    assert positions == tuple(sorted(positions))
    assert result.index("<Picture 1> is") < result.index("<Video 1> is") < result.index(
        "<Audio 1> is"
    )
    assert "<Picture 1>: fully_preserved" in result
    assert "<Audio 1>: fully_copy" in result
    assert "summary:\n[reference generation]" in result


def test_dialogue_literal_language_and_stable_speaker_ids_are_deterministic():
    literal = "你好，世界"
    dialogues = (
        ({
            "literal_index": 1,
            "speaker_key": "host",
            "speaker_description": "The host",
            "delivery": "says warmly",
        },),
        (),
    )
    semantic = parse_semantic_response(
        base_semantic(("The host faces camera.", "The host smiles."), starts=(0, 2), dialogues=dialogues),
        mode="T2VA",
        protected_dialogue_count=1,
        expected_shot_count=2,
    )
    result = format_official_prompt(
        semantic,
        mode="T2VA",
        duration_seconds=5,
        protected_dialogues=(literal,),
    )
    assert result.count(literal) == 1
    assert f"The host (S1) says warmly: <d>[Chinese] {literal}</d>" in result
    assert literal not in result.split("overall_soundscape:", 1)[1]


def test_protected_dialogue_cannot_be_copied_into_semantic_prose():
    literal = "老师好"
    response = base_semantic(
        (f"The presenter says {literal} to camera.",),
        dialogues=(({
            "literal_index": 1,
            "speaker_key": "host",
            "speaker_description": "The presenter",
            "delivery": "says warmly",
        },),),
    )
    with pytest.raises(H3SemanticError, match="must not copy protected dialogue literal"):
        parse_semantic_response(
            response,
            mode="T2VA",
            protected_dialogue_count=1,
            protected_dialogues=(literal,),
        )


@pytest.mark.parametrize(
    "description",
    (
        "She likely starts a video call.",
        "She touches her hair or arches her back.",
        "She appears to wait for her teacher.",
        "She perhaps blushes.",
    ),
)
def test_semantic_prose_rejects_speculation_and_alternative_actions(description):
    response = base_semantic((description,))
    with pytest.raises(H3SemanticError, match="must be decisive"):
        parse_semantic_response(response, mode="T2VA")


def test_repeated_identical_dialogue_occurrences_are_preserved_by_index():
    literal = "好"
    dialogues = (
        ({
            "literal_index": 1,
            "speaker_key": "host",
            "speaker_description": "The host",
            "delivery": "says",
        },),
        ({
            "literal_index": 2,
            "speaker_key": "host",
            "speaker_description": "The host",
            "delivery": "repeats",
        },),
    )
    semantic = parse_semantic_response(
        base_semantic(("The host nods.", "The host nods again."), starts=(0, 2), dialogues=dialogues),
        mode="T2VA",
        protected_dialogue_count=2,
        protected_dialogues=(literal, literal),
    )
    result = format_official_prompt(
        semantic,
        mode="T2VA",
        duration_seconds=5,
        protected_dialogues=(literal, literal),
    )
    assert result.count(f"<d>[Chinese] {literal}</d>") == 2


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda raw: raw.update({"unknown": True}), "unknown field"),
        (lambda raw: raw["shots"][0].update({"dialogues": [{"literal_index": 2, "speaker_key": "a", "speaker_description": "A", "delivery": "says"}]}), "literal_index"),
        (lambda raw: raw.update({"references": [{"label": "<Picture 2>", "definition": "x", "retention": "fully_preserved", "retention_detail": "x"}]}), "exact registered label order"),
    ],
)
def test_schema_rejects_model_control_of_contract_fields(mutate, match):
    raw = json.loads(base_semantic())
    mode = "T2VA"
    kwargs = {}
    mutate(raw)
    if raw.get("references"):
        mode = "Ref2VA"
        raw["style"] = "style"
        raw["task_types"] = ["reference generation"]
        raw["summary"] = "summary"
        kwargs["allowed_labels"] = ("<Picture 1>",)
    with pytest.raises(H3SemanticError, match=match):
        parse_semantic_response(
            json.dumps(raw), mode=mode, protected_dialogue_count=1 if "literal_index" in match else 0,
            **kwargs,
        )


def test_audio_and_visible_retention_taxonomies_are_not_interchangeable():
    raw = json.loads(ref_semantic(("<Audio 1>",)))
    raw["references"][0]["retention"] = "fully_preserved"
    with pytest.raises(H3SemanticError, match="fully_copy"):
        parse_semantic_response(
            json.dumps(raw), mode="Ref2VA", allowed_labels=("<Audio 1>",)
        )


@pytest.mark.parametrize(
    ("label", "alias_name", "alias_value", "irrelevant_name", "irrelevant_value"),
    [
        ("<Picture 1>", "visible_retention", "fully_preserved", "audio_retention", "fully_copy"),
        ("<Video 1>", "visible_retention", "partially_preserved", "audio_retention", "reference"),
        ("<Audio 1>", "audio_retention", "reference", "visible_retention", "weak_reference"),
    ],
)
def test_known_reference_aliases_are_normalized_by_media_family(
    label, alias_name, alias_value, irrelevant_name, irrelevant_value
):
    raw = json.loads(ref_semantic((label,)))
    reference = raw["references"][0]
    reference["reference_label"] = reference.pop("label")
    reference.pop("retention")
    reference[alias_name] = alias_value
    reference[irrelevant_name] = irrelevant_value

    semantic = parse_semantic_response(
        json.dumps(raw), mode="Ref2VA", allowed_labels=(label,)
    )

    assert semantic.references[0].label == label
    assert semantic.references[0].retention == alias_value


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        (
            {"reference_label": "<Picture 2>"},
            "conflicting values for label and reference_label",
        ),
        (
            {"visible_retention": "partially_preserved"},
            "conflicting values for retention and visible_retention",
        ),
    ],
)
def test_known_reference_alias_conflicts_are_rejected(updates, match):
    raw = json.loads(ref_semantic(("<Picture 1>",)))
    raw["references"][0].update(updates)

    with pytest.raises(H3SemanticError, match=match):
        parse_semantic_response(
            json.dumps(raw), mode="Ref2VA", allowed_labels=("<Picture 1>",)
        )


def test_unrelated_unknown_reference_field_remains_rejected():
    raw = json.loads(ref_semantic(("<Picture 1>",)))
    raw["references"][0]["made_up_field"] = "value"

    with pytest.raises(H3SemanticError, match="unknown field.*made_up_field"):
        parse_semantic_response(
            json.dumps(raw), mode="Ref2VA", allowed_labels=("<Picture 1>",)
        )


def test_formatter_rejects_missing_later_timing_without_director_authority():
    semantic = parse_semantic_response(
        base_semantic(("First.", "Second."), starts=(0, None)), mode="T2VA"
    )
    with pytest.raises(H3OfficialFormatError, match="start_seconds is required"):
        format_official_prompt(semantic, mode="T2VA", duration_seconds=5)


def test_fenced_json_is_accepted_but_extra_prose_is_not():
    semantic = base_semantic()
    assert parse_semantic_response(f"```json\n{semantic}\n```", mode="T2VA")
    with pytest.raises(H3SemanticError, match="not valid JSON"):
        parse_semantic_response(f"Here is JSON: {semantic}", mode="T2VA")
