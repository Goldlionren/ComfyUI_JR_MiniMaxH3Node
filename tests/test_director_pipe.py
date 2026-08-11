from dataclasses import FrozenInstanceError, replace

import pytest
from ComfyUI_JR_MiniMaxH3Node.utils.director_pipe import (
    PIPE_SCHEMA,
    DirectorPipe,
    RuntimeMedia,
    build_director_pipe,
    validate_director_pipe,
)
from ComfyUI_JR_MiniMaxH3Node.utils.director_prompt_compiler import ReferenceRecord
from ComfyUI_JR_MiniMaxH3Node.utils.director_state import (
    AssetDescriptor,
    VisualState,
    default_director_state,
)


def test_pipe_is_typed_immutable_and_prompt_matches_output():
    state = default_director_state()
    marker = object()
    pipe = build_director_pipe(
        state,
        runtime_resolver=lambda _state: (),
    )
    assert isinstance(pipe, DirectorPipe)
    assert pipe.schema == PIPE_SCHEMA
    assert pipe.schema_version == 2
    assert pipe.optimized_prompt == pipe.reviewed_prompt == ""
    assert pipe.compiled_director_prompt.startswith("GLOBAL DIRECTION")
    assert validate_director_pipe(pipe) is pipe
    with pytest.raises(FrozenInstanceError):
        pipe.global_direction = "mutated"
    assert marker is marker


def test_runtime_media_is_separate_from_persisted_state():
    state = default_director_state()
    pipe = build_director_pipe(state)
    persisted = pipe.to_persisted()
    assert "runtime_media" not in persisted
    assert "compiled_director_prompt" not in persisted
    assert "optimized_prompt" not in persisted
    assert "reviewed_prompt" not in persisted
    assert pipe.runtime_media == ()


def test_pipe_rejects_tampered_prompt_registry_and_runtime_item():
    pipe = build_director_pipe(default_director_state())
    with pytest.raises(ValueError, match="schema_version"):
        validate_director_pipe(replace(pipe, schema_version=True))
    with pytest.raises(ValueError, match="immutable tuple"):
        validate_director_pipe(replace(pipe, runtime_media=[]))
    with pytest.raises(ValueError, match="compiled_director_prompt"):
        validate_director_pipe(replace(pipe, compiled_director_prompt="tampered"))
    fake = ReferenceRecord("<Picture 1>", "Picture", "x", "a", "reference_image", 0, 1, "", "")
    with pytest.raises(ValueError, match="reference_registry"):
        validate_director_pipe(replace(pipe, reference_registry=(fake,)))
    bad_runtime = RuntimeMedia("asset", "missing-item", "image", object())
    with pytest.raises(ValueError, match="unknown item"):
        validate_director_pipe(replace(pipe, runtime_media=(bad_runtime,)))


def test_pipe_rejects_runtime_asset_or_kind_mismatch():
    state = default_director_state()
    asset = AssetDescriptor(
        id="asset-1", kind="image", filename="frame.png", subfolder="", folder_type="input",
        display_name="frame.png",
    )
    visual = VisualState(
        id="visual-1", kind="image", role="reference_image", start=0.0, end=10.0,
        source_in=None, source_out=None, direction="", notes="", registry_order=1, asset=asset,
    )
    pipe = build_director_pipe(replace(state, visual_items=(visual,)))
    with pytest.raises(ValueError, match="asset_id"):
        validate_director_pipe(replace(
            pipe, runtime_media=(RuntimeMedia("wrong", "visual-1", "image", object()),),
        ))
    with pytest.raises(ValueError, match="kind"):
        validate_director_pipe(replace(
            pipe, runtime_media=(RuntimeMedia("asset-1", "visual-1", "video", object()),),
        ))


def test_pipe_stage_derivation_is_immutable_and_preserves_runtime_identity():
    state = default_director_state()
    pipe0 = build_director_pipe(state)
    pipe1 = pipe0.derive(optimized_prompt=" optimized ", reviewed_prompt="")
    pipe2 = pipe1.derive(reviewed_prompt=" reviewed ")
    assert pipe0 is not pipe1 and pipe1 is not pipe2
    assert pipe0.optimized_prompt == pipe0.reviewed_prompt == ""
    assert pipe1.final_prompt() == " optimized "
    assert pipe2.final_prompt() == " reviewed "
    assert pipe0.timeline is pipe1.timeline is pipe2.timeline
    assert pipe0.shots is pipe1.shots is pipe2.shots
    assert pipe0.runtime_media is pipe1.runtime_media is pipe2.runtime_media
    pipe3 = pipe2.derive(optimized_prompt="new", reviewed_prompt="")
    assert pipe3.final_prompt() == "new"
