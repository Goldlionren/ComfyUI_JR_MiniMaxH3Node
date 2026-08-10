"""Timeline-aware multimodal Director Desk for MiniMax H3 workflows."""

from __future__ import annotations

from ..utils.director_media import resolve_runtime_media
from ..utils.director_pipe import build_director_pipe
from ..utils.director_prompt_compiler import validate_director_state
from ..utils.director_state import DEFAULT_DIRECTOR_STATE_JSON, director_state_from_json


class JR_H3_DirectorDesk:
    CATEGORY = "JR MiniMax H3/Director"
    FUNCTION = "compose"
    RETURN_TYPES = ("STRING", "JR_H3_DIRECTOR_PIPE")
    RETURN_NAMES = ("director_prompt", "pip")
    DESCRIPTION = (
        "Edits a multimodal timeline and deterministically compiles a raw Director Prompt "
        "plus an immutable JR_H3_DIRECTOR_PIPE. This node does not call an LLM."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "director_state_json": (
                    "STRING",
                    {"multiline": True, "default": DEFAULT_DIRECTOR_STATE_JSON},
                ),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, director_state_json):
        try:
            validate_director_state(director_state_from_json(director_state_json))
        except (TypeError, ValueError) as error:
            return str(error)
        return True

    @classmethod
    def IS_CHANGED(cls, director_state_json):
        # Descriptors point at mutable files. Re-resolve on every queue so a replaced,
        # corrupted, or deleted asset cannot reuse a stale runtime IMAGE payload.
        return float("nan")

    def compose(self, director_state_json):
        state = director_state_from_json(director_state_json)
        pipe = build_director_pipe(state, runtime_resolver=resolve_runtime_media)
        return pipe.compiled_director_prompt, pipe


__all__ = ["JR_H3_DirectorDesk"]
