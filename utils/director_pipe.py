"""Immutable runtime pipe produced by the Director Desk node."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from .director_prompt_compiler import (
    ReferenceRecord,
    build_reference_registry,
    compile_director_prompt,
    validate_director_state,
)
from .director_state import (
    AudioState,
    DirectorState,
    DirectorUIState,
    ShotState,
    TimelineState,
    VisualState,
    director_state_to_dict,
)

PIPE_SCHEMA = "jr_h3_director_pipe"
PIPE_VERSION = 2
MAX_STAGE_PROMPT_CHARS = 512 * 1024
_UNSET = object()


@dataclass(frozen=True, slots=True)
class RuntimeMediaFile:
    """Validated runtime-only file reference; never serialized into workflow JSON."""

    path: str = field(repr=False)
    kind: str

    def as_path(self) -> Path:
        return Path(self.path)


@dataclass(frozen=True, slots=True)
class RuntimeMedia:
    asset_id: str
    item_id: str
    kind: str
    payload: Any = field(repr=False)
    metadata: tuple[tuple[str, Any], ...] = ()

    def metadata_dict(self) -> dict[str, Any]:
        return dict(self.metadata)


@dataclass(frozen=True, slots=True)
class DirectorPipe:
    schema: str
    schema_version: int
    timeline: TimelineState
    global_direction: str
    shots: tuple[ShotState, ...]
    visual_items: tuple[VisualState, ...]
    audio_items: tuple[AudioState, ...]
    compiled_director_prompt: str
    optimized_prompt: str
    reviewed_prompt: str
    reference_registry: tuple[ReferenceRecord, ...]
    runtime_media: tuple[RuntimeMedia, ...]

    def media_for_item(self, item_id: str) -> RuntimeMedia | None:
        return next((item for item in self.runtime_media if item.item_id == item_id), None)

    def derive(
        self,
        *,
        optimized_prompt: str | None | object = _UNSET,
        reviewed_prompt: str | None | object = _UNSET,
    ) -> DirectorPipe:
        """Return a new stage value without changing timeline or runtime media."""

        updated = replace(
            self,
            optimized_prompt=(
                self.optimized_prompt if optimized_prompt is _UNSET else str(optimized_prompt or "")
            ),
            reviewed_prompt=(
                self.reviewed_prompt if reviewed_prompt is _UNSET else str(reviewed_prompt or "")
            ),
        )
        validate_director_pipe(updated)
        return updated

    def prompt_for_review(self) -> str:
        return self.optimized_prompt if self.optimized_prompt.strip() else self.compiled_director_prompt

    def final_prompt(self) -> str:
        if self.reviewed_prompt.strip():
            return self.reviewed_prompt
        if self.optimized_prompt.strip():
            return self.optimized_prompt
        return self.compiled_director_prompt

    def to_persisted(self) -> dict[str, Any]:
        state = DirectorState(
            schema="jr_h3_director_state",
            schema_version=1,
            timeline=self.timeline,
            global_direction=self.global_direction,
            shots=self.shots,
            visual_items=self.visual_items,
            audio_items=self.audio_items,
            ui=DirectorUIState(),
        )
        return director_state_to_dict(state, include_ui=False)


RuntimeResolver = Callable[[DirectorState], Iterable[RuntimeMedia]]


def build_director_pipe(
    state: DirectorState,
    runtime_resolver: RuntimeResolver | None = None,
) -> DirectorPipe:
    validate_director_state(state)
    registry = build_reference_registry(state)
    prompt = compile_director_prompt(state, registry)
    runtime = tuple(runtime_resolver(state)) if runtime_resolver is not None else ()
    pipe = DirectorPipe(
        schema=PIPE_SCHEMA,
        schema_version=PIPE_VERSION,
        timeline=state.timeline,
        global_direction=state.global_direction,
        shots=tuple(state.shots),
        visual_items=tuple(state.visual_items),
        audio_items=tuple(state.audio_items),
        compiled_director_prompt=prompt,
        optimized_prompt="",
        reviewed_prompt="",
        reference_registry=registry,
        runtime_media=runtime,
    )
    validate_director_pipe(pipe)
    return pipe


def validate_director_pipe(value: Any) -> DirectorPipe:
    if not isinstance(value, DirectorPipe):
        raise ValueError("pip must be a JR_H3_DIRECTOR_PIPE produced by JR MiniMax H3 Director Desk.")
    if value.schema != PIPE_SCHEMA:
        raise ValueError(f"Unsupported Director PIP schema: {value.schema!r}.")
    if (
        isinstance(value.schema_version, bool)
        or not isinstance(value.schema_version, int)
        or value.schema_version != PIPE_VERSION
    ):
        raise ValueError(f"Unsupported Director PIP schema_version: {value.schema_version!r}.")
    for field_name in ("shots", "visual_items", "audio_items", "reference_registry", "runtime_media"):
        if not isinstance(getattr(value, field_name), tuple):
            raise ValueError(f"Director PIP {field_name} must be an immutable tuple.")
    state = DirectorState(
        schema="jr_h3_director_state", schema_version=1,
        timeline=value.timeline, global_direction=value.global_direction,
        shots=value.shots, visual_items=value.visual_items, audio_items=value.audio_items,
        ui=DirectorUIState(),
    )
    validate_director_state(state)
    expected_registry = build_reference_registry(state)
    if value.reference_registry != expected_registry:
        raise ValueError("Director PIP reference_registry does not match its timeline state.")
    expected_prompt = compile_director_prompt(state, expected_registry)
    if value.compiled_director_prompt != expected_prompt:
        raise ValueError("Director PIP compiled_director_prompt does not match its timeline state.")
    for field_name in ("optimized_prompt", "reviewed_prompt"):
        field_value = getattr(value, field_name)
        if not isinstance(field_value, str):
            raise ValueError(f"Director PIP {field_name} must be text.")
        if len(field_value) > MAX_STAGE_PROMPT_CHARS or "\x00" in field_value:
            raise ValueError(f"Director PIP {field_name} is too large or contains a NUL character.")
    items_by_id = {item.id: item for item in (*value.visual_items, *value.audio_items)}
    seen = set()
    for media in value.runtime_media:
        if not isinstance(media, RuntimeMedia) or not isinstance(media.metadata, tuple):
            raise ValueError("Director PIP runtime_media entries must use immutable RuntimeMedia containers.")
        item = items_by_id.get(media.item_id)
        if item is None:
            raise ValueError(f"Director PIP runtime media references unknown item {media.item_id!r}.")
        if media.item_id in seen:
            raise ValueError(f"Director PIP has duplicate runtime media for item {media.item_id!r}.")
        if media.asset_id != item.asset.id:
            raise ValueError(f"Director PIP runtime media asset_id does not match item {media.item_id!r}.")
        if media.kind != item.asset.kind:
            raise ValueError(f"Director PIP runtime media kind does not match item {media.item_id!r}.")
        seen.add(media.item_id)
    return value


__all__ = [
    "DirectorPipe", "PIPE_SCHEMA", "PIPE_VERSION", "RuntimeMedia", "RuntimeMediaFile",
    "build_director_pipe", "validate_director_pipe",
]
