"""Deterministic formatter for the pinned MiniMax H3 prompt specification."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from .h3_official_prompt_schema import H3SemanticPrompt, detect_dialogue_language


class H3OfficialFormatError(ValueError):
    """Semantic data cannot be represented by the official H3 format."""


def _single_line(value: str, field: str) -> str:
    result = " ".join(str(value).split())
    if not result:
        raise H3OfficialFormatError(f"{field} must not be empty.")
    return result


def _timestamp(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise H3OfficialFormatError("Shot timestamp must be finite and non-negative.")
    millis = int(round(seconds * 1000))
    minutes, remainder = divmod(millis, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _shot_starts(semantic: H3SemanticPrompt, authoritative: Iterable[float], duration: float) -> tuple[float, ...]:
    supplied = tuple(float(value) for value in authoritative)
    if supplied:
        if len(supplied) != len(semantic.shots):
            raise H3OfficialFormatError("Director shot count does not match semantic shot count.")
        starts = supplied
    else:
        pending: list[float] = []
        for index, shot in enumerate(semantic.shots):
            if shot.start_seconds is None:
                if index == 0:
                    pending.append(0.0)
                    continue
                raise H3OfficialFormatError(
                    f"shots[{index}].start_seconds is required when Director timing is unavailable."
                )
            pending.append(float(shot.start_seconds))
        starts = tuple(pending)
    if not starts or not math.isclose(starts[0], 0.0, abs_tol=1e-9):
        raise H3OfficialFormatError("Shot 1 must begin at 0.0 seconds.")
    previous = -1.0
    for index, value in enumerate(starts, 1):
        if not math.isfinite(value) or value < 0 or value >= duration:
            raise H3OfficialFormatError(f"Shot {index} start is outside the target duration.")
        if value <= previous:
            raise H3OfficialFormatError("Shot start times must be strictly increasing.")
        previous = value
    return starts


def _format_shots(
    semantic: H3SemanticPrompt,
    starts: tuple[float, ...],
    protected_dialogues: tuple[str, ...],
) -> str:
    speaker_ids: dict[str, int] = {}
    lines: list[str] = []
    for shot_index, (shot, start) in enumerate(zip(semantic.shots, starts), 1):
        header = f"[Shot {shot_index}]"
        if shot_index > 1:
            header += f" At {_timestamp(start)},"
        description = _single_line(shot.description, f"shots[{shot_index - 1}].description")
        parts = [f"{header} {description}"]
        for dialogue in shot.dialogues:
            try:
                literal = protected_dialogues[dialogue.literal_index - 1]
            except IndexError:
                raise H3OfficialFormatError(
                    f"Dialogue literal_index {dialogue.literal_index} is not registered."
                ) from None
            speaker_id = speaker_ids.setdefault(dialogue.speaker_key, len(speaker_ids) + 1)
            speaker = _single_line(dialogue.speaker_description, "speaker_description")
            delivery = _single_line(dialogue.delivery, "delivery").rstrip(" :")
            language = detect_dialogue_language(literal)
            parts.append(
                f"{speaker} (S{speaker_id}) {delivery}: <d>[{language}] {literal}</d>"
            )
        lines.append(" ".join(parts))
    return " ".join(lines)


def _alignment(mode: str, duration: float, final_shot: int) -> str:
    duration_text = f"{duration:.2f}"
    if mode == "I2VA":
        return (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    if mode == "FL2VA":
        return (
            "How the reference pictures align with the target video — Picture 1 (from Shot 1) "
            "aligns with the 0.00-second mark of the target video; Picture 2 "
            f"(from Shot {final_shot}) aligns with the {duration_text}-second mark of the target video."
        )
    if mode == "L2VA":
        return (
            "How the reference pictures align with the target video — "
            f"<Picture 1> (from [Shot {final_shot}]) aligns with the "
            f"{duration_text}-second mark of the target video."
        )
    return ""


def format_official_prompt(
    semantic: H3SemanticPrompt,
    *,
    mode: str,
    duration_seconds: float,
    protected_dialogues: Iterable[str] = (),
    authoritative_shot_starts: Iterable[float] = (),
) -> str:
    """Render semantic data into the pinned official Base or Ref2VA layout."""

    duration = float(duration_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise H3OfficialFormatError("duration_seconds must be finite and greater than zero.")
    dialogues = tuple(str(value) for value in protected_dialogues)
    starts = _shot_starts(semantic, authoritative_shot_starts, duration)
    shot_text = _format_shots(semantic, starts, dialogues)
    soundscape = _single_line(semantic.overall_soundscape, "overall_soundscape")
    music = _single_line(semantic.non_diegetic_music, "non_diegetic_music")
    if "<d>" in soundscape.casefold() or any(value in soundscape for value in dialogues):
        raise H3OfficialFormatError("overall_soundscape must not contain protected dialogue.")
    if "<d>" in music.casefold() or any(value in music for value in dialogues):
        raise H3OfficialFormatError("non_diegetic_music must not contain protected dialogue.")

    if mode != "Ref2VA":
        body = (
            f"integrated_multimodal_description: {shot_text}\n\n"
            f"overall_soundscape: {soundscape}\n\n"
            f"non_diegetic_music: {music}"
        )
        preamble = _alignment(mode, duration, len(semantic.shots))
        return f"{preamble}\n\n{body}" if preamble else body

    definitions = "\n".join(
        f"{item.label} is {_single_line(item.definition, 'reference definition')}"
        for item in semantic.references
    )
    retention = "\n".join(
        f"{item.label}: {item.retention} - {_single_line(item.retention_detail, 'retention detail')}"
        for item in semantic.references
    )
    task_prefix = " + ".join(semantic.task_types)
    summary = _single_line(semantic.summary, "summary")
    style = _single_line(semantic.style, "style")
    if re.match(r"^\[[^\]]+\]", summary):
        raise H3OfficialFormatError("Semantic summary must not include the task-type prefix.")
    return (
        f"subject_definitions:\n{definitions}\n\n"
        f"summary:\n[{task_prefix}] {summary}\n\n"
        f"retention_analysis:\n{retention}\n\n"
        f"detailed_description:\n{style}\n{shot_text}\n\n"
        f"overall_soundscape:\n{soundscape}\n\n"
        f"non_diegetic_music:\n{music}"
    )


__all__ = ["H3OfficialFormatError", "format_official_prompt"]
