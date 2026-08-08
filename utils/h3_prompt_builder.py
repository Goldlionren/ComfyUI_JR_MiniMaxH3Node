"""Build clean-room H3-oriented prompts without redistributing MiniMax guides."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .h3_official_resources import BASE_SECTIONS, REF_SECTIONS, get_spec_for_mode

JR_DIRECTOR_PROFILES = {
    "Standard": (
        "Plan a concise, readable scene with natural motion, physical continuity, "
        "clear causality, balanced camera choices, and a visible final state."
    ),
    "Cinematic Drama": (
        "Prioritize character relationships, blocking, dialogue timing, micro-expressions, "
        "reactions, emotional progression, and deliberate shot intention."
    ),
    "Action": (
        "Plan action as readable cause and effect: preparation, weight transfer, movement, "
        "contact, impact, recoil, counteraction, and spatially continuous results."
    ),
    "Character Consistency": (
        "Prioritize stable identity, face, hairstyle, body proportions, clothing, accessories, "
        "left/right placement, props, and spatial continuity."
    ),
}

_MODE_VALUES = {"T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"}
_DIALOGUE_BLOCK_RE = re.compile(r"<d>\s*(?:\[[^\]\r\n]+\]\s*)?(.*?)</d>", re.I | re.S)
_QUOTED_PATTERNS = (
    re.compile(r"“([^”\r\n]+)”"),
    re.compile(r"「([^」\r\n]+)」"),
    re.compile(r"『([^』\r\n]+)』"),
    re.compile(r'"([^"\r\n]+)"'),
)


@dataclass(frozen=True)
class PromptBuildContext:
    """Values used to assemble one H3 prompt-writing request."""

    original_prompt: str
    profile: str
    mode: str
    duration_seconds: int
    target_width: int
    target_height: int
    registry_text: str = "No reference media is registered."
    reference_instructions: str = ""


def extract_preserved_literals(text: str) -> tuple[str, ...]:
    """Extract explicit quoted/dialogue literals that static validation can protect."""

    source = str(text or "")
    matches: list[tuple[int, str]] = []
    for pattern in (_DIALOGUE_BLOCK_RE, *_QUOTED_PATTERNS):
        for match in pattern.finditer(source):
            value = match.group(1)
            if value:
                matches.append((match.start(1), value))
    seen: set[str] = set()
    ordered: list[str] = []
    for _, value in sorted(matches, key=lambda item: item[0]):
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _mode_name(mode: object) -> str:
    value = getattr(mode, "value", mode)
    text = str(value)
    if text not in _MODE_VALUES:
        raise ValueError(f"Unsupported resolved H3 input mode: {text!r}.")
    return text


def _alignment_contract(mode: str, duration_seconds: int) -> str:
    duration = f"{float(duration_seconds):.2f}"
    if mode == "T2VA":
        return "Begin directly with integrated_multimodal_description; do not add an image-alignment line."
    if mode == "I2VA":
        return (
            "The first line must be exactly: For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced. Then add one blank line."
        )
    if mode == "FL2VA":
        return (
            "The first line must use this exact template with the real final shot number replacing N: "
            "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with "
            f"the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the {duration}-second "
            "mark of the target video. Then add one blank line."
        )
    if mode == "L2VA":
        return (
            "The first line must use this exact template with the real final shot number replacing N: "
            "How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with "
            f"the {duration}-second mark of the target video. Then add one blank line."
        )
    return "Do not add a base-mode image-alignment preamble; begin with subject_definitions:."


def _mode_planning(mode: str) -> str:
    return {
        "T2VA": "Construct the complete visible and audible timeline from the user's text.",
        "I2VA": (
            "Treat <Picture 1> as the actual 0.00-second frame. Preserve its observable anchors and describe "
            "mainly what develops after it."
        ),
        "FL2VA": (
            "Treat <Picture 1> as the opening and <Picture 2> as the ending. Describe observable intermediate "
            "motion and progressive convergence; prefer one continuous shot unless the user requires cuts."
        ),
        "L2VA": (
            "Treat <Picture 1> as the final frame. Infer a plausible prior state and make pose, props, camera, "
            "lighting, and composition converge to it at the exact end."
        ),
        "Ref2VA": (
            "Define reusable visible content as <Subject N>; use <Picture N> only for source images or concrete "
            "frame/planning anchors. Keep <Video N> for whole-video structure and <Audio N> for audio assets."
        ),
    }[mode]


def _output_skeleton(mode: str) -> str:
    if mode == "Ref2VA":
        return """subject_definitions: <definitions or None>
summary: <concise audiovisual plan>
retention_analysis: <registered label relationships or None>
detailed_description: [Shot 1] <visible action, performance, and camera description>
overall_soundscape: <diegetic ambience, effects, and vocal delivery>
non_diegetic_music: <audience-only score, or explicitly none>"""
    return """integrated_multimodal_description: [Shot 1] <visible action, performance, and camera description>
overall_soundscape: <diegetic ambience, effects, and vocal delivery>
non_diegetic_music: <audience-only score, or explicitly none>"""


def build_system_prompt(context: PromptBuildContext) -> str:
    """Assemble the authoritative system message for the selected mode."""

    mode = _mode_name(context.mode)
    if context.profile not in JR_DIRECTOR_PROFILES:
        raise ValueError(f"Unknown JR director profile: {context.profile!r}.")
    sections = get_spec_for_mode(mode).sections
    section_contract = " -> ".join(f"{name}:" for name in sections)
    reference_notes = context.reference_instructions.strip() or "No additional reference relationship instructions."
    return f"""Mission
Rewrite the user's request into one directly usable MiniMax H3 audiovisual prompt. Output only the finished prompt: no analysis, title, Markdown fence, JSON, or answer prefix.

Authority and preservation
The user's explicit intent has highest priority. Exact dialogue, lyrics, visible scene text, proper names, and technical terms must remain verbatim and in their original language. Do not add or alter identity, outcome, dialogue, or hard constraints. Narrative structure is English; dialogue/lyrics/visible text retain their source language. Never invent facts hidden or absent from reference media.

Official H3 interoperability contract (clean-room implementation)
Resolved mode: {mode}
Required field order: {section_contract}
Use exact lowercase field names followed by a colon. [Shot 1] has no timestamp. Every later cut is sequential and begins '[Shot N] At MM:SS.mmm,' with a strictly increasing time below {int(context.duration_seconds)} seconds. Camera motion is natural prose, not a tag list. Stable vocal sources use (S1), (S2), etc.; spoken or sung content uses <d>[Language] exact content</d>. Describe ambient/physical sound in overall_soundscape and audience-only score in non_diegetic_music.
After any required alignment preamble, follow this minimum syntactic skeleton exactly, replacing every angle-bracket placeholder with concrete content and adding shots only when useful:
{_output_skeleton(mode)}
Alignment rule: {_alignment_contract(mode, int(context.duration_seconds))}
Mode planning rule: {_mode_planning(mode)}
For Ref2VA, retention_analysis uses only fully_preserved, partially_preserved, attribute_transfer, or weak_reference for visible labels; and fully_copy, partially_copy, reference, or weak_reference for Audio labels. All reference labels must be defined and remain stable.

JR creative director layer
Selected profile: {context.profile}
Direction: {JR_DIRECTOR_PROFILES[context.profile]}
This layer may improve staging, performance, motion, and continuity, but it must never override field names, field order, timing syntax, reference syntax, user intent, or verbatim content.

Reference registry
{context.registry_text}

User-supplied reference relationships
{reference_notes}

Target
Duration: {int(context.duration_seconds)} seconds. Canvas reference: {int(context.target_width)}x{int(context.target_height)}. Do not create a cut at or beyond the duration."""


def build_user_prompt(context: PromptBuildContext, preserved_literals: Iterable[str] = ()) -> str:
    """Build the text part of the multimodal user message."""

    mode = _mode_name(context.mode)
    literals = tuple(str(value) for value in preserved_literals if str(value))
    protected = "\n".join(f"- {value}" for value in literals) or "- None detected; still preserve all explicit names and text."
    return f"""Write the final H3 prompt now.
Resolved input mode: {mode}
Target duration: {int(context.duration_seconds)} seconds
Canvas reference: {int(context.target_width)}x{int(context.target_height)}

Registered reference media:
{context.registry_text}

Verbatim literals detected in the user request:
{protected}

Original user request:
{context.original_prompt}"""


def registry_as_text(entries: Iterable[object]) -> str:
    """Serialize registry entries without depending on a concrete registry class."""

    lines = []
    for entry in entries:
        label = getattr(entry, "label", getattr(entry, "identifier", "<unknown>"))
        source = getattr(entry, "source_input", getattr(entry, "source", "unknown"))
        role = getattr(entry, "role", "reference")
        subject = getattr(entry, "subject_binding", None)
        suffix = f"; subject_binding={subject}" if subject else ""
        lines.append(f"{label}: source_input={source}; role={role}{suffix}")
    return "\n".join(lines) or "No reference media is registered."


__all__ = [
    "BASE_SECTIONS",
    "JR_DIRECTOR_PROFILES",
    "PromptBuildContext",
    "REF_SECTIONS",
    "build_system_prompt",
    "build_user_prompt",
    "extract_preserved_literals",
    "registry_as_text",
]
