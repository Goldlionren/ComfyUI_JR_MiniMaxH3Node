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
    duration_seconds: float
    target_width: int
    target_height: int
    registry_text: str = "No reference media is registered."
    reference_instructions: str = ""
    shot_starts: tuple[float, ...] = ()
    reference_labels: tuple[str, ...] = ()
    protected_dialogues: tuple[str, ...] = ()


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


def extract_protected_dialogues(text: str) -> tuple[str, ...]:
    """Extract explicit spoken literals while leaving ordinary visible text alone."""

    source = str(text or "")
    matches: list[tuple[int, str]] = []
    for match in _DIALOGUE_BLOCK_RE.finditer(source):
        if match.group(1):
            matches.append((match.start(1), match.group(1)))
    speech_hint = re.compile(
        r"(?:dialogue|spoken line|says?|asks?|repl(?:y|ies)|shouts?|whispers?|"
        r"groans?|moans?|murmurs?|mutters?|sings?|chants?|"
        r"台词|对白|说|问|喊|回答|回应|低语|耳语|呢喃|喃喃|嘟囔|呻吟|唱|念|叫)",
        re.I,
    )
    for pattern in _QUOTED_PATTERNS:
        for match in pattern.finditer(source):
            if speech_hint.search(source[max(0, match.start() - 64):match.start()]):
                matches.append((match.start(1), match.group(1)))
    line_literal = re.compile(r"^[ \t]*(?:dialogue|spoken line|台词|对白)[ \t]*[:：][ \t]*(.+?)\s*$", re.I | re.M)
    for match in line_literal.finditer(source):
        value = match.group(1).strip().strip('“”「」『』"')
        if value:
            matches.append((match.start(1), value))
    ordered: list[str] = []
    seen_positions: set[tuple[int, str]] = set()
    for position, value in sorted(matches, key=lambda item: item[0]):
        key = (position, value)
        if key not in seen_positions:
            seen_positions.add(key)
            ordered.append(value)
    return tuple(ordered)


def _mode_name(mode: object) -> str:
    value = getattr(mode, "value", mode)
    text = str(value)
    if text not in _MODE_VALUES:
        raise ValueError(f"Unsupported resolved H3 input mode: {text!r}.")
    return text


def _alignment_contract(mode: str, duration_seconds: float) -> str:
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
        return '{"style":"...","shots":[{"description":"...","start_seconds":0,"dialogues":[]}],"overall_soundscape":"...","non_diegetic_music":"N/A","task_types":["reference generation"],"summary":"...","references":[{"label":"<Picture 1>","definition":"...","retention":"fully_preserved","retention_detail":"..."}]}'
    return '{"style":"","shots":[{"description":"...","start_seconds":0,"dialogues":[]}],"overall_soundscape":"...","non_diegetic_music":"N/A","task_types":[],"summary":"","references":[]}'


def build_system_prompt(context: PromptBuildContext) -> str:
    """Request semantic JSON; Python owns every official output-format token."""

    mode = _mode_name(context.mode)
    if context.profile not in JR_DIRECTOR_PROFILES:
        raise ValueError(f"Unknown JR director profile: {context.profile!r}.")
    sections = get_spec_for_mode(mode).sections
    section_contract = " -> ".join(f"{name}:" for name in sections)
    reference_notes = context.reference_instructions.strip() or "No additional reference relationship instructions."
    duration_text = f"{float(context.duration_seconds):g}"
    starts = ", ".join(f"{value:g}" for value in context.shot_starts) or "model-proposed"
    labels = ", ".join(context.reference_labels) or "none"
    return f"""Mission
Return one JSON object containing audiovisual semantics for a MiniMax H3 prompt. Do not output the final H3 prompt, section headings, Shot headers, timestamps, reference numbering, speaker IDs, dialogue text, retention spelling variants, Markdown, or analysis. Python formats and validates the official output.

Authority and preservation
The user's explicit intent has highest priority. Do not invent or rewrite dialogue. Protected dialogue is supplied by index; place every literal_index exactly once in a semantic shot and provide only speaker_key, speaker_description, and delivery. Python inserts the byte-exact text, language tag, and stable speaker ID.

Closed-world faithful rewrite contract
- Treat the original request, Director shot direction/notes/timing, registered reference relationships, and facts directly visible in supplied reference images as the complete source of truth.
- Every concrete action, pose, gesture, gaze, facial expression, emotion, relationship, prop interaction, setting change, camera move, sound, and music cue must be traceable to that source. Do not add, replace, intensify, or dramatize one.
- Dialogue meaning must not be used to invent an off-screen participant, relationship, video-call context, motivation, or story goal. For example, a greeting to a teacher does not prove that a teacher is present or that a video call is occurring.
- Reference images may supply directly observable stable appearance, clothing, environment, and visible-prop facts. Do not assume an initial pose or prop state persists across later shots unless the Director explicitly requires it.
- A stated action may receive only the minimal physical consequence needed to make that same action readable; it must not gain a second action. If a detail is unspecified or ambiguous, omit it instead of completing it creatively.
- Use one definite action, not alternatives or speculation. Avoid "or", "either", "likely", "possibly", "perhaps", "maybe", "appears to", and "seems to" in semantic prose.
- Summary, reference retention, every shot, soundscape, and music must agree. Never claim that a pose, prop, or state is preserved throughout when later shots do not preserve it.

Pinned official H3 contract (Python-owned)
Resolved mode: {mode}
Final field order (informational only; never emit these headings): {section_contract}
Authoritative Director shot starts: {starts}
Allowed reference labels in exact order: {labels}
Return this JSON shape, using JSON strings and arrays only:
{_output_skeleton(mode)}
Mode planning rule: {_mode_planning(mode)}
Each shot object has description, start_seconds, and dialogues. A dialogue item has literal_index, speaker_key, speaker_description, and delivery; never include its text. If Director starts are supplied, return exactly that many shots and copy those start_seconds values. For Ref2VA, references must contain every allowed label exactly once and in order. Visible retention is one of fully_preserved, partially_preserved, attribute_transfer, weak_reference; Audio retention is one of fully_copy, partially_copy, reference, weak_reference. task_types uses only keyframe completion, reference generation, video editing, video continuation, audio reuse, audio reference. Base modes must return empty task_types, summary, and references.

JR creative director layer
Selected profile: {context.profile}
Direction: {JR_DIRECTOR_PROFILES[context.profile]}
The profile controls emphasis and prose style only. It may clarify staging, performance, motion, and continuity already present in the source, but it must not create new story facts, actions, emotions, poses, relationships, or prop behavior. It must not emit final formatting or override timing, labels, user intent, or protected literals.

Reference registry
{context.registry_text}

User-supplied reference relationships
{reference_notes}

Target
Duration: {duration_text} seconds. Canvas reference: {int(context.target_width)}x{int(context.target_height)}. Shot descriptions, soundscape, music, summary, definitions, and retention details are English semantic prose. Do not place <d> tags or protected dialogue text in any prose field."""


def build_user_prompt(context: PromptBuildContext, preserved_literals: Iterable[str] = ()) -> str:
    """Build the semantic JSON request sent to the model."""

    mode = _mode_name(context.mode)
    literals = tuple(str(value) for value in preserved_literals if str(value))
    protected = "\n".join(f"- {value}" for value in literals) or "- None detected."
    dialogues = "\n".join(
        f"- literal_index={index}: {value}" for index, value in enumerate(context.protected_dialogues, 1)
    ) or "- None"
    duration_text = f"{float(context.duration_seconds):g}"
    return f"""Return the semantic JSON object now.
Resolved input mode: {mode}
Target duration: {duration_text} seconds
Canvas reference: {int(context.target_width)}x{int(context.target_height)}

Registered reference media:
{context.registry_text}

Verbatim literals detected in the user request:
{protected}

Protected dialogue literals (reference by literal_index; do not copy their text into JSON):
{dialogues}

Fidelity reminder:
Perform a conservative rewrite, not a creative continuation. Keep every Director action and note authoritative; omit unspecified details and do not infer relationships, calls, motives, gestures, emotions, or alternative actions.

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
    "extract_protected_dialogues",
    "extract_preserved_literals",
    "registry_as_text",
]
