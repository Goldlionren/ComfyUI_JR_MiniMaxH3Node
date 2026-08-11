"""OpenAI-compatible H3 prompt optimizer with clean-room official formatting."""

from __future__ import annotations

import re

from ..utils.director_pipe_adapter import pipe_to_optimizer_context, validate_legacy_conflicts
from ..utils.h3_prompt_builder import (
    JR_DIRECTOR_PROFILES,
    REF_SECTIONS,
    PromptBuildContext,
    build_system_prompt,
    build_user_prompt,
    extract_preserved_literals,
    registry_as_text,
)
from ..utils.h3_prompt_modes import H3InputMode, find_reference_label_tokens, route_h3_mode
from ..utils.h3_prompt_validator import ValidationResult, validate_prompt
from ..utils.h3_reference_registry import ReferenceRegistry
from ..utils.image_conversion import image_batch_to_jpeg_data_urls
from ..utils.openai_compat import (
    discover_model,
    normalize_api_urls,
    normalize_picture_markers,
    parse_chat_content,
    request_chat,
)
from ..utils.safe_logging import safe_error

_PROFILES = JR_DIRECTOR_PROFILES
_H3_INPUT_MODES = [mode.value for mode in H3InputMode]
_REPAIR_TEMPERATURE = 0.1
_MAX_LEGACY_PROMPT_BYTES = 512 * 1024
_MAX_REFERENCE_INSTRUCTIONS_BYTES = 64 * 1024


def _validate_text_size(value, field, limit):
    try:
        size = len(str(value).encode("utf-8"))
    except UnicodeEncodeError:
        raise ValueError(f"{field} must be valid UTF-8 text.") from None
    if size > limit:
        raise ValueError(f"{field} exceeds the {limit}-byte limit.")


class _FinalPromptValidationError(ValueError):
    """A candidate still violates the H3 contract after one repair attempt."""

    def __init__(self, validation):
        self.concise_reason = validation.errors[0] if validation.errors else "unknown validation error"
        super().__init__(
            "H3 prompt validation failed after one format-repair attempt: "
            f"{_validation_summary(validation)}"
        )


def _context(prompt, profile, duration, width, height, mode="T2VA", registry_text="No reference media is registered.", reference_instructions=""):
    return PromptBuildContext(
        original_prompt=str(prompt), profile=profile, mode=mode,
        duration_seconds=float(duration), target_width=int(width), target_height=int(height),
        registry_text=registry_text, reference_instructions=reference_instructions,
    )


def _system_prompt(profile, duration, width, height, h3_input_mode="T2VA", registry_text="No reference media is registered.", reference_instructions=""):
    """Compatibility wrapper for the previous private helper."""
    return build_system_prompt(_context("", profile, duration, width, height, h3_input_mode, registry_text, reference_instructions))


def _user_prompt(prompt, profile, duration, width, height, image_count, h3_input_mode="T2VA"):
    """Compatibility wrapper for the previous private helper."""
    labels = "\n".join(
        f"<Picture {index}>: source_input=legacy_ref_image; role=reference"
        for index in range(1, int(image_count) + 1)
    ) or "No reference media is registered."
    normalized = normalize_picture_markers(str(prompt))
    context = _context(normalized, profile, duration, width, height, h3_input_mode, labels)
    return build_user_prompt(context, extract_preserved_literals(normalized))


def _reference_slot_count(kwargs):
    return sum(kwargs.get(f"ref_image_{index}") is not None for index in range(1, 10))


def _encode_and_register_images(registry, *, first_frame, last_frame, image_send_size, kwargs):
    encoded = []

    def add_anchor(value, source_input, role):
        if value is None:
            return
        urls = image_batch_to_jpeg_data_urls(value, image_send_size)
        if len(urls) != 1:
            raise ValueError(f"{source_input} must contain exactly one IMAGE, got batch size {len(urls)}.")
        entry = registry.register_picture(source_input, role, source_key=source_input)
        encoded.append((entry.label, urls[0]))

    add_anchor(first_frame, "first_frame", "first_frame")
    add_anchor(last_frame, "last_frame", "last_frame")
    for slot_index in range(1, 10):
        source_input = f"ref_image_{slot_index}"
        value = kwargs.get(source_input)
        if value is None:
            continue
        for batch_index, data_url in enumerate(image_batch_to_jpeg_data_urls(value, image_send_size), 1):
            entry = registry.register_picture(
                source_input, "reference", source_key=f"{source_input}#{batch_index}"
            )
            encoded.append((entry.label, data_url))
    return encoded


def _register_instruction_only_references(registry, instructions):
    """Track downstream Video/Audio/Subject labels declared by user instructions."""
    for token in dict.fromkeys(find_reference_label_tokens(instructions)):
        if registry.resolve(token) is not None:
            continue
        family = token[1:].split(" ", 1)[0]
        if family == "Picture":
            raise ValueError(f"reference_instructions uses {token}, but no matching image is connected.")
        register = {
            "Subject": registry.register_subject,
            "Video": registry.register_video,
            "Audio": registry.register_audio,
        }[family]
        register(
            "reference_instructions", "subject" if family == "Subject" else "source",
            source_key=f"reference_instructions:{token}", identifier=token,
        )


def _validation_summary(validation):
    summary = "; ".join(validation.errors[:5])
    if len(validation.errors) > 5:
        summary += f"; plus {len(validation.errors) - 5} more error(s)"
    return summary


def _shield_preserved_literals(candidate, preserved_literals):
    shielded = candidate
    shields = []
    for index, literal in enumerate(preserved_literals, 1):
        token = f"__JR_H3_PRESERVED_LITERAL_{index:02d}__"
        count = shielded.count(literal)
        if count:
            shielded = shielded.replace(literal, token)
        else:
            characters = [re.escape(character) for character in literal if not character.isspace()]
            if characters:
                whitespace_tolerant = r"[ \t]*".join(characters)
                shielded, count = re.subn(whitespace_tolerant, token, shielded)
        if count:
            shields.append((token, literal, count))
    return shielded, tuple(shields)


def _restore_preserved_literals(candidate, shields):
    restored = candidate
    errors = []
    for token, literal, expected_count in shields:
        token_count = restored.count(token)
        literal_count = restored.count(literal)
        if token_count + literal_count != expected_count:
            errors.append(f"repair changed protected literal sentinel for {literal!r}")
        restored = restored.replace(token, literal)
    return restored, tuple(errors)


def _normalize_repaired_ref2va_sections(candidate, mode):
    if mode != "Ref2VA":
        return candidate
    names = "|".join(re.escape(name) for name in REF_SECTIONS)
    heading = re.compile(
        rf"^[ \t]*(?:#{{1,6}}[ \t]*)?(?:\*\*|__)?(?P<name>{names}):"
        rf"(?:\*\*|__)?(?:[ \t]*(?P<body>.*))?$",
        re.IGNORECASE,
    )
    normalized = []
    for line in candidate.splitlines():
        match = heading.fullmatch(line)
        if match is None:
            normalized.append(line)
            continue
        canonical = next(
            name for name in REF_SECTIONS
            if name.casefold() == match.group("name").casefold()
        )
        normalized.append(f"{canonical}:")
        body = (match.group("body") or "").strip()
        if body:
            normalized.append(body)
    return "\n".join(normalized)


def _normalize_repaired_ref2va_retention(candidate, mode):
    if mode != "Ref2VA":
        return candidate
    visible_from_audio = {
        "fully_copy": "fully_preserved",
        "partially_copy": "partially_preserved",
        "reference": "attribute_transfer",
    }
    audio_from_visible = {
        "fully_preserved": "fully_copy",
        "partially_preserved": "partially_copy",
        "attribute_transfer": "reference",
    }
    retention = re.compile(
        r"^(?P<prefix>[ \t]*<(?P<family>Subject|Picture|Video|Audio) [1-9]\d*>"
        r"(?:[ \t]*\([^\r\n:]*\))?[ \t]*:[ \t]*)"
        r"(?P<value>[A-Za-z][A-Za-z0-9_-]*)(?P<suffix>.*)$"
    )
    normalized = []
    for line in candidate.splitlines():
        match = retention.fullmatch(line)
        if match is None:
            normalized.append(line)
            continue
        mapping = (
            audio_from_visible
            if match.group("family") == "Audio"
            else visible_from_audio
        )
        value = mapping.get(match.group("value"), match.group("value"))
        normalized.append(f"{match.group('prefix')}{value}{match.group('suffix')}")
    return "\n".join(normalized)


def _normalize_repaired_ref2va_subject_definitions(candidate, mode):
    if mode != "Ref2VA":
        return candidate
    normalized = []
    in_definitions = False
    definition = re.compile(
        r"^(?P<label>[ \t]*<Subject [1-9]\d*>)[ \t]*:[ \t]*(?P<body>.+)$"
    )
    for line in candidate.splitlines():
        if line == "subject_definitions:":
            in_definitions = True
            normalized.append(line)
            continue
        if line == "summary:":
            in_definitions = False
            normalized.append(line)
            continue
        match = definition.fullmatch(line) if in_definitions else None
        if match is None:
            normalized.append(line)
        else:
            normalized.append(f"{match.group('label')} is {match.group('body')}")
    return "\n".join(normalized)


def _repair_payload(*, model, context, candidate, validation, preserved_literals, max_tokens):
    protected = "\n".join(f"- {literal}" for literal in preserved_literals) or "- None"
    errors = "\n".join(f"- {error}" for error in validation.errors)
    repair_system = f"""You perform one constrained MiniMax H3 format repair.
Output only the repaired prompt. Repair syntax and structure only: field names, field order, shot markers, timestamps, alignment syntax, reference syntax, and other deterministic format violations. Do not rewrite, expand, summarize, embellish, translate, or otherwise change the story, actions, camera intent, sounds, dialogue, lyrics, visible text, names, or technical terms. Preserve every protected literal exactly. Any __JR_H3_PRESERVED_LITERAL_NN__ token is immutable: copy it exactly once at its existing location and never expand, remove, rename, or interpret it. If content is already valid, leave it unchanged.

Authoritative format contract:
{build_system_prompt(context)}"""
    repair_user = f"""Repair the candidate only for the listed validation errors.

Validation errors:
{errors}

Protected user literals (must remain byte-for-byte present):
{protected}

Candidate prompt:
{candidate}"""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": repair_system},
            {"role": "user", "content": repair_user},
        ],
        "temperature": _REPAIR_TEMPERATURE,
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
        "stream": False,
    }


class JR_H3_OpenAICompatiblePromptOptimizer:
    CATEGORY = "JR MiniMax H3/Prompt"
    FUNCTION = "optimize"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "JR_H3_DIRECTOR_PIPE")
    RETURN_NAMES = ("optimized_prompt", "original_prompt", "status", "pip")
    DESCRIPTION = "Builds and validates mode-aware MiniMax H3 prompts through an OpenAI-compatible endpoint."

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "enable": ("BOOLEAN", {"default": True}),
            "api_base_url": ("STRING", {"default": "http://127.0.0.1:10000"}),
            "model": ("STRING", {"default": ""}),
            "prompt_profile": (list(_PROFILES), {"default": "Standard"}),
            "duration_seconds": ("INT", {"default": 10, "min": 1, "max": 60}),
            "target_width": ("INT", {"default": 768, "min": 64, "max": 8192}),
            "target_height": ("INT", {"default": 1152, "min": 64, "max": 8192}),
            "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
            "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
            "max_tokens": ("INT", {"default": 1800, "min": 32, "max": 32768}),
            "timeout_seconds": ("INT", {"default": 180, "min": 1, "max": 1800}),
            "image_send_size": ("INT", {"default": 768, "min": 64, "max": 4096}),
            "fail_mode": (["Return Original", "Stop Workflow"], {"default": "Return Original"}),
            "disable_reasoning": ("BOOLEAN", {"default": True}),
            # Appended after every legacy widget to preserve saved workflow positions.
            "h3_input_mode": (_H3_INPUT_MODES, {"default": "Auto"}),
            "reference_instructions": ("STRING", {"multiline": True, "default": ""}),
        }
        optional = {"api_key": ("STRING", {"default": ""})}
        optional.update({f"ref_image_{index}": ("IMAGE",) for index in range(1, 10)})
        optional["first_frame"] = ("IMAGE",)
        optional["last_frame"] = ("IMAGE",)
        optional["pip"] = ("JR_H3_DIRECTOR_PIPE",)
        return {"required": required, "optional": optional}

    def optimize(
        self, prompt, enable, api_base_url, model, prompt_profile, duration_seconds,
        target_width, target_height, temperature, top_p, max_tokens, timeout_seconds,
        image_send_size, fail_mode, disable_reasoning, h3_input_mode="Auto",
        reference_instructions="", api_key="", first_frame=None, last_frame=None, pip=None, **kwargs,
    ):
        original = str(prompt)
        output_pipe = None
        if not enable and pip is None:
            return original, original, "Disabled: original prompt returned", None
        try:
            _validate_text_size(prompt, "prompt", _MAX_LEGACY_PROMPT_BYTES)
            _validate_text_size(
                reference_instructions,
                "reference_instructions",
                _MAX_REFERENCE_INSTRUCTIONS_BYTES,
            )
            source_suffix = ""
            if pip is not None:
                from ..utils.director_pipe import validate_director_pipe

                pipe = validate_director_pipe(pip)
                output_pipe = pipe
                original = pipe.compiled_director_prompt
                validate_legacy_conflicts(
                    pipe, prompt=str(prompt), reference_instructions=reference_instructions,
                    first_frame=first_frame, last_frame=last_frame,
                    reference_image_count=_reference_slot_count(kwargs),
                    duration_seconds=duration_seconds,
                )
                if not enable:
                    return original, original, "Disabled: original prompt returned", pipe
                adapted = pipe_to_optimizer_context(pipe, int(image_send_size))
                instructions = adapted.reference_instructions
                registry = adapted.registry
                encoded_images = list(adapted.encoded_images)
                duration_seconds = adapted.duration_seconds
                has_first_frame = adapted.has_first_frame
                has_last_frame = adapted.has_last_frame
                reference_image_count = adapted.reference_image_count
                source_suffix = ", source=pip"
            else:
                instructions = normalize_picture_markers(str(reference_instructions or ""))
                registry = ReferenceRegistry()
                encoded_images = _encode_and_register_images(
                    registry, first_frame=first_frame, last_frame=last_frame,
                    image_send_size=int(image_send_size), kwargs=kwargs,
                )
                _register_instruction_only_references(registry, instructions)
                registry.validate_references(instructions)
                has_first_frame = first_frame is not None
                has_last_frame = last_frame is not None
                reference_image_count = _reference_slot_count(kwargs)
            selected_mode = route_h3_mode(
                h3_input_mode,
                has_first_frame=has_first_frame,
                has_last_frame=has_last_frame,
                reference_image_count=reference_image_count,
                reference_instructions=instructions,
            )

            normalized_original = normalize_picture_markers(original)
            preserved_literals = extract_preserved_literals(normalized_original)
            context = _context(
                normalized_original, prompt_profile, duration_seconds, target_width, target_height,
                selected_mode.value, registry_as_text(registry.entries()), instructions,
            )
            models_url, chat_url = normalize_api_urls(api_base_url)
            selected_model = model.strip() or discover_model(models_url, timeout_seconds, api_key)
            user_content = [{"type": "text", "text": build_user_prompt(context, preserved_literals)}]
            for label, data_url in encoded_images:
                user_content.append({"type": "text", "text": f"[{label[1:-1]}]"})
                user_content.append({"type": "image_url", "image_url": {"url": data_url}})
            payload = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": build_system_prompt(context)},
                    {"role": "user", "content": user_content},
                ],
                "temperature": float(temperature), "top_p": float(top_p),
                "max_tokens": int(max_tokens), "stream": False,
            }
            if disable_reasoning:
                payload["reasoning_effort"] = "none"
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            response = request_chat(chat_url, payload, timeout_seconds, api_key, disable_reasoning)
            raw_prompt = parse_chat_content(response)
            validation = validate_prompt(
                raw_prompt, mode=selected_mode.value, duration_seconds=duration_seconds,
                allowed_labels=registry.labels(), preserved_literals=preserved_literals,
            )
            if not validation.valid:
                shielded_prompt, literal_shields = _shield_preserved_literals(
                    raw_prompt, preserved_literals
                )
                repair = _repair_payload(
                    model=selected_model, context=context, candidate=shielded_prompt,
                    validation=validation, preserved_literals=preserved_literals,
                    max_tokens=max_tokens,
                )
                # No reasoning extensions are sent here, so request_chat cannot perform
                # its compatibility retry. This is exactly one format-repair request.
                repaired_response = request_chat(
                    chat_url, repair, timeout_seconds, api_key, False,
                )
                repaired_prompt, shield_errors = _restore_preserved_literals(
                    parse_chat_content(repaired_response), literal_shields
                )
                repaired_prompt = _normalize_repaired_ref2va_sections(
                    repaired_prompt, selected_mode.value
                )
                repaired_prompt = _normalize_repaired_ref2va_subject_definitions(
                    repaired_prompt, selected_mode.value
                )
                repaired_prompt = _normalize_repaired_ref2va_retention(
                    repaired_prompt, selected_mode.value
                )
                repaired_validation = validate_prompt(
                    repaired_prompt, mode=selected_mode.value,
                    duration_seconds=duration_seconds, allowed_labels=registry.labels(),
                    preserved_literals=preserved_literals,
                )
                if shield_errors:
                    repaired_validation = ValidationResult(
                        cleaned_prompt=repaired_validation.cleaned_prompt,
                        valid=False,
                        errors=shield_errors + repaired_validation.errors,
                    )
                if not repaired_validation.valid:
                    raise _FinalPromptValidationError(repaired_validation)
                return (
                    repaired_validation.cleaned_prompt, original,
                    f"Success: model={selected_model}, mode={selected_mode.value}, repaired=1{source_suffix}",
                    (
                        pipe.derive(
                            optimized_prompt=repaired_validation.cleaned_prompt,
                            reviewed_prompt="",
                        )
                        if pip is not None else None
                    ),
                )
            return (
                validation.cleaned_prompt, original,
                f"Success: model={selected_model}, mode={selected_mode.value}, repaired=0{source_suffix}",
                (
                    pipe.derive(optimized_prompt=validation.cleaned_prompt, reviewed_prompt="")
                    if pip is not None else None
                ),
            )
        except _FinalPromptValidationError as error:
            if fail_mode == "Stop Workflow":
                raise ValueError(str(error)) from None
            return original, original, f"Fallback: {error.concise_reason}", output_pipe
        except Exception as error:
            message = safe_error(error, api_key)
            if fail_mode == "Stop Workflow":
                raise RuntimeError(message) from None
            return original, original, f"Fallback: {message}", output_pipe


__all__ = ["JR_H3_OpenAICompatiblePromptOptimizer", "_system_prompt", "_user_prompt"]
