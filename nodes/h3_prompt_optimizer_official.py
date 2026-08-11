"""OpenAI-compatible H3 prompt optimizer with clean-room official formatting."""

from __future__ import annotations

from ..utils.director_pipe_adapter import pipe_to_optimizer_context, validate_legacy_conflicts
from ..utils.h3_official_prompt_formatter import H3OfficialFormatError, format_official_prompt
from ..utils.h3_official_prompt_schema import H3SemanticError, parse_semantic_response
from ..utils.h3_prompt_builder import (
    JR_DIRECTOR_PROFILES,
    PromptBuildContext,
    build_system_prompt,
    build_user_prompt,
    extract_preserved_literals,
    extract_protected_dialogues,
    registry_as_text,
)
from ..utils.h3_prompt_modes import H3InputMode, find_reference_label_tokens, route_h3_mode
from ..utils.h3_prompt_validator import validate_prompt
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
            "Deterministic official H3 formatter output failed validation: "
            f"{_validation_summary(validation)}"
        )


class _FinalSemanticError(ValueError):
    """Structured semantic JSON is still invalid after one repair."""

    def __init__(self, error):
        self.concise_reason = str(error)
        super().__init__(f"H3 semantic JSON failed after one structured repair: {error}")


def _context(
    prompt, profile, duration, width, height, mode="T2VA",
    registry_text="No reference media is registered.", reference_instructions="",
    shot_starts=(), reference_labels=(), protected_dialogues=(),
):
    return PromptBuildContext(
        original_prompt=str(prompt), profile=profile, mode=mode,
        duration_seconds=float(duration), target_width=int(width), target_height=int(height),
        registry_text=registry_text, reference_instructions=reference_instructions,
        shot_starts=tuple(shot_starts), reference_labels=tuple(reference_labels),
        protected_dialogues=tuple(protected_dialogues),
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


def _repair_payload(*, model, context, candidate, error, max_tokens):
    repair_system = f"""You perform one constrained semantic JSON repair.
Output exactly one JSON object. Do not output a final H3 prompt, Markdown, analysis, section headings, Shot headers, timestamps, speaker IDs, dialogue text, or reference labels not present in the contract. Preserve the candidate's semantic intent. Fix only JSON/schema errors. Python owns all final MiniMax H3 formatting.

Authoritative semantic contract:
{build_system_prompt(context)}"""
    repair_user = f"""Repair this semantic JSON candidate once.

Validation error:
- {error}

Candidate semantic response:
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
                shot_starts = adapted.shot_starts
                protected_dialogues = adapted.protected_dialogues
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
                shot_starts = ()
                protected_dialogues = extract_protected_dialogues(original)
            selected_mode = route_h3_mode(
                h3_input_mode,
                has_first_frame=has_first_frame,
                has_last_frame=has_last_frame,
                reference_image_count=reference_image_count,
                reference_instructions=instructions,
            )

            normalized_original = normalize_picture_markers(original)
            preserved_literals = tuple(dict.fromkeys(
                (*extract_preserved_literals(normalized_original), *protected_dialogues)
            ))
            context = _context(
                normalized_original, prompt_profile, duration_seconds, target_width, target_height,
                selected_mode.value, registry_as_text(registry.entries()), instructions,
                shot_starts=shot_starts, reference_labels=registry.labels(),
                protected_dialogues=protected_dialogues,
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
            raw_semantic = parse_chat_content(response)
            repaired = 0
            try:
                semantic = parse_semantic_response(
                    raw_semantic,
                    mode=selected_mode.value,
                    allowed_labels=registry.labels(),
                    protected_dialogue_count=len(protected_dialogues),
                    protected_dialogues=protected_dialogues,
                    expected_shot_count=len(shot_starts) if shot_starts else None,
                )
            except H3SemanticError as first_error:
                repair = _repair_payload(
                    model=selected_model, context=context, candidate=raw_semantic,
                    error=first_error, max_tokens=max_tokens,
                )
                # No reasoning extensions are sent here, so this is exactly one
                # structured semantic-repair request.
                repaired_response = request_chat(
                    chat_url, repair, timeout_seconds, api_key, False,
                )
                try:
                    semantic = parse_semantic_response(
                        parse_chat_content(repaired_response),
                        mode=selected_mode.value,
                        allowed_labels=registry.labels(),
                        protected_dialogue_count=len(protected_dialogues),
                        protected_dialogues=protected_dialogues,
                        expected_shot_count=len(shot_starts) if shot_starts else None,
                    )
                except H3SemanticError as second_error:
                    raise _FinalSemanticError(second_error) from None
                repaired = 1

            try:
                formatted_prompt = format_official_prompt(
                    semantic,
                    mode=selected_mode.value,
                    duration_seconds=duration_seconds,
                    protected_dialogues=protected_dialogues,
                    authoritative_shot_starts=shot_starts,
                )
            except H3OfficialFormatError as error:
                raise _FinalSemanticError(error) from None
            validation = validate_prompt(
                formatted_prompt, mode=selected_mode.value, duration_seconds=duration_seconds,
                allowed_labels=registry.labels(), preserved_literals=preserved_literals,
                protected_dialogues=protected_dialogues,
            )
            if not validation.valid:
                raise _FinalPromptValidationError(validation)
            return (
                validation.cleaned_prompt, original,
                f"Success: model={selected_model}, mode={selected_mode.value}, repaired={repaired}{source_suffix}",
                (
                    pipe.derive(optimized_prompt=validation.cleaned_prompt, reviewed_prompt="")
                    if pip is not None else None
                ),
            )
        except (_FinalPromptValidationError, _FinalSemanticError) as error:
            if fail_mode == "Stop Workflow":
                raise ValueError(str(error)) from None
            return original, original, f"Fallback: {error.concise_reason}", output_pipe
        except Exception as error:
            message = safe_error(error, api_key)
            if fail_mode == "Stop Workflow":
                raise RuntimeError(message) from None
            return original, original, f"Fallback: {message}", output_pipe


__all__ = ["JR_H3_OpenAICompatiblePromptOptimizer", "_system_prompt", "_user_prompt"]
