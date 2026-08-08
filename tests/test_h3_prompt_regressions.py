"""Offline regression coverage for the MiniMax H3 prompt node boundary."""

from __future__ import annotations

import importlib

import pytest
from ComfyUI_JR_MiniMaxH3Node.nodes import h3_prompt_optimizer_official as optimizer_module
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official import (
    JR_H3_OpenAICompatiblePromptOptimizer,
)
from ComfyUI_JR_MiniMaxH3Node.utils.openai_compat import OpenAICompatError, normalize_api_urls


def _legacy_args(**overrides: object) -> dict[str, object]:
    """The pre-H3-mode optimize keyword set, intentionally without new fields."""

    values: dict[str, object] = {
        "prompt": "A paper boat crosses a quiet pool.",
        "enable": False,
        "api_base_url": "http://127.0.0.1:10000",
        "model": "test-model",
        "prompt_profile": "Standard",
        "duration_seconds": 5,
        "target_width": 768,
        "target_height": 1152,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1800,
        "timeout_seconds": 2,
        "image_send_size": 768,
        "fail_mode": "Return Original",
        "disable_reasoning": True,
        "api_key": "",
    }
    values.update(overrides)
    return values


def _valid_t2va_response() -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": (
                        "integrated_multimodal_description: [Shot 1] A paper boat crosses the pool.\n"
                        "overall_soundscape: Water moves softly.\n"
                        "non_diegetic_music: N/A"
                    )
                }
            }
        ]
    }


def test_legacy_optimize_kwargs_use_defaults_without_new_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Saved callers that omit appended mode/anchor arguments remain callable."""

    calls: list[str] = []

    def fake_request(url: str, payload: dict[str, object], *args: object) -> dict[str, object]:
        calls.append(url)
        return _valid_t2va_response()

    monkeypatch.setattr(optimizer_module, "request_chat", fake_request)
    optimized, original, status = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_legacy_args(enable=True)
    )

    assert optimized.startswith("integrated_multimodal_description:")
    assert original == _legacy_args()["prompt"]
    assert "mode=T2VA" in status
    assert calls == ["http://127.0.0.1:10000/v1/chat/completions"]


def test_disabled_legacy_call_returns_three_outputs_without_http(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fail_http(*args: object, **kwargs: object) -> object:
        calls.append("http")
        raise AssertionError("disabled node must not perform HTTP or model discovery")

    monkeypatch.setattr(optimizer_module, "normalize_api_urls", fail_http)
    monkeypatch.setattr(optimizer_module, "discover_model", fail_http)
    monkeypatch.setattr(optimizer_module, "request_chat", fail_http)

    output = JR_H3_OpenAICompatiblePromptOptimizer().optimize(**_legacy_args())

    assert output == (
        "A paper boat crosses a quiet pool.",
        "A paper boat crosses a quiet pool.",
        "Disabled: original prompt returned",
    )
    assert calls == []


def test_required_and_optional_input_order_keeps_legacy_prefix() -> None:
    inputs = JR_H3_OpenAICompatiblePromptOptimizer.INPUT_TYPES()
    legacy_required = [
        "prompt",
        "enable",
        "api_base_url",
        "model",
        "prompt_profile",
        "duration_seconds",
        "target_width",
        "target_height",
        "temperature",
        "top_p",
        "max_tokens",
        "timeout_seconds",
        "image_send_size",
        "fail_mode",
        "disable_reasoning",
    ]
    legacy_optional = ["api_key", *(f"ref_image_{index}" for index in range(1, 10))]

    assert list(inputs["required"]) == legacy_required + ["h3_input_mode", "reference_instructions"]
    assert list(inputs["optional"]) == legacy_optional + ["first_frame", "last_frame"]


def test_optimizer_node_metadata_and_registration_are_stable() -> None:
    package = importlib.import_module("ComfyUI_JR_MiniMaxH3Node")
    node = package.NODE_CLASS_MAPPINGS["JR_H3_OpenAICompatiblePromptOptimizer"]

    assert node is JR_H3_OpenAICompatiblePromptOptimizer
    assert (
        package.NODE_DISPLAY_NAME_MAPPINGS["JR_H3_OpenAICompatiblePromptOptimizer"]
        == "JR MiniMax H3 Prompt Optimizer (OpenAI Compatible)"
    )
    assert node.RETURN_TYPES == ("STRING", "STRING", "STRING")
    assert node.FUNCTION == "optimize"
    assert node.CATEGORY == "JR MiniMax H3/Prompt"


def test_network_error_returns_original_or_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_request(*args: object, **kwargs: object) -> object:
        raise OpenAICompatError("connection failed")

    monkeypatch.setattr(optimizer_module, "request_chat", fail_request)
    fallback = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        **_legacy_args(enable=True, fail_mode="Return Original")
    )
    assert fallback[:2] == (_legacy_args()["prompt"], _legacy_args()["prompt"])
    assert fallback[2].startswith("Fallback: OpenAICompatError: connection failed")

    with pytest.raises(RuntimeError, match="OpenAICompatError: connection failed"):
        JR_H3_OpenAICompatiblePromptOptimizer().optimize(
            **_legacy_args(enable=True, fail_mode="Stop Workflow")
        )


@pytest.mark.parametrize("fail_mode", ["Return Original", "Stop Workflow"])
def test_validation_error_obeys_fail_mode(
    monkeypatch: pytest.MonkeyPatch, fail_mode: str
) -> None:
    def invalid_response(*args: object, **kwargs: object) -> dict[str, object]:
        return {"choices": [{"message": {"content": "not an H3 prompt"}}]}

    monkeypatch.setattr(optimizer_module, "request_chat", invalid_response)
    call = lambda: JR_H3_OpenAICompatiblePromptOptimizer().optimize(  # noqa: E731
        **_legacy_args(enable=True, fail_mode=fail_mode)
    )

    if fail_mode == "Return Original":
        fallback = call()
        assert fallback[:2] == (_legacy_args()["prompt"], _legacy_args()["prompt"])
        assert fallback[2].startswith("Fallback: missing required section")
    else:
        with pytest.raises(ValueError, match="after one format-repair attempt"):
            call()


def test_full_models_url_is_normalized_without_duplicate_v1() -> None:
    assert normalize_api_urls("http://127.0.0.1:10000/v1/models") == (
        "http://127.0.0.1:10000/v1/models",
        "http://127.0.0.1:10000/v1/chat/completions",
    )
    assert normalize_api_urls("https://host.example/prefix/v1/models/") == (
        "https://host.example/prefix/v1/models",
        "https://host.example/prefix/v1/chat/completions",
    )
