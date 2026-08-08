"""Minimal OpenAI-compatible HTTP client built on the standard library."""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request


class OpenAICompatError(RuntimeError):
    """Safe error for an OpenAI-compatible endpoint."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


def normalize_api_urls(api_base_url: str) -> tuple[str, str]:
    raw = (api_base_url or "").strip()
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("api_base_url must be an absolute http:// or https:// URL.")
    path = parsed.path.rstrip("/")
    lower = path.lower()
    if lower.endswith("/v1/chat/completions"):
        root = path[: -len("/v1/chat/completions")]
    elif lower.endswith("/v1/models"):
        root = path[: -len("/v1/models")]
    elif lower.endswith("/v1"):
        root = path[: -len("/v1")]
    else:
        root = path
    base = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, root.rstrip("/"), "", ""))
    return base + "/v1/models", base + "/v1/chat/completions"


def _request_json(url: str, timeout: float, *, method: str = "GET", payload=None, api_key: str = ""):
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            body = response.read(4 * 1024 * 1024)
    except urllib.error.HTTPError as error:
        body = error.read(2000).decode("utf-8", errors="replace")
        body = re.sub(r"data:image/[^\s\"]+", "<image-data-redacted>", body)
        raise OpenAICompatError(f"HTTP {error.code} from {url}: {body[:2000]}", status=error.code) from None
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as error:
        raise OpenAICompatError(f"{type(error).__name__} while requesting {url}: {error}") from None
    if not body:
        raise OpenAICompatError(f"Empty response from {url}.")
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenAICompatError(f"Invalid JSON from {url}: {type(error).__name__}") from None


def discover_model(models_url: str, timeout: float, api_key: str = "") -> str:
    data = _request_json(models_url, timeout, api_key=api_key)
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list) or not models or not isinstance(models[0], dict) or not models[0].get("id"):
        raise OpenAICompatError(f"No model IDs returned by {models_url}.")
    return str(models[0]["id"])


def request_chat(chat_url: str, payload: dict, timeout: float, api_key: str = "", retry_reasoning_400: bool = False):
    try:
        return _request_json(chat_url, timeout, method="POST", payload=payload, api_key=api_key)
    except OpenAICompatError as error:
        compatibility_error = re.search(
            r"unknown\s+(?:field|parameter)|unsupported\s+(?:field|parameter)|unrecognized\s+(?:field|parameter)",
            str(error),
            re.IGNORECASE,
        )
        if error.status != 400 or not retry_reasoning_400 or compatibility_error is None:
            raise
        fallback = dict(payload)
        fallback.pop("reasoning_effort", None)
        fallback.pop("chat_template_kwargs", None)
        return _request_json(chat_url, timeout, method="POST", payload=fallback, api_key=api_key)


_PICTURE_RE = re.compile(r"<(?:image|picture)\s*(\d+)>|\[(?:image|picture)\s*(\d+)\]", re.I)


def normalize_picture_markers(text: str) -> str:
    return _PICTURE_RE.sub(lambda m: f"<Picture {m.group(1) or m.group(2)}>", text)


def parse_chat_content(data: object) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (TypeError, KeyError, IndexError):
        raise OpenAICompatError("Response is missing choices[0].message.content.") from None
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text")
    else:
        raise OpenAICompatError("Response message content is neither text nor a text-part array.")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    text = re.sub(r"^\s*```(?:text|markdown)?\s*|\s*```\s*$", "", text, flags=re.I)
    text = re.sub(r"^\s*(?:final answer|optimized prompt|优化后的提示词|最终提示词)\s*[:：]\s*", "", text, flags=re.I)
    text = normalize_picture_markers(text).strip()
    if not text:
        raise OpenAICompatError("The service returned an empty prompt after cleanup.")
    return text
