"""Small helpers that keep secrets and image payloads out of messages."""

import re


def safe_error(error: BaseException, api_key: str = "", limit: int = 500) -> str:
    text = f"{type(error).__name__}: {error}"
    text = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "<image-data-redacted>", text)
    if api_key:
        text = text.replace(api_key, "<api-key-redacted>")
    return text[:limit]
