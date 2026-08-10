import asyncio

import pytest
from aiohttp import web
from ComfyUI_JR_MiniMaxH3Node.server.director_media_routes import (
    _read_bounded_json,
    probe_director_asset,
)
from ComfyUI_JR_MiniMaxH3Node.utils.director_state import MAX_STATE_BYTES


class _Content:
    def __init__(self, payload):
        self.payload = payload
        self.position = 0

    async def read(self, limit):
        chunk = self.payload[self.position:self.position + limit]
        self.position += len(chunk)
        return chunk


class _Request:
    def __init__(self, payload, content_length=None):
        self.content_length = content_length
        self.content = _Content(payload)


def test_bounded_json_accepts_small_chunked_request():
    result = asyncio.run(_read_bounded_json(_Request(b'{"kind":"image"}')))
    assert result == {"kind": "image"}


def test_bounded_json_rejects_oversized_chunked_request():
    request = _Request(b"x" * (MAX_STATE_BYTES + 1))
    with pytest.raises(web.HTTPRequestEntityTooLarge):
        asyncio.run(_read_bounded_json(request))


def test_bounded_json_rejects_invalid_utf8_json():
    with pytest.raises(web.HTTPBadRequest) as caught:
        asyncio.run(_read_bounded_json(_Request(b"\xff")))
    assert "valid UTF-8 JSON" in caught.value.text


def test_bounded_json_rejects_excessive_json_nesting():
    payload = ("[" * 5000 + "0" + "]" * 5000).encode("utf-8")
    with pytest.raises(web.HTTPBadRequest):
        asyncio.run(_read_bounded_json(_Request(payload)))


def test_bounded_json_rejects_unsupported_large_integer():
    with pytest.raises(web.HTTPBadRequest):
        asyncio.run(_read_bounded_json(_Request(("9" * 5000).encode("utf-8"))))


def test_probe_route_preserves_bounded_request_http_error():
    request = _Request(b"x" * (MAX_STATE_BYTES + 1))
    with pytest.raises(web.HTTPRequestEntityTooLarge):
        asyncio.run(probe_director_asset(request))
