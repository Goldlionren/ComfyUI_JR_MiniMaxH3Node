"""Bounded media-inspection route used by the Director Desk frontend."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from ..utils.director_media import PROBE_TIMEOUT_SECONDS, DirectorMediaError, probe_asset
from ..utils.director_state import (
    MAX_STATE_BYTES,
    DirectorStateError,
    asset_descriptor_from_dict,
    asset_descriptor_to_dict,
)

_REGISTERED = False
_PROBE_SEMAPHORE = asyncio.Semaphore(2)


async def _read_bounded_json(request):
    from aiohttp import web

    content_length = request.content_length
    if content_length is not None and content_length > MAX_STATE_BYTES:
        raise web.HTTPRequestEntityTooLarge(max_size=MAX_STATE_BYTES, actual_size=content_length)
    chunks = bytearray()
    while True:
        chunk = await request.content.read(min(64 * 1024, MAX_STATE_BYTES + 1 - len(chunks)))
        if not chunk:
            break
        chunks.extend(chunk)
        if len(chunks) > MAX_STATE_BYTES:
            raise web.HTTPRequestEntityTooLarge(max_size=MAX_STATE_BYTES, actual_size=len(chunks))
    try:
        return json.loads(chunks.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise web.HTTPBadRequest(text="Director media request must be valid UTF-8 JSON.") from error


async def _probe_with_budget(asset):
    await _PROBE_SEMAPHORE.acquire()
    task = asyncio.create_task(asyncio.to_thread(probe_asset, asset))
    release_on_completion = False
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=PROBE_TIMEOUT_SECONDS + 2)
    except (TimeoutError, asyncio.CancelledError):
        release_on_completion = True
        def finish_background_probe(completed):
            try:
                completed.exception()
            except asyncio.CancelledError:
                pass
            finally:
                _PROBE_SEMAPHORE.release()

        task.add_done_callback(finish_background_probe)
        raise
    finally:
        if not release_on_completion:
            _PROBE_SEMAPHORE.release()


async def probe_director_asset(request):
    from aiohttp import web

    try:
        body = await _read_bounded_json(request)
        asset = asset_descriptor_from_dict(body)
        updated, metadata = await _probe_with_budget(asset)
    except web.HTTPException:
        raise
    except TimeoutError as error:
        raise web.HTTPGatewayTimeout(text="Director media inspection timed out.") from error
    except DirectorStateError as error:
        raise web.HTTPBadRequest(text=str(error)) from error
    except DirectorMediaError as error:
        message = str(error)
        if "missing" in message.lower():
            raise web.HTTPNotFound(text=message) from error
        raise web.HTTPUnprocessableEntity(text=message) from error
    except Exception as error:
        raise web.HTTPInternalServerError(text="Director media inspection failed safely.") from error
    result = asset_descriptor_to_dict(replace(updated, status=metadata.get("status", updated.status)))
    result["metadata"] = metadata
    return web.json_response(result)


def register_director_media_routes() -> bool:
    global _REGISTERED
    try:
        from server import PromptServer
    except ImportError:
        return False
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return False
    guard = "_jr_h3_director_media_routes_registered"
    if getattr(instance, guard, False):
        _REGISTERED = True
        return True
    instance.routes.post("/jr-h3/director/probe")(probe_director_asset)
    setattr(instance, guard, True)
    _REGISTERED = True
    return True


ROUTES_REGISTERED = register_director_media_routes()

__all__ = ["ROUTES_REGISTERED", "probe_director_asset", "register_director_media_routes"]
