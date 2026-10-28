"""Bounded local editor endpoints. No arbitrary filesystem paths or remote URLs."""

import asyncio
import json
from urllib.parse import quote

from ..utils.cut_audio import CutAudioError, analyze_audio, commit_cut, list_audio_inputs, media_path, resolve_audio

_SLOTS = asyncio.Semaphore(1)


async def _body(request):
    from aiohttp import web

    raw = bytearray()
    while chunk := await request.content.read(4097 - len(raw)):
        raw.extend(chunk)
        if len(raw) > 4096:
            raise web.HTTPRequestEntityTooLarge(max_size=4096, actual_size=len(raw))
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, RecursionError) as error:
        raise web.HTTPBadRequest(text="Expected a small JSON object.") from error


async def _work(function, *args):
    from aiohttp import web

    # Do not build an unbounded queue of expensive decoder requests. A cancelled
    # browser request does not release its slot until the worker actually exits.
    if _SLOTS.locked():
        raise web.HTTPServiceUnavailable(text="Audio editor is busy; retry shortly.")
    await _SLOTS.acquire()
    task = asyncio.create_task(asyncio.to_thread(function, *args))

    def done(finished):
        try:
            finished.exception()
        except asyncio.CancelledError:
            pass
        finally:
            _SLOTS.release()

    task.add_done_callback(done)
    try:
        return await asyncio.shield(task)
    except CutAudioError as error:
        raise web.HTTPBadRequest(text=str(error)) from error
    except Exception as error:
        raise web.HTTPInternalServerError(text="JR Cut Audio could not process the local audio file.") from error


async def audio_files(request):
    from aiohttp import web

    return web.json_response({"files": await _work(list_audio_inputs)})


async def audio_info(request):
    from aiohttp import web

    data = await _body(request)
    return web.json_response(await _work(analyze_audio, data.get("audio")))


async def audio_cut(request):
    from aiohttp import web

    data = await _body(request)
    return web.json_response(await _work(commit_cut, data.get("audio"), data.get("start"),
                                        data.get("end"), data.get("source_id")))


async def audio_media(request):
    from aiohttp import web

    try:
        path = media_path(request.match_info["token"])
    except CutAudioError as error:
        raise web.HTTPBadRequest(text=str(error)) from error
    if not path.is_file():
        raise web.HTTPNotFound(text="Preview expired. Reload the audio or press Cut again.")
    response = web.FileResponse(path)
    response.content_type = "audio/wav"
    if request.query.get("download") == "1":
        response.headers["Content-Disposition"] = f'attachment; filename="JR_Cut_Audio_{path.stem[:12]}.wav"'
    return response


async def audio_original(request):
    from aiohttp import web

    try:
        path = resolve_audio(request.query.get("audio"))
    except CutAudioError as error:
        raise web.HTTPBadRequest(text=str(error)) from error
    response = web.FileResponse(path)
    response.headers["Content-Disposition"] = "attachment; filename*=UTF-8''" + quote(path.name, safe="")
    return response


def register_cut_audio_routes():
    try:
        from server import PromptServer
    except ImportError:
        return False
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return False
    guard = "_jr_cut_audio_routes_registered"
    if getattr(instance, guard, False):
        return True
    instance.routes.get("/jr-cut-audio/files")(audio_files)
    instance.routes.post("/jr-cut-audio/info")(audio_info)
    instance.routes.post("/jr-cut-audio/cut")(audio_cut)
    instance.routes.get("/jr-cut-audio/media/{token}.wav")(audio_media)
    instance.routes.get("/jr-cut-audio/original")(audio_original)
    setattr(instance, guard, True)
    return True
