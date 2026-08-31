"""Server-side auto-continue for sequential audio chunks.

Mirrors the frontend requeue contract (js/sequential_audio.js) without a
browser: wait until the committing prompt finishes successfully, then re-post
its own API prompt through the public /prompt route. Coordination is per
source prompt — one replay covers every sequential chain inside it — so total
posts stay bounded by the manifests' total_chunks.
"""

from __future__ import annotations

import asyncio
import logging
import threading

_log = logging.getLogger(__name__)

_POLL_SECONDS = 1.0
_LOST_PROMPT_GRACE_SECONDS = 30.0
_POST_TIMEOUT_SECONDS = 30.0

_lock = threading.Lock()
_pending_prompts: set[str] = set()


def _server():
    try:
        from server import PromptServer
    except ImportError:
        return None
    return getattr(PromptServer, "instance", None)


def _contains_sequential_output(api_prompt: dict) -> bool:
    return any(
        isinstance(node, dict) and node.get("class_type") == "JR_H3_SequentialVideoOutput"
        for node in api_prompt.values()
    )


def _prompt_in_queue(queue, prompt_id: str) -> bool:
    try:
        running, pending = queue.get_current_queue()
    except Exception:
        return False
    for item in list(running) + list(pending):
        if len(item) > 1 and str(item[1]) == prompt_id:
            return True
    return False


async def _post_prompt(port: int, api_prompt: dict, extra_data: dict | None) -> None:
    import aiohttp

    body: dict = {"prompt": api_prompt}
    if extra_data:
        body["extra_data"] = extra_data
    url = f"http://127.0.0.1:{port}/prompt"
    timeout = aiohttp.ClientTimeout(total=_POST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=body) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"/prompt returned {resp.status}: {text[:300]}")


async def _watch_and_continue(prompt_id: str, job_id: str, port: int) -> None:
    server = _server()
    queue = getattr(server, "prompt_queue", None)
    lost_since = None
    try:
        while True:
            history = queue.get_history(prompt_id=prompt_id) if queue else {}
            if history:
                entry = history[prompt_id]
                status = entry.get("status") or {}
                if status.get("status_str") == "success" and status.get("completed"):
                    item = entry["prompt"]
                    api_prompt = item[2]
                    extra_data = item[3] if len(item) > 3 and isinstance(item[3], dict) else None
                    if not _contains_sequential_output(api_prompt):
                        _log.info(
                            "[JR H3 Sequential Audio] server auto-continue skipped for %s: prompt %s "
                            "does not contain the sequential output node (wrapped execution, e.g. a "
                            "stage orchestrator) — re-posting it would hit the execution cache. "
                            "Re-run from the submitting client to continue.", job_id, prompt_id,
                        )
                        return
                    await _post_prompt(port, api_prompt, extra_data)
                    _log.info(
                        "[JR H3 Sequential Audio] server auto-continue: replayed prompt %s "
                        "(scheduled by %s)", prompt_id, job_id,
                    )
                else:
                    _log.info(
                        "[JR H3 Sequential Audio] server auto-continue stopped: prompt %s ended "
                        "with status %r", prompt_id, status.get("status_str"),
                    )
                return
            if queue is None or not _prompt_in_queue(queue, prompt_id):
                now = asyncio.get_event_loop().time()
                if lost_since is None:
                    lost_since = now
                elif now - lost_since > _LOST_PROMPT_GRACE_SECONDS:
                    _log.warning(
                        "[JR H3 Sequential Audio] server auto-continue stopped: prompt %s left "
                        "the queue without a history entry", prompt_id,
                    )
                    return
            else:
                lost_since = None
            await asyncio.sleep(_POLL_SECONDS)
    except Exception as error:
        _log.warning(
            "[JR H3 Sequential Audio] server auto-continue failed for %s: %s", job_id, error,
        )
    finally:
        with _lock:
            _pending_prompts.discard(prompt_id)


def schedule_server_continue(*, job_id: str, chunk_index: int, total_chunks: int) -> str:
    if int(chunk_index) + 1 >= int(total_chunks):
        return "Server auto-continue: last chunk, nothing to queue."
    server = _server()
    loop = getattr(server, "loop", None)
    if server is None or loop is None:
        return "Server auto-continue unavailable: no running PromptServer."
    prompt_id = getattr(server, "last_prompt_id", None)
    port = getattr(server, "port", None)
    if not prompt_id or not port:
        return "Server auto-continue unavailable: prompt id or port unknown."
    prompt_id = str(prompt_id)
    with _lock:
        if prompt_id in _pending_prompts:
            return (
                "Server auto-continue already armed for this prompt; "
                "one replay covers every sequential chain in it."
            )
        _pending_prompts.add(prompt_id)
    watcher = _watch_and_continue(prompt_id, job_id, int(port))
    try:
        asyncio.run_coroutine_threadsafe(watcher, loop)
    except RuntimeError as error:
        watcher.close()
        with _lock:
            _pending_prompts.discard(prompt_id)
        return f"Server auto-continue unavailable: could not schedule watcher ({error})."
    return f"Server auto-continue: next chunk queues after prompt {prompt_id} completes."
