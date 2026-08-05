"""Interactive Prompt Review & Continue node."""

from __future__ import annotations

import time

from ..utils.prompt_review_state import PROMPT_REVIEW_STORE

_REQUEST_EVENT = "jr_h3_prompt_review_requested"
_STATUS_EVENT = "jr_h3_prompt_review_status"
_NO_BROWSER_ERROR = (
    "JR MiniMax H3 Prompt Review & Continue requires an active ComfyUI browser client.\n"
    "Interactive review is not available in unattended API mode."
)
_TIMEOUT_ERROR = "JR MiniMax H3 Prompt Review & Continue timed out while waiting for approval."


def _prompt_server():
    try:
        from server import PromptServer
    except ImportError as error:
        raise RuntimeError(_NO_BROWSER_ERROR) from error
    return getattr(PromptServer, "instance", None)


def _active_client(server):
    if server is None:
        return None
    client_id = getattr(server, "client_id", None)
    socket = getattr(server, "sockets", {}).get(client_id) if client_id else None
    if socket is None or getattr(socket, "closed", False):
        return None
    return str(client_id)


def _check_interruption() -> None:
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def _send_status(server, client_id: str, pending, status: str) -> None:
    server.send_sync(
        _STATUS_EVENT,
        {"review_id": pending.review_id, "node_id": pending.node_id, "status": status},
        client_id,
    )


class JR_H3_PromptReviewPause:
    CATEGORY = "JR MiniMax H3/Prompt"
    FUNCTION = "review"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("reviewed_prompt",)
    DESCRIPTION = "Pauses execution until the active ComfyUI browser approves an editable prompt."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "timeout_seconds": ("INT", {"default": 3600, "min": 60, "max": 86400, "step": 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def review(self, prompt, timeout_seconds=3600, unique_id=None):
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a STRING.")
        server = _prompt_server()
        client_id = _active_client(server)
        if client_id is None:
            raise RuntimeError(_NO_BROWSER_ERROR)
        timeout = min(86400, max(60, int(timeout_seconds)))
        pending = PROMPT_REVIEW_STORE.create(str(unique_id), client_id, prompt, timeout)
        server.send_sync(
            _REQUEST_EVENT,
            {
                "review_id": pending.review_id,
                "node_id": pending.node_id,
                "text": prompt,
                "timeout_seconds": timeout,
            },
            client_id,
        )
        terminal_status = "Cancelled"
        try:
            while True:
                _check_interruption()
                remaining = pending.deadline - time.monotonic()
                if remaining <= 0:
                    PROMPT_REVIEW_STORE.mark_terminal(pending.review_id, "timed_out")
                    terminal_status = "Timed out"
                    _send_status(server, client_id, pending, terminal_status)
                    raise RuntimeError(_TIMEOUT_ERROR)
                if pending.event.wait(timeout=min(0.25, remaining)):
                    break
            if pending.status != "approved" or pending.result_text is None:
                raise RuntimeError("JR MiniMax H3 Prompt Review & Continue was cancelled.")
            terminal_status = "Approved"
            _send_status(server, client_id, pending, terminal_status)
            return (pending.result_text,)
        except BaseException:
            if pending.status == "pending":
                PROMPT_REVIEW_STORE.mark_terminal(pending.review_id, "cancelled")
                _send_status(server, client_id, pending, terminal_status)
            raise
        finally:
            PROMPT_REVIEW_STORE.cleanup(pending.review_id)
