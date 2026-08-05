"""HTTP routes for completing and recovering interactive prompt reviews."""

from __future__ import annotations

from ..utils.prompt_review_state import (
    MAX_REVIEW_TEXT_LENGTH,
    PROMPT_REVIEW_STORE,
    InvalidReviewText,
    ReviewAlreadyCompleted,
    ReviewNotFound,
    ReviewTextTooLong,
)

try:
    from aiohttp import web
except ImportError:  # Import tests and non-Comfy tooling may not provide aiohttp.
    web = None


def _json_error(code: str, status: int):
    return web.json_response({"success": False, "error": code}, status=status)


async def continue_prompt_review(request, store=None):
    state = store or PROMPT_REVIEW_STORE
    try:
        payload = await request.json()
    except Exception:
        return _json_error("invalid_json", 400)
    if not isinstance(payload, dict):
        return _json_error("invalid_json", 400)
    review_id = payload.get("review_id")
    if not isinstance(review_id, str) or not review_id:
        return _json_error("missing_review_id", 400)
    if "text" not in payload or not isinstance(payload["text"], str):
        return _json_error("missing_text", 400)
    text = payload["text"]
    if len(text) > MAX_REVIEW_TEXT_LENGTH:
        return _json_error("text_too_long", 413)
    try:
        state.submit(review_id, text)
    except InvalidReviewText:
        return _json_error("empty_text", 400)
    except ReviewTextTooLong:
        return _json_error("text_too_long", 413)
    except ReviewNotFound:
        return _json_error("review_not_found", 404)
    except ReviewAlreadyCompleted:
        return _json_error("review_already_completed", 409)
    except Exception:
        return _json_error("internal_error", 500)
    return web.json_response({"success": True})


async def pending_prompt_reviews(request, store=None):
    state = store or PROMPT_REVIEW_STORE
    client_id = request.rel_url.query.get("client_id", "")
    if not client_id:
        return _json_error("missing_client_id", 400)
    return web.json_response({"success": True, "reviews": state.pending_for_client(client_id)})


def register_prompt_review_routes() -> bool:
    if web is None:
        return False
    try:
        from server import PromptServer
    except ImportError:
        return False
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return False
    guard = "_jr_h3_prompt_review_routes_registered"
    if getattr(instance, guard, False):
        return True
    setattr(instance, guard, True)
    instance.routes.post("/jr_h3/prompt-review/continue")(continue_prompt_review)
    instance.routes.get("/jr_h3/prompt-review/pending")(pending_prompt_reviews)
    return True


ROUTES_REGISTERED = register_prompt_review_routes()
