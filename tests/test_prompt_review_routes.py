import asyncio
import json
from types import SimpleNamespace

from ComfyUI_JR_MiniMaxH3Node.server.prompt_review_routes import (
    continue_prompt_review,
    pending_prompt_reviews,
)
from ComfyUI_JR_MiniMaxH3Node.utils.prompt_review_state import PromptReviewStore


class FakeRequest:
    def __init__(self, payload=None, query=None, json_error=None):
        self.payload = payload
        self.rel_url = SimpleNamespace(query=query or {})
        self.json_error = json_error

    async def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


def call(handler, request, store):
    return asyncio.run(handler(request, store))


def body(response):
    return json.loads(response.body.decode("utf-8"))


def test_continue_route_approves_unicode_multiline_text():
    store = PromptReviewStore()
    pending = store.create("7", "client", "original", 60)
    edited = "第一行\n第二行，保留 <Picture 1>。"
    response = call(continue_prompt_review, FakeRequest({"review_id": pending.review_id, "text": edited}), store)
    assert response.status == 200 and body(response) == {"success": True}
    assert pending.event.is_set() and pending.result_text == edited


def test_continue_route_validation_statuses():
    store = PromptReviewStore()
    invalid_json = call(continue_prompt_review, FakeRequest(json_error=ValueError("bad")), store)
    missing_id = call(continue_prompt_review, FakeRequest({"text": "x"}), store)
    missing_text = call(continue_prompt_review, FakeRequest({"review_id": "x"}), store)
    empty = store.create("1", "client", "x", 60)
    empty_response = call(continue_prompt_review, FakeRequest({"review_id": empty.review_id, "text": " \n"}), store)
    unknown = call(continue_prompt_review, FakeRequest({"review_id": "unknown", "text": "x"}), store)
    oversized = call(
        continue_prompt_review,
        FakeRequest({"review_id": empty.review_id, "text": "x" * 100_001}),
        store,
    )
    assert [invalid_json.status, missing_id.status, missing_text.status] == [400, 400, 400]
    assert empty_response.status == 400
    assert unknown.status == 404
    assert oversized.status == 413


def test_duplicate_approval_returns_conflict_even_after_cleanup():
    store = PromptReviewStore()
    pending = store.create("1", "client", "x", 60)
    request = FakeRequest({"review_id": pending.review_id, "text": "approved"})
    assert call(continue_prompt_review, request, store).status == 200
    assert call(continue_prompt_review, request, store).status == 409
    store.cleanup(pending.review_id)
    assert call(continue_prompt_review, request, store).status == 409


def test_pending_route_is_scoped_to_client_id_and_preserves_text():
    store = PromptReviewStore()
    visible = store.create("1", "client-a", "多行\n<Picture 2>", 60)
    store.create("2", "client-b", "secret-other-client", 60)
    response = call(pending_prompt_reviews, FakeRequest(query={"client_id": "client-a"}), store)
    payload = body(response)
    assert response.status == 200
    assert len(payload["reviews"]) == 1
    assert payload["reviews"][0]["review_id"] == visible.review_id
    assert payload["reviews"][0]["text"] == "多行\n<Picture 2>"
    assert call(pending_prompt_reviews, FakeRequest(), store).status == 400
