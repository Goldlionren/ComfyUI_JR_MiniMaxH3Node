import concurrent.futures
import time

import pytest
from ComfyUI_JR_MiniMaxH3Node.nodes import prompt_review_pause as node_module
from ComfyUI_JR_MiniMaxH3Node.nodes.prompt_review_pause import JR_H3_PromptReviewPause
from ComfyUI_JR_MiniMaxH3Node.utils.prompt_review_state import (
    PROMPT_REVIEW_STORE,
    InvalidReviewText,
    ReviewAlreadyCompleted,
    ReviewNotFound,
)


class FakeSocket:
    closed = False


class FakeServer:
    def __init__(self, client_id="browser-client"):
        self.client_id = client_id
        self.sockets = {client_id: FakeSocket()} if client_id else {}
        self.events = []

    def send_sync(self, event, data, sid=None):
        self.events.append((event, data, sid))


@pytest.fixture(autouse=True)
def clean_store():
    PROMPT_REVIEW_STORE.clear()
    yield
    PROMPT_REVIEW_STORE.clear()


def wait_for_request(server, count=1):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        requests = [item for item in server.events if item[0] == "jr_h3_prompt_review_requested"]
        if len(requests) >= count:
            return requests
        time.sleep(0.01)
    raise AssertionError("review request event was not sent")


def test_input_output_and_cache_contract():
    inputs = JR_H3_PromptReviewPause.INPUT_TYPES()
    assert inputs["required"]["prompt"] == ("STRING", {"multiline": True, "forceInput": True})
    assert inputs["required"]["timeout_seconds"][1] == {
        "default": 3600, "min": 60, "max": 86400, "step": 1,
    }
    assert inputs["hidden"] == {"unique_id": "UNIQUE_ID"}
    assert JR_H3_PromptReviewPause.RETURN_TYPES == ("STRING",)
    assert JR_H3_PromptReviewPause.RETURN_NAMES == ("reviewed_prompt",)
    assert JR_H3_PromptReviewPause.CATEGORY == "JR MiniMax H3/Prompt"
    assert JR_H3_PromptReviewPause.IS_CHANGED() != JR_H3_PromptReviewPause.IS_CHANGED()


def test_waits_then_returns_exact_edited_unicode(monkeypatch):
    server = FakeServer()
    monkeypatch.setattr(node_module, "_prompt_server", lambda: server)
    incoming = "原始第一行\n保留 <Picture 1>，标点！"
    edited = "修改后第一行\n继续保留 <Picture 1>，标点！"
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(JR_H3_PromptReviewPause().review, incoming, 3600, "17")
        request = wait_for_request(server)[0]
        assert not future.done()
        assert request[1]["text"] == incoming
        assert request[1]["node_id"] == "17"
        assert request[2] == server.client_id
        PROMPT_REVIEW_STORE.submit(request[1]["review_id"], edited)
        assert future.result(timeout=3) == (edited,)
    assert PROMPT_REVIEW_STORE.pending_count() == 0


def test_same_input_always_creates_independent_reviews(monkeypatch):
    server = FakeServer()
    monkeypatch.setattr(node_module, "_prompt_server", lambda: server)
    review_ids = []
    for result_text in ("first approval", "second approval"):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(JR_H3_PromptReviewPause().review, "same input", 3600, "5")
            request = wait_for_request(server, len(review_ids) + 1)[-1]
            review_ids.append(request[1]["review_id"])
            PROMPT_REVIEW_STORE.submit(review_ids[-1], result_text)
            assert future.result(timeout=3) == (result_text,)
    assert review_ids[0] != review_ids[1]


def test_two_waiting_nodes_do_not_cross_results(monkeypatch):
    server = FakeServer()
    monkeypatch.setattr(node_module, "_prompt_server", lambda: server)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(JR_H3_PromptReviewPause().review, "A", 3600, "1")
        second = pool.submit(JR_H3_PromptReviewPause().review, "B", 3600, "2")
        requests = wait_for_request(server, 2)
        by_node = {item[1]["node_id"]: item[1]["review_id"] for item in requests}
        PROMPT_REVIEW_STORE.submit(by_node["2"], "edited B")
        PROMPT_REVIEW_STORE.submit(by_node["1"], "edited A")
        assert first.result(timeout=3) == ("edited A",)
        assert second.result(timeout=3) == ("edited B",)


def test_timeout_stops_and_cleans_pending_state(monkeypatch):
    server = FakeServer()
    monkeypatch.setattr(node_module, "_prompt_server", lambda: server)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(JR_H3_PromptReviewPause().review, "wait", 60, "3")
        review_id = wait_for_request(server)[0][1]["review_id"]
        PROMPT_REVIEW_STORE.get(review_id).deadline = time.monotonic() + 0.02
        with pytest.raises(RuntimeError, match="timed out"):
            future.result(timeout=3)
    assert PROMPT_REVIEW_STORE.pending_count() == 0
    assert any(item[1].get("status") == "Timed out" for item in server.events)


def test_interruption_cancels_and_cleans_pending_state(monkeypatch):
    server = FakeServer()
    monkeypatch.setattr(node_module, "_prompt_server", lambda: server)
    interrupted = False

    def check_interruption():
        nonlocal interrupted
        if interrupted:
            raise RuntimeError("processing interrupted")

    monkeypatch.setattr(node_module, "_check_interruption", check_interruption)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(JR_H3_PromptReviewPause().review, "wait", 3600, "4")
        wait_for_request(server)
        interrupted = True
        with pytest.raises(RuntimeError, match="processing interrupted"):
            future.result(timeout=3)
    assert PROMPT_REVIEW_STORE.pending_count() == 0
    assert any(item[1].get("status") == "Cancelled" for item in server.events)


def test_headless_mode_fails_immediately(monkeypatch):
    monkeypatch.setattr(node_module, "_prompt_server", lambda: FakeServer(client_id=None))
    with pytest.raises(RuntimeError, match="active ComfyUI browser client"):
        JR_H3_PromptReviewPause().review("prompt", 3600, "1")
    assert PROMPT_REVIEW_STORE.pending_count() == 0


def test_state_rejects_invalid_unknown_and_duplicate_without_leaking_text(capsys):
    pending = PROMPT_REVIEW_STORE.create("1", "client", "private incoming text", 60)
    with pytest.raises(InvalidReviewText):
        PROMPT_REVIEW_STORE.submit(pending.review_id, "  ")
    with pytest.raises(ReviewNotFound):
        PROMPT_REVIEW_STORE.submit("unknown", "valid")
    PROMPT_REVIEW_STORE.submit(pending.review_id, "private approved text")
    with pytest.raises(ReviewAlreadyCompleted):
        PROMPT_REVIEW_STORE.submit(pending.review_id, "second")
    captured = capsys.readouterr()
    assert "private incoming text" not in captured.out + captured.err
    assert "private approved text" not in captured.out + captured.err


def test_frontend_contract_exists_and_does_not_log_prompt():
    source = node_module.__file__.replace("nodes\\prompt_review_pause.py", "js\\prompt_review_pause.js")
    text = open(source, encoding="utf-8").read()
    for marker in (
        "jr_h3_prompt_review_requested", "/jr_h3/prompt-review/continue", "review_text",
        "Next / Continue", "Waiting for review", "Timed out", "Cancelled", "serialize: false",
        "api.clientId", "/jr_h3/prompt-review/pending",
    ):
        assert marker in text
    assert "console.log" not in text
