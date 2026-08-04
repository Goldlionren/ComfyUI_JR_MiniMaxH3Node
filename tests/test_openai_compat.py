import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from ComfyUI_JR_MiniMaxH3Node.utils.openai_compat import (
    OpenAICompatError,
    discover_model,
    normalize_api_urls,
    normalize_picture_markers,
    parse_chat_content,
    request_chat,
)


@pytest.mark.parametrize("base", [
    "http://127.0.0.1:10000", "http://127.0.0.1:10000/", "http://127.0.0.1:10000/v1",
    "http://127.0.0.1:10000/v1/", "http://127.0.0.1:10000/v1/chat/completions",
])
def test_url_variants(base):
    assert normalize_api_urls(base) == (
        "http://127.0.0.1:10000/v1/models", "http://127.0.0.1:10000/v1/chat/completions")


def test_url_preserves_path_prefix():
    assert normalize_api_urls("https://host.example/llama") == (
        "https://host.example/llama/v1/models", "https://host.example/llama/v1/chat/completions")


@pytest.mark.parametrize("bad", ["", "127.0.0.1:10000", "ftp://host/x"])
def test_invalid_url(bad):
    with pytest.raises(ValueError): normalize_api_urls(bad)


def test_content_string_list_and_cleanup():
    assert parse_chat_content({"choices": [{"message": {"content": "<think>x</think>\nFinal answer: <image 2> shot"}}]}) == "<Picture 2> shot"
    assert parse_chat_content({"choices": [{"message": {"content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}}]}) == "AB"


@pytest.mark.parametrize("value", ["<image1>", "<image 1>", "<picture1>", "<picture 1>", "[picture 1]"])
def test_picture_normalization(value):
    assert normalize_picture_markers(value) == "<Picture 1>"


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"message": {}}]}, {"choices": [{"message": {"content": []}}]}])
def test_bad_or_empty_content(body):
    with pytest.raises(OpenAICompatError): parse_chat_content(body)


class _Handler(BaseHTTPRequestHandler):
    requests = []
    mode = "ok"

    def log_message(self, *_): pass

    def _json(self, status, data):
        raw = json.dumps(data).encode(); self.send_response(status); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        self.__class__.requests.append(("GET", self.path, dict(self.headers), None))
        if self.__class__.mode == "nonjson": self.send_response(200); self.end_headers(); self.wfile.write(b"nope"); return
        self._json(200, {"data": [{"id": "local-model"}]})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        self.__class__.requests.append(("POST", self.path, dict(self.headers), payload))
        if self.__class__.mode == "retry" and "reasoning_effort" in payload:
            self._json(400, {"error": "unknown field"}); return
        if self.__class__.mode == "401": self._json(401, {"error": "denied"}); return
        if self.__class__.mode == "slow": time.sleep(0.2)
        self._json(200, {"choices": [{"message": {"content": "final"}}]})


@pytest.fixture
def server():
    _Handler.requests = []; _Handler.mode = "ok"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown(); thread.join()


def test_discovery_and_authorization(server):
    model = discover_model(server + "/v1/models", 2, "secret")
    assert model == "local-model"
    assert _Handler.requests[0][2]["Authorization"] == "Bearer secret"


def test_no_authorization_when_key_empty(server):
    discover_model(server + "/v1/models", 2)
    assert "Authorization" not in _Handler.requests[0][2]


def test_reasoning_400_retries_once_without_extensions(server):
    _Handler.mode = "retry"
    payload = {"reasoning_effort": "none", "chat_template_kwargs": {}, "messages": []}
    request_chat(server + "/v1/chat/completions", payload, 2, retry_reasoning_400=True)
    posts = [x for x in _Handler.requests if x[0] == "POST"]
    assert len(posts) == 2 and "reasoning_effort" in posts[0][3] and "reasoning_effort" not in posts[1][3]


def test_http_error_is_bounded_and_has_no_authorization(server):
    _Handler.mode = "401"
    with pytest.raises(OpenAICompatError) as caught:
        request_chat(server + "/v1/chat/completions", {"messages": []}, 2, "super-secret")
    assert "super-secret" not in str(caught.value) and caught.value.status == 401


def test_non_json(server):
    _Handler.mode = "nonjson"
    with pytest.raises(OpenAICompatError, match="Invalid JSON"): discover_model(server + "/v1/models", 2)


def test_timeout(server):
    _Handler.mode = "slow"
    with pytest.raises(OpenAICompatError): request_chat(server + "/v1/chat/completions", {}, 0.01)
