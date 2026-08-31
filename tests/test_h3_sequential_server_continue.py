import asyncio
import sys
import types
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1].name


def _module():
    return __import__(
        f"{PACKAGE}.utils.h3_sequential_server_continue",
        fromlist=["schedule_server_continue"],
    )


def _install_fake_server(monkeypatch, *, prompt_id="p-1", port=8188, loop=None):
    mod = _module()
    server = types.SimpleNamespace(
        loop=loop,
        last_prompt_id=prompt_id,
        port=port,
        prompt_queue=None,
    )
    monkeypatch.setattr(mod, "_server", lambda: server)
    return mod, server


@pytest.fixture(autouse=True)
def _clear_prompt_states():
    mod = _module()
    with mod._lock:
        mod._prompt_states.clear()
    yield
    with mod._lock:
        mod._prompt_states.clear()


def _run(coroutine):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


def test_last_chunk_schedules_nothing(monkeypatch):
    loop = asyncio.new_event_loop()
    try:
        mod, _ = _install_fake_server(monkeypatch, loop=loop)
        scheduled = []
        monkeypatch.setattr(
            mod.asyncio,
            "run_coroutine_threadsafe",
            lambda coro, running_loop: (scheduled.append(coro), coro.close()),
        )
        note = mod.schedule_server_continue(job_id="j/r1", chunk_index=1, total_chunks=2)
        assert "last chunk" in note
        assert len(scheduled) == 1
        assert mod._prompt_states["p-1"].needs_replay is False
    finally:
        loop.close()

    class FakeQueue:
        def get_history(self, prompt_id=None):
            return {
                prompt_id: {
                    "prompt": (
                        0,
                        prompt_id,
                        {"1": {"class_type": "JR_H3_SequentialVideoOutput", "inputs": {}}},
                        {},
                        [],
                    ),
                    "status": {"status_str": "success", "completed": True},
                }
            }

    posts = []

    async def fake_post(port, api_prompt, extra_data):
        posts.append((port, api_prompt, extra_data))

    monkeypatch.setattr(mod, "_post_prompt", fake_post)
    monkeypatch.setattr(mod, "_server", lambda: types.SimpleNamespace(prompt_queue=FakeQueue()))
    _run(mod._watch_and_continue("p-1", "j/r1", 8188))
    assert posts == []
    assert "p-1" not in mod._prompt_states


def test_no_server_reports_unavailable(monkeypatch):
    mod = _module()
    monkeypatch.setattr(mod, "_server", lambda: None)
    note = mod.schedule_server_continue(job_id="j/r1", chunk_index=0, total_chunks=2)
    assert "unavailable" in note


def test_missing_prompt_id_reports_unavailable(monkeypatch):
    loop = asyncio.new_event_loop()
    try:
        mod, server = _install_fake_server(monkeypatch, prompt_id=None, loop=loop)
        note = mod.schedule_server_continue(job_id="j/r1", chunk_index=0, total_chunks=2)
        assert "unavailable" in note
    finally:
        loop.close()


def test_same_job_duplicate_schedule_arms_one_watcher(monkeypatch):
    loop = asyncio.new_event_loop()
    try:
        mod, server = _install_fake_server(monkeypatch, loop=loop)
        scheduled = []
        monkeypatch.setattr(
            mod.asyncio, "run_coroutine_threadsafe",
            lambda coro, running_loop: (scheduled.append(coro), coro.close()),
        )
        first = mod.schedule_server_continue(job_id="job-a/r1", chunk_index=0, total_chunks=3)
        second = mod.schedule_server_continue(job_id="job-a/r1", chunk_index=0, total_chunks=3)
        assert "queues after prompt p-1" in first
        assert "already armed for job-a/r1" in second
        assert len(scheduled) == 1
    finally:
        loop.close()

    class FakeQueue:
        def get_history(self, prompt_id=None):
            return {
                prompt_id: {
                    "prompt": (
                        0,
                        prompt_id,
                        {"1": {"class_type": "JR_H3_SequentialVideoOutput", "inputs": {}}},
                        {},
                        [],
                    ),
                    "status": {"status_str": "success", "completed": True},
                }
            }

    posts = []

    async def fake_post(port, api_prompt, extra_data):
        posts.append((port, api_prompt, extra_data))

    monkeypatch.setattr(mod, "_post_prompt", fake_post)
    monkeypatch.setattr(mod, "_server", lambda: types.SimpleNamespace(prompt_queue=FakeQueue()))
    _run(mod._watch_and_continue("p-1", "job-a/r1", 8188))
    assert len(posts) == 1


def test_closed_loop_clears_pending(monkeypatch):
    loop = asyncio.new_event_loop()
    loop.close()
    mod, server = _install_fake_server(monkeypatch, loop=loop)
    note = mod.schedule_server_continue(job_id="j/r1", chunk_index=0, total_chunks=2)
    assert "unavailable" in note
    assert "p-1" not in mod._prompt_states


def test_watcher_reposts_only_on_success(monkeypatch):
    mod = _module()

    class FakeQueue:
        def __init__(self, status_str, class_type="JR_H3_SequentialVideoOutput", completed=True):
            self.entry = {
                "prompt": (
                    0, "p-1",
                    {"1": {"class_type": class_type, "inputs": {}}},
                    {"extra_pnginfo": {"workflow": {}}, "client_id": "c-1"},
                    [],
                ),
                "status": {"status_str": status_str, "completed": completed},
            }

        def get_history(self, prompt_id=None):
            return {prompt_id: self.entry}

        def get_current_queue(self):
            return [], []

    posts = []

    async def fake_post(port, api_prompt, extra_data):
        posts.append((port, api_prompt, extra_data))

    monkeypatch.setattr(mod, "_post_prompt", fake_post)

    cases = (
        ("success", "JR_H3_SequentialVideoOutput", 1),
        ("error", "JR_H3_SequentialVideoOutput", 0),
        ("success", "ComfyTV.VideoStage", 0),
    )
    for status, class_type, expected_posts in cases:
        posts.clear()
        server = types.SimpleNamespace(prompt_queue=FakeQueue(status, class_type))
        monkeypatch.setattr(mod, "_server", lambda s=server: s)
        mod._prompt_states["p-1"] = mod._PromptContinuationState(
            first_job_id="j/r1", needs_replay=True
        )
        _run(mod._watch_and_continue("p-1", "j/r1", 8188))
        assert len(posts) == expected_posts
        assert "p-1" not in mod._prompt_states
    server = types.SimpleNamespace(prompt_queue=FakeQueue("success"))
    monkeypatch.setattr(mod, "_server", lambda s=server: s)
    mod._prompt_states["p-1"] = mod._PromptContinuationState(
        first_job_id="j/r1", needs_replay=True
    )
    _run(mod._watch_and_continue("p-1", "j/r1", 8188))
    assert posts[-1][2] == {"extra_pnginfo": {"workflow": {}}, "client_id": "c-1"}


def test_single_job_success_replays_exactly_once(monkeypatch):
    mod = _module()

    class FakeQueue:
        def get_history(self, prompt_id=None):
            return {
                prompt_id: {
                    "prompt": (
                        0,
                        prompt_id,
                        {"1": {"class_type": "JR_H3_SequentialVideoOutput", "inputs": {}}},
                        {"client_id": "c-1"},
                        [],
                    ),
                    "status": {"status_str": "success", "completed": True},
                }
            }

    posts = []

    async def fake_post(port, api_prompt, extra_data):
        posts.append((port, api_prompt, extra_data))

    monkeypatch.setattr(mod, "_post_prompt", fake_post)
    monkeypatch.setattr(mod, "_server", lambda: types.SimpleNamespace(prompt_queue=FakeQueue()))
    mod._prompt_states["p-1"] = mod._PromptContinuationState(
        first_job_id="job-a/r1", needs_replay=True
    )
    _run(mod._watch_and_continue("p-1", "job-a/r1", 8188))
    assert len(posts) == 1
    assert posts[0][2] == {"client_id": "c-1"}
    assert "p-1" not in mod._prompt_states


def test_different_jobs_block_armed_replay_and_cleanup(monkeypatch):
    loop = asyncio.new_event_loop()
    try:
        mod, server = _install_fake_server(monkeypatch, loop=loop)
        scheduled = []
        monkeypatch.setattr(
            mod.asyncio,
            "run_coroutine_threadsafe",
            lambda coro, running_loop: (scheduled.append(coro), coro.close()),
        )
        first = mod.schedule_server_continue(job_id="job-a/r1", chunk_index=0, total_chunks=3)
        second = mod.schedule_server_continue(job_id="job-b/r1", chunk_index=0, total_chunks=5)
        assert "queues after prompt p-1" in first
        assert "multiple server-auto Sequential chains" in second
        assert len(scheduled) == 1
        assert mod._prompt_states["p-1"].blocked is True
    finally:
        loop.close()

    class FakeQueue:
        def get_history(self, prompt_id=None):
            return {
                prompt_id: {
                    "prompt": (
                        0,
                        prompt_id,
                        {"1": {"class_type": "JR_H3_SequentialVideoOutput", "inputs": {}}},
                        {},
                        [],
                    ),
                    "status": {"status_str": "success", "completed": True},
                }
            }

    posts = []

    async def fake_post(port, api_prompt, extra_data):
        posts.append((port, api_prompt, extra_data))

    monkeypatch.setattr(mod, "_post_prompt", fake_post)
    monkeypatch.setattr(mod, "_server", lambda: types.SimpleNamespace(prompt_queue=FakeQueue()))
    _run(mod._watch_and_continue("p-1", "job-a/r1", 8188))
    assert posts == []
    assert "p-1" not in mod._prompt_states


def test_different_total_chunks_fail_closed_before_replay(monkeypatch):
    loop = asyncio.new_event_loop()
    try:
        mod, server = _install_fake_server(monkeypatch, loop=loop)
        scheduled = []
        monkeypatch.setattr(
            mod.asyncio,
            "run_coroutine_threadsafe",
            lambda coro, running_loop: (scheduled.append(coro), coro.close()),
        )
        first = mod.schedule_server_continue(job_id="short/r1", chunk_index=0, total_chunks=1)
        note = mod.schedule_server_continue(job_id="long/r1", chunk_index=0, total_chunks=4)
        assert "last chunk" in first
        assert "paused" in note
        assert mod._prompt_states["p-1"].blocked is True
        assert len(scheduled) == 1
    finally:
        loop.close()


def test_blocked_prompt_does_not_affect_new_prompt(monkeypatch):
    loop = asyncio.new_event_loop()
    try:
        mod, server = _install_fake_server(monkeypatch, loop=loop)
        scheduled = []
        monkeypatch.setattr(
            mod.asyncio,
            "run_coroutine_threadsafe",
            lambda coro, running_loop: (scheduled.append(coro), coro.close()),
        )
        mod.schedule_server_continue(job_id="job-a/r1", chunk_index=0, total_chunks=3)
        mod.schedule_server_continue(job_id="job-b/r1", chunk_index=0, total_chunks=5)
        assert mod._prompt_states["p-1"].blocked is True
        server.prompt_queue = types.SimpleNamespace(
            get_history=lambda prompt_id=None: {
                prompt_id: {
                    "prompt": (
                        0,
                        prompt_id,
                        {"1": {"class_type": "JR_H3_SequentialVideoOutput", "inputs": {}}},
                        {},
                        [],
                    ),
                    "status": {"status_str": "success", "completed": True},
                }
            }
        )
        _run(mod._watch_and_continue("p-1", "job-a/r1", 8188))
        assert "p-1" not in mod._prompt_states
        server.last_prompt_id = "p-2"
        note = mod.schedule_server_continue(job_id="job-c/r1", chunk_index=0, total_chunks=2)
        assert "queues after prompt p-2" in note
        assert mod._prompt_states["p-2"].blocked is False
        assert len(scheduled) == 2
    finally:
        loop.close()


def test_final_chunk_output_still_registers_topology(monkeypatch):
    node_module = __import__(
        f"{PACKAGE}.nodes.h3_sequential_audio",
        fromlist=["JR_H3_SequentialVideoOutput"],
    )
    continue_module = _module()
    calls = []
    monkeypatch.setattr(
        node_module,
        "commit_decoded_chunk",
        lambda **kwargs: ("final.mp4", "Committed final chunk.", False),
    )
    monkeypatch.setattr(
        continue_module,
        "schedule_server_continue",
        lambda **kwargs: calls.append(kwargs) or "Server auto-continue: last chunk, nothing to queue.",
    )
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(send_sync=lambda *args, **kwargs: None, client_id=None)
    )
    monkeypatch.setitem(sys.modules, "server", fake_server)
    context = types.SimpleNamespace(job_id="short/r1", chunk_index=0, total_chunks=1)
    _filename, status = node_module.JR_H3_SequentialVideoOutput().commit(
        images=object(),
        chunk_context=context,
        auto_queue_next=True,
        aggressive_memory_cleanup=False,
        server_auto_continue=True,
        unique_id="node-1",
    )
    assert calls == [{"job_id": "short/r1", "chunk_index": 0, "total_chunks": 1}]
    assert "last chunk" in status


def test_output_node_contract():
    nodes = __import__(f"{PACKAGE}.nodes.h3_sequential_audio", fromlist=["JR_H3_SequentialVideoOutput"])
    inputs = nodes.JR_H3_SequentialVideoOutput.INPUT_TYPES()
    assert "server_auto_continue" in inputs["optional"]
    assert inputs["optional"]["server_auto_continue"][1]["default"] is False
