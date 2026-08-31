import asyncio
import types
from pathlib import Path

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


def test_last_chunk_schedules_nothing(monkeypatch):
    mod, _ = _install_fake_server(monkeypatch)
    note = mod.schedule_server_continue(job_id="j/r1", chunk_index=1, total_chunks=2)
    assert "last chunk" in note


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


def test_one_replay_per_source_prompt(monkeypatch):
    loop = asyncio.new_event_loop()
    try:
        mod, server = _install_fake_server(monkeypatch, loop=loop)
        scheduled = []
        monkeypatch.setattr(
            mod.asyncio, "run_coroutine_threadsafe",
            lambda coro, running_loop: (scheduled.append(coro), coro.close()),
        )
        first = mod.schedule_server_continue(job_id="job-a/r1", chunk_index=0, total_chunks=3)
        second = mod.schedule_server_continue(job_id="job-b/r1", chunk_index=0, total_chunks=5)
        assert "queues after prompt p-1" in first
        assert "already armed" in second
        assert len(scheduled) == 1
        mod._pending_prompts.discard("p-1")
    finally:
        loop.close()


def test_closed_loop_clears_pending(monkeypatch):
    loop = asyncio.new_event_loop()
    loop.close()
    mod, server = _install_fake_server(monkeypatch, loop=loop)
    note = mod.schedule_server_continue(job_id="j/r1", chunk_index=0, total_chunks=2)
    assert "unavailable" in note
    assert "p-1" not in mod._pending_prompts


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
        mod._pending_prompts.add("p-1")
        asyncio.new_event_loop().run_until_complete(
            mod._watch_and_continue("p-1", "j/r1", 8188)
        )
        assert len(posts) == expected_posts
        assert "p-1" not in mod._pending_prompts
    server = types.SimpleNamespace(prompt_queue=FakeQueue("success"))
    monkeypatch.setattr(mod, "_server", lambda s=server: s)
    mod._pending_prompts.add("p-1")
    asyncio.new_event_loop().run_until_complete(
        mod._watch_and_continue("p-1", "j/r1", 8188)
    )
    assert posts[-1][2] == {"extra_pnginfo": {"workflow": {}}, "client_id": "c-1"}


def test_output_node_contract():
    nodes = __import__(f"{PACKAGE}.nodes.h3_sequential_audio", fromlist=["JR_H3_SequentialVideoOutput"])
    inputs = nodes.JR_H3_SequentialVideoOutput.INPUT_TYPES()
    assert "server_auto_continue" in inputs["optional"]
    assert inputs["optional"]["server_auto_continue"][1]["default"] is False
