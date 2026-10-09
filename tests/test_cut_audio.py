import asyncio
import importlib
import json
import shutil

import pytest
import torch


@pytest.fixture
def audio_env(package_name, tmp_path, monkeypatch):
    import folder_paths

    mod = importlib.import_module(f"{package_name}.utils.cut_audio")
    source, temp = tmp_path / "input", tmp_path / "temp"
    source.mkdir()
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(source))
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(temp))
    rate = 8000
    t = torch.arange(rate * 3) / rate
    wave = torch.stack([.65 * torch.sin(t * 440 * 2 * torch.pi), -.65 * torch.sin(t * 440 * 2 * torch.pi)])
    token = mod.write_wav(wave, rate, "a" * 64)
    shutil.copyfile(mod.media_path(token), source / "音乐.wav")
    return mod, source, wave, rate


def test_exact_pcm_cut_and_download(audio_env, package_name):
    mod, _, wave, rate = audio_env
    info = mod.analyze_audio("音乐.wav")
    assert info["channels"] == 2 and info["duration"] == 3
    assert any(lo < -.5 and hi > .5 for lo, hi in info["peaks"])
    result = mod.commit_cut("音乐.wav", .123456, 2.54321, info["source_id"])
    selected = json.loads(result["selection"])
    node = importlib.import_module(f"{package_name}.nodes.cut_audio").JR_CutAudio
    assert node.VALIDATE_INPUTS("音乐.wav", result["selection"]) is True
    audio, duration, status = node().cut("音乐.wav", result["selection"])
    expected = wave[:, selected["start_sample"]:selected["end_sample"]]
    torch.testing.assert_close(audio["waveform"][0], expected, rtol=0, atol=0)
    download, download_rate = mod.decode_audio(mod.media_path(result["media_token"]))
    torch.testing.assert_close(download, expected, rtol=0, atol=0)
    assert audio["sample_rate"] == download_rate == rate
    assert duration == result["duration"] == expected.shape[-1] / rate
    assert "unchanged PCM gain" in status
    assert mod.commit_cut("音乐.wav", selected["start_sample"] / rate, selected["end_sample"] / rate,
                          info["source_id"])["selection"] == result["selection"]


@pytest.mark.parametrize("ext,codec", [("mp3", "libmp3lame"), ("flac", "flac"), ("ogg", "libvorbis"), ("m4a", "aac"), ("aiff", "pcm_s16be")])
def test_mainstream_formats(audio_env, ext, codec):
    import av

    mod, source, wave, rate = audio_env
    if codec not in av.codecs_available:
        pytest.skip(f"Test fixture encoder {codec} unavailable in installed PyAV; decoder support is independent")
    with av.open(str(source / f"test.{ext}"), "w") as container:
        stream = container.add_stream(codec, rate=rate)
        stream.layout = "stereo"
        frame = av.AudioFrame.from_ndarray(wave.numpy(), format="fltp", layout="stereo")
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    info = mod.analyze_audio(f"test.{ext}")
    assert info["channels"] == 2 and info["sample_rate"] == rate and info["duration"] > 2.9
    locked = mod.commit_cut(f"test.{ext}", .5, 2.0, info["source_id"])
    output = mod.selection_audio(f"test.{ext}", locked["selection"])[0]
    assert output["waveform"].shape == (1, 2, 12000)


@pytest.mark.parametrize("name", ["../music.wav", "C:/test.wav", "a\\b.wav", "https://x/a.wav", "/foo.wav", "bad.exe", "", None])
def test_reject_paths(audio_env, name):
    mod, *_ = audio_env
    with pytest.raises(mod.CutAudioError):
        mod.resolve_audio(name)


@pytest.mark.parametrize("start,end", [(0, 0), (-1, 1), (1, .5), (0, 4), (float("nan"), 1), (0, float("inf")), (True, 2), ("0", 1), (0, .00001)])
def test_invalid_range(audio_env, start, end):
    mod, *_ = audio_env
    with pytest.raises(mod.CutAudioError):
        mod.sample_range(start, end, 8000, 24000)


def test_changed_source_and_invalid_selection(audio_env, package_name):
    mod, source, _, _ = audio_env
    info = mod.analyze_audio("音乐.wav")
    result = mod.commit_cut("音乐.wav", 0, 1, info["source_id"])
    node = importlib.import_module(f"{package_name}.nodes.cut_audio").JR_CutAudio
    before = node.IS_CHANGED("音乐.wav", result["selection"])
    with (source / "音乐.wav").open("ab") as handle:
        handle.write(b"changed")
    assert before != node.IS_CHANGED("音乐.wav", result["selection"])
    with pytest.raises(mod.CutAudioError, match="changed"):
        mod.selection_audio("音乐.wav", result["selection"])
    assert node.VALIDATE_INPUTS("音乐.wav", "") is not True
    selected = json.loads(result["selection"])
    for field, value in [("start_sample", True), ("end_sample", 999999), ("audio", "other.wav"), ("version", True)]:
        invalid = dict(selected, **{field: value})
        with pytest.raises(mod.CutAudioError):
            mod.parse_selection(json.dumps(invalid), "音乐.wav")


def test_decoder_budget_and_corruption(audio_env, monkeypatch):
    mod, source, *_ = audio_env
    monkeypatch.setattr(mod, "MAX_PCM_BYTES", 100)
    with pytest.raises(mod.CutAudioError, match="exceeds"):
        mod.analyze_audio("音乐.wav")
    (source / "bad.mp3").write_bytes(b"not audio")
    with pytest.raises(mod.CutAudioError, match="Cannot decode"):
        mod.analyze_audio("bad.mp3")


def test_http_routes(audio_env, package_name):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    mod, source, *_ = audio_env
    routes = importlib.import_module(f"{package_name}.server.cut_audio_routes")

    async def run():
        app = web.Application()
        app.router.add_get("/files", routes.audio_files)
        app.router.add_post("/info", routes.audio_info)
        app.router.add_post("/cut", routes.audio_cut)
        app.router.add_get("/media/{token}.wav", routes.audio_media)
        app.router.add_get("/original", routes.audio_original)
        async with TestClient(TestServer(app)) as client:
            assert (await (await client.get("/files")).json())["files"] == ["音乐.wav"]
            response = await client.post("/info", json={"audio": "音乐.wav"})
            assert response.status == 200
            info = await response.json()
            response = await client.post("/cut", json={"audio": "音乐.wav", "start": .3, "end": 1.3, "source_id": info["source_id"]})
            assert response.status == 200
            cut = await response.json()
            response = await client.get(f"/media/{cut['media_token']}.wav?download=1")
            assert response.status == 200 and "attachment" in response.headers["Content-Disposition"]
            assert await response.read() == mod.media_path(cut["media_token"]).read_bytes()
            response = await client.get(f"/media/{cut['media_token']}.wav", headers={"Range": "bytes=0-55"})
            assert response.status == 206 and len(await response.read()) == 56
            response = await client.get("/original", params={"audio": "音乐.wav"})
            assert await response.read() == (source / "音乐.wav").read_bytes()
            assert (await client.post("/info", data="[]")).status == 400
            assert (await client.post("/info", data="x" * 4097)).status == 413
            assert (await client.post("/info", json={"audio": "../no.wav"})).status == 400
            assert (await client.get("/media/not-a-token.wav")).status == 400
    asyncio.run(run())


def test_cancelled_request_keeps_worker_slot(package_name, monkeypatch):
    from aiohttp import web

    routes = importlib.import_module(f"{package_name}.server.cut_audio_routes")

    async def run():
        started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def worker(*args):
            started.set()
            await release.wait()
            finished.set()
            return 1
        monkeypatch.setattr(routes.asyncio, "to_thread", worker)
        monkeypatch.setattr(routes, "_SLOTS", asyncio.Semaphore(1))
        request = asyncio.create_task(routes._work(lambda: None))
        await started.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert routes._SLOTS.locked()
        with pytest.raises(web.HTTPServiceUnavailable):
            await routes._work(lambda: None)
        release.set()
        await finished.wait()
        await asyncio.sleep(0)
        assert not routes._SLOTS.locked()
    asyncio.run(run())


def test_mono_last_sample_and_duration(audio_env):
    mod, source, wave, rate = audio_env
    token = mod.write_wav(wave[:1], rate, "b" * 64)
    shutil.copyfile(mod.media_path(token), source / "mono.wav")
    info = mod.analyze_audio("mono.wav")
    cut = mod.commit_cut("mono.wav", 3 - 1 / rate, 3, info["source_id"])
    audio, duration, _ = mod.selection_audio("mono.wav", cut["selection"])
    assert audio["waveform"].shape == (1, 1, 1)
    assert duration == 1 / rate
    assert audio["waveform"][0, 0, 0] == wave[0, -1]
