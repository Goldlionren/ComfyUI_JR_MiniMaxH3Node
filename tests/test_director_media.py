import importlib
import io
import sys
import types

import pytest
from ComfyUI_JR_MiniMaxH3Node.utils.director_media import (
    DirectorMediaError,
    probe_asset,
    resolve_asset_path,
)
from ComfyUI_JR_MiniMaxH3Node.utils.director_state import AssetDescriptor
from PIL import Image


def _asset(filename="参考 图.png", subfolder="镜头 一", kind="image"):
    return AssetDescriptor(
        id="asset-1", kind=kind, filename=filename, subfolder=subfolder,
        folder_type="input", display_name=filename, mime_type="", status="ready",
    )


def _install_folder_paths(monkeypatch, root):
    module = types.SimpleNamespace(get_directory_by_type=lambda kind: str(root) if kind == "input" else None)
    monkeypatch.setitem(sys.modules, "folder_paths", module)


def test_resolver_accepts_unicode_relative_asset_and_probe_is_bounded(monkeypatch, tmp_path):
    root = tmp_path / "input"
    folder = root / "镜头 一"
    folder.mkdir(parents=True)
    path = folder / "参考 图.png"
    Image.new("RGB", (12, 8), (10, 20, 30)).save(path)
    _install_folder_paths(monkeypatch, root)
    assert resolve_asset_path(_asset()) == path.resolve()
    updated, metadata = probe_asset(_asset())
    assert (updated.width, updated.height, updated.status) == (12, 8, "ready")
    assert metadata["size_bytes"] == path.stat().st_size


def test_resolver_rejects_escape_and_missing_file(monkeypatch, tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    _install_folder_paths(monkeypatch, root)
    with pytest.raises(DirectorMediaError, match="outside"):
        resolve_asset_path(_asset(filename="secret.png", subfolder=".."))
    with pytest.raises(DirectorMediaError, match="missing"):
        resolve_asset_path(_asset(filename="missing.png", subfolder=""))


def test_corrupt_image_is_a_clear_media_error(monkeypatch, tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    (root / "bad.png").write_bytes(b"not an image")
    _install_folder_paths(monkeypatch, root)
    with pytest.raises(DirectorMediaError, match="corrupt or unsupported"):
        probe_asset(_asset(filename="bad.png", subfolder=""))


def test_ffprobe_uses_devnull_stdin_and_portable_options(monkeypatch, tmp_path):
    import ComfyUI_JR_MiniMaxH3Node.utils.director_media as module

    root = tmp_path / "input"
    root.mkdir()
    (root / "clip.mp4").write_bytes(b"probe fixture")
    _install_folder_paths(monkeypatch, root)
    monkeypatch.setattr(module, "_find_ffprobe", lambda: "ffprobe")
    captured = {}

    class FakeProcess:
        def __init__(self, command, **kwargs):
            payload = b'{"streams":[{"codec_type":"video","codec_name":"h264","width":16,"height":16,"avg_frame_rate":"24/1"}],"format":{"duration":"1.0"}}'
            captured.update(command=command, kwargs=kwargs)
            self.stdout = io.BytesIO(payload)
            self.stderr = io.BytesIO()

        def wait(self, timeout):
            return 0

        def kill(self):
            return None

    def fake_popen(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return FakeProcess(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    updated, metadata = probe_asset(_asset(filename="clip.mp4", subfolder="", kind="video"))
    assert updated.status == "ready"
    assert metadata["duration_seconds"] == 1.0
    assert "-nostdin" not in captured["command"]
    assert captured["kwargs"]["stdin"] is module.subprocess.DEVNULL
    assert captured["kwargs"]["shell"] is False


def test_director_media_import_never_runs_subprocess(monkeypatch):
    import ComfyUI_JR_MiniMaxH3Node.utils.director_media as module

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("import ran ffprobe"))
    importlib.reload(module)
