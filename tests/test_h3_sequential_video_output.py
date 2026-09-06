from types import SimpleNamespace

import pytest
from comfy_api.latest import VideoFromFile
from ComfyUI_JR_MiniMaxH3Node.nodes import h3_sequential_audio as nodes


@pytest.mark.parametrize(("filename", "has_next", "ready"), [
    ("", True, False),
    ("", False, False),  # duplicate intermediate commit
    ("complete.mp4", False, True),
    ("complete.mp4", True, False),
])
def test_only_completed_video_reaches_downstream(monkeypatch, filename, has_next, ready):
    from comfy_execution.graph import ExecutionBlocker

    monkeypatch.setattr(nodes, "commit_decoded_chunk", lambda **kwargs: (filename, "status", has_next))
    result = nodes.JR_H3_SequentialVideoOutput().commit(
        images=object(), chunk_context=SimpleNamespace(job_id="test", chunk_index=0, total_chunks=2),
        aggressive_memory_cleanup=False,
    )
    assert result[:2] == (filename, "status")
    assert nodes.JR_H3_SequentialVideoOutput.RETURN_NAMES == ("filename", "status", "video")
    assert nodes.JR_H3_SequentialVideoOutput.RETURN_TYPES == ("STRING", "STRING", "VIDEO")
    if ready:
        assert isinstance(result[2], VideoFromFile)
        assert result[2].get_stream_source() == filename
    else:
        assert isinstance(result[2], ExecutionBlocker)
        assert result[2].message is None


def test_completed_video_connects_to_native_save_video_with_audio(tmp_path, monkeypatch):
    import shutil
    import subprocess

    import av
    import folder_paths
    from comfy_extras.nodes_video import SaveVideo

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is required for the real VIDEO integration smoke test")
    source = tmp_path / "completed.mp4"
    subprocess.run([
        ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=size=32x32:rate=24:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=24000:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source),
    ], check=True, capture_output=True)
    monkeypatch.setattr(nodes, "commit_decoded_chunk", lambda **kwargs: (str(source), "complete", False))
    _, _, video = nodes.JR_H3_SequentialVideoOutput().commit(
        images=object(), chunk_context=SimpleNamespace(job_id="test", chunk_index=1, total_chunks=2),
        aggressive_memory_cleanup=False,
    )
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    saver = SaveVideo.PREPARE_CLASS_CLONE(None)
    result = saver.execute(video, filename_prefix="native_save", format="mp4")
    assert result.result[0] is video
    saved, = tmp_path.glob("native_save_*.mp4")
    with av.open(str(saved)) as container:
        assert len(container.streams.video) == len(container.streams.audio) == 1
        assert container.streams.video[0].frames == 24
    assert result.ui is not None
