import sys
import types
from pathlib import Path

import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.nodes.enhanced_video_combine import (
    JR_H3_EnhancedVideoCombine,
    find_ffmpeg,
    pingpong_frames,
)
from ComfyUI_JR_MiniMaxH3Node.nodes.last_frame import JR_H3_LastFrame


def test_pingpong_count_and_order():
    frames = torch.arange(4).reshape(4,1,1,1)
    assert pingpong_frames(frames, True).flatten().tolist() == [0,1,2,3,2,1]


@pytest.mark.parametrize("enabled,expected", [(False,4),(True,6)])
def test_pingpong_lengths(enabled, expected):
    assert pingpong_frames(torch.zeros(4,1,1,3), enabled).shape[0] == expected


@pytest.mark.skipif(find_ffmpeg() is None, reason="FFmpeg unavailable")
@pytest.mark.parametrize("directory_name", ["space output", "Unicode-输出"])
def test_ffmpeg_smoke_and_frame_contract(tmp_path, directory_name):
    output = tmp_path / directory_name; output.mkdir()
    stub = types.SimpleNamespace(get_output_directory=lambda: str(output), get_temp_directory=lambda: str(output))
    old = sys.modules.get("folder_paths"); sys.modules["folder_paths"] = stub
    try:
        frames = torch.zeros(4, 16, 24, 3); frames[-1, ..., 1] = 1
        result = JR_H3_EnhancedVideoCombine().combine(
            frames, 4.0, "H.264", "MP4", 28, True, False, "测试 video", True, True,
            False, True, True,
        )
        returned, filename = result["result"]
        assert Path(filename).exists() and Path(filename).stat().st_size > 0
        assert returned.shape[0] == 6
        assert Path(filename).with_name(Path(filename).stem + "-first-frame.png").exists()
        assert Path(filename).with_name(Path(filename).stem + "-last-frame.png").exists()
        assert JR_H3_LastFrame().extract(returned)[0].shape == (1,16,24,3)
    finally:
        if old is None: sys.modules.pop("folder_paths", None)
        else: sys.modules["folder_paths"] = old


def test_pass_frames_false_can_feed_clear_last_frame_error():
    empty = torch.zeros(3,2,2,3)[:0]
    with pytest.raises(ValueError, match="pass_frames"): JR_H3_LastFrame().extract(empty)
