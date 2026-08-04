import sys
import types
from datetime import datetime
from pathlib import Path

import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.nodes.enhanced_video_combine import (
    AUDIO_BITRATES,
    AUDIO_CODECS,
    BIT_DEPTHS,
    CODECS,
    CONTAINERS,
    JR_H3_EnhancedVideoCombine,
    _available_video_encoders,
    _codec_order,
    _container_order,
    _format_date_tokens,
    _preview_source_path,
    _resolve_bit_depth,
    _safe_relative_prefix,
    _write_audio,
    detect_bit_depth,
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
    stub = types.SimpleNamespace(
        get_output_directory=lambda: str(output),
        get_temp_directory=lambda: str(output),
        get_directory_by_type=lambda _kind: str(output),
    )
    old = sys.modules.get("folder_paths"); sys.modules["folder_paths"] = stub
    try:
        frames = torch.zeros(4, 16, 24, 3); frames[-1, ..., 1] = 1
        result = JR_H3_EnhancedVideoCombine().combine(
            images=frames, frame_rate=4.0, codec="H.264", container="MP4", bit_depth="8-bit",
            quality=28, log_level="Standard", pingpong=True, save_metadata=False,
            filename_prefix="nested/测试 video", save_output=True, pass_frames=True,
            crop_to_audio=False, audio_codec="Auto", audio_bitrate="192k",
            save_first_frame=True, save_last_frame=True,
        )
        returned, filename = result["result"]
        assert Path(filename).exists() and Path(filename).stat().st_size > 0
        assert returned.shape[0] == 6
        assert Path(filename).with_name(Path(filename).stem + "-first-frame.png").exists()
        assert Path(filename).with_name(Path(filename).stem + "-last-frame.png").exists()
        assert JR_H3_LastFrame().extract(returned)[0].shape == (1,16,24,3)
        assert result["ui"]["gifs"][0] == {
            "filename": Path(filename).name,
            "subfolder": "nested",
            "type": "output",
            "format": "video/mp4",
            "width": 24,
            "height": 16,
            "codec": "H.264",
            "bit_depth": 8,
            "container": "MP4",
            "fps": 4.0,
        }
        assert len(result["ui"]["images"]) == 3
    finally:
        if old is None: sys.modules.pop("folder_paths", None)
        else: sys.modules["folder_paths"] = old


def test_pass_frames_false_can_feed_clear_last_frame_error():
    empty = torch.zeros(3,2,2,3)[:0]
    with pytest.raises(ValueError, match="pass_frames"): JR_H3_LastFrame().extract(empty)


def test_input_surface_matches_enhanced_reference_features():
    required = JR_H3_EnhancedVideoCombine.INPUT_TYPES()["required"]
    expected = {
        "images", "frame_rate", "codec", "container", "bit_depth", "quality", "log_level",
        "pingpong", "save_metadata", "filename_prefix", "save_output", "pass_frames",
        "crop_to_audio", "audio_codec", "audio_bitrate", "save_first_frame", "save_last_frame",
    }
    assert set(required) == expected
    assert required["codec"][0] == CODECS
    assert required["container"][0] == CONTAINERS
    assert required["bit_depth"][0] == BIT_DEPTHS
    assert required["audio_codec"][0] == AUDIO_CODECS
    assert required["audio_bitrate"][0] == AUDIO_BITRATES
    assert required["filename_prefix"][1]["default"].startswith("video/%date:")


def test_auto_orders_and_explicit_container_orders():
    assert _codec_order("Auto") == ("AV1", "VP9", "H.264")
    assert "H.265 (HEVC)" not in _codec_order("Auto")
    assert _container_order("AV1", "Auto", True) == ("WebM",)
    assert _container_order("AV1", "Auto", False) == ("WebM", "MKV", "MP4")
    assert _container_order("H.265 (HEVC)", "Auto", False) == ("MP4", "MKV")


def test_bit_depth_detection_and_auto_policy():
    eight_bit = torch.tensor([0, 1 / 255, 2 / 255, 1], dtype=torch.float32)
    ten_bit = torch.tensor([1 / 1023, 2 / 1023, 7 / 1023], dtype=torch.float32)
    assert detect_bit_depth(eight_bit) == 8
    assert detect_bit_depth(ten_bit) == 10
    assert _resolve_bit_depth("Auto", "Auto", ten_bit) == 8
    assert _resolve_bit_depth("AV1", "Auto", ten_bit) == 10
    assert _resolve_bit_depth("AV1", "8-bit", ten_bit) == 8


def test_date_tokens_and_prefix_safety():
    fixed = datetime(2026, 8, 4, 13, 2, 9)
    assert _format_date_tokens("video/%date:yyyy-MM-dd%/%date:hhmmss%", fixed) == "video/2026-08-04/130209"
    assert _safe_relative_prefix("../../outside/clip") == "outside/clip"
    assert _safe_relative_prefix("C:\\unsafe\\a:b") == "C/unsafe/ab"


def test_audio_conversion_supports_comfy_shapes_and_cleans_up():
    audio_info, duration = _write_audio({"waveform": torch.zeros(1, 2, 800), "sample_rate": 8000})
    try:
        assert audio_info[1:] == (8000, 2)
        assert Path(audio_info[0]).stat().st_size == 800 * 2 * 4
        assert duration == pytest.approx(0.1)
    finally:
        Path(audio_info[0]).unlink(missing_ok=True)


def test_preview_path_blocks_traversal(tmp_path):
    output = tmp_path / "output"; output.mkdir()
    video = output / "safe.mp4"; video.write_bytes(b"video")
    stub = types.SimpleNamespace(get_directory_by_type=lambda kind: str(output) if kind == "output" else None)
    old = sys.modules.get("folder_paths"); sys.modules["folder_paths"] = stub
    try:
        assert _preview_source_path("safe.mp4", "", "output") == video.resolve()
        assert _preview_source_path("../safe.mp4", "", "output") is None
        assert _preview_source_path("safe.mp4", "../outside", "output") is None
        assert _preview_source_path("safe.mp4", "", "unknown") is None
    finally:
        if old is None: sys.modules.pop("folder_paths", None)
        else: sys.modules["folder_paths"] = old


def test_frontend_extension_contains_preview_save_and_download_contract():
    source = Path(__file__).parents[1].joinpath("js", "enhanced_video_combine_preview.js").read_text(encoding="utf-8")
    for marker in (
        "JR_H3_EnhancedVideoCombine", "addDOMWidget", 'createElement("video")', "Save first frame",
        "Save last frame", "Autoplay", "Download", "/jr-h3/enhanced-video-preview",
    ):
        assert marker in source


@pytest.mark.skipif(find_ffmpeg() is None, reason="FFmpeg unavailable")
def test_ffmpeg_reports_required_h264_software_fallback():
    assert "libx264" in _available_video_encoders(find_ffmpeg())
