import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.nodes.rtx_upscaler_refiner import (
    JR_H3_RTXUpscalerRefiner,
    _quality_level,
    target_size,
)


class FakeQualityLevel:
    LOW = "vsr-low"
    DENOISE_HIGH = "denoise-high"
    DEBLUR_ULTRA = "deblur-ultra"
    HIGHBITRATE_MEDIUM = "high-bitrate-medium"


@pytest.mark.parametrize(
    "operation,quality,expected_name,expected_value",
    [
        ("VSR", "Low", "LOW", "vsr-low"),
        ("Denoise", "High", "DENOISE_HIGH", "denoise-high"),
        ("Deblur", "Ultra", "DEBLUR_ULTRA", "deblur-ultra"),
        ("High Bitrate", "Medium", "HIGHBITRATE_MEDIUM", "high-bitrate-medium"),
    ],
)
def test_quality_level_uses_single_video_super_res_enum(operation, quality, expected_name, expected_value):
    assert _quality_level(FakeQualityLevel, operation, quality) == (expected_value, expected_name)


def test_missing_effect_quality_is_explicit():
    with pytest.raises(RuntimeError, match="does not support Denoise"):
        _quality_level(FakeQualityLevel, "Denoise", "Ultra")


def test_target_size_modes():
    assert target_size(100, 50, "Scale", 2, 1, 1, 1, 8, "16:9") == (200, 96)
    assert target_size(100, 50, "Manual", 2, 1, 640, 480, 32, "16:9") == (640, 480)


def test_no_effects_passes_rgb_without_dependency():
    node = JR_H3_RTXUpscalerRefiner()
    image = torch.rand(1, 8, 8, 4)
    result = node.execute(image, False,"Low",False,"Low","Off","Low","Same Size",1,1,8,8,"8","1:1","Center Crop (Fill)",0)[0]
    assert result.shape == (1,8,8,3)


@pytest.mark.skipif(torch.cuda.is_available(), reason="This verifies the non-CUDA dependency path")
def test_missing_cuda_is_friendly():
    node = JR_H3_RTXUpscalerRefiner()
    with pytest.raises(RuntimeError, match="CUDA.*NVIDIA RTX"):
        node.execute(torch.rand(1,8,8,3), False,"Low",False,"Low","VSR","Low","Scale",2,1,8,8,"8","1:1","Center Crop (Fill)",0)
