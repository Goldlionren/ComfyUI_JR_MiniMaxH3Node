import pytest
from ComfyUI_JR_MiniMaxH3Node.nodes.resolution_scale_calculator import (
    JR_H3_ResolutionScaleCalculator,
    calculate_resolution,
)


@pytest.mark.parametrize("w,h,aspect", [(1920,1080,"Source"),(1080,1920,"Source"),(1024,1024,"1:1"),(800,1200,"2:3"),(1200,800,"3:2")])
@pytest.mark.parametrize("divisor", [8, 16, 32])
def test_resolution_is_deterministic_divisible(w, h, aspect, divisor):
    result = calculate_resolution(w, h, 1.0, divisor, aspect)
    assert result[0] % divisor == 0 and result[1] % divisor == 0
    assert result == calculate_resolution(w, h, 1.0, divisor, aspect)


def test_tiny_resolution_clamps_to_divisor():
    width, height, *_ = calculate_resolution(1, 1, 0.001, 32)
    assert width >= 32 and height >= 32


@pytest.mark.parametrize("w,h", [(0,1),(1,0),(-1,5)])
def test_invalid_source(w, h):
    with pytest.raises(ValueError): calculate_resolution(w, h, 1, 8)


@pytest.mark.parametrize("mp", [0.0001, 300])
def test_target_area_boundaries(mp):
    with pytest.raises(ValueError): calculate_resolution(100, 100, mp, 8)


def test_scale_and_area_are_positive():
    width, height, scale, area = calculate_resolution(768, 1152, 0.88, 32)
    assert width > 0 and height > 0 and scale > 0 and area == width * height / 1_000_000


def test_divisor_combo_uses_string_values():
    divisor_spec = JR_H3_ResolutionScaleCalculator.INPUT_TYPES()["required"]["divisor"]
    assert divisor_spec == (["8", "16", "32"], {"default": "32"})


@pytest.mark.parametrize("divisor", [8, 16, 32, "8", "16", "32"])
def test_divisor_validation_accepts_new_and_legacy_values(divisor):
    assert JR_H3_ResolutionScaleCalculator.VALIDATE_INPUTS(divisor) is True
    width, height, *_ = calculate_resolution(768, 1152, 2.1, divisor)
    normalized = int(divisor)
    assert width % normalized == 0 and height % normalized == 0


@pytest.mark.parametrize("divisor", [None, "", "64", 64])
def test_divisor_validation_rejects_unavailable_values(divisor):
    result = JR_H3_ResolutionScaleCalculator.VALIDATE_INPUTS(divisor)
    assert result == "divisor must be 8, 16, or 32."
