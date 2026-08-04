import pytest
from ComfyUI_JR_MiniMaxH3Node.nodes.resolution_scale_calculator import calculate_resolution


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
