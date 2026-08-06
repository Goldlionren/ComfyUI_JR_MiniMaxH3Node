import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.utils.h3_cache_metrics import relative_delta


def test_identical_small_and_large_changes():
    base = torch.ones(2, 3, 4)
    assert relative_delta(base, base, 2) == pytest.approx(0.0)
    assert relative_delta(base + 0.01, base, 2) < relative_delta(base + 1.0, base, 2)


def test_zero_bfloat16_non_contiguous_and_tiny_are_safe():
    zero = torch.zeros(2, 3, dtype=torch.bfloat16)
    assert relative_delta(zero, zero, 99) == pytest.approx(0.0)
    non_contiguous = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2)
    assert relative_delta(non_contiguous, non_contiguous.clone(), 3) == pytest.approx(0.0)
    assert relative_delta(torch.tensor([1.0]), torch.tensor([1.0]), 1000) == pytest.approx(0.0)


def test_audio_and_video_scores_are_independent():
    video = torch.ones(8)
    audio = torch.ones(8)
    assert relative_delta(video, video, 2) == 0.0
    assert relative_delta(audio * 2, audio, 2) > 0.0


def test_invalid_stride_rejected():
    with pytest.raises(ValueError, match="stride"):
        relative_delta(torch.ones(2), torch.ones(2), 0)
