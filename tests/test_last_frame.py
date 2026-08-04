import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.nodes.last_frame import JR_H3_LastFrame


@pytest.mark.parametrize("batch", [1, 2, 10])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_last_frame_preserves_contract(batch, channels, dtype):
    frames = torch.arange(batch * 3 * 4 * channels, dtype=dtype).reshape(batch, 3, 4, channels)
    before = frames.clone()
    output = JR_H3_LastFrame().extract(frames)[0]
    assert output.shape == (1, 3, 4, channels)
    assert output.dtype == dtype and output.device == frames.device
    assert torch.equal(output[0], frames[-1]) and torch.equal(frames, before)


def test_empty_mentions_pass_frames():
    with pytest.raises(ValueError, match="pass_frames"): JR_H3_LastFrame().extract(torch.empty(0, 2, 2, 3))


@pytest.mark.parametrize("shape", [(2,2,3),(1,1,2,2,3)])
def test_bad_dimensions(shape):
    with pytest.raises(ValueError): JR_H3_LastFrame().extract(torch.zeros(shape))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_preserved():
    frames = torch.rand(2, 3, 4, 3, device="cuda")
    assert JR_H3_LastFrame().extract(frames)[0].device.type == "cuda"
