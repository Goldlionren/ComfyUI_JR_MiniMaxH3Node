import base64
import io

import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.utils.image_conversion import image_batch_to_jpeg_data_urls
from PIL import Image


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("batch", [1, 3])
def test_jpeg_data_urls(channels, batch):
    tensor = torch.linspace(-1, 2, batch * 40 * 80 * channels).reshape(batch, 40, 80, channels)
    urls = image_batch_to_jpeg_data_urls(tensor, 32 if False else 64)
    assert len(urls) == batch
    decoded = Image.open(io.BytesIO(base64.b64decode(urls[0].split(",", 1)[1])))
    assert decoded.mode == "RGB" and max(decoded.size) == 64 and decoded.size == (64, 32)


@pytest.mark.parametrize("tensor", [torch.empty(0, 4, 4, 3), torch.zeros(4, 4, 3), torch.zeros(1, 2, 3, 5)])
def test_invalid_images(tensor):
    with pytest.raises((ValueError, TypeError)): image_batch_to_jpeg_data_urls(tensor, 768)
