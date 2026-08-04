"""Convert ComfyUI IMAGE batches to bounded JPEG data URLs."""

import base64
import io

import numpy as np
import torch
from PIL import Image


def image_batch_to_jpeg_data_urls(images: torch.Tensor, max_side: int, quality: int = 88) -> list[str]:
    if not isinstance(images, torch.Tensor):
        raise TypeError("Reference image must be a torch.Tensor in ComfyUI IMAGE format.")
    if images.ndim != 4:
        raise ValueError("Reference image must have shape [B,H,W,C].")
    if images.shape[0] < 1:
        raise ValueError("Reference image batch is empty.")
    if images.shape[1] < 1 or images.shape[2] < 1 or images.shape[3] not in (3, 4):
        raise ValueError("Reference image must contain non-empty RGB or RGBA frames.")
    if not 64 <= int(max_side) <= 4096:
        raise ValueError("image_send_size must be between 64 and 4096.")

    urls = []
    for frame in images:
        array = frame.detach().to(device="cpu", dtype=torch.float32).clamp(0, 1).numpy()
        if array.shape[-1] == 4:
            rgb, alpha = array[..., :3], array[..., 3:4]
            array = rgb * alpha + (1.0 - alpha)
        pil = Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB")
        scale = min(1.0, float(max_side) / max(pil.size))
        if scale < 1.0:
            size = (max(1, round(pil.width * scale)), max(1, round(pil.height * scale)))
            pil = pil.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=quality, optimize=True)
        urls.append("data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    return urls
