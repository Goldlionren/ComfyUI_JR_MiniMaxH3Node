"""Low-overhead tensor metrics used independently for H3 video and audio."""

from __future__ import annotations

import torch


def sample_tensor(tensor, stride: int):
    if stride <= 0:
        raise ValueError("stride must be greater than zero")
    flat = tensor.reshape(-1) if tensor.is_contiguous() else tensor.contiguous().view(-1)
    return flat[:: min(stride, max(1, flat.numel()))]


def metric_sample(tensor, stride: int):
    """Return a small, graph-free fp32 metric snapshot on the tensor's device."""
    sampled = sample_tensor(tensor, stride)
    return sampled.detach().to(device=tensor.device, dtype=torch.float32).clone()


def relative_delta(current, previous, stride: int = 1, epsilon: float = 1e-6) -> float:
    """Sample on the existing device, compute in fp32, and synchronize only once."""
    if current.device != previous.device:
        raise RuntimeError(
            "H3 cache metric history device mismatch: "
            f"current={current.device}, previous={previous.device}. "
            "Metric history must remain on the active compute device."
        )
    if current.shape != previous.shape:
        return float("inf")
    current_sample = sample_tensor(current, stride).float()
    previous_sample = sample_tensor(previous, stride).float()
    if current_sample.numel() == 0:
        return 0.0
    score = (current_sample.sub(previous_sample).abs().mean()
             / previous_sample.abs().mean().add(float(epsilon)))
    return float(score.detach().item())


def tensor_signature(tensor) -> tuple:
    return tuple(tensor.shape), str(tensor.dtype), str(tensor.device), int(tensor.shape[0]) if tensor.ndim else 0
