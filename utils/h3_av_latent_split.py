"""Strict splitting of an official MiniMax H3 AV NestedTensor LATENT."""

from __future__ import annotations

from typing import Any

import torch

ERROR_PREFIX = "JR MiniMax H3 Split AV Latent:"
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_STREAM_CHANNELS = 2


class H3AVLatentSplitError(ValueError):
    """Raised when an input cannot be split into valid H3 video/audio latents."""


def _error(message: str) -> H3AVLatentSplitError:
    return H3AVLatentSplitError(f"{ERROR_PREFIX}\n{message}")


def _shape(tensor: torch.Tensor) -> str:
    return "[" + ",".join(str(int(value)) for value in tensor.shape) + "]"


def _official_nested_tensor_type():
    try:
        from comfy.nested_tensor import NestedTensor
    except ImportError:
        raise RuntimeError(
            f"{ERROR_PREFIX}\nThe installed ComfyUI does not provide comfy.nested_tensor.NestedTensor."
        ) from None
    return NestedTensor


def _validate_tensor_storage(tensor: torch.Tensor, name: str) -> None:
    if tensor.layout != torch.strided:
        raise _error(f"{name} must be a strided tensor, received {tensor.layout}.")
    if tensor.device.type == "meta":
        raise _error(f"{name} cannot be a meta tensor because it has no materialized latent values.")
    if not tensor.is_floating_point():
        raise _error(f"{name} must be a floating-point tensor, received {tensor.dtype}.")


def _validate_video(video: torch.Tensor) -> None:
    if video.ndim != 5 or video.shape[1] != VIDEO_CHANNELS:
        raise _error(f"video stream must have shape [B,24,T,H,W], received {_shape(video)}.")
    if video.shape[0] <= 0 or video.shape[2] <= 0 or video.shape[3] <= 0 or video.shape[4] <= 0:
        raise _error("video stream dimensions B, T, H and W must all be greater than zero.")
    _validate_tensor_storage(video, "video stream")


def _validate_audio(audio: torch.Tensor) -> None:
    if audio.ndim != 4 or audio.shape[1] != AUDIO_CHANNELS or audio.shape[2] != AUDIO_STREAM_CHANNELS:
        raise _error(f"audio stream must have shape [B,32,2,T], received {_shape(audio)}.")
    if audio.shape[0] <= 0 or audio.shape[3] <= 0:
        raise _error("audio stream dimensions B and T must both be greater than zero.")
    _validate_tensor_storage(audio, "audio stream")


def _check_finite(tensor: torch.Tensor, name: str) -> None:
    try:
        finite = bool(torch.isfinite(tensor).all().item())
    except (RuntimeError, TypeError):
        raise _error(f"{name} could not be checked for finite values.") from None
    if not finite:
        raise _error(f"{name} contains NaN or Inf values.")


def split_h3_av_latent(av_latent: Any) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return the two original H3 stream tensors in standard LATENT mappings."""

    if not isinstance(av_latent, dict):
        raise _error("av_latent must be a LATENT dictionary containing 'samples'.")
    if "samples" not in av_latent:
        raise _error("av_latent is missing required 'samples'.")

    samples = av_latent["samples"]
    nested_tensor_type = _official_nested_tensor_type()
    if type(samples) is not nested_tensor_type:
        raise _error("av_latent 'samples' must be the official comfy.nested_tensor.NestedTensor type.")

    streams = samples.unbind()
    if len(streams) != 2:
        raise _error(f"H3 AV NestedTensor must contain exactly 2 streams (video, audio), received {len(streams)}.")
    video, audio = streams
    if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
        raise _error("H3 AV NestedTensor streams must both be torch.Tensor objects.")

    _validate_video(video)
    _validate_audio(audio)
    if video.shape[0] != audio.shape[0]:
        raise _error(
            "video/audio batch mismatch.\n"
            f"video batch: {video.shape[0]}\n"
            f"audio batch: {audio.shape[0]}"
        )
    _check_finite(video, "video stream")
    _check_finite(audio, "audio stream")

    # Intentionally preserve the exact stream tensor objects. Save Latent can
    # consume each returned mapping through its normal samples.contiguous() path.
    return {"samples": video}, {"samples": audio}


__all__ = ["ERROR_PREFIX", "H3AVLatentSplitError", "split_h3_av_latent"]
