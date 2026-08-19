"""Strict MiniMax H3 video/audio latent validation and AV assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

ERROR_PREFIX = "JR MiniMax H3 AV Latent Builder:"
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_STREAM_CHANNELS = 2
H3_FPS = 24
AUDIO_LATENT_FPS = 40
AUDIO_TEMPORAL_TOLERANCE = 1


class H3AVLatentBuilderError(ValueError):
    """Raised when separate latent streams cannot form a valid H3 AV latent."""


@dataclass(frozen=True, slots=True)
class H3TimelineMatch:
    frame_count: int
    video_latent_t: int
    audio_latent_t: int
    expected_audio_t: int
    audio_delta: int


def _error(message: str) -> H3AVLatentBuilderError:
    return H3AVLatentBuilderError(f"{ERROR_PREFIX}\n{message}")


def _shape(tensor: torch.Tensor) -> str:
    return "[" + ",".join(str(int(value)) for value in tensor.shape) + "]"


def _extract_latent_samples(latent: Any, name: str) -> torch.Tensor:
    if not isinstance(latent, Mapping):
        raise _error(f"{name} is invalid. Expected a LATENT mapping containing 'samples'.")
    if "samples" not in latent:
        raise _error(f"{name} is invalid. Missing required 'samples'.")
    samples = latent["samples"]
    if not isinstance(samples, torch.Tensor):
        raise _error(f"{name} is invalid. 'samples' must be a torch.Tensor.")
    return samples


def _validate_tensor_storage(tensor: torch.Tensor, name: str) -> None:
    if tensor.layout != torch.strided:
        raise _error(f"{name} is invalid. Only strided tensors are supported, received {tensor.layout}.")
    if tensor.device.type == "meta":
        raise _error(f"{name} is invalid. Meta tensors have no materialized latent values.")
    if not tensor.is_floating_point():
        raise _error(f"{name} is invalid. Expected a floating-point tensor, received {tensor.dtype}.")


def _validate_video_latent(video: torch.Tensor) -> None:
    expected = "Expected a H3 video latent tensor with shape [B,24,T,H,W]"
    if video.ndim != 5:
        raise _error(f"video_latent is invalid. {expected}, but received shape {_shape(video)}.")
    if video.shape[1] != VIDEO_CHANNELS:
        raise _error(f"video_latent is invalid. {expected}, but received shape {_shape(video)}.")
    if video.shape[0] <= 0:
        raise _error("video_latent is invalid. Batch size must be greater than zero.")
    if video.shape[2] <= 0:
        raise _error("video_latent is invalid. Temporal dimension T must be greater than zero.")
    if video.shape[3] <= 0 or video.shape[4] <= 0:
        raise _error("video_latent is invalid. Latent spatial dimensions H and W must be greater than zero.")
    _validate_tensor_storage(video, "video_latent")


def _validate_audio_latent(audio: torch.Tensor) -> None:
    expected = "Expected a H3 audio latent tensor with shape [B,32,2,T]"
    if audio.ndim != 4:
        raise _error(f"audio_latent is invalid. {expected}, but received shape {_shape(audio)}.")
    if audio.shape[1] != AUDIO_CHANNELS or audio.shape[2] != AUDIO_STREAM_CHANNELS:
        raise _error(f"audio_latent is invalid. {expected}, but received shape {_shape(audio)}.")
    if audio.shape[0] <= 0:
        raise _error("audio_latent is invalid. Batch size must be greater than zero.")
    if audio.shape[3] <= 0:
        raise _error("audio_latent is invalid. Temporal dimension T must be greater than zero.")
    _validate_tensor_storage(audio, "audio_latent")


def _check_pair_compatibility(video: torch.Tensor, audio: torch.Tensor) -> None:
    if video.shape[0] != audio.shape[0]:
        raise _error(
            "video/audio batch mismatch.\n"
            f"video batch: {video.shape[0]}\n"
            f"audio batch: {audio.shape[0]}"
        )
    if video.device != audio.device:
        raise _error(
            "video/audio device mismatch. Automatic device transfers are intentionally disabled.\n"
            f"video device: {video.device}\n"
            f"audio device: {audio.device}"
        )
    if video.dtype != audio.dtype:
        raise _error(
            "video/audio dtype mismatch. Automatic casting is intentionally disabled.\n"
            f"video dtype: {video.dtype}\n"
            f"audio dtype: {audio.dtype}"
        )


def _check_finite(tensor: torch.Tensor, name: str) -> None:
    try:
        finite = bool(torch.isfinite(tensor).all().item())
    except (RuntimeError, TypeError) as error:
        raise _error(f"{name} could not be checked for finite values: {type(error).__name__}.") from None
    if not finite:
        raise _error(f"{name} is invalid. Tensor contains NaN or Inf values.")


def _check_temporal_compatibility(video_t: int, audio_t: int) -> H3TimelineMatch:
    if video_t < 2 or (video_t - 2) % 5 != 0:
        raise _error(
            "video_latent has an invalid H3 temporal grid. "
            "Expected T_video = 5k + 2, corresponding to the official 17k + 5 frame grid, "
            f"but received T_video={video_t}."
        )
    block_count = (video_t - 2) // 5
    frame_count = 5 + 17 * block_count
    expected_audio_t = round(frame_count * AUDIO_LATENT_FPS / H3_FPS)
    delta = audio_t - expected_audio_t
    if abs(delta) > AUDIO_TEMPORAL_TOLERANCE:
        raise _error(
            "video/audio temporal mismatch.\n\n"
            f"video latent T: {video_t} (official timeline: {frame_count} frames at {H3_FPS} fps)\n"
            f"audio latent T: {audio_t}\n"
            f"expected audio latent T: {expected_audio_t} ± {AUDIO_TEMPORAL_TOLERANCE}\n\n"
            "These two latents do not appear to represent the same H3 timeline."
        )
    return H3TimelineMatch(
        frame_count=frame_count,
        video_latent_t=video_t,
        audio_latent_t=audio_t,
        expected_audio_t=expected_audio_t,
        audio_delta=delta,
    )


def _build_nested_latent(video: torch.Tensor, audio: torch.Tensor) -> dict[str, Any]:
    try:
        from comfy.nested_tensor import NestedTensor
    except ImportError:
        raise RuntimeError(
            f"{ERROR_PREFIX}\nThe installed ComfyUI does not provide comfy.nested_tensor.NestedTensor."
        ) from None
    return {"samples": NestedTensor((video, audio))}


def _format_status(video: torch.Tensor, audio: torch.Tensor, timeline: H3TimelineMatch) -> str:
    video_seconds = timeline.frame_count / H3_FPS
    audio_seconds = timeline.audio_latent_t / AUDIO_LATENT_FPS
    return "\n".join(
        (
            "Success",
            f"video: {_shape(video)}",
            f"audio: {_shape(audio)}",
            f"batch: {video.shape[0]}",
            f"dtype: {video.dtype} / {audio.dtype}",
            f"device: {video.device} / {audio.device}",
            (
                "temporal check: passed "
                f"({timeline.frame_count} frames @ {H3_FPS} fps; "
                f"video≈{video_seconds:.3f}s, audio≈{audio_seconds:.3f}s, "
                f"audio delta={timeline.audio_delta:+d})"
            ),
            "H3 AV latent: valid",
        )
    )


def build_h3_av_latent(video_latent: Any, audio_latent: Any) -> tuple[dict[str, Any], str]:
    """Validate two separately encoded H3 streams and wrap them without copying."""

    video = _extract_latent_samples(video_latent, "video_latent")
    audio = _extract_latent_samples(audio_latent, "audio_latent")
    _validate_video_latent(video)
    _validate_audio_latent(audio)
    _check_pair_compatibility(video, audio)
    timeline = _check_temporal_compatibility(int(video.shape[2]), int(audio.shape[3]))
    _check_finite(video, "video_latent")
    _check_finite(audio, "audio_latent")
    return _build_nested_latent(video, audio), _format_status(video, audio, timeline)


__all__ = [
    "AUDIO_TEMPORAL_TOLERANCE",
    "ERROR_PREFIX",
    "H3AVLatentBuilderError",
    "H3TimelineMatch",
    "build_h3_av_latent",
]
