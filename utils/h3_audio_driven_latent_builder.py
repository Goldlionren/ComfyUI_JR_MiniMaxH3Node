"""Inject and lock an external audio latent inside an official MiniMax H3 AV latent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

ERROR_PREFIX = "JR MiniMax H3 Audio Driven Latent Builder:"
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_STREAM_CHANNELS = 2
H3_FPS = 24
AUDIO_LATENT_FPS = 40
AUDIO_TEMPORAL_TOLERANCE = 1


class H3AudioDrivenLatentBuilderError(ValueError):
    """Raised when an audio drive latent cannot safely replace the H3 audio stream."""


@dataclass(frozen=True, slots=True)
class AudioDriveFit:
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    time_fit: str
    batch_fit: str
    device_fit: str
    dtype_fit: str


def _error(message: str) -> H3AudioDrivenLatentBuilderError:
    return H3AudioDrivenLatentBuilderError(f"{ERROR_PREFIX}\n{message}")


def _shape(value: torch.Tensor | tuple[int, ...]) -> str:
    shape = value.shape if isinstance(value, torch.Tensor) else value
    return "[" + ",".join(str(int(item)) for item in shape) + "]"


def _official_nested_tensor_type():
    try:
        from comfy.nested_tensor import NestedTensor
    except ImportError:
        raise RuntimeError(
            f"{ERROR_PREFIX}\nThe installed ComfyUI does not provide comfy.nested_tensor.NestedTensor."
        ) from None
    return NestedTensor


def _validate_storage(tensor: torch.Tensor, name: str) -> None:
    if tensor.layout != torch.strided:
        raise _error(f"{name} must be a strided tensor, received {tensor.layout}.")
    if tensor.device.type == "meta":
        raise _error(f"{name} cannot be a meta tensor because it has no materialized values.")
    if not tensor.is_floating_point():
        raise _error(f"{name} must be floating point, received {tensor.dtype}.")


def _check_finite(tensor: torch.Tensor, name: str) -> None:
    try:
        finite = bool(torch.isfinite(tensor).all().item())
    except (RuntimeError, TypeError):
        raise _error(f"{name} could not be checked for NaN or Inf values.") from None
    if not finite:
        raise _error(f"{name} contains NaN or Inf values.")


def _validate_video(video: torch.Tensor) -> None:
    if video.ndim != 5 or video.shape[1] != VIDEO_CHANNELS:
        raise _error(f"AV video stream must have shape [B,24,T,H,W], received {_shape(video)}.")
    if any(int(video.shape[index]) <= 0 for index in (0, 2, 3, 4)):
        raise _error("AV video stream dimensions B, T, H and W must all be greater than zero.")
    _validate_storage(video, "AV video stream")
    _check_finite(video, "AV video stream")


def _validate_audio(audio: torch.Tensor, name: str) -> None:
    if audio.ndim != 4 or audio.shape[1] != AUDIO_CHANNELS or audio.shape[2] != AUDIO_STREAM_CHANNELS:
        raise _error(f"{name} must have shape [B,32,2,T], received {_shape(audio)}.")
    if audio.shape[0] <= 0 or audio.shape[3] <= 0:
        raise _error(f"{name} dimensions B and T must both be greater than zero.")
    _validate_storage(audio, name)
    _check_finite(audio, name)


def _validate_h3_timeline(video: torch.Tensor, audio: torch.Tensor) -> None:
    video_t = int(video.shape[2])
    if video_t < 2 or (video_t - 2) % 5 != 0:
        raise _error(
            "AV video stream has an invalid H3 temporal grid. "
            f"Expected T_video = 5k + 2, received T_video={video_t}."
        )
    frame_count = 5 + 17 * ((video_t - 2) // 5)
    expected_audio_t = round(frame_count * AUDIO_LATENT_FPS / H3_FPS)
    actual_audio_t = int(audio.shape[-1])
    if abs(actual_audio_t - expected_audio_t) > AUDIO_TEMPORAL_TOLERANCE:
        raise _error(
            "AV video/template-audio temporal mismatch.\n"
            f"Video: {frame_count} frames at {H3_FPS} fps\n"
            f"Template audio T: {actual_audio_t}\n"
            f"Expected audio T: {expected_audio_t} ± {AUDIO_TEMPORAL_TOLERANCE}"
        )


def _extract_av_streams(av_latent: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(av_latent, Mapping):
        raise _error("'av_latent' must be a LATENT mapping containing 'samples'.")
    if "samples" not in av_latent:
        raise _error("'av_latent' is missing required 'samples'.")
    samples = av_latent["samples"]
    nested_tensor_type = _official_nested_tensor_type()
    if type(samples) is not nested_tensor_type:
        raise _error(
            "'av_latent' does not contain a valid MiniMax H3 joint AV NestedTensor. "
            "Expected the exact official comfy.nested_tensor.NestedTensor type."
        )
    streams = samples.unbind()
    if len(streams) != 2:
        raise _error(f"H3 AV NestedTensor must contain exactly 2 streams (video, audio), received {len(streams)}.")
    video, audio = streams
    if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
        raise _error("H3 AV NestedTensor streams must both be torch.Tensor objects.")
    _validate_video(video)
    _validate_audio(audio, "AV template audio stream")
    if video.shape[0] != audio.shape[0]:
        raise _error(
            "AV video/template-audio batch mismatch.\n"
            f"Video batch: {video.shape[0]}\n"
            f"Template audio batch: {audio.shape[0]}"
        )
    if video.device != audio.device:
        raise _error(
            "AV video/template-audio device mismatch.\n"
            f"Video device: {video.device}\n"
            f"Template audio device: {audio.device}"
        )
    if video.dtype != audio.dtype:
        raise _error(
            "AV video/template-audio dtype mismatch.\n"
            f"Video dtype: {video.dtype}\n"
            f"Template audio dtype: {audio.dtype}"
        )
    _validate_h3_timeline(video, audio)
    return video, audio


def _extract_audio_drive(audio_drive_latent: Any) -> torch.Tensor:
    if not isinstance(audio_drive_latent, Mapping):
        raise _error("'audio_drive_latent' must be a LATENT mapping containing 'samples'.")
    if "samples" not in audio_drive_latent:
        raise _error("'audio_drive_latent' is missing required 'samples'.")
    samples = audio_drive_latent["samples"]
    if not isinstance(samples, torch.Tensor):
        raise _error("Audio Drive latent 'samples' must be a torch.Tensor.")
    _validate_audio(samples, "Audio Drive latent")
    return samples


def _validate_mask_tensor(mask: Any, name: str, *, rank: int, target_batch: int) -> None:
    if not isinstance(mask, torch.Tensor):
        raise _error(f"{name} must be a torch.Tensor.")
    if mask.ndim != rank:
        raise _error(f"{name} must have rank {rank}, received shape {_shape(mask)}.")
    if mask.shape[0] not in {1, target_batch}:
        raise _error(
            f"{name} batch size {mask.shape[0]} cannot be matched to AV batch size {target_batch}. "
            f"Expected batch size 1 or {target_batch}."
        )
    if any(int(size) <= 0 for size in mask.shape):
        raise _error(f"{name} dimensions must all be greater than zero, received {_shape(mask)}.")
    _validate_storage(mask, name)
    _check_finite(mask, name)


def _resolve_video_mask(av_latent: Mapping[str, Any], video: torch.Tensor) -> tuple[torch.Tensor, str]:
    incoming = av_latent.get("noise_mask")
    if incoming is None:
        return torch.ones_like(video), "generated ones_like(video)"

    nested_tensor_type = _official_nested_tensor_type()
    if type(incoming) is not nested_tensor_type:
        raise _error("Incoming AV 'noise_mask' must be the official two-stream NestedTensor or None.")
    masks = incoming.unbind()
    if len(masks) != 2:
        raise _error(f"Incoming AV noise_mask must contain exactly 2 streams, received {len(masks)}.")
    video_mask, audio_mask = masks
    _validate_mask_tensor(video_mask, "Incoming video noise mask", rank=5, target_batch=int(video.shape[0]))
    _validate_mask_tensor(audio_mask, "Incoming audio noise mask", rank=4, target_batch=int(video.shape[0]))
    return video_mask, "preserved"


def _fit_audio_drive(audio_drive: torch.Tensor, template_audio: torch.Tensor) -> tuple[torch.Tensor, AudioDriveFit]:
    input_shape = tuple(int(value) for value in audio_drive.shape)
    target_batch = int(template_audio.shape[0])
    input_batch = int(audio_drive.shape[0])
    input_t = int(audio_drive.shape[-1])
    target_t = int(template_audio.shape[-1])

    if input_batch not in {1, target_batch}:
        raise _error(
            f"Audio Drive latent batch size {input_batch} cannot be matched to AV batch size {target_batch}.\n"
            f"Expected batch size 1 or {target_batch}."
        )

    original_device = audio_drive.device
    original_dtype = audio_drive.dtype
    fitted = audio_drive.to(device=template_audio.device, dtype=template_audio.dtype)
    device_fit = "unchanged" if original_device == template_audio.device else f"{original_device} -> {template_audio.device}"
    dtype_fit = "unchanged" if original_dtype == template_audio.dtype else f"{original_dtype} -> {template_audio.dtype}"

    if input_batch == target_batch:
        batch_fit = "unchanged"
    elif input_batch == 1:
        fitted = fitted.expand(target_batch, -1, -1, -1)
        batch_fit = f"replicated 1 -> {target_batch}"

    if input_t == target_t:
        time_fit = "unchanged"
    elif input_t > target_t:
        fitted = fitted[..., :target_t]
        time_fit = f"trimmed {input_t} -> {target_t}"
    else:
        pad = fitted.new_zeros((*fitted.shape[:-1], target_t - input_t))
        fitted = torch.cat((fitted, pad), dim=-1)
        time_fit = f"padded {input_t} -> {target_t}"

    _check_finite(fitted, "Fitted Audio Drive latent")
    return fitted, AudioDriveFit(
        input_shape=input_shape,
        output_shape=tuple(int(value) for value in fitted.shape),
        time_fit=time_fit,
        batch_fit=batch_fit,
        device_fit=device_fit,
        dtype_fit=dtype_fit,
    )


def _format_status(
    video: torch.Tensor,
    template_audio: torch.Tensor,
    fitted_audio: torch.Tensor,
    fit: AudioDriveFit,
    video_mask_status: str,
) -> str:
    return "\n".join(
        (
            "JR MiniMax H3 Audio Driven Latent Builder",
            "",
            "Status: READY",
            "",
            "Video:",
            f"  Shape: {_shape(video)}",
            f"  Noise Mask: {video_mask_status}",
            "",
            "Template Audio:",
            f"  Shape: {_shape(template_audio)}",
            "",
            "Audio Drive:",
            f"  Input Shape: {_shape(fit.input_shape)}",
            f"  Output Shape: {_shape(fit.output_shape)}",
            f"  Time Fit: {fit.time_fit}",
            f"  Batch Fit: {fit.batch_fit}",
            f"  Device Fit: {fit.device_fit}",
            f"  Dtype Fit: {fit.dtype_fit}",
            f"  Device: {fitted_audio.device}",
            f"  Dtype: {fitted_audio.dtype}",
            "",
            "Drive Mode:",
            "  Video: GENERATE",
            "  Audio: LOCKED",
            "  Audio Noise Mask: 0.0",
            "",
            "Output:",
            "  H3 AV Latent: VALID",
        )
    )


def build_h3_audio_driven_latent(
    av_latent: Any,
    audio_drive_latent: Any,
) -> tuple[dict[str, Any], str]:
    """Replace the H3 audio stream and lock it while preserving video semantics."""

    video, template_audio = _extract_av_streams(av_latent)
    audio_drive = _extract_audio_drive(audio_drive_latent)
    video_mask, video_mask_status = _resolve_video_mask(av_latent, video)
    fitted_audio, fit = _fit_audio_drive(audio_drive, template_audio)

    nested_tensor_type = _official_nested_tensor_type()
    audio_mask = torch.zeros_like(fitted_audio)
    output = dict(av_latent)
    output["samples"] = nested_tensor_type((video, fitted_audio))
    output["noise_mask"] = nested_tensor_type((video_mask, audio_mask))
    status = _format_status(video, template_audio, fitted_audio, fit, video_mask_status)
    return output, status


__all__ = [
    "ERROR_PREFIX",
    "AudioDriveFit",
    "H3AudioDrivenLatentBuilderError",
    "build_h3_audio_driven_latent",
]
