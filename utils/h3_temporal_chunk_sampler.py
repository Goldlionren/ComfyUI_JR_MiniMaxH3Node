"""Sequential temporal chunk sampling for official MiniMax H3 AV latents.

The sampler deliberately delegates each chunk to ComfyUI's native
SamplerCustomAdvanced implementation.  This module only owns H3 timeline
planning, stream slicing, CPU preallocation, and lifecycle control.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

ERROR_PREFIX = "JR MiniMax H3 Temporal Chunk Sampler:"
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_STREAM_CHANNELS = 2
H3_FPS = 24
AUDIO_LATENT_FPS = 40
AUDIO_TEMPORAL_TOLERANCE = 1
FRAMES_PER_VIDEO_TOKEN = (1, 4, 4, 4, 4)
FRAMES_PER_CYCLE = sum(FRAMES_PER_VIDEO_TOKEN)
VIDEO_TOKENS_PER_CYCLE = len(FRAMES_PER_VIDEO_TOKEN)
UINT64_MASK = (1 << 64) - 1


class H3TemporalChunkSamplerError(ValueError):
    """Raised when an input cannot be sampled as an H3 AV timeline."""


@dataclass(frozen=True, slots=True)
class H3TemporalChunk:
    """One half-open, time-aligned video/audio latent interval."""

    index: int
    video_start: int
    video_end: int
    audio_start: int
    audio_end: int
    frame_start: int
    frame_end: int

    @property
    def video_tokens(self) -> int:
        return self.video_end - self.video_start

    @property
    def audio_tokens(self) -> int:
        return self.audio_end - self.audio_start

    @property
    def frames(self) -> int:
        return self.frame_end - self.frame_start


@dataclass(frozen=True, slots=True)
class H3TemporalChunkPlan:
    """Validated full timeline and its sequential sampling chunks."""

    frame_count: int
    video_latent_t: int
    audio_latent_t: int
    expected_audio_t: int
    audio_delta: int
    requested_chunk_seconds: float
    target_video_tokens: int
    chunks: tuple[H3TemporalChunk, ...]


SampleChunk = Callable[..., Mapping[str, Any]]
NestedFactory = Callable[[tuple[torch.Tensor, torch.Tensor]], Any]
CleanupCallback = Callable[[], None]
ChunkNoiseFactory = Callable[[H3TemporalChunk], Any]


def _error(message: str) -> H3TemporalChunkSamplerError:
    return H3TemporalChunkSamplerError(f"{ERROR_PREFIX}\n{message}")


def _shape(tensor: torch.Tensor) -> str:
    return "[" + ",".join(str(int(value)) for value in tensor.shape) + "]"


def frame_boundary_for_video_token(video_token_index: int) -> int:
    """Map a global H3 video-token boundary to its 24-fps frame boundary."""

    if isinstance(video_token_index, bool) or not isinstance(video_token_index, int):
        raise TypeError("video_token_index must be an integer")
    if video_token_index < 0:
        raise ValueError("video_token_index must be non-negative")
    cycles, remainder = divmod(video_token_index, VIDEO_TOKENS_PER_CYCLE)
    return cycles * FRAMES_PER_CYCLE + sum(FRAMES_PER_VIDEO_TOKEN[:remainder])


def derive_chunk_seed(base_seed: int, frame_start: int) -> int:
    """Derive a stable uint64 seed from the base seed and absolute H3 frame start.

    SplitMix64's finalizer is a permutation over uint64.  For one base seed,
    distinct frame starts therefore produce distinct derived seeds without
    addition overflow or dependence on Python's randomized hash function.
    """

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or not 0 <= base_seed <= UINT64_MASK:
        raise _error(f"standard RandomNoise seed must be an integer in [0, {UINT64_MASK}].")
    if isinstance(frame_start, bool) or not isinstance(frame_start, int) or not 0 <= frame_start <= UINT64_MASK:
        raise _error(f"chunk frame_start must be an integer in [0, {UINT64_MASK}].")

    mixed = (frame_start + 0x9E3779B97F4A7C15) & UINT64_MASK
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    mixed ^= mixed >> 31
    return (base_seed ^ mixed) & UINT64_MASK


def _validate_full_timeline(video_t: int, audio_t: int) -> tuple[int, int, int]:
    if video_t < 2 or (video_t - 2) % VIDEO_TOKENS_PER_CYCLE != 0:
        raise _error(
            "video stream has an invalid H3 temporal grid. Expected T_video = 5k + 2 "
            f"(official 17k + 5 frame grid), received T_video={video_t}."
        )
    frame_count = frame_boundary_for_video_token(video_t)
    expected_audio_t = round(frame_count * AUDIO_LATENT_FPS / H3_FPS)
    audio_delta = audio_t - expected_audio_t
    if abs(audio_delta) > AUDIO_TEMPORAL_TOLERANCE:
        raise _error(
            "video/audio temporal mismatch.\n"
            f"video: T={video_t}, {frame_count} frames at {H3_FPS} fps\n"
            f"audio: T={audio_t}\n"
            f"expected audio T={expected_audio_t} ± {AUDIO_TEMPORAL_TOLERANCE}."
        )
    return frame_count, expected_audio_t, audio_delta


def plan_h3_temporal_chunks(
    video_t: int,
    audio_t: int,
    chunk_duration_seconds: float,
) -> H3TemporalChunkPlan:
    """Create deterministic, non-overlapping H3 AV temporal chunks.

    Internal video boundaries are aligned to complete five-token / seventeen-frame
    cycles.  Audio boundaries are derived from the same global frame boundary,
    not from the video's latent-token count.  The final audio boundary is forced
    to the actual validated stream length so a permitted encoder ±1 tick is kept.
    """

    if isinstance(chunk_duration_seconds, bool) or not isinstance(chunk_duration_seconds, (int, float)):
        raise _error("chunk_duration_seconds must be a finite positive number.")
    requested_seconds = float(chunk_duration_seconds)
    if not math.isfinite(requested_seconds) or requested_seconds <= 0:
        raise _error("chunk_duration_seconds must be a finite positive number.")
    if isinstance(video_t, bool) or not isinstance(video_t, int) or video_t <= 0:
        raise _error("video temporal length must be a positive integer.")
    if isinstance(audio_t, bool) or not isinstance(audio_t, int) or audio_t <= 0:
        raise _error("audio temporal length must be a positive integer.")

    frame_count, expected_audio_t, audio_delta = _validate_full_timeline(video_t, audio_t)
    cycles_per_chunk = max(1, math.floor(requested_seconds * H3_FPS / FRAMES_PER_CYCLE))
    target_video_tokens = cycles_per_chunk * VIDEO_TOKENS_PER_CYCLE

    video_boundaries = [0]
    cursor = 0
    minimum_tail_tokens = max(VIDEO_TOKENS_PER_CYCLE, target_video_tokens // 2)
    while video_t - cursor - 2 > target_video_tokens:
        tail_after_split = video_t - cursor - target_video_tokens - 2
        if tail_after_split < minimum_tail_tokens:
            break
        cursor += target_video_tokens
        video_boundaries.append(cursor)
    video_boundaries.append(video_t)

    chunks: list[H3TemporalChunk] = []
    for index, (video_start, video_end) in enumerate(zip(video_boundaries, video_boundaries[1:])):
        frame_start = frame_boundary_for_video_token(video_start)
        frame_end = frame_boundary_for_video_token(video_end)
        audio_start = round(frame_start * AUDIO_LATENT_FPS / H3_FPS)
        audio_end = audio_t if video_end == video_t else round(frame_end * AUDIO_LATENT_FPS / H3_FPS)
        if video_end <= video_start or audio_end <= audio_start or frame_end <= frame_start:
            raise _error("the requested chunk size produced an empty temporal interval.")
        chunks.append(
            H3TemporalChunk(
                index=index,
                video_start=video_start,
                video_end=video_end,
                audio_start=audio_start,
                audio_end=audio_end,
                frame_start=frame_start,
                frame_end=frame_end,
            )
        )

    return H3TemporalChunkPlan(
        frame_count=frame_count,
        video_latent_t=video_t,
        audio_latent_t=audio_t,
        expected_audio_t=expected_audio_t,
        audio_delta=audio_delta,
        requested_chunk_seconds=requested_seconds,
        target_video_tokens=target_video_tokens,
        chunks=tuple(chunks),
    )


def _extract_h3_streams(latent_image: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(latent_image, Mapping):
        raise _error("latent_image must be a LATENT mapping containing 'samples'.")
    if "samples" not in latent_image:
        raise _error("latent_image is missing required 'samples'.")
    samples = latent_image["samples"]
    if not getattr(samples, "is_nested", False) or not callable(getattr(samples, "unbind", None)):
        raise _error("latent_image must contain the official two-stream H3 NestedTensor.")
    streams = list(samples.unbind())
    if len(streams) != 2 or not all(isinstance(stream, torch.Tensor) for stream in streams):
        raise _error("latent_image must contain exactly two tensor streams: video and audio.")
    video, audio = streams
    if video.ndim != 5 or video.shape[1] != VIDEO_CHANNELS:
        raise _error(f"video stream must have shape [B,24,T,H,W], received {_shape(video)}.")
    if audio.ndim != 4 or audio.shape[1] != AUDIO_CHANNELS or audio.shape[2] != AUDIO_STREAM_CHANNELS:
        raise _error(f"audio stream must have shape [B,32,2,T], received {_shape(audio)}.")
    if video.shape[0] <= 0 or video.shape[2] <= 0 or video.shape[3] <= 0 or video.shape[4] <= 0:
        raise _error("video stream dimensions must all be non-zero.")
    if audio.shape[0] <= 0 or audio.shape[3] <= 0:
        raise _error("audio stream dimensions must all be non-zero.")
    for name, stream in (("video", video), ("audio", audio)):
        if stream.layout != torch.strided or stream.device.type == "meta" or not stream.is_floating_point():
            raise _error(f"{name} stream must be a materialized, strided floating-point tensor.")
    if video.shape[0] != audio.shape[0]:
        raise _error(f"video/audio batch mismatch: {video.shape[0]} != {audio.shape[0]}.")
    if video.dtype != audio.dtype:
        raise _error(f"video/audio dtype mismatch: {video.dtype} != {audio.dtype}.")
    if video.device != audio.device:
        raise _error(f"video/audio device mismatch: {video.device} != {audio.device}.")
    for name, stream, temporal_dim in (("video", video, 2), ("audio", audio, 3)):
        try:
            finite = True
            for start in range(0, int(stream.shape[temporal_dim]), 64):
                temporal_slice = stream.narrow(
                    temporal_dim,
                    start,
                    min(64, int(stream.shape[temporal_dim]) - start),
                )
                if not bool(torch.isfinite(temporal_slice).all().item()):
                    finite = False
                    break
        except (RuntimeError, TypeError) as error:
            raise _error(f"{name} stream could not be checked for finite values: {type(error).__name__}.") from None
        if not finite:
            raise _error(f"{name} stream contains NaN or Inf values.")
    if "noise_mask" in latent_image:
        raise _error(
            "noise_mask is not supported in phase 1 because its temporal mapping across the packed H3 AV streams "
            "cannot be inferred safely. Remove the mask or use the native monolithic sampler."
        )
    return video, audio


def _guider_has_temporal_keyframes(guider: Any) -> bool:
    """Inspect the public current-Comfy guider condition store without mutating it."""

    original_conds = getattr(guider, "original_conds", None)
    if not isinstance(original_conds, Mapping):
        return False
    for cond_group in original_conds.values():
        if not isinstance(cond_group, (tuple, list)):
            continue
        for cond in cond_group:
            if isinstance(cond, Mapping) and cond.get("minimax_keyframes"):
                return True
    return False


def _resolve_chunk_noise(noise: Any, plan: H3TemporalChunkPlan) -> tuple[ChunkNoiseFactory, str]:
    """Select a safe native NOISE strategy without inspecting custom internals."""

    if len(plan.chunks) == 1:
        return lambda _chunk: noise, "native_single"

    try:
        from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
    except ImportError:
        raise RuntimeError(
            f"{ERROR_PREFIX}\nThe installed ComfyUI does not provide its standard NOISE implementations."
        ) from None

    if type(noise) is Noise_RandomNoise:
        base_seed = noise.seed

        def chunk_random_noise(chunk: H3TemporalChunk):
            return Noise_RandomNoise(derive_chunk_seed(base_seed, chunk.frame_start))

        return chunk_random_noise, "chunk_derived"
    if type(noise) is Noise_EmptyNoise:
        return lambda _chunk: noise, "native_zero"

    raise _error(
        "multi-chunk sampling cannot safely derive temporal substreams for this generic/custom NOISE object. "
        "The current ComfyUI NOISE contract exposes no standard clone, seed-derivation, offset or substream API. "
        "Use the official RandomNoise or DisableNoise node, or use the native monolithic sampler."
    )


def _make_official_nested(streams: tuple[torch.Tensor, torch.Tensor]):
    try:
        from comfy.nested_tensor import NestedTensor
    except ImportError:
        raise RuntimeError(
            f"{ERROR_PREFIX}\nThe installed ComfyUI does not provide comfy.nested_tensor.NestedTensor."
        ) from None
    return NestedTensor(streams)


def _sample_with_native_advanced_sampler(*, noise, guider, sampler, sigmas, latent_image):
    try:
        from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
    except ImportError:
        raise RuntimeError(
            f"{ERROR_PREFIX}\nThe installed ComfyUI does not provide SamplerCustomAdvanced."
        ) from None

    node_output = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent_image)
    values = node_output.result if hasattr(node_output, "result") else node_output
    if not isinstance(values, (tuple, list)) or not values or not isinstance(values[0], Mapping):
        raise RuntimeError(f"{ERROR_PREFIX}\nSamplerCustomAdvanced returned an unexpected result.")
    sampled_latent = values[0]
    del values, node_output
    return sampled_latent


def _default_cleanup() -> None:
    from comfy.model_management import soft_empty_cache

    soft_empty_cache()


def _validate_sampled_chunk(
    sampled_latent: Any,
    source_video: torch.Tensor,
    source_audio: torch.Tensor,
    chunk: H3TemporalChunk,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(sampled_latent, Mapping) or "samples" not in sampled_latent:
        raise _error(f"native sampler returned an invalid LATENT for chunk {chunk.index + 1}.")
    samples = sampled_latent["samples"]
    if not getattr(samples, "is_nested", False) or not callable(getattr(samples, "unbind", None)):
        raise _error(f"native sampler did not return a two-stream NestedTensor for chunk {chunk.index + 1}.")
    streams = list(samples.unbind())
    if len(streams) != 2 or not all(isinstance(stream, torch.Tensor) for stream in streams):
        raise _error(f"native sampler returned the wrong stream count for chunk {chunk.index + 1}.")
    video, audio = streams
    expected_video_shape = list(source_video.shape)
    expected_video_shape[2] = chunk.video_tokens
    expected_audio_shape = list(source_audio.shape)
    expected_audio_shape[3] = chunk.audio_tokens
    if list(video.shape) != expected_video_shape:
        raise _error(
            f"native sampler changed the video shape for chunk {chunk.index + 1}: "
            f"expected {expected_video_shape}, received {list(video.shape)}."
        )
    if list(audio.shape) != expected_audio_shape:
        raise _error(
            f"native sampler changed the audio shape for chunk {chunk.index + 1}: "
            f"expected {expected_audio_shape}, received {list(audio.shape)}."
        )
    if not video.is_floating_point() or not audio.is_floating_point():
        raise _error(f"native sampler returned a non-floating tensor for chunk {chunk.index + 1}.")
    return video, audio


def _format_status(
    plan: H3TemporalChunkPlan,
    source_device: torch.device,
    output_video: torch.Tensor,
    noise_mode: str,
) -> str:
    chunk_ranges = ", ".join(
        f"#{chunk.index + 1} v[{chunk.video_start}:{chunk.video_end}] "
        f"a[{chunk.audio_start}:{chunk.audio_end}]"
        for chunk in plan.chunks
    )
    return "\n".join(
        (
            "Success: sequential native temporal chunk sampling",
            f"timeline: {plan.frame_count} frames @ {H3_FPS} fps ({plan.frame_count / H3_FPS:.3f}s)",
            f"chunks: {len(plan.chunks)} (requested {plan.requested_chunk_seconds:g}s)",
            f"source device: {source_device}",
            f"output: CPU preallocated, dtype={output_video.dtype}",
            f"noise_mode={noise_mode}",
            f"audio timeline delta: {plan.audio_delta:+d} tick(s)",
            f"ranges: {chunk_ranges}",
            "phase 1: no overlap, no temporal hidden-state carry, no global position offset",
        )
    )


def sample_h3_temporal_chunks(
    *,
    noise: Any,
    guider: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    latent_image: Mapping[str, Any],
    chunk_duration_seconds: float,
    aggressive_memory_cleanup: bool = False,
    sample_chunk: SampleChunk | None = None,
    nested_factory: NestedFactory | None = None,
    cleanup: CleanupCallback | None = None,
) -> tuple[dict[str, Any], str]:
    """Sample one H3 AV temporal chunk at a time and reassemble on CPU."""

    video, audio = _extract_h3_streams(latent_image)
    plan = plan_h3_temporal_chunks(int(video.shape[2]), int(audio.shape[3]), chunk_duration_seconds)
    if len(plan.chunks) > 1 and _guider_has_temporal_keyframes(guider):
        raise _error(
            "multi-chunk sampling cannot safely consume minimax_keyframes. Current H3 keyframes store absolute "
            "full-timeline frame indices, while the native sampler exposes no public per-chunk position-offset "
            "contract. Use text/reference conditioning without keyframes or the native monolithic sampler."
        )
    chunk_noise_factory, noise_mode = _resolve_chunk_noise(noise, plan)
    sample_chunk = sample_chunk or _sample_with_native_advanced_sampler
    nested_factory = nested_factory or _make_official_nested
    cleanup = cleanup or _default_cleanup

    output_video: torch.Tensor | None = None
    output_audio: torch.Tensor | None = None

    for chunk in plan.chunks:
        chunk_noise = chunk_noise_factory(chunk)
        chunk_video = video[:, :, chunk.video_start : chunk.video_end, :, :]
        chunk_audio = audio[:, :, :, chunk.audio_start : chunk.audio_end]
        chunk_latent = dict(latent_image)
        chunk_latent["samples"] = nested_factory((chunk_video, chunk_audio))

        sampled_latent = sample_chunk(
            noise=chunk_noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=chunk_latent,
        )
        sampled_video, sampled_audio = _validate_sampled_chunk(sampled_latent, video, audio, chunk)

        if output_video is None:
            output_video_shape = list(video.shape)
            output_audio_shape = list(audio.shape)
            output_video = torch.empty(output_video_shape, dtype=sampled_video.dtype, device="cpu")
            output_audio = torch.empty(output_audio_shape, dtype=sampled_audio.dtype, device="cpu")

        output_video[:, :, chunk.video_start : chunk.video_end, :, :].copy_(sampled_video)
        output_audio[:, :, :, chunk.audio_start : chunk.audio_end].copy_(sampled_audio)

        del sampled_video, sampled_audio, sampled_latent
        del chunk_latent, chunk_video, chunk_audio, chunk_noise
        if aggressive_memory_cleanup:
            gc.collect()
            cleanup()

    if output_video is None or output_audio is None:  # defensive; a valid H3 timeline always has at least one chunk
        raise _error("no temporal chunks were produced.")

    output = dict(latent_image)
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = nested_factory((output_video, output_audio))
    return output, _format_status(plan, video.device, output_video, noise_mode)


__all__ = [
    "AUDIO_LATENT_FPS",
    "ERROR_PREFIX",
    "FRAMES_PER_VIDEO_TOKEN",
    "H3_FPS",
    "H3TemporalChunk",
    "H3TemporalChunkPlan",
    "H3TemporalChunkSamplerError",
    "derive_chunk_seed",
    "frame_boundary_for_video_token",
    "plan_h3_temporal_chunks",
    "sample_h3_temporal_chunks",
]
