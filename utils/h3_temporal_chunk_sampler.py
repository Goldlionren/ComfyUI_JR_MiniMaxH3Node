"""Sequential temporal chunk sampling for official MiniMax H3 AV latents.

The sampler deliberately delegates each chunk to ComfyUI's native
SamplerCustomAdvanced implementation.  This module only owns H3 timeline
planning, stream slicing, CPU preallocation, and lifecycle control.
"""

from __future__ import annotations

import gc
import math
import sys
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
LEGACY_INDEPENDENT_MODE = "Legacy Independent Chunks"
HARD_AV_PREFIX_MODE = "Hard AV Latent Prefix"
CONTINUITY_MODES = (HARD_AV_PREFIX_MODE, LEGACY_INDEPENDENT_MODE)
HARD_WINDOW_FRAMES = 345
HARD_VIDEO_WINDOW_T = 102
HARD_AUDIO_WINDOW_T = 575
HARD_OVERLAP_FRAMES = 39
HARD_VIDEO_PREFIX_T = 12
HARD_AUDIO_PREFIX_T = 65
HARD_STRIDE_FRAMES = 306
HARD_VIDEO_FRESH_T = 90
HARD_AUDIO_FRESH_T = 510
HARD_CHUNK_SECONDS = HARD_WINDOW_FRAMES / H3_FPS
HARD_CHUNK_PRESETS = (
    ("5.875s / 141 frames / 235 ticks", 141, 42, 235),
    ("8.000s / 192 frames / 320 ticks", 192, 57, 320),
    ("10.125s / 243 frames / 405 ticks", 243, 72, 405),
    ("14.375s / 345 frames / 575 ticks", 345, 102, 575),
)
HARD_CHUNK_PRESET_LABELS = tuple(profile[0] for profile in HARD_CHUNK_PRESETS)
DEFAULT_HARD_CHUNK_PRESET = HARD_CHUNK_PRESET_LABELS[0]
_HARD_PROFILE_BY_LABEL = {profile[0]: profile[1:] for profile in HARD_CHUNK_PRESETS}
TEMPORAL_MODE_A = "A - Legacy No Overlap"
TEMPORAL_MODE_B = "B - Exact H3 Source Overlap"
TEMPORAL_MODE_C = "C - Exact H3 Refined Overlap"
TEMPORAL_MODES = (TEMPORAL_MODE_A, TEMPORAL_MODE_B, TEMPORAL_MODE_C)
MIN_EXACT_WINDOW_INDEX = 2
MAX_EXACT_WINDOW_INDEX = 6


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


@dataclass(frozen=True, slots=True)
class H3TemporalOverlapWindow:
    """One exact local H3 AV sampling window and its non-duplicated keep range."""

    sample: H3TemporalChunk
    keep_video_start: int
    keep_video_end: int
    keep_audio_start: int
    keep_audio_end: int

    @property
    def local_keep_video_start(self) -> int:
        return self.keep_video_start - self.sample.video_start

    @property
    def local_keep_audio_start(self) -> int:
        return self.keep_audio_start - self.sample.audio_start


@dataclass(frozen=True, slots=True)
class H3TemporalOverlapPlan:
    """Validated exact-window overlap plan for the controlled B/C experiment."""

    frame_count: int
    video_latent_t: int
    audio_latent_t: int
    expected_audio_t: int
    audio_delta: int
    requested_chunk_seconds: float
    stride_video_tokens: int
    stride_frames: int
    window_video_tokens: int
    window_frames: int
    window_audio_tokens: int
    overlap_video_tokens: int
    overlap_frames: int
    windows: tuple[H3TemporalOverlapWindow, ...]


@dataclass(frozen=True, slots=True)
class H3HardAVPrefixPlan:
    """Exact selectable-profile H3 windows and their non-duplicated global keep ranges."""

    frame_count: int
    video_latent_t: int
    audio_latent_t: int
    preset_label: str
    window_frames: int
    window_video_t: int
    window_audio_t: int
    stride_frames: int
    fresh_video_t: int
    fresh_audio_t: int
    windows: tuple[H3TemporalOverlapWindow, ...]


SampleChunk = Callable[..., Mapping[str, Any]]
NestedFactory = Callable[[tuple[torch.Tensor, torch.Tensor]], Any]
CleanupCallback = Callable[[], None]
ChunkNoiseFactory = Callable[[H3TemporalChunk], Any]
BuildGuider = Callable[[Any, Any], Any]
ApplyGuide = Callable[..., Any]
DecodeLastFrame = Callable[[Any, torch.Tensor], torch.Tensor]


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


def plan_h3_hard_av_prefix_windows(
    video_t: int,
    audio_t: int,
    hard_chunk_preset: str,
) -> H3HardAVPrefixPlan:
    """Plan one of the exact hard-prefix AV profiles selected by its stable label."""

    if not isinstance(hard_chunk_preset, str) or hard_chunk_preset not in _HARD_PROFILE_BY_LABEL:
        raise _error(
            f"hard_chunk_preset must be one of {list(HARD_CHUNK_PRESET_LABELS)}, "
            f"received {hard_chunk_preset!r}."
        )
    if isinstance(video_t, bool) or not isinstance(video_t, int) or video_t <= 0:
        raise _error("video temporal length must be a positive integer.")
    if isinstance(audio_t, bool) or not isinstance(audio_t, int) or audio_t <= 0:
        raise _error("audio temporal length must be a positive integer.")

    window_frames, window_video_t, window_audio_t = _HARD_PROFILE_BY_LABEL[hard_chunk_preset]
    stride_frames = window_frames - HARD_OVERLAP_FRAMES
    fresh_video_t = window_video_t - HARD_VIDEO_PREFIX_T
    fresh_audio_t = window_audio_t - HARD_AUDIO_PREFIX_T
    if (
        frame_boundary_for_video_token(window_video_t) != window_frames
        or frame_boundary_for_video_token(HARD_VIDEO_PREFIX_T) != HARD_OVERLAP_FRAMES
        or frame_boundary_for_video_token(fresh_video_t) != stride_frames
        or round(window_frames * AUDIO_LATENT_FPS / H3_FPS) != window_audio_t
        or round(HARD_OVERLAP_FRAMES * AUDIO_LATENT_FPS / H3_FPS) != HARD_AUDIO_PREFIX_T
        or round(stride_frames * AUDIO_LATENT_FPS / H3_FPS) != fresh_audio_t
    ):
        raise _error(f"hard_chunk_preset {hard_chunk_preset!r} is internally inconsistent with the H3 AV grid.")

    frame_count, _expected_audio_t, _audio_delta = _validate_full_timeline(video_t, audio_t)
    remaining_video_t = max(0, video_t - window_video_t)
    remaining_audio_t = max(0, audio_t - window_audio_t)
    video_window_count = (remaining_video_t + fresh_video_t - 1) // fresh_video_t + 1
    audio_window_count = (remaining_audio_t + fresh_audio_t - 1) // fresh_audio_t + 1
    if video_window_count != audio_window_count:
        raise _error(
            f"{HARD_AV_PREFIX_MODE} preset {hard_chunk_preset!r} cannot cover the permitted video/audio boundary "
            f"delta with the same number of windows: video needs {video_window_count}, audio needs "
            f"{audio_window_count}."
        )
    window_count = video_window_count

    windows: list[H3TemporalOverlapWindow] = []
    committed_video_end = 0
    committed_audio_end = 0
    for index in range(window_count):
        video_start = index * fresh_video_t
        audio_start = index * fresh_audio_t
        frame_start = index * stride_frames
        video_end = video_start + window_video_t
        audio_end = audio_start + window_audio_t
        frame_end = frame_start + window_frames
        keep_video_start = video_start if index == 0 else video_start + HARD_VIDEO_PREFIX_T
        keep_audio_start = audio_start if index == 0 else audio_start + HARD_AUDIO_PREFIX_T
        keep_video_end = min(video_end, video_t)
        keep_audio_end = min(audio_end, audio_t)
        if keep_video_start != committed_video_end or keep_audio_start != committed_audio_end:
            raise _error("hard-prefix windows would create an AV gap or duplicate before the final padded window.")
        if keep_video_end <= keep_video_start or keep_audio_end <= keep_audio_start:
            raise _error("hard-prefix planning produced an empty fresh AV range.")
        windows.append(
            H3TemporalOverlapWindow(
                sample=H3TemporalChunk(
                    index=index,
                    video_start=video_start,
                    video_end=video_end,
                    audio_start=audio_start,
                    audio_end=audio_end,
                    frame_start=frame_start,
                    frame_end=frame_end,
                ),
                keep_video_start=keep_video_start,
                keep_video_end=keep_video_end,
                keep_audio_start=keep_audio_start,
                keep_audio_end=keep_audio_end,
            )
        )
        committed_video_end = keep_video_end
        committed_audio_end = keep_audio_end

    if committed_video_end != video_t or committed_audio_end != audio_t:
        raise _error("fixed hard-prefix windows did not cover the complete global AV timeline.")
    return H3HardAVPrefixPlan(
        frame_count=frame_count,
        video_latent_t=video_t,
        audio_latent_t=audio_t,
        preset_label=hard_chunk_preset,
        window_frames=window_frames,
        window_video_t=window_video_t,
        window_audio_t=window_audio_t,
        stride_frames=stride_frames,
        fresh_video_t=fresh_video_t,
        fresh_audio_t=fresh_audio_t,
        windows=tuple(windows),
    )


def _exact_h3_av_window_for_stride(stride_video_tokens: int) -> tuple[int, int, int]:
    """Return the smallest conservative, audio-exact H3 window above a stride.

    Exact local windows follow ``T=15n+12``, ``frames=51n+39`` and
    ``audio=85n+65``.  ``n=2..6`` is the subset inside the installed native
    H3 node's documented approximately 124--362 frame trained range.
    """

    window_index = max(
        MIN_EXACT_WINDOW_INDEX,
        math.ceil((stride_video_tokens - 11) / 15),
    )
    if window_index > MAX_EXACT_WINDOW_INDEX:
        raise _error(
            "B/C exact-overlap mode cannot construct a non-zero-overlap local window inside the "
            "conservative H3 trained range (maximum 102 video tokens / 345 frames / 14.375s). "
            "Choose a smaller chunk_duration_seconds or use A - Legacy No Overlap."
        )
    window_video_tokens = 15 * window_index + 12
    window_frames = 51 * window_index + 39
    window_audio_tokens = 85 * window_index + 65
    if window_video_tokens <= stride_video_tokens:
        raise _error("B/C exact-overlap planning produced a zero-overlap window.")
    return window_video_tokens, window_frames, window_audio_tokens


def plan_h3_temporal_overlap_windows(
    video_t: int,
    audio_t: int,
    chunk_duration_seconds: float,
) -> H3TemporalOverlapPlan:
    """Plan globally aligned exact H3 AV windows for modes B and C."""

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
    cycles_per_stride = max(1, math.floor(requested_seconds * H3_FPS / FRAMES_PER_CYCLE))
    stride_video_tokens = cycles_per_stride * VIDEO_TOKENS_PER_CYCLE
    stride_frames = cycles_per_stride * FRAMES_PER_CYCLE
    window_video_tokens, window_frames, window_audio_tokens = _exact_h3_av_window_for_stride(
        stride_video_tokens
    )
    if video_t <= window_video_tokens:
        raise _error(
            "B/C exact-overlap mode requires a global timeline longer than its exact local window "
            f"({window_video_tokens} video tokens / {window_frames} frames) so a real overlap seam exists. "
            "Use A - Legacy No Overlap for this short timeline."
        )

    final_start = video_t - window_video_tokens
    if final_start < 0 or final_start % VIDEO_TOKENS_PER_CYCLE != 0:
        raise _error("the final exact H3 window cannot be aligned to the global token lattice.")
    starts = [0]
    while starts[-1] + window_video_tokens < video_t:
        regular_start = starts[-1] + stride_video_tokens
        next_start = min(regular_start, final_start)
        if next_start <= starts[-1]:
            raise _error("B/C exact-overlap planning could not advance the global timeline safely.")
        starts.append(next_start)

    windows: list[H3TemporalOverlapWindow] = []
    committed_video_end = 0
    committed_audio_end = 0
    for index, video_start in enumerate(starts):
        video_end = video_start + window_video_tokens
        frame_start = frame_boundary_for_video_token(video_start)
        frame_end = frame_boundary_for_video_token(video_end)
        audio_start = round(frame_start * AUDIO_LATENT_FPS / H3_FPS)
        audio_end = audio_t if video_end == video_t else round(frame_end * AUDIO_LATENT_FPS / H3_FPS)
        keep_video_start = committed_video_end
        keep_video_end = video_end
        keep_audio_start = committed_audio_end
        keep_audio_end = audio_end
        if video_start % VIDEO_TOKENS_PER_CYCLE != 0:
            raise _error("an exact H3 sampling window start is not on a five-token cycle boundary.")
        if frame_end - frame_start != window_frames:
            raise _error("an exact H3 sampling window does not preserve the expected frame length.")
        expected_window_audio = window_audio_tokens + (audio_delta if video_end == video_t else 0)
        if audio_end - audio_start != expected_window_audio:
            raise _error("an exact H3 sampling window does not preserve the expected audio length.")
        if not video_start <= keep_video_start < keep_video_end:
            raise _error("a B/C keep range would create a video gap or fail to advance.")
        if not audio_start <= keep_audio_start < keep_audio_end:
            raise _error("a B/C keep range would create an audio gap or fail to advance.")
        sample = H3TemporalChunk(
            index=index,
            video_start=video_start,
            video_end=video_end,
            audio_start=audio_start,
            audio_end=audio_end,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        windows.append(
            H3TemporalOverlapWindow(
                sample=sample,
                keep_video_start=keep_video_start,
                keep_video_end=keep_video_end,
                keep_audio_start=keep_audio_start,
                keep_audio_end=keep_audio_end,
            )
        )
        committed_video_end = keep_video_end
        committed_audio_end = keep_audio_end

    if committed_video_end != video_t or committed_audio_end != audio_t:
        raise _error("B/C exact-overlap windows did not cover the complete global AV timeline.")
    overlap_video_tokens = window_video_tokens - stride_video_tokens
    overlap_frames = window_frames - stride_frames
    return H3TemporalOverlapPlan(
        frame_count=frame_count,
        video_latent_t=video_t,
        audio_latent_t=audio_t,
        expected_audio_t=expected_audio_t,
        audio_delta=audio_delta,
        requested_chunk_seconds=requested_seconds,
        stride_video_tokens=stride_video_tokens,
        stride_frames=stride_frames,
        window_video_tokens=window_video_tokens,
        window_frames=window_frames,
        window_audio_tokens=window_audio_tokens,
        overlap_video_tokens=overlap_video_tokens,
        overlap_frames=overlap_frames,
        windows=tuple(windows),
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
    return video, audio


def _reject_legacy_noise_mask(latent_image: Mapping[str, Any]) -> None:
    if "noise_mask" in latent_image:
        raise _error(
            "noise_mask is not supported in the legacy path because its temporal mapping across the packed H3 AV "
            "streams cannot be inferred safely. Remove the mask or use the native monolithic sampler."
        )


def _tensor_is_all_one(tensor: torch.Tensor, temporal_dim: int) -> bool:
    try:
        for start in range(0, int(tensor.shape[temporal_dim]), 64):
            temporal_slice = tensor.narrow(
                temporal_dim,
                start,
                min(64, int(tensor.shape[temporal_dim]) - start),
            )
            if not bool(torch.eq(temporal_slice, 1).all().item()):
                return False
    except (RuntimeError, TypeError):
        return False
    return True


def _validate_trivial_hard_input_mask(
    latent_image: Mapping[str, Any],
    video: torch.Tensor,
    audio: torch.Tensor,
) -> None:
    """Accept only absent/None or exact all-one AV masks before replacing them."""

    if "noise_mask" not in latent_image or latent_image["noise_mask"] is None:
        return
    noise_mask = latent_image["noise_mask"]
    if not getattr(noise_mask, "is_nested", False) or not callable(getattr(noise_mask, "unbind", None)):
        raise _error(
            f"{HARD_AV_PREFIX_MODE} accepts only an official two-stream all-one noise_mask. "
            "Nontrivial or unknown mask semantics fail closed; Audio Driven latents are not supported."
        )
    masks = list(noise_mask.unbind())
    if len(masks) != 2 or not all(isinstance(mask, torch.Tensor) for mask in masks):
        raise _error(f"{HARD_AV_PREFIX_MODE} noise_mask must contain exactly video and audio tensor streams.")
    video_mask, audio_mask = masks
    if list(video_mask.shape) != list(video.shape) or list(audio_mask.shape) != list(audio.shape):
        raise _error(
            f"{HARD_AV_PREFIX_MODE} all-one noise_mask shapes must match the AV samples exactly; "
            f"video mask {_shape(video_mask)} vs {_shape(video)}, audio mask {_shape(audio_mask)} vs {_shape(audio)}."
        )
    if not _tensor_is_all_one(video_mask, 2) or not _tensor_is_all_one(audio_mask, 3):
        raise _error(
            f"{HARD_AV_PREFIX_MODE} refuses an existing nontrivial noise_mask. Remove it and provide a normal "
            "generated H3 AV latent; Audio Driven compatibility belongs to Sequential Audio."
        )


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


def _positive_has_temporal_keyframes(positive: Any) -> bool:
    """Return whether raw CONDITIONING already contains native H3 frame guides."""

    if not isinstance(positive, (tuple, list)):
        return False
    for item in positive:
        if not isinstance(item, (tuple, list)) or len(item) < 2 or not isinstance(item[1], Mapping):
            continue
        if item[1].get("minimax_keyframes"):
            return True
    return False


def _unwrap_noise_node_output(node_output: Any, node_id: str) -> Any:
    values = node_output.result if hasattr(node_output, "result") else node_output
    if not isinstance(values, (tuple, list)) or len(values) != 1:
        raise RuntimeError(f"{ERROR_PREFIX}\nThe registered ComfyUI {node_id} node returned an unexpected result.")
    provider = values[0]
    if not callable(getattr(provider, "generate_noise", None)):
        raise RuntimeError(f"{ERROR_PREFIX}\nThe registered ComfyUI {node_id} node returned an invalid NOISE provider.")
    return provider


def _registered_noise_factory(node_id: str) -> Callable[..., Any] | None:
    """Return the factory from ComfyUI's live node registry, if initialized.

    ComfyUI loads built-in extra nodes through ``load_custom_node`` under a
    path-derived module name.  Importing the same file through its package name
    can therefore create a second, non-identical Python class object.  The live
    registry is authoritative for objects arriving through a workflow.
    """

    comfy_nodes = sys.modules.get("nodes")
    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(mappings, Mapping):
        return None
    node_class = mappings.get(node_id)
    execute = getattr(node_class, "execute", None)
    if not callable(execute):
        return None

    def factory(*args: Any) -> Any:
        return _unwrap_noise_node_output(execute(*args), node_id)

    return factory


def _append_noise_factory(
    candidates: list[tuple[type[Any], Callable[..., Any]]],
    factory: Callable[..., Any] | None,
    *probe_args: Any,
) -> None:
    if factory is None:
        return
    provider = factory(*probe_args)
    provider_type = type(provider)
    if all(existing_type is not provider_type for existing_type, _existing_factory in candidates):
        candidates.append((provider_type, factory))


def _resolve_chunk_noise(noise: Any, plan: H3TemporalChunkPlan) -> tuple[ChunkNoiseFactory, str]:
    """Select a safe native NOISE strategy without inspecting custom internals."""

    if len(plan.chunks) == 1:
        return lambda _chunk: noise, "native_single"

    random_factories: list[tuple[type[Any], Callable[..., Any]]] = []
    empty_factories: list[tuple[type[Any], Callable[..., Any]]] = []
    _append_noise_factory(random_factories, _registered_noise_factory("RandomNoise"), 0)
    _append_noise_factory(empty_factories, _registered_noise_factory("DisableNoise"))

    if not random_factories or not empty_factories:
        try:
            from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
        except ImportError:
            raise RuntimeError(
                f"{ERROR_PREFIX}\nThe installed ComfyUI does not provide its standard NOISE implementations."
            ) from None
        if not random_factories:
            _append_noise_factory(random_factories, Noise_RandomNoise, 0)
        if not empty_factories:
            _append_noise_factory(empty_factories, Noise_EmptyNoise)

    for random_type, random_factory in random_factories:
        if type(noise) is random_type:
            base_seed = noise.seed
            derive_chunk_seed(base_seed, plan.chunks[0].frame_start)

            def chunk_random_noise(chunk: H3TemporalChunk):
                return random_factory(derive_chunk_seed(base_seed, chunk.frame_start))

            return chunk_random_noise, "chunk_derived"
    if any(type(noise) is empty_type for empty_type, _empty_factory in empty_factories):
        return lambda _chunk: noise, "native_zero"

    noise_type = f"{type(noise).__module__}.{type(noise).__qualname__}"
    raise _error(
        f"multi-chunk sampling cannot safely derive temporal substreams for generic/custom NOISE type "
        f"{noise_type!r}. "
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


def _build_native_basic_guider(model: Any, positive: Any) -> Any:
    """Build a new official Basic Guider without mutating a guider from upstream."""

    try:
        from comfy_extras.nodes_custom_sampler import BasicGuider
    except ImportError:
        raise RuntimeError(
            f"{ERROR_PREFIX}\nThe installed ComfyUI does not provide BasicGuider."
        ) from None
    node_output = BasicGuider.execute(model, positive)
    values = node_output.result if hasattr(node_output, "result") else node_output
    if not isinstance(values, (tuple, list)) or len(values) != 1:
        raise RuntimeError(f"{ERROR_PREFIX}\nBasicGuider returned an unexpected result.")
    guider = values[0]
    if not callable(guider):
        raise RuntimeError(f"{ERROR_PREFIX}\nBasicGuider returned an invalid GUIDER.")
    return guider


def _apply_native_previous_frame_guide(
    *,
    positive: Any,
    latent: Mapping[str, Any],
    vae: Any,
    image: torch.Tensor,
) -> Any:
    """Apply the previous decoded terminal frame at local pixel frame zero."""

    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide
    except ImportError:
        raise RuntimeError(
            f"{ERROR_PREFIX}\nCurrent ComfyUI MiniMax H3 guide implementation is unavailable."
        ) from None
    node_output = MiniMaxH3AddGuide.execute(
        positive=positive,
        latent=latent,
        frame_idx=0,
        vae=vae,
        audio_vae=None,
        image=image,
        audio=None,
    )
    values = node_output.result if hasattr(node_output, "result") else node_output
    if not isinstance(values, (tuple, list)) or len(values) != 1:
        raise RuntimeError(f"{ERROR_PREFIX}\nMiniMaxH3AddGuide returned an unexpected result.")
    return values[0]


def _decode_terminal_frame(vae: Any, sampled_video: torch.Tensor) -> torch.Tensor:
    """Decode one complete sampled chunk and retain only its final RGB frame on CPU."""

    decode = getattr(vae, "decode", None)
    if not callable(decode):
        raise _error("vae must provide the standard ComfyUI decode(samples) API.")
    images = decode(sampled_video)
    if not isinstance(images, torch.Tensor) or images.ndim not in (4, 5):
        shape = getattr(images, "shape", None)
        raise _error(f"VAE decode returned an incompatible IMAGE tensor: {shape}.")
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    if images.shape[0] < 1 or images.shape[-1] < 3:
        raise _error(f"VAE decode returned an empty or non-RGB IMAGE tensor: {_shape(images)}.")
    frame = images[-1:, ..., :3].detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(frame).all().item()):
        raise _error("VAE decoded terminal frame contains NaN or Inf values.")
    return frame


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
            f"temporal_mode={TEMPORAL_MODE_A}",
            f"global: video T={plan.video_latent_t}, audio T={plan.audio_latent_t}, frames={plan.frame_count}",
            f"planner stride target: {plan.target_video_tokens} video tokens / "
            f"{frame_boundary_for_video_token(plan.target_video_tokens)} frames / "
            f"{frame_boundary_for_video_token(plan.target_video_tokens) / H3_FPS:.3f}s",
            "context=none",
            "phase 1: no overlap, no temporal hidden-state carry, no global position offset",
        )
    )


def _format_guided_status(
    plan: H3TemporalChunkPlan,
    source_device: torch.device,
    output_video: torch.Tensor,
    noise_mode: str,
) -> str:
    base = _format_status(plan, source_device, output_video, noise_mode).splitlines()
    return "\n".join(
        (
            *base[:8],
            f"continuity_mode={LEGACY_INDEPENDENT_MODE}",
            "continuity=Previous Last Frame -> MiniMaxH3AddGuide(frame_idx=0)",
            f"guides_applied={max(0, len(plan.chunks) - 1)}",
            "guider=official BasicGuider rebuilt per chunk",
            *base[9:11],
            "context=decoded previous terminal frame; no latent overlap or hidden-state carry",
        )
    )


def _format_overlap_status(
    plan: H3TemporalOverlapPlan,
    temporal_mode: str,
    source_device: torch.device,
    output_video: torch.Tensor,
    noise_mode: str,
) -> str:
    context = "source" if temporal_mode == TEMPORAL_MODE_B else "previous_refined"
    ranges = [
        (
            f"#{window.sample.index + 1} sample v[{window.sample.video_start}:{window.sample.video_end}] "
            f"f[{window.sample.frame_start}:{window.sample.frame_end}] "
            f"a[{window.sample.audio_start}:{window.sample.audio_end}] "
            f"keep v[{window.keep_video_start}:{window.keep_video_end}] "
            f"a[{window.keep_audio_start}:{window.keep_audio_end}]"
        )
        for window in plan.windows
    ]
    return "\n".join(
        (
            "Success: sequential native exact-overlap temporal sampling",
            f"temporal_mode={temporal_mode}",
            f"requested_chunk={plan.requested_chunk_seconds:g}s",
            f"global: video T={plan.video_latent_t}, audio T={plan.audio_latent_t}, frames={plan.frame_count}",
            f"stride: {plan.stride_video_tokens} video tokens / {plan.stride_frames} frames / "
            f"{plan.stride_frames / H3_FPS:.3f}s",
            f"window: {plan.window_video_tokens} video tokens / {plan.window_frames} frames / "
            f"{plan.window_audio_tokens} audio ticks / {plan.window_frames / H3_FPS:.3f}s",
            f"nominal overlap: {plan.overlap_video_tokens} video tokens / {plan.overlap_frames} frames / "
            f"{plan.overlap_frames / H3_FPS:.3f}s",
            f"windows: {len(plan.windows)}",
            f"context={context}",
            f"source device: {source_device}",
            f"output: CPU preallocated, dtype={output_video.dtype}",
            f"noise_mode={noise_mode}",
            f"audio timeline delta: {plan.audio_delta:+d} tick(s)",
            "ranges:",
            *ranges,
            "no overlap lock, no hidden-state carry, no global position offset",
        )
    )


def _overlap_noise_plan(plan: H3TemporalOverlapPlan) -> H3TemporalChunkPlan:
    """Adapt exact sampling windows to the existing, unchanged noise resolver."""

    return H3TemporalChunkPlan(
        frame_count=plan.frame_count,
        video_latent_t=plan.video_latent_t,
        audio_latent_t=plan.audio_latent_t,
        expected_audio_t=plan.expected_audio_t,
        audio_delta=plan.audio_delta,
        requested_chunk_seconds=plan.requested_chunk_seconds,
        target_video_tokens=plan.stride_video_tokens,
        chunks=tuple(window.sample for window in plan.windows),
    )


def _sample_h3_temporal_overlap_windows(
    *,
    noise: Any,
    guider: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    latent_image: Mapping[str, Any],
    chunk_duration_seconds: float,
    aggressive_memory_cleanup: bool,
    temporal_mode: str,
    sample_chunk: SampleChunk | None,
    nested_factory: NestedFactory | None,
    cleanup: CleanupCallback | None,
) -> tuple[dict[str, Any], str]:
    video, audio = _extract_h3_streams(latent_image)
    _reject_legacy_noise_mask(latent_image)
    plan = plan_h3_temporal_overlap_windows(
        int(video.shape[2]),
        int(audio.shape[3]),
        chunk_duration_seconds,
    )
    if _guider_has_temporal_keyframes(guider):
        raise _error(
            "multi-window sampling cannot safely consume minimax_keyframes. Current H3 keyframes store absolute "
            "full-timeline frame indices, while the native sampler exposes no public per-window position-offset "
            "contract. Use text/reference conditioning without keyframes or the native monolithic sampler."
        )
    chunk_noise_factory, noise_mode = _resolve_chunk_noise(noise, _overlap_noise_plan(plan))
    sample_chunk = sample_chunk or _sample_with_native_advanced_sampler
    nested_factory = nested_factory or _make_official_nested
    cleanup = cleanup or _default_cleanup

    output_video: torch.Tensor | None = None
    output_audio: torch.Tensor | None = None

    for window in plan.windows:
        chunk = window.sample
        chunk_noise = chunk_noise_factory(chunk)
        source_video_window = video[:, :, chunk.video_start : chunk.video_end, :, :]
        source_audio_window = audio[:, :, :, chunk.audio_start : chunk.audio_end]

        if temporal_mode == TEMPORAL_MODE_C and chunk.index > 0:
            if output_video is None or output_audio is None:
                raise _error("refined overlap was requested before the CPU output range was initialized.")
            if output_video.dtype != video.dtype or output_audio.dtype != audio.dtype:
                raise _error(
                    "C refined-overlap mode requires native sampled dtype to match the source dtype so previous "
                    "context is not silently converted."
                )
            chunk_video = torch.empty_like(source_video_window)
            chunk_audio = torch.empty_like(source_audio_window)
            chunk_video.copy_(source_video_window)
            chunk_audio.copy_(source_audio_window)
            video_overlap = window.local_keep_video_start
            audio_overlap = window.local_keep_audio_start
            if video_overlap <= 0 or audio_overlap <= 0:
                raise _error("C refined-overlap mode received an empty previous-context range.")
            chunk_video[:, :, :video_overlap, :, :].copy_(
                output_video[:, :, chunk.video_start : window.keep_video_start, :, :]
            )
            chunk_audio[:, :, :, :audio_overlap].copy_(
                output_audio[:, :, :, chunk.audio_start : window.keep_audio_start]
            )
        else:
            chunk_video = source_video_window
            chunk_audio = source_audio_window

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
        if temporal_mode == TEMPORAL_MODE_C and (
            sampled_video.dtype != video.dtype or sampled_audio.dtype != audio.dtype
        ):
            raise _error(
                "C refined-overlap mode requires native sampled dtype to match the source dtype across windows."
            )

        if output_video is None:
            output_video_shape = list(video.shape)
            output_audio_shape = list(audio.shape)
            output_video = torch.empty(output_video_shape, dtype=sampled_video.dtype, device="cpu")
            output_audio = torch.empty(output_audio_shape, dtype=sampled_audio.dtype, device="cpu")

        output_video[:, :, window.keep_video_start : window.keep_video_end, :, :].copy_(
            sampled_video[:, :, window.local_keep_video_start :, :, :]
        )
        output_audio[:, :, :, window.keep_audio_start : window.keep_audio_end].copy_(
            sampled_audio[:, :, :, window.local_keep_audio_start :]
        )

        del sampled_video, sampled_audio, sampled_latent
        del chunk_latent, chunk_video, chunk_audio, source_video_window, source_audio_window, chunk_noise
        if aggressive_memory_cleanup:
            gc.collect()
            cleanup()

    if output_video is None or output_audio is None:
        raise _error("no exact temporal windows were produced.")

    output = dict(latent_image)
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = nested_factory((output_video, output_audio))
    return output, _format_overlap_status(plan, temporal_mode, video.device, output_video, noise_mode)


def _hard_noise_plan(plan: H3HardAVPrefixPlan) -> H3TemporalChunkPlan:
    return H3TemporalChunkPlan(
        frame_count=plan.frame_count,
        video_latent_t=plan.video_latent_t,
        audio_latent_t=plan.audio_latent_t,
        expected_audio_t=plan.audio_latent_t,
        audio_delta=0,
        requested_chunk_seconds=plan.window_frames / H3_FPS,
        target_video_tokens=plan.fresh_video_t,
        chunks=tuple(window.sample for window in plan.windows),
    )


def _format_hard_prefix_status(
    plan: H3HardAVPrefixPlan,
    source_device: torch.device,
    output_video: torch.Tensor,
    noise_mode: str,
    video_prefix_drift_corrections: int,
    audio_prefix_drift_corrections: int,
) -> str:
    ranges = [
        (
            f"#{window.sample.index + 1} raw f[{window.sample.frame_start}:{window.sample.frame_end}) "
            f"v[{window.sample.video_start}:{window.sample.video_end}) "
            f"a[{window.sample.audio_start}:{window.sample.audio_end}) -> "
            f"fresh v[{window.keep_video_start}:{window.keep_video_end}) "
            f"a[{window.keep_audio_start}:{window.keep_audio_end}); "
            f"tail_pad v={max(0, window.sample.video_end - plan.video_latent_t)} "
            f"a={max(0, window.sample.audio_end - plan.audio_latent_t)}"
        )
        for window in plan.windows
    ]
    return "\n".join(
        (
            "Success: sequential native Hard AV Latent Prefix sampling",
            f"continuity_mode={HARD_AV_PREFIX_MODE}",
            f"hard_chunk_preset={plan.preset_label}",
            f"global: video T={plan.video_latent_t}, audio T={plan.audio_latent_t}, frames={plan.frame_count}",
            f"window: {plan.window_frames} frames / video T={plan.window_video_t} / audio T={plan.window_audio_t}",
            f"hard prefix: {HARD_OVERLAP_FRAMES} frames / video T={HARD_VIDEO_PREFIX_T} / "
            f"audio T={HARD_AUDIO_PREFIX_T}",
            f"fresh stride: {plan.stride_frames} frames / video T={plan.fresh_video_t} / "
            f"audio T={plan.fresh_audio_t}",
            f"chunks={len(plan.windows)}; prefixes_applied={max(0, len(plan.windows) - 1)}",
            "prefix_source=previous sampled AV tail; prefix_mask=0; fresh_mask=1",
            "post_sample_prefix_relock=bit-identical; "
            f"native_drift_corrected video_chunks={video_prefix_drift_corrections} "
            f"audio_chunks={audio_prefix_drift_corrections}",
            "continuation_guide=none; MiniMaxH3AddGuide/VAE decode not used",
            "guider=official BasicGuider rebuilt per chunk from original positive",
            f"source device: {source_device}",
            f"output: CPU preallocated, dtype={output_video.dtype}; overlap copied once",
            f"noise_mode={noise_mode}; seed_basis=absolute raw frame_start",
            "ranges:",
            *ranges,
        )
    )


def _materialize_hard_window(
    source: torch.Tensor,
    *,
    temporal_dim: int,
    start: int,
    length: int,
    writable: bool,
) -> tuple[torch.Tensor, int]:
    total = int(source.shape[temporal_dim])
    if start < 0 or start >= total:
        raise _error(f"hard-prefix local window starts outside its source stream: start={start}, T={total}.")
    available = min(length, total - start)
    source_window = source.narrow(temporal_dim, start, available)
    padding = length - available
    if padding == 0:
        return (source_window.clone() if writable else source_window), 0

    shape = list(source.shape)
    shape[temporal_dim] = length
    window = torch.zeros(shape, dtype=source.dtype, device=source.device)
    window.narrow(temporal_dim, 0, available).copy_(source_window)
    return window, padding


def _sample_h3_hard_av_prefix(
    *,
    model: Any,
    positive: Any,
    noise: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    latent_image: Mapping[str, Any],
    hard_chunk_preset: str,
    aggressive_memory_cleanup: bool,
    sample_chunk: SampleChunk | None,
    nested_factory: NestedFactory | None,
    cleanup: CleanupCallback | None,
    build_guider: BuildGuider | None,
) -> tuple[dict[str, Any], str]:
    video, audio = _extract_h3_streams(latent_image)
    _validate_trivial_hard_input_mask(latent_image, video, audio)
    plan = plan_h3_hard_av_prefix_windows(
        int(video.shape[2]),
        int(audio.shape[3]),
        hard_chunk_preset,
    )
    if len(plan.windows) > 1 and _positive_has_temporal_keyframes(positive):
        raise _error(
            f"{HARD_AV_PREFIX_MODE} cannot be combined with minimax_keyframes/AddGuide conditioning. "
            "Provide normal H3 positive conditioning without temporal keyframes."
        )

    chunk_noise_factory, noise_mode = _resolve_chunk_noise(noise, _hard_noise_plan(plan))
    sample_chunk = sample_chunk or _sample_with_native_advanced_sampler
    nested_factory = nested_factory or _make_official_nested
    cleanup = cleanup or _default_cleanup
    build_guider = build_guider or _build_native_basic_guider

    output_video: torch.Tensor | None = None
    output_audio: torch.Tensor | None = None
    previous_video_tail: torch.Tensor | None = None
    previous_audio_tail: torch.Tensor | None = None
    video_prefix_drift_corrections = 0
    audio_prefix_drift_corrections = 0

    for window in plan.windows:
        chunk = window.sample
        chunk_noise = chunk_noise_factory(chunk)
        source_video_window, _video_padding = _materialize_hard_window(
            video,
            temporal_dim=2,
            start=chunk.video_start,
            length=plan.window_video_t,
            writable=chunk.index > 0,
        )
        source_audio_window, _audio_padding = _materialize_hard_window(
            audio,
            temporal_dim=3,
            start=chunk.audio_start,
            length=plan.window_audio_t,
            writable=chunk.index > 0,
        )

        chunk_latent = dict(latent_image)
        if chunk.index == 0:
            chunk_video = source_video_window
            chunk_audio = source_audio_window
            chunk_latent.pop("noise_mask", None)
        else:
            if previous_video_tail is None or previous_audio_tail is None:
                raise _error("previous sampled AV tails were not available for hard-prefix continuation.")
            if previous_video_tail.dtype != video.dtype or previous_audio_tail.dtype != audio.dtype:
                raise _error("previous sampled AV tail dtype changed and cannot be copied bit-identically.")
            expected_video_tail_shape = list(video.shape)
            expected_video_tail_shape[2] = HARD_VIDEO_PREFIX_T
            expected_audio_tail_shape = list(audio.shape)
            expected_audio_tail_shape[3] = HARD_AUDIO_PREFIX_T
            if list(previous_video_tail.shape) != expected_video_tail_shape:
                raise _error(
                    f"previous sampled video tail has an incompatible shape: expected {expected_video_tail_shape}, "
                    f"received {list(previous_video_tail.shape)}."
                )
            if list(previous_audio_tail.shape) != expected_audio_tail_shape:
                raise _error(
                    f"previous sampled audio tail has an incompatible shape: expected {expected_audio_tail_shape}, "
                    f"received {list(previous_audio_tail.shape)}."
                )

            chunk_video = source_video_window
            chunk_audio = source_audio_window
            chunk_video[:, :, :HARD_VIDEO_PREFIX_T, :, :].copy_(previous_video_tail)
            chunk_audio[:, :, :, :HARD_AUDIO_PREFIX_T].copy_(previous_audio_tail)
            video_mask = torch.ones_like(chunk_video)
            audio_mask = torch.ones_like(chunk_audio)
            video_mask[:, :, :HARD_VIDEO_PREFIX_T, :, :] = 0
            audio_mask[:, :, :, :HARD_AUDIO_PREFIX_T] = 0
            chunk_latent["noise_mask"] = nested_factory((video_mask, audio_mask))

        chunk_latent["samples"] = nested_factory((chunk_video, chunk_audio))
        chunk_guider = build_guider(model, positive)
        sampled_latent = sample_chunk(
            noise=chunk_noise,
            guider=chunk_guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=chunk_latent,
        )
        sampled_video, sampled_audio = _validate_sampled_chunk(sampled_latent, video, audio, chunk)
        if sampled_video.dtype != video.dtype or sampled_audio.dtype != audio.dtype:
            raise _error("native sampler changed AV dtype; hard-prefix continuation requires bit-identical dtype.")

        if chunk.index > 0:
            sampled_video_prefix = sampled_video[:, :, :HARD_VIDEO_PREFIX_T, :, :].detach().to(device="cpu")
            sampled_audio_prefix = sampled_audio[:, :, :, :HARD_AUDIO_PREFIX_T].detach().to(device="cpu")
            if not torch.equal(sampled_video_prefix, previous_video_tail):
                video_prefix_drift_corrections += 1
            if not torch.equal(sampled_audio_prefix, previous_audio_tail):
                audio_prefix_drift_corrections += 1
            del sampled_video_prefix, sampled_audio_prefix

            # The native sampler performs its work in float32 and H3 applies latent
            # in/out transforms around the packed AV stream.  A zero denoise mask
            # therefore locks the prefix semantically during denoising, but the
            # returned tensor can still differ by floating-point round-trip bits.
            # Reassert the sampled tail after native sampling so the continuation
            # contract is bit-identical without changing native sampler behavior.
            sampled_video[:, :, :HARD_VIDEO_PREFIX_T, :, :].copy_(
                previous_video_tail.to(device=sampled_video.device)
            )
            sampled_audio[:, :, :, :HARD_AUDIO_PREFIX_T].copy_(
                previous_audio_tail.to(device=sampled_audio.device)
            )

            relocked_video_prefix = sampled_video[:, :, :HARD_VIDEO_PREFIX_T, :, :].detach().to(device="cpu")
            relocked_audio_prefix = sampled_audio[:, :, :, :HARD_AUDIO_PREFIX_T].detach().to(device="cpu")
            if not torch.equal(relocked_video_prefix, previous_video_tail):
                raise _error(
                    f"failed to restore the locked {HARD_VIDEO_PREFIX_T}-step video prefix "
                    f"bit-identically for chunk {chunk.index + 1}."
                )
            if not torch.equal(relocked_audio_prefix, previous_audio_tail):
                raise _error(
                    f"failed to restore the locked {HARD_AUDIO_PREFIX_T}-tick audio prefix "
                    f"bit-identically for chunk {chunk.index + 1}."
                )
            del relocked_video_prefix, relocked_audio_prefix

        if output_video is None:
            output_video = torch.empty(list(video.shape), dtype=sampled_video.dtype, device="cpu")
            output_audio = torch.empty(list(audio.shape), dtype=sampled_audio.dtype, device="cpu")
        if output_audio is None:
            raise _error("CPU audio output buffer was not initialized.")

        local_video_start = window.local_keep_video_start
        local_audio_start = window.local_keep_audio_start
        keep_video_t = window.keep_video_end - window.keep_video_start
        keep_audio_t = window.keep_audio_end - window.keep_audio_start
        output_video[:, :, window.keep_video_start : window.keep_video_end, :, :].copy_(
            sampled_video[:, :, local_video_start : local_video_start + keep_video_t, :, :]
        )
        output_audio[:, :, :, window.keep_audio_start : window.keep_audio_end].copy_(
            sampled_audio[:, :, :, local_audio_start : local_audio_start + keep_audio_t]
        )

        if chunk.index + 1 < len(plan.windows):
            previous_video_tail = (
                sampled_video[:, :, -HARD_VIDEO_PREFIX_T:, :, :].detach().to(device="cpu").clone().contiguous()
            )
            previous_audio_tail = (
                sampled_audio[:, :, :, -HARD_AUDIO_PREFIX_T:].detach().to(device="cpu").clone().contiguous()
            )
        else:
            previous_video_tail = None
            previous_audio_tail = None

        del sampled_video, sampled_audio, sampled_latent, chunk_guider, chunk_noise
        del chunk_latent, chunk_video, chunk_audio, source_video_window, source_audio_window
        if aggressive_memory_cleanup:
            gc.collect()
            cleanup()

    if output_video is None or output_audio is None:
        raise _error("no hard-prefix temporal windows were produced.")

    output = dict(latent_image)
    output.pop("noise_mask", None)
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = nested_factory((output_video, output_audio))
    return output, _format_hard_prefix_status(
        plan,
        video.device,
        output_video,
        noise_mode,
        video_prefix_drift_corrections,
        audio_prefix_drift_corrections,
    )


def sample_h3_temporal_chunks(
    *,
    model: Any = None,
    positive: Any = None,
    vae: Any = None,
    noise: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    latent_image: Mapping[str, Any],
    chunk_duration_seconds: float,
    aggressive_memory_cleanup: bool = False,
    sample_chunk: SampleChunk | None = None,
    nested_factory: NestedFactory | None = None,
    cleanup: CleanupCallback | None = None,
    build_guider: BuildGuider | None = None,
    apply_guide: ApplyGuide | None = None,
    decode_last_frame: DecodeLastFrame | None = None,
    guider: Any = None,
    temporal_mode: str = TEMPORAL_MODE_A,
    continuity_mode: str = LEGACY_INDEPENDENT_MODE,
    hard_chunk_preset: str = DEFAULT_HARD_CHUNK_PRESET,
) -> tuple[dict[str, Any], str]:
    """Sample H3 chunks using the selected production continuity strategy."""

    if continuity_mode not in CONTINUITY_MODES:
        raise _error(f"continuity_mode must be one of {list(CONTINUITY_MODES)}, received {continuity_mode!r}.")
    guided_inputs = (model is not None, positive is not None, vae is not None)
    guided_mode = all(guided_inputs)
    if any(guided_inputs) and not guided_mode:
        raise _error("guided sampling requires model, positive, and vae together.")
    if not guided_mode:
        if continuity_mode == HARD_AV_PREFIX_MODE:
            raise _error(f"{HARD_AV_PREFIX_MODE} requires the node's model, positive, and vae inputs.")
        if temporal_mode not in TEMPORAL_MODES:
            raise _error(f"temporal_mode must be one of {list(TEMPORAL_MODES)}, received {temporal_mode!r}.")
        if temporal_mode != TEMPORAL_MODE_A:
            return _sample_h3_temporal_overlap_windows(
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent_image,
                chunk_duration_seconds=chunk_duration_seconds,
                aggressive_memory_cleanup=aggressive_memory_cleanup,
                temporal_mode=temporal_mode,
                sample_chunk=sample_chunk,
                nested_factory=nested_factory,
                cleanup=cleanup,
            )

    if guided_mode and continuity_mode == HARD_AV_PREFIX_MODE:
        return _sample_h3_hard_av_prefix(
            model=model,
            positive=positive,
            noise=noise,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent_image,
            hard_chunk_preset=hard_chunk_preset,
            aggressive_memory_cleanup=aggressive_memory_cleanup,
            sample_chunk=sample_chunk,
            nested_factory=nested_factory,
            cleanup=cleanup,
            build_guider=build_guider,
        )

    video, audio = _extract_h3_streams(latent_image)
    _reject_legacy_noise_mask(latent_image)
    plan = plan_h3_temporal_chunks(int(video.shape[2]), int(audio.shape[3]), chunk_duration_seconds)
    if len(plan.chunks) > 1:
        if guided_mode and _positive_has_temporal_keyframes(positive):
            raise _error(
                "multi-chunk Previous Last Frame sampling requires original positive conditioning without existing "
                "minimax_keyframes. Existing absolute/full-timeline guides cannot be safely combined with the local "
                "frame-0 continuation guide. Use Reference-to-Video conditioning without image keyframes."
            )
        if not guided_mode and _guider_has_temporal_keyframes(guider):
            raise _error(
                "multi-chunk sampling cannot safely consume minimax_keyframes. Current H3 keyframes store absolute "
                "full-timeline frame indices, while the native sampler exposes no public per-window position-offset "
                "contract. Use text/reference conditioning without keyframes or the native monolithic sampler."
            )
    chunk_noise_factory, noise_mode = _resolve_chunk_noise(noise, plan)
    sample_chunk = sample_chunk or _sample_with_native_advanced_sampler
    nested_factory = nested_factory or _make_official_nested
    cleanup = cleanup or _default_cleanup
    build_guider = build_guider or _build_native_basic_guider
    apply_guide = apply_guide or _apply_native_previous_frame_guide
    decode_last_frame = decode_last_frame or _decode_terminal_frame

    output_video: torch.Tensor | None = None
    output_audio: torch.Tensor | None = None
    previous_last_frame: torch.Tensor | None = None

    for chunk in plan.chunks:
        chunk_noise = chunk_noise_factory(chunk)
        chunk_video = video[:, :, chunk.video_start : chunk.video_end, :, :]
        chunk_audio = audio[:, :, :, chunk.audio_start : chunk.audio_end]
        chunk_latent = dict(latent_image)
        chunk_latent["samples"] = nested_factory((chunk_video, chunk_audio))

        if not guided_mode:
            chunk_positive = None
            chunk_guider = guider
        elif chunk.index == 0:
            chunk_positive = positive
            chunk_guider = build_guider(model, chunk_positive)
        else:
            if previous_last_frame is None:
                raise _error("previous chunk terminal frame was not available for continuation.")
            chunk_positive = apply_guide(
                positive=positive,
                latent=chunk_latent,
                vae=vae,
                image=previous_last_frame,
            )
            chunk_guider = build_guider(model, chunk_positive)

        sampled_latent = sample_chunk(
            noise=chunk_noise,
            guider=chunk_guider,
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

        if guided_mode and chunk.index + 1 < len(plan.chunks):
            previous_last_frame = decode_last_frame(vae, sampled_video)

        del sampled_video, sampled_audio, sampled_latent
        del chunk_latent, chunk_video, chunk_audio, chunk_noise, chunk_guider, chunk_positive
        if aggressive_memory_cleanup:
            gc.collect()
            cleanup()

    if output_video is None or output_audio is None:  # defensive; a valid H3 timeline always has at least one chunk
        raise _error("no temporal chunks were produced.")

    output = dict(latent_image)
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = nested_factory((output_video, output_audio))
    status = (
        _format_guided_status(plan, video.device, output_video, noise_mode)
        if guided_mode
        else _format_status(plan, video.device, output_video, noise_mode)
    )
    return output, status


__all__ = [
    "AUDIO_LATENT_FPS",
    "CONTINUITY_MODES",
    "DEFAULT_HARD_CHUNK_PRESET",
    "ERROR_PREFIX",
    "FRAMES_PER_VIDEO_TOKEN",
    "HARD_AUDIO_FRESH_T",
    "HARD_AUDIO_PREFIX_T",
    "HARD_AUDIO_WINDOW_T",
    "HARD_AV_PREFIX_MODE",
    "HARD_CHUNK_SECONDS",
    "HARD_CHUNK_PRESETS",
    "HARD_CHUNK_PRESET_LABELS",
    "HARD_OVERLAP_FRAMES",
    "HARD_STRIDE_FRAMES",
    "HARD_VIDEO_FRESH_T",
    "HARD_VIDEO_PREFIX_T",
    "HARD_VIDEO_WINDOW_T",
    "HARD_WINDOW_FRAMES",
    "H3_FPS",
    "H3HardAVPrefixPlan",
    "H3TemporalChunk",
    "H3TemporalChunkPlan",
    "H3TemporalOverlapPlan",
    "H3TemporalOverlapWindow",
    "H3TemporalChunkSamplerError",
    "TEMPORAL_MODE_A",
    "TEMPORAL_MODE_B",
    "TEMPORAL_MODE_C",
    "TEMPORAL_MODES",
    "LEGACY_INDEPENDENT_MODE",
    "derive_chunk_seed",
    "frame_boundary_for_video_token",
    "plan_h3_hard_av_prefix_windows",
    "plan_h3_temporal_chunks",
    "plan_h3_temporal_overlap_windows",
    "sample_h3_temporal_chunks",
]
