"""Bounded VIDEO reads and phase-aligned H3 regeneration windows."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .h3_temporal_chunk_sampler import frame_boundary_for_video_token

CHUNK_MODES = ("off", "5s", "8s", "10s")
WINDOW_TOKENS = {"5s": 37, "8s": 57, "10s": 72}
PREFIX_TOKENS = 12


@dataclass(frozen=True)
class RegenerateWindow:
    start: int
    end: int
    prefix: int

    @property
    def frames(self):
        return frame_boundary_for_video_token(self.end - self.start)


def plan_windows(video_t, chunk_mode):
    if video_t < 2 or (video_t - 2) % 5:
        raise ValueError("H3 video requires T = 5k + 2")
    if chunk_mode not in CHUNK_MODES:
        raise ValueError(f"Unknown chunk mode: {chunk_mode}")
    limit = WINDOW_TOKENS.get(chunk_mode, video_t)
    windows, start = [], 0
    while True:
        end = min(start + limit, video_t)
        windows.append(RegenerateWindow(start, end, PREFIX_TOKENS if start else 0))
        if end == video_t:
            return tuple(windows)
        start = end - PREFIX_TOKENS


class _VideoOnlyContainer:
    """Let ComfyUI perform its native crop/rotation/color decode, without audio."""
    def __init__(self, container):
        self.streams = SimpleNamespace(video=container.streams.video, audio=[])
        self.metadata = container.metadata
        self.demux = container.demux
        self.seek = container.seek


def read_original_audio(video):
    """Decode the active file's audio once, retaining its rate and channel layout."""
    import av
    import numpy as np
    from comfy_api.latest._input_impl.video_types import last_decodable_audio_stream

    start, duration = video.get_active_trim_window()
    with av.open(video.get_stream_source()) as container:
        stream = last_decodable_audio_stream(container)
        if stream is None:
            return None
        # Only change sample representation to planar float, never sample rate/layout.
        resampler = av.AudioResampler(format="fltp")
        parts, rate, channels, cursor = [], None, None, 0
        end = start + duration if duration else math.inf
        for packet in container.demux(stream):
            for decoded in packet.decode():
                for frame in resampler.resample(decoded):
                    rate = frame.sample_rate
                    channels = frame.layout.nb_channels
                    timestamp = float(frame.time) if frame.time is not None else start + cursor / rate
                    if timestamp >= end:
                        break
                    data = frame.to_ndarray()
                    left = max(0, round((start - timestamp) * rate))
                    right = min(frame.samples, round((end - timestamp) * rate)) if duration else frame.samples
                    if right <= left:
                        continue
                    position = max(0, round((timestamp - start) * rate) + left)
                    if position > cursor:
                        parts.append(np.zeros((channels, position - cursor), dtype=np.float32))
                        cursor = position
                    left += max(0, cursor - position)
                    if left < right:
                        parts.append(data[:, left:right])
                        cursor += right - left
            if rate and packet.pts is not None and float(packet.pts * stream.time_base) >= end:
                break
        if not parts:
            return None
        return {"waveform": torch.from_numpy(np.concatenate(parts, axis=1)).unsqueeze(0),
                "sample_rate": rate}


class VideoWindowReader:
    def __init__(self, video):
        from comfy_api.input_impl import VideoFromComponents, VideoFromFile

        self.video = video
        self.components = None
        if isinstance(video, VideoFromComponents):
            self.components = video.get_components()  # already resident; no copy
            self.fps = float(self.components.frame_rate)
            source_count = len(self.components.images)
            self.audio = self.components.audio
        elif isinstance(video, VideoFromFile):
            self.fps = float(video.get_frame_rate())
            source_count = video.get_frame_count()
            self.audio = read_original_audio(video)
        else:
            raise ValueError("Regenerate requires native LoadVideo / Video Slice / CreateVideo output")
        if not math.isfinite(self.fps) or self.fps <= 0 or source_count < 1:
            raise ValueError("VIDEO must contain frames and a positive frame rate")
        self.source_count = source_count
        self.count = max(1, round(source_count * 24 / self.fps))
        self.video_t = max(0, math.ceil((max(5, self.count) - 5) / 17)) * 5 + 2

    def read(self, window):
        import av
        start = frame_boundary_for_video_token(window.start)
        count = window.frames
        # Resample on the GLOBAL 24-fps grid: overlapping frames use identical indices.
        indices = ((torch.arange(start, start + count, dtype=torch.float64).clamp(max=self.count - 1)
                    * self.fps / 24).floor().long().clamp(max=self.source_count - 1))
        if self.components is not None:
            images = self.components.images
            return images.index_select(0, indices.to(images.device))[..., :3]
        first, last = int(indices[0]), int(indices[-1])
        trimmed = self.video.as_trimmed(first / self.fps, (last - first + 1) / self.fps,
                                       strict_duration=False)
        with av.open(trimmed.get_stream_source()) as container:
            components = trimmed.get_components_internal(_VideoOnlyContainer(container))
        images = components.images
        if len(images) == 0:
            raise ValueError(f"No VIDEO frames decoded for window at frame {start}")
        local_indices = (indices - first).clamp(max=len(images) - 1)
        return images.index_select(0, local_indices)[..., :3]


def encode_window(reader, window, vae):
    images = reader.read(window)
    height, width = (max(32, round(n / 32) * 32) for n in images.shape[1:3])
    if images.shape[1:3] != (height, width):
        images = F.interpolate(images.movedim(-1, 1), size=(height, width), mode="bicubic",
                               align_corners=False).movedim(1, -1).clamp(0, 1)
    return vae.encode(images).cpu()
