"""Immutable, model-free TaoMate temporal geometry (independent implementation)."""

from dataclasses import dataclass
from fractions import Fraction

VIDEO_FPS = 24
AUDIO_LATENT_RATE = 40
VIDEO_PREFIX_FRAMES, VIDEO_PREFIX_LATENTS = 5, 2
VIDEO_GROUP_FRAMES, VIDEO_GROUP_LATENTS = 17, 5
GROUP_COUNTS = (2, 2, 2, 1)
PRESET = "TaoMate 5s Canonical"


def round_half_even_ratio(numerator: int, denominator: int) -> int:
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ValueError("Expected integer numerator and positive integer denominator")
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(2 * remainder > denominator or (2 * remainder == denominator and quotient % 2 != 0))


def audio_boundary(frame: int) -> int:
    return round_half_even_ratio(frame * AUDIO_LATENT_RATE, VIDEO_FPS)


@dataclass(frozen=True)
class StreamPhase:
    phase_index: int
    group_count: int
    frame_start: int
    frame_stop: int
    video_latent_start: int
    video_latent_stop: int
    audio_latent_start: int
    audio_latent_stop: int

    @property
    def frame_count(self):
        return self.frame_stop - self.frame_start

    @property
    def video_latent_count(self):
        return self.video_latent_stop - self.video_latent_start

    @property
    def audio_latent_count(self):
        return self.audio_latent_stop - self.audio_latent_start

    @property
    def duration_seconds(self):
        return Fraction(self.frame_count, VIDEO_FPS)


@dataclass(frozen=True)
class StreamPlan:
    native_frame_count: int
    phases: tuple[StreamPhase, ...]
    preset: str = PRESET
    schema_version: int = 1
    request_index: int = 0
    native_frame_offset: int = 0
    video_latent_offset: int = 0
    audio_latent_offset: int = 0
    media_time_origin: Fraction = Fraction(0)

    @property
    def video_latent_count(self):
        return self.phases[-1].video_latent_stop

    @property
    def audio_latent_count(self):
        return self.phases[-1].audio_latent_stop

    def status(self):
        lines = ["JR TaoMate Streaming Plan", f"Preset: {self.preset}", "FPS: 24; audio latent rate: 40 Hz",
                 f"Native frames: {self.native_frame_count}; duration: {self.native_frame_count / VIDEO_FPS:.6f}s",
                 f"Video latents: {self.video_latent_count}; audio latents: {self.audio_latent_count}",
                 f"Phases: {len(self.phases)}; single request only"]
        for p in self.phases:
            lines.append(f"Phase {p.phase_index}: frames {p.frame_start}->{p.frame_stop}; "
                         f"video {p.video_latent_start}->{p.video_latent_stop}; "
                         f"audio {p.audio_latent_start}->{p.audio_latent_stop}")
        return "\n".join(lines)


def canonical_plan(preset=PRESET):
    if preset != PRESET:
        raise ValueError(f"JR H3 Streaming: unsupported preset {preset!r}")
    phases = []
    frame = video = 0
    for index, groups in enumerate(GROUP_COUNTS):
        next_frame = frame + groups * VIDEO_GROUP_FRAMES + (VIDEO_PREFIX_FRAMES if index == 0 else 0)
        next_video = video + groups * VIDEO_GROUP_LATENTS + (VIDEO_PREFIX_LATENTS if index == 0 else 0)
        phases.append(StreamPhase(index, groups, frame, next_frame, video, next_video,
                                  audio_boundary(frame), audio_boundary(next_frame)))
        frame, video = next_frame, next_video
    return StreamPlan(frame, tuple(phases))


def validate_plan(plan):
    if type(plan) is not StreamPlan or plan != canonical_plan():
        raise ValueError("JR H3 Streaming: invalid stream plan; only the unmodified single-request canonical plan is supported")
    return plan
