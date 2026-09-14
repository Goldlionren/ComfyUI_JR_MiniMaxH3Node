"""Immutable, model-free TaoMate temporal geometry (independent implementation)."""

from dataclasses import dataclass
from fractions import Fraction

VIDEO_FPS = 24
AUDIO_LATENT_RATE = 40
VIDEO_PREFIX_FRAMES, VIDEO_PREFIX_LATENTS = 5, 2
VIDEO_GROUP_FRAMES, VIDEO_GROUP_LATENTS = 17, 5
GROUP_COUNTS = (2, 2, 2, 1)
PRESET = "TaoMate 5s Canonical"
PRESET_WINDOWS = {PRESET: 1, "JR 10s Experimental": 2, "JR 15s Experimental": 3}
PRESETS = tuple(PRESET_WINDOWS)


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

    @property
    def windows(self):
        """Logical ~5s groups, not independent requests or decoder clips."""
        size = len(GROUP_COUNTS)
        return tuple(self.phases[i:i + size] for i in range(0, len(self.phases), size))

    def status(self):
        lines = ["JR TaoMate Streaming Plan", f"Preset: {self.preset}", "FPS: 24; audio latent rate: 40 Hz",
                 f"Native frames: {self.native_frame_count}; duration: {self.native_frame_count / VIDEO_FPS:.6f}s",
                 f"Video latents: {self.video_latent_count}; audio latents: {self.audio_latent_count}",
                 f"Phases: {len(self.phases)}; windows: {len(self.windows)}; one complete AV timeline",
                 "Pass 1 stays full-length. Pass 2 shares global positions/noise and bounded KV across windows.",
                 "Geometry only: connect external SIGMAS; no schedule is generated or changed."]
        for index, phases in enumerate(self.windows):
            first, last = phases[0], phases[-1]
            lines.append(f"Window {index}: frames {first.frame_start}->{last.frame_stop}; "
                         f"video {first.video_latent_start}->{last.video_latent_stop}; "
                         f"audio {first.audio_latent_start}->{last.audio_latent_stop}")
        for p in self.phases:
            lines.append(f"Phase {p.phase_index}: frames {p.frame_start}->{p.frame_stop}; "
                         f"video {p.video_latent_start}->{p.video_latent_stop}; "
                         f"audio {p.audio_latent_start}->{p.audio_latent_stop}")
        return "\n".join(lines)


def canonical_plan(preset=PRESET):
    if not isinstance(preset, str) or preset not in PRESET_WINDOWS:
        raise ValueError(f"JR H3 Streaming: unsupported preset {preset!r}")
    phases = []
    frame = video = 0
    # The affine prefix belongs only to the start of the full video. Later
    # windows add 119 frames / 35 video latents, not another 124-frame clip.
    for index, groups in enumerate(GROUP_COUNTS * PRESET_WINDOWS[preset]):
        next_frame = frame + groups * VIDEO_GROUP_FRAMES + (VIDEO_PREFIX_FRAMES if index == 0 else 0)
        next_video = video + groups * VIDEO_GROUP_LATENTS + (VIDEO_PREFIX_LATENTS if index == 0 else 0)
        phases.append(StreamPhase(index, groups, frame, next_frame, video, next_video,
                                  audio_boundary(frame), audio_boundary(next_frame)))
        frame, video = next_frame, next_video
    return StreamPlan(frame, tuple(phases), preset=preset)


def validate_plan(plan):
    if (type(plan) is not StreamPlan or not isinstance(plan.preset, str) or plan.preset not in PRESET_WINDOWS
            or plan != canonical_plan(plan.preset)):
        raise ValueError("JR H3 Streaming: invalid stream plan; use an unmodified 5s/10s/15s planner preset")
    return plan
