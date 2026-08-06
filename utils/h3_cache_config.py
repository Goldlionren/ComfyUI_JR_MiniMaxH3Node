"""Immutable, validated configuration for the clean-room JR H3 cache."""

from __future__ import annotations

from dataclasses import dataclass, replace

SCHEMA_VERSION = 1
PROFILES = ("visual_fast", "dialogue_safe", "action_safe", "balanced", "off")
QUALITY_LEVELS = ("Conservative", "Balanced", "Aggressive", "Custom")
CACHE_DEVICES = ("Auto", "GPU", "CPU")
AUDIO_CONTENTS = ("Auto", "None", "Speech", "Singing", "Music", "Ambient")


@dataclass(frozen=True)
class H3CacheConfig:
    schema_version: int
    source: str
    profile: str
    quality_level: str
    start_percent: float
    end_percent: float
    warmup_steps: int
    front_blocks: int
    back_blocks: int
    video_threshold: float
    audio_threshold: float
    fast_path_threshold: float
    probe_path_threshold: float
    max_full_step_hits: int
    max_block_hits: int
    video_metric_stride: int
    audio_metric_stride: int
    cache_device: str
    gpu_reserve_mb: int
    audio_content: str = "Auto"
    confidence: float = 0.0
    analysis_summary: str = ""

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported JR H3 cache config schema_version: {self.schema_version}")
        if self.profile not in PROFILES:
            raise ValueError(f"Invalid JR H3 cache profile: {self.profile}")
        if self.quality_level not in QUALITY_LEVELS:
            raise ValueError(f"Invalid JR H3 cache quality level: {self.quality_level}")
        if self.cache_device not in CACHE_DEVICES:
            raise ValueError(f"Invalid JR H3 cache device: {self.cache_device}")
        if self.audio_content not in AUDIO_CONTENTS:
            raise ValueError(f"Invalid audio_content: {self.audio_content}")
        if not 0.0 <= self.start_percent < self.end_percent <= 1.0:
            raise ValueError("Cache window must satisfy 0 <= start_percent < end_percent <= 1.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative.")
        if self.front_blocks < 0 or self.back_blocks < 0 or self.front_blocks + self.back_blocks >= 50:
            raise ValueError("front_blocks/back_blocks must leave at least one middle block in a 50-block H3 model.")
        for name in ("video_threshold", "audio_threshold", "fast_path_threshold", "probe_path_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.fast_path_threshold > self.probe_path_threshold:
            raise ValueError("fast_path_threshold cannot exceed probe_path_threshold.")
        if self.max_full_step_hits < 0 or self.max_block_hits < 0:
            raise ValueError("Consecutive cache hit limits cannot be negative.")
        if self.video_metric_stride <= 0 or self.audio_metric_stride <= 0:
            raise ValueError("Metric stride values must be greater than zero.")
        if self.gpu_reserve_mb < 0:
            raise ValueError("gpu_reserve_mb cannot be negative.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if len(self.analysis_summary) > 512:
            raise ValueError("analysis_summary is too long.")


_BASE_PRESETS = {
    "visual_fast": dict(start_percent=0.06, end_percent=0.94, warmup_steps=2, front_blocks=1, back_blocks=1,
                        video_threshold=0.140, audio_threshold=0.120, fast_path_threshold=0.070,
                        probe_path_threshold=0.180, max_full_step_hits=3, max_block_hits=2,
                        video_metric_stride=16, audio_metric_stride=8),
    "dialogue_safe": dict(start_percent=0.12, end_percent=0.88, warmup_steps=3, front_blocks=1, back_blocks=2,
                          video_threshold=0.120, audio_threshold=0.100, fast_path_threshold=0.060,
                          probe_path_threshold=0.160, max_full_step_hits=0, max_block_hits=1,
                          video_metric_stride=16, audio_metric_stride=4),
    "action_safe": dict(start_percent=0.15, end_percent=0.85, warmup_steps=3, front_blocks=2, back_blocks=2,
                        video_threshold=0.100, audio_threshold=0.080, fast_path_threshold=0.050,
                        probe_path_threshold=0.140, max_full_step_hits=0, max_block_hits=1,
                        video_metric_stride=8, audio_metric_stride=4),
    "balanced": dict(start_percent=0.10, end_percent=0.90, warmup_steps=2, front_blocks=1, back_blocks=2,
                     video_threshold=0.120, audio_threshold=0.100, fast_path_threshold=0.070,
                     probe_path_threshold=0.180, max_full_step_hits=1, max_block_hits=2,
                     video_metric_stride=12, audio_metric_stride=6),
    "off": dict(start_percent=0.0, end_percent=1.0, warmup_steps=0, front_blocks=0, back_blocks=0,
                video_threshold=0.0, audio_threshold=0.0, fast_path_threshold=0.0,
                probe_path_threshold=0.0, max_full_step_hits=0, max_block_hits=0,
                video_metric_stride=1, audio_metric_stride=1),
}


def _quality_adjust(values: dict, quality_level: str) -> dict:
    values = dict(values)
    if quality_level in ("Balanced", "Custom") or values["max_block_hits"] == 0:
        return values
    if quality_level == "Conservative":
        scale = 0.65
        values["start_percent"] = min(values["end_percent"] - 0.05, values["start_percent"] + 0.05)
        values["end_percent"] = max(values["start_percent"] + 0.05, values["end_percent"] - 0.05)
        values["warmup_steps"] += 1
        values["front_blocks"] = min(47, values["front_blocks"] + 1)
        values["back_blocks"] = min(47 - values["front_blocks"], values["back_blocks"] + 1)
        values["max_full_step_hits"] = min(values["max_full_step_hits"], 1)
        values["max_block_hits"] = min(values["max_block_hits"], 1)
    elif quality_level == "Aggressive":
        scale = 1.4
        values["start_percent"] = max(0.0, values["start_percent"] - 0.04)
        values["end_percent"] = min(1.0, values["end_percent"] + 0.04)
        values["warmup_steps"] = max(1, values["warmup_steps"] - 1)
        values["front_blocks"] = max(1, values["front_blocks"] - 1)
        values["back_blocks"] = max(1, values["back_blocks"] - 1)
        values["max_full_step_hits"] += 1
        values["max_block_hits"] += 1
    else:
        raise ValueError(f"Unknown quality level: {quality_level}")
    for key in ("video_threshold", "audio_threshold", "fast_path_threshold", "probe_path_threshold"):
        values[key] = min(1.0, values[key] * scale)
    return values


def build_preset_config(profile: str, quality_level: str = "Balanced", *, source: str = "manual",
                        cache_device: str = "Auto", gpu_reserve_mb: int = 2048,
                        audio_content: str = "Auto", confidence: float = 0.0, analysis_summary: str = "") -> H3CacheConfig:
    if profile not in _BASE_PRESETS:
        raise ValueError(f"Unknown profile: {profile}")
    if quality_level == "Custom":
        raise ValueError("Custom configurations must be built from explicit manual values.")
    values = _quality_adjust(_BASE_PRESETS[profile], quality_level)
    return H3CacheConfig(SCHEMA_VERSION, source, profile, quality_level, cache_device=cache_device,
                         gpu_reserve_mb=int(gpu_reserve_mb), audio_content=audio_content, confidence=float(confidence),
                         analysis_summary=str(analysis_summary)[:512], **values)


def build_custom_config(profile: str, *, source: str = "manual", quality_level: str = "Custom", **values) -> H3CacheConfig:
    return H3CacheConfig(schema_version=SCHEMA_VERSION, source=source, profile=profile,
                         quality_level=quality_level, confidence=float(values.pop("confidence", 0.0)),
                         analysis_summary=str(values.pop("analysis_summary", ""))[:512], **values)


def with_profile(config: H3CacheConfig, profile: str) -> H3CacheConfig:
    """Return a deterministic replacement profile while preserving deployment choices."""
    if profile == config.profile:
        return config
    if config.quality_level == "Custom":
        return replace(config, profile=profile)
    return build_preset_config(profile, config.quality_level, source=config.source,
                               cache_device=config.cache_device, gpu_reserve_mb=config.gpu_reserve_mb,
                               audio_content=config.audio_content, confidence=config.confidence,
                               analysis_summary=config.analysis_summary)


def select_manual_profile(mode: str, audio_content: str, profile_hint: str = "") -> str:
    explicit = {
        "Visual Fast": "visual_fast", "Dialogue Safe": "dialogue_safe", "Action Safe": "action_safe",
        "Balanced": "balanced", "Off": "off",
    }
    if mode in explicit:
        return explicit[mode]
    hint = str(profile_hint).strip().lower()
    if hint in PROFILES:
        return hint
    return {
        "Speech": "dialogue_safe", "Singing": "dialogue_safe", "Music": "visual_fast",
        "Ambient": "visual_fast", "None": "visual_fast", "Auto": "balanced",
    }.get(audio_content, "balanced")
