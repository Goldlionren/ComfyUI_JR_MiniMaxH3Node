"""Per-model H3 cache state and ComfyUI wrapper callbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .h3_cache_config import H3CacheConfig
from .h3_cache_metrics import metric_sample, relative_delta, tensor_signature


@dataclass
class ScoreStats:
    count: int = 0
    total: float = 0.0
    minimum: float = float("inf")
    maximum: float = 0.0

    def observe(self, value: float):
        if value == float("inf"):
            return
        value = float(value)
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def minimum_or_zero(self) -> float:
        return self.minimum if self.count else 0.0


@dataclass
class CacheStats:
    total_steps: int = 0
    full_forward_count: int = 0
    full_step_cache_hits: int = 0
    block_cache_hits: int = 0
    forced_refresh_count: int = 0
    audio_veto_count: int = 0
    video_veto_count: int = 0
    cache_resets: int = 0
    cache_bytes: int = 0
    residual_to_cpu: int = 0
    residual_to_gpu: int = 0
    metric_migrations: int = 0
    input_video_scores: ScoreStats = field(default_factory=ScoreStats)
    input_audio_scores: ScoreStats = field(default_factory=ScoreStats)
    probe_video_scores: ScoreStats = field(default_factory=ScoreStats)
    probe_audio_scores: ScoreStats = field(default_factory=ScoreStats)

    @property
    def cpu_gpu_transfers(self) -> int:
        """Backward-compatible aggregate for benchmark consumers."""
        return self.residual_to_cpu + self.residual_to_gpu


class H3AdaptiveCacheRuntime:
    """State belongs to one patched ModelPatcher clone, never to a module global."""

    def __init__(self, config: H3CacheConfig, block_count: int, *, audio_required: bool, verbose: bool = False):
        self.config = config
        self.block_count = int(block_count)
        self.audio_required = bool(audio_required)
        self.verbose = bool(verbose)
        self.stats = CacheStats()
        self._resolved_device = None
        self._metric_device = None
        self._sample_signature = None
        self._previous_timestep = None
        self._last_counted_timestep = None
        self._previous_input_metrics = None
        self._previous_outputs = None
        self._previous_output_metrics = None
        self._full_hit_streak = 0
        self._block_hit_streak = 0
        self._path = "full"
        self._layout = None
        self._video_slice = None
        self._audio_slice = None
        self._probe_video = None
        self._probe_audio = None
        self._middle_entry = None
        self._middle_residual = None
        self._skip_middle = False
        self._video_cumulative_change = 0.0
        self._audio_cumulative_change = 0.0
        self.metrics = {}

    @property
    def front_blocks(self):
        return min(self.config.front_blocks, self.block_count - 1)

    @property
    def back_blocks(self):
        return min(self.config.back_blocks, self.block_count - self.front_blocks - 1)

    @property
    def middle_start(self):
        return self.front_blocks

    @property
    def middle_end(self):
        return self.block_count - self.back_blocks - 1

    def reset(self, reason: str = "reset", *, keep_stats: bool = True):
        if keep_stats:
            self.stats.cache_resets += 1
        self._sample_signature = None
        self._previous_timestep = None
        self._last_counted_timestep = None
        self._previous_input_metrics = None
        self._previous_outputs = None
        self._previous_output_metrics = None
        self._probe_video = None
        self._probe_audio = None
        self._metric_device = None
        self._middle_entry = None
        self._middle_residual = None
        self._full_hit_streak = 0
        self._block_hit_streak = 0
        self._video_cumulative_change = 0.0
        self._audio_cumulative_change = 0.0
        self._path = "full"
        self._skip_middle = False
        if self.verbose:
            logging.info("JR H3 Adaptive Cache reset: %s", reason)

    def cleanup(self, *_args, **_kwargs):
        s = self.stats
        reduction = 0.0
        possible = max(1, s.total_steps * self.block_count)
        skipped = s.full_step_cache_hits * self.block_count + s.block_cache_hits * max(0, self.block_count - self.front_blocks - self.back_blocks)
        reduction = 100.0 * skipped / possible
        logging.info(
            "JR H3 Adaptive Cache summary: steps=%d full=%d full_hits=%d block_hits=%d forced=%d "
            "audio_veto=%d video_veto=%d resets=%d cache_bytes=%d residual_device=%s metric_device=%s "
            "residual_to_cpu=%d residual_to_gpu=%d metric_migrations=%d "
            "input_video[n=%d min=%.5f avg=%.5f max=%.5f] "
            "input_audio[n=%d min=%.5f avg=%.5f max=%.5f] "
            "probe_video[n=%d min=%.5f avg=%.5f max=%.5f] "
            "probe_audio[n=%d min=%.5f avg=%.5f max=%.5f] compute_reduction=%.1f%%",
            s.total_steps, s.full_forward_count, s.full_step_cache_hits, s.block_cache_hits,
            s.forced_refresh_count, s.audio_veto_count, s.video_veto_count, s.cache_resets,
            s.cache_bytes, self._resolved_device or self.config.cache_device,
            self._metric_device or "uninitialized", s.residual_to_cpu, s.residual_to_gpu,
            s.metric_migrations,
            s.input_video_scores.count, s.input_video_scores.minimum_or_zero,
            s.input_video_scores.average, s.input_video_scores.maximum,
            s.input_audio_scores.count, s.input_audio_scores.minimum_or_zero,
            s.input_audio_scores.average, s.input_audio_scores.maximum,
            s.probe_video_scores.count, s.probe_video_scores.minimum_or_zero,
            s.probe_video_scores.average, s.probe_video_scores.maximum,
            s.probe_audio_scores.count, s.probe_audio_scores.minimum_or_zero,
            s.probe_audio_scores.average, s.probe_audio_scores.maximum, reduction,
        )
        self.reset("sampling cleanup", keep_stats=False)
        self.stats = CacheStats()

    def _make_signature(self, video, audio, context, payload, model_obj) -> tuple:
        def condition_items(items):
            signature = []
            for item in items or ():
                tensors = []
                for key in ("latent", "audio_latent"):
                    tensor = item.get(key) if isinstance(item, dict) else None
                    if tensor is not None:
                        tensors.append((key, tensor_signature(tensor)))
                metadata = tuple(
                    (key, item.get(key)) for key in (
                        "kind", "resolved_frame_index", "latent_t", "latent_h", "latent_w", "ref_audio_t"
                    ) if isinstance(item, dict) and item.get(key) is not None
                )
                signature.append((metadata, tuple(tensors)))
            return tuple(signature)
        layout = payload.get("layout")
        return (
            id(model_obj), tensor_signature(video), tensor_signature(audio),
            tensor_signature(context),
            payload.get("seed"), condition_items(payload.get("refs")), condition_items(payload.get("keyframes")),
            getattr(layout, "signature", None), tuple(getattr(layout, "segments", ())),
        )

    def _begin_forward(self, video, audio, timestep, context, payload, model_obj):
        value = float(timestep.flatten()[0].detach().float().item())
        signature = self._make_signature(video, audio, context, payload, model_obj)
        restarted = self._previous_timestep is not None and value > self._previous_timestep + 1e-5
        if self._sample_signature is None:
            self._sample_signature = signature
        elif signature != self._sample_signature or restarted:
            self.reset("new sampling signature" if signature != self._sample_signature else "timestep restart")
            self._sample_signature = signature
        self._previous_timestep = value
        if self._last_counted_timestep is None or abs(value - self._last_counted_timestep) > 1e-6:
            self.stats.total_steps += 1
            self._last_counted_timestep = value
        self._layout = payload.get("layout")
        self._set_target_slices()
        return max(0.0, min(1.0, 1.0 - value / 1000.0))

    def _set_target_slices(self):
        self._video_slice = self._audio_slice = None
        if self._layout is None:
            return
        for start, stop, kind in self._layout.segments:
            if kind == "video":
                self._video_slice = slice(start, stop)
            elif kind == "audio":
                self._audio_slice = slice(start, stop)

    def _stream_scores(self, video, audio):
        if self._previous_input_metrics is None:
            return float("inf"), float("inf")
        prev_video, prev_audio = self._previous_input_metrics
        current_video = metric_sample(video, self.config.video_metric_stride)
        video_score = relative_delta(current_video, prev_video)
        if self.audio_required:
            current_audio = metric_sample(audio, self.config.audio_metric_stride)
            audio_score = relative_delta(current_audio, prev_audio)
        else:
            audio_score = 0.0
        self.stats.input_video_scores.observe(video_score)
        if self.audio_required:
            self.stats.input_audio_scores.observe(audio_score)
        self._video_cumulative_change += video_score if video_score != float("inf") else 0.0
        self._audio_cumulative_change += audio_score if audio_score != float("inf") else 0.0
        self.metrics.update(video_input_change=video_score, audio_input_change=audio_score,
                            video_predicted_change=video_score, audio_predicted_change=audio_score,
                            video_cumulative_change=self._video_cumulative_change,
                            audio_cumulative_change=self._audio_cumulative_change)
        return video_score, audio_score

    def choose_path(self, video, audio, progress: float) -> str:
        cfg = self.config
        if self.stats.total_steps <= cfg.warmup_steps or not cfg.start_percent <= progress <= cfg.end_percent:
            self._path = "full"
            return self._path
        profile = cfg.profile
        if profile in ("dialogue_safe", "action_safe"):
            self._path = "probe"
            return self._path
        video_score, audio_score = self._stream_scores(video, audio)
        audio_ok = not self.audio_required or audio_score < cfg.audio_threshold
        video_ok = video_score < cfg.video_threshold
        if profile == "visual_fast":
            if not audio_ok:
                self.stats.audio_veto_count += 1
            if not video_ok:
                self.stats.video_veto_count += 1
            self._path = "fast" if video_ok and audio_ok else "full"
        elif profile == "balanced":
            score = max(video_score, audio_score if self.audio_required else 0.0)
            if score < cfg.fast_path_threshold:
                self._path = "fast"
            elif score < cfg.probe_path_threshold:
                self._path = "probe"
            else:
                self._path = "full"
                if video_score >= cfg.probe_path_threshold:
                    self.stats.video_veto_count += 1
                if self.audio_required and audio_score >= cfg.probe_path_threshold:
                    self.stats.audio_veto_count += 1
        else:
            self._path = "full"
        return self._path

    def _cache_size(self, tensors) -> int:
        return sum(int(t.numel() * t.element_size()) for t in tensors if t is not None)

    def _resolve_cache_device(self, tensors) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        requested = self.config.cache_device
        if requested != "Auto":
            self._resolved_device = requested
            return requested
        first = next((t for t in tensors if t is not None), None)
        if first is None or first.device.type != "cuda":
            self._resolved_device = "CPU"
            return self._resolved_device
        try:
            import torch
            free_bytes, _ = torch.cuda.mem_get_info(first.device)
            reserve = self.config.gpu_reserve_mb * 1024 * 1024
            self._resolved_device = "GPU" if free_bytes - reserve > self._cache_size(tensors) * 2 else "CPU"
        except Exception:
            self._resolved_device = "CPU"
        return self._resolved_device

    def _store_residual(self, tensor):
        if tensor is None:
            return None
        target = self._resolve_cache_device([tensor])
        stored = tensor.detach().clone()
        if target == "CPU" and stored.device.type != "cpu":
            stored = stored.to("cpu")
            self.stats.residual_to_cpu += 1
        return stored

    def _restore_residual(self, tensor, like):
        if tensor is None:
            return None
        if tensor.device != like.device:
            if tensor.device.type == "cpu" and like.device.type != "cpu":
                self.stats.residual_to_gpu += 1
            elif tensor.device.type != "cpu" and like.device.type == "cpu":
                self.stats.residual_to_cpu += 1
        return tensor.to(device=like.device, dtype=like.dtype)

    def _store_metric(self, tensor, stride: int):
        if tensor is None:
            return None
        stored = metric_sample(tensor, stride)
        self._metric_device = str(tensor.device)
        return stored

    def _remember_inputs(self, video, audio):
        self._previous_input_metrics = (
            self._store_metric(video, self.config.video_metric_stride),
            self._store_metric(audio, self.config.audio_metric_stride) if self.audio_required else None,
        )

    def can_full_hit(self, video, audio) -> bool:
        if self._path != "fast" or self._previous_outputs is None or self.config.max_full_step_hits <= 0:
            return False
        if self._full_hit_streak >= self.config.max_full_step_hits:
            self.stats.forced_refresh_count += 1
            self._full_hit_streak = 0
            return False
        self._full_hit_streak += 1
        self._block_hit_streak = 0
        self.stats.full_step_cache_hits += 1
        self._remember_inputs(video, audio)
        return True

    def full_hit_output(self, video, audio):
        return [self._restore_residual(self._previous_outputs[0], video),
                self._restore_residual(self._previous_outputs[1], audio)]

    def remember_full_forward(self, video, audio, outputs, *, counted_as_full: bool = True):
        current_output_metrics = (
            self._store_metric(outputs[0], self.config.video_metric_stride),
            self._store_metric(outputs[1], self.config.audio_metric_stride),
        )
        if self._previous_output_metrics is not None:
            self.metrics["video_output_change"] = relative_delta(
                current_output_metrics[0], self._previous_output_metrics[0])
            self.metrics["audio_output_change"] = relative_delta(
                current_output_metrics[1], self._previous_output_metrics[1])
        self._previous_output_metrics = current_output_metrics
        self._previous_outputs = (self._store_residual(outputs[0]), self._store_residual(outputs[1]))
        self._remember_inputs(video, audio)
        self._full_hit_streak = 0
        if counted_as_full:
            self.stats.full_forward_count += 1
        self.stats.cache_bytes = self._cache_size([*(self._previous_outputs or ()), self._middle_residual])

    def begin_middle(self, h):
        self._skip_middle = False
        if self._path != "probe" or self._video_slice is None or self._audio_slice is None:
            self._middle_entry = None
            return False
        video = h[self._video_slice]
        audio = h[self._audio_slice]
        current_video = self._store_metric(video, self.config.video_metric_stride)
        current_audio = self._store_metric(audio, self.config.audio_metric_stride) if self.audio_required else None
        video_score = relative_delta(current_video, self._probe_video) if self._probe_video is not None else float("inf")
        audio_score = (relative_delta(current_audio, self._probe_audio)
                       if self._probe_audio is not None and self.audio_required else 0.0)
        self.stats.probe_video_scores.observe(video_score)
        if self.audio_required:
            self.stats.probe_audio_scores.observe(audio_score)
        video_ok = video_score < self.config.video_threshold
        audio_ok = not self.audio_required or audio_score < self.config.audio_threshold
        if not video_ok:
            self.stats.video_veto_count += 1
        if not audio_ok:
            self.stats.audio_veto_count += 1
        limit_ok = self._block_hit_streak < self.config.max_block_hits
        if not limit_ok and self._middle_residual is not None:
            self.stats.forced_refresh_count += 1
        if self._middle_residual is not None and video_ok and audio_ok and limit_ok:
            self._block_hit_streak += 1
            self._full_hit_streak = 0
            self.stats.block_cache_hits += 1
            self._probe_video = current_video
            self._probe_audio = current_audio
            self._skip_middle = True
            return True
        self._block_hit_streak = 0
        # This entry is consumed later in the same forward. Keeping it on the
        # compute device avoids a CPU round trip that is unrelated to a hit.
        self._middle_entry = h.detach().clone()
        self._probe_video = current_video
        self._probe_audio = current_audio
        return False

    def apply_middle_hit(self, h):
        return h + self._restore_residual(self._middle_residual, h)

    def finish_middle(self, h):
        if self._middle_entry is not None:
            self._middle_residual = self._store_residual(h - self._middle_entry)
            self.stats.cache_bytes = self._cache_size([self._middle_residual, *(self._previous_outputs or ())])
        self._middle_entry = None

    def diffusion_wrapper(self, executor, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
        transformer_options = transformer_options or {}
        payload = minimax_payload or {}
        video, audio = x[0], x[1]
        try:
            progress = self._begin_forward(video, audio, timestep, context, payload, executor.class_obj)
            self.choose_path(video, audio, progress)
            if self.can_full_hit(video, audio):
                return self.full_hit_output(video, audio)
            block_hits_before = self.stats.block_cache_hits
            outputs = executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
            self.remember_full_forward(video, audio, outputs,
                                       counted_as_full=self.stats.block_cache_hits == block_hits_before)
            return outputs
        except Exception:
            self.reset("forward exception")
            raise

    def block_wrapper(self, index: int):
        def wrapper(args, extra):
            original = extra["original_block"]
            h = args["img"]
            if index == self.middle_start:
                if self.begin_middle(h):
                    return {"img": self.apply_middle_hit(h)}
            elif self._skip_middle and self.middle_start < index <= self.middle_end:
                return {"img": h}
            result = original(args)
            if index == self.middle_end and not self._skip_middle:
                self.finish_middle(result["img"])
            return result
        return wrapper

    def status(self) -> str:
        return (f"{self.config.profile} | configured | Residual cache device: "
                f"{self._resolved_device or self.config.cache_device} | Metric state device: "
                f"{self._metric_device or 'active compute device'} | Configuration source: {self.config.source}")
