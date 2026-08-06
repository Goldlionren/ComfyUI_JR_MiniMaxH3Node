"""Per-model H3 cache state and ComfyUI wrapper callbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .h3_cache_config import H3CacheConfig
from .h3_cache_metrics import relative_delta, tensor_signature


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
    cpu_gpu_transfers: int = 0


class H3AdaptiveCacheRuntime:
    """State belongs to one patched ModelPatcher clone, never to a module global."""

    def __init__(self, config: H3CacheConfig, block_count: int, *, audio_required: bool, verbose: bool = False):
        self.config = config
        self.block_count = int(block_count)
        self.audio_required = bool(audio_required)
        self.verbose = bool(verbose)
        self.stats = CacheStats()
        self._resolved_device = None
        self._sample_signature = None
        self._previous_timestep = None
        self._last_counted_timestep = None
        self._previous_inputs = None
        self._previous_outputs = None
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
        self._previous_inputs = None
        self._previous_outputs = None
        self._probe_video = None
        self._probe_audio = None
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
            "audio_veto=%d video_veto=%d resets=%d cache_bytes=%d transfers=%d compute_reduction=%.1f%%",
            s.total_steps, s.full_forward_count, s.full_step_cache_hits, s.block_cache_hits,
            s.forced_refresh_count, s.audio_veto_count, s.video_veto_count, s.cache_resets,
            s.cache_bytes, s.cpu_gpu_transfers, reduction,
        )
        self.reset("sampling cleanup")

    def _make_signature(self, video, audio, context, payload, model_obj) -> tuple:
        def condition_items(items):
            signature = []
            for item in items or ():
                tensors = []
                for key in ("latent", "audio_latent"):
                    tensor = item.get(key) if isinstance(item, dict) else None
                    if tensor is not None:
                        tensors.append((key, tensor_signature(tensor), int(tensor.data_ptr())))
                signature.append((item.get("kind") if isinstance(item, dict) else None,
                                  item.get("resolved_frame_index") if isinstance(item, dict) else None,
                                  tuple(tensors)))
            return tuple(signature)
        return (
            id(model_obj), tensor_signature(video), tensor_signature(audio),
            tuple(context.shape), str(context.dtype), str(context.device), int(context.data_ptr()),
            payload.get("seed"), condition_items(payload.get("refs")), condition_items(payload.get("keyframes")),
            getattr(payload.get("layout"), "signature", None),
        )

    def _begin_forward(self, video, audio, timestep, context, payload, model_obj):
        value = float(timestep.flatten()[0].detach().float().item())
        signature = self._make_signature(video, audio, context, payload, model_obj)
        restarted = self._previous_timestep is not None and value > self._previous_timestep + 1e-5
        if signature != self._sample_signature or restarted:
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
        if self._previous_inputs is None:
            return float("inf"), float("inf")
        prev_video, prev_audio = self._previous_inputs
        video_score = relative_delta(video, prev_video, self.config.video_metric_stride)
        audio_score = relative_delta(audio, prev_audio, self.config.audio_metric_stride) if self.audio_required else 0.0
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
        video_score, audio_score = self._stream_scores(video, audio)
        audio_ok = not self.audio_required or audio_score < cfg.audio_threshold
        video_ok = video_score < cfg.video_threshold
        if not audio_ok:
            self.stats.audio_veto_count += 1
        if not video_ok:
            self.stats.video_veto_count += 1
        profile = cfg.profile
        if profile == "visual_fast":
            self._path = "fast" if video_ok and audio_ok else "full"
        elif profile in ("dialogue_safe", "action_safe"):
            self._path = "probe"
        elif profile == "balanced":
            score = max(video_score, audio_score if self.audio_required else 0.0)
            if score < cfg.fast_path_threshold:
                self._path = "fast"
            elif score < cfg.probe_path_threshold:
                self._path = "probe"
            else:
                self._path = "full"
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

    def _store(self, tensor):
        if tensor is None:
            return None
        target = self._resolve_cache_device([tensor])
        stored = tensor.detach().clone()
        if target == "CPU" and stored.device.type != "cpu":
            stored = stored.to("cpu")
            self.stats.cpu_gpu_transfers += 1
        return stored

    def _restore(self, tensor, like):
        if tensor is None:
            return None
        if tensor.device != like.device:
            self.stats.cpu_gpu_transfers += 1
        return tensor.to(device=like.device, dtype=like.dtype)

    def _remember_inputs(self, video, audio):
        self._previous_inputs = (self._store(video), self._store(audio))

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
        return [self._restore(self._previous_outputs[0], video), self._restore(self._previous_outputs[1], audio)]

    def remember_full_forward(self, video, audio, outputs, *, counted_as_full: bool = True):
        if self._previous_outputs is not None:
            self.metrics["video_output_change"] = relative_delta(outputs[0], self._previous_outputs[0].to(outputs[0].device), self.config.video_metric_stride)
            self.metrics["audio_output_change"] = relative_delta(outputs[1], self._previous_outputs[1].to(outputs[1].device), self.config.audio_metric_stride)
        self._previous_outputs = (self._store(outputs[0]), self._store(outputs[1]))
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
        video_score = relative_delta(video, self._probe_video.to(video.device), self.config.video_metric_stride) if self._probe_video is not None else float("inf")
        audio_score = relative_delta(audio, self._probe_audio.to(audio.device), self.config.audio_metric_stride) if self._probe_audio is not None and self.audio_required else 0.0
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
            self._probe_video = self._store(video)
            self._probe_audio = self._store(audio)
            self._skip_middle = True
            return True
        self._block_hit_streak = 0
        self._middle_entry = self._store(h)
        self._probe_video = self._store(video)
        self._probe_audio = self._store(audio)
        return False

    def apply_middle_hit(self, h):
        return h + self._restore(self._middle_residual, h)

    def finish_middle(self, h):
        if self._middle_entry is not None:
            entry = self._restore(self._middle_entry, h)
            self._middle_residual = self._store(h - entry)
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
        return (f"{self.config.profile} | configured | {self._resolved_device or self.config.cache_device} cache | "
                f"Configuration source: {self.config.source}")
