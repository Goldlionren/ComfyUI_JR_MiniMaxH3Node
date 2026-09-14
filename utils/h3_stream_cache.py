"""Execution-owned transactional BF16 AV cache. No global model/tensor state."""

from dataclasses import dataclass, replace

import torch

RETENTIONS = ("previous_only", "sink_plus_recent_1", "sink_plus_recent_2")
LAYER_POLICIES = ("all", "every_2", "every_4", "custom")


def select_layers(policy, total, custom=""):
    if type(total) is not int or total < 1 or policy not in LAYER_POLICIES:
        raise ValueError("JR H3 Streaming: invalid layer policy/count")
    if policy == "custom":
        try:
            indices = [int(part.strip()) for part in custom.split(",")]
        except (ValueError, AttributeError) as exc:
            raise ValueError("JR H3 Streaming: custom layers must be comma-separated integers") from exc
        if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= total:
            raise ValueError("JR H3 Streaming: custom layer indices must be unique and in range")
        return tuple(sorted(indices))
    return tuple(range(0, total, {"all": 1, "every_2": 2, "every_4": 4}[policy]))


@dataclass(frozen=True)
class AVCommit:
    index: int
    video_k: torch.Tensor
    video_v: torch.Tensor
    audio_k: torch.Tensor | None
    audio_v: torch.Tensor | None

    @property
    def video_tokens(self):
        return self.video_k.shape[2]

    @property
    def audio_tokens(self):
        return 0 if self.audio_k is None else self.audio_k.shape[2]

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.video_k, self.video_v, self.audio_k, self.audio_v) if t is not None)

    def without_audio(self):
        return replace(self, audio_k=None, audio_v=None)


class JR_H3_CleanAVKVCache:
    def __init__(self, layers, heads, *, retention="sink_plus_recent_2", storage="cpu", max_bytes=8 * 1024**3):
        if retention not in RETENTIONS or storage not in ("cpu", "cuda"):
            raise ValueError("JR H3 Streaming: unsupported cache retention/storage")
        if not layers or len(set(layers)) != len(layers) or heads < 1 or max_bytes <= 0:
            raise ValueError("JR H3 Streaming: invalid cache contract")
        self.layers, self.heads = tuple(layers), heads
        self.retention, self.storage, self.max_bytes = retention, storage, max_bytes
        self._committed = {}
        self._staged = {}
        self._active = None
        self._last_index = -1

    @property
    def active(self):
        return self._active is not None

    def begin_clean_commit(self, index):
        if self.active or type(index) is not int or index <= self._last_index:
            raise RuntimeError("JR H3 Streaming: invalid/reentrant clean commit")
        self._active = index
        self._staged = {}

    def stage(self, layer, head_start, video_k, video_v, audio_k, audio_v):
        if not self.active or layer not in self.layers:
            raise RuntimeError("JR H3 Streaming: stage requires active selected clean layer")
        tensors = (video_k, video_v, audio_k, audio_v)
        if any(t.ndim != 4 or t.shape[0] != 1 or t.shape[2] < 1 for t in tensors):
            raise ValueError("JR H3 Streaming: invalid AV KV shapes")
        if video_k.shape != video_v.shape or audio_k.shape != audio_v.shape:
            raise ValueError("JR H3 Streaming: K/V shape mismatch")
        if any((t.shape[1], t.shape[3], t.device, t.dtype) !=
               (video_k.shape[1], video_k.shape[3], video_k.device, video_k.dtype) for t in tensors):
            raise ValueError("JR H3 Streaming: AV KV dtype/device/head mismatch")
        parts = self._staged.get(layer, ())
        expected_start = sum(p.video_k.shape[1] for p in parts)
        if head_start != expected_start or head_start + video_k.shape[1] > self.heads:
            raise RuntimeError("JR H3 Streaming: duplicate, missing or reordered head group")
        if parts and (parts[0].video_tokens != video_k.shape[2] or parts[0].audio_tokens != audio_k.shape[2]):
            raise RuntimeError("JR H3 Streaming: changing AV rows during commit")
        if any(not t.is_floating_point() or not bool(torch.isfinite(t).all()) for t in tensors):
            raise ValueError("JR H3 Streaming: non-finite KV")
        added = sum(t.numel() * 2 for t in tensors)
        if self.nbytes + self.staged_bytes + added > self.max_bytes:
            raise ValueError("JR H3 Streaming: KV budget exceeded; reduce cached layers/resolution or change retention")
        # Own compact storage, including on same device/dtype: never hold a slice
        # retaining the entire noisy qkv allocation. No autograd history survives.
        device = torch.device("cpu") if self.storage == "cpu" else video_k.device
        if self.storage == "cuda" and device.type != "cuda":
            raise ValueError("JR H3 Streaming: CUDA cache requires CUDA attention")
        owned = [t.detach().to(device=device, dtype=torch.bfloat16, copy=True).contiguous() for t in tensors]
        part = AVCommit(self._active, *owned)
        self._staged[layer] = tuple(parts) + (part,)

    @property
    def staged_bytes(self):
        return sum(p.nbytes for parts in self._staged.values() for p in parts)

    @property
    def nbytes(self):
        return sum(p.nbytes for parts in self._committed.values() for p in parts)

    def commit(self):
        if not self.active or set(self._staged) != set(self.layers):
            raise RuntimeError("JR H3 Streaming: incomplete clean layers; rollback required")
        if any(sum(p.video_k.shape[1] for p in parts) != self.heads for parts in self._staged.values()):
            raise RuntimeError("JR H3 Streaming: incomplete clean heads; rollback required")
        shapes = {(p.video_tokens, p.audio_tokens, p.video_k.shape[-1])
                  for parts in self._staged.values() for p in parts}
        if len(shapes) != 1:
            raise RuntimeError("JR H3 Streaming: inconsistent AV rows across clean layers")
        candidate = dict(self._committed)
        for layer, parts in self._staged.items():
            fields = ("video_k", "video_v", "audio_k", "audio_v")
            merged = parts[0] if len(parts) == 1 else AVCommit(self._active, *[
                torch.cat([getattr(p, name) for p in parts], dim=1) for name in fields])
            candidate[layer] = candidate.get(layer, ()) + (merged,)
        # Only publish after every layer is complete and allocated successfully.
        self._committed = candidate
        self._last_index = self._active
        self._active = None
        self._staged = {}

    def rollback(self):
        self._active = None
        self._staged = {}

    def trim(self):
        if self.active:
            raise RuntimeError("JR H3 Streaming: cannot trim staged cache")
        recent = 1 if self.retention != "sink_plus_recent_2" else 2
        candidate = {}
        for layer, parts in self._committed.items():
            keep = parts[-recent:]
            if self.retention != "previous_only" and parts and parts[0].index != keep[0].index:
                sink = parts[0].without_audio()
                if all(p.index != sink.index for p in keep):
                    keep = (sink,) + keep
            candidate[layer] = keep
        self._committed = candidate

    def drop_audio_history(self):
        if self.active:
            raise RuntimeError("JR H3 Streaming: cannot reset audio during commit")
        self._committed = {layer: tuple(p.without_audio() for p in parts)
                           for layer, parts in self._committed.items()}

    def history(self, layer, head_start, heads, *, device, dtype):
        keys, values = [], []
        for p in self._committed.get(layer, ()):
            for k, v in ((p.audio_k, p.audio_v), (p.video_k, p.video_v)):
                if k is not None:
                    keys.append(k[:, head_start:head_start + heads].to(device=device, dtype=dtype))
                    values.append(v[:, head_start:head_start + heads].to(device=device, dtype=dtype))
        return keys, values

    def metrics(self):
        parts = next(iter(self._committed.values()), ())
        video, audio = sum(p.video_tokens for p in parts), sum(p.audio_tokens for p in parts)
        return {"history_tokens": video + audio, "video_history_tokens": video, "audio_history_tokens": audio,
                "kv_bytes": self.nbytes, "kv_mib": self.nbytes / 1024**2, "kv_gib": self.nbytes / 1024**3,
                "cached_layers": len(self._committed), "retained_commits": len(parts)}

    def clear(self):
        self.rollback()
        self._committed = {}
        self._last_index = -1
