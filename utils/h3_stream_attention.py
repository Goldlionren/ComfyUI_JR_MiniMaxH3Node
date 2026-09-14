"""Native H3 layout slicing and execution-local persistent-history attention."""

import time

import torch

from .h3_temporal_transport import layer_step_weight, schedule_progress, transform_video_q


def phase_layout(plan, phase, text_len, height, width, payload):
    from comfy.ldm.minimax.model import PackedLayout

    common = dict(keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    full = PackedLayout(text_len, plan.video_latent_count, height, width, plan.audio_latent_count, **common)
    local = PackedLayout(text_len, phase.video_latent_count, height, width, phase.audio_latent_count, **common)
    positions = []
    for start, stop, kind in full.segments:
        pos = full.position_ids[start:stop]
        if kind == "video":
            pos = pos.reshape(plan.video_latent_count, -1, 3)[phase.video_latent_start:phase.video_latent_stop].reshape(-1, 3)
        elif kind == "audio":
            pos = pos.reshape(2, plan.audio_latent_count, 3)[:, phase.audio_latent_start:phase.audio_latent_stop].reshape(-1, 3)
        positions.append(pos)
    local.position_ids = torch.cat(positions)
    if local.position_ids.shape != (local.seq_len, 3):
        raise ValueError("JR H3 Streaming: phase position rows mismatch")
    return local


class StreamingRuntime:
    def __init__(self, plan, cache, total_layers, heads, previous, *, use_history, collect, clean_enabled,
                 tst_strength=0., schedule=()):
        self.plan, self.cache = plan, cache
        self.total_layers, self.heads, self.previous = total_layers, heads, previous
        self.use_history, self.collect, self.clean_enabled = use_history, collect, clean_enabled
        self.tst_strength, self.schedule = tst_strength, schedule
        self.phase = None
        self.clean = False
        self.active = False
        self.head_counts = {}
        self.layout = None
        self.progress = 0.
        self.denoise_forwards = self.clean_forwards = self.kernel_calls = 0
        self.attention_host_seconds = self.commit_seconds = self.trim_seconds = 0.

    def forward(self, executor, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
        if self.active or self.phase is None:
            raise RuntimeError("JR H3 Streaming: invalid forward lifecycle")
        opts = dict(transformer_options or {})
        if opts.get("optimized_attention_override") is not self:
            raise RuntimeError("JR H3 Streaming: another patch replaced streaming attention")
        payload = dict(minimax_payload or {})
        v, a = x
        if v.shape[2] != self.phase.video_latent_count or a.shape[-1] != self.phase.audio_latent_count:
            raise ValueError("JR H3 Streaming: active phase AV shape mismatch")
        self.layout = phase_layout(self.plan, self.phase, context.shape[1], (v.shape[3] + 1) // 2 * 2,
                                   (v.shape[4] + 1) // 2 * 2, payload)
        payload["layout"] = self.layout
        self.head_counts = {}
        self.progress = schedule_progress(self.schedule, float(timestep.flatten()[0]) / 1000.)
        if self.clean:
            if bool(timestep.ne(0).any()):
                raise RuntimeError("JR H3 Streaming: clean commit requires sigma zero")
            self.clean_forwards += 1
        else:
            self.denoise_forwards += 1
        self.active = True
        try:
            result = executor(x, timestep, context, opts, minimax_payload=payload, **kwargs)
            if self.head_counts != dict.fromkeys(range(self.total_layers), self.heads):
                raise RuntimeError("JR H3 Streaming: attention skipped a layer/head group; cache cannot be trusted")
            return result
        finally:
            self.active = False
            self.head_counts = {}
            self.layout = None

    def __call__(self, func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        if not self.active or mask is not None or not skip_reshape or q.ndim != 4 or q.shape != k.shape or k.shape != v.shape:
            raise RuntimeError("JR H3 Streaming: requires native unmasked head-major QKV")
        options = kwargs.get("transformer_options", {})
        layer = options.get("block_index")
        if type(layer) is not int or not 0 <= layer < self.total_layers or q.shape[1] != heads:
            raise RuntimeError("JR H3 Streaming: invalid native layer/head contract")
        start_head = self.head_counts.get(layer, 0)
        if start_head + heads > self.heads or q.shape[2] != self.layout.seq_len:
            raise RuntimeError("JR H3 Streaming: repeated/reordered attention or layout mismatch")
        self.head_counts[layer] = start_head + heads
        aa, ab, _ = next(s for s in self.layout.segments if s[2] == "audio")
        va, vb, _ = next(s for s in self.layout.segments if s[2] == "video")
        if not 0 < aa < ab == va < vb == q.shape[2]:
            raise RuntimeError("JR H3 Streaming: expected conditioning, stereo audio, video order")
        if self.clean and self.collect and layer in self.cache.layers:
            self.cache.stage(layer, start_head, k[:, :, va:vb], v[:, :, va:vb], k[:, :, aa:ab], v[:, :, aa:ab])
        # Only noisy current-video Q is transported. Clean K/V is captured
        # before any transport and the entire clean pass bypasses TST.
        if not self.clean and self.tst_strength:
            q = transform_video_q(q, k, start=va, stop=vb, frames=self.phase.video_latent_count,
                                  strength=self.tst_strength,
                                  weight=layer_step_weight(layer, self.total_layers, self.progress), scale=kwargs.get("scale"))

        def delegate(qv, kv, vv):
            begin = time.perf_counter()
            # Comfy removes the override before calling its underlying function.
            # Keep this explicit for third-party delegates to avoid recursion.
            clean_options = dict(options)
            clean_options.pop("optimized_attention_override", None)
            if self.use_history:
                clean_options.pop("minimax_h3_layout", None)  # no stale Sol sink offsets on sliced rows
            call_kwargs = dict(kwargs, transformer_options=clean_options)
            target = func if self.previous is None else lambda *args, **kw: self.previous(func, *args, **kw)
            out = target(qv, kv, vv, heads, mask=None, attn_precision=attn_precision,
                         skip_reshape=True, skip_output_reshape=True, **call_kwargs)
            self.attention_host_seconds += time.perf_counter() - begin
            self.kernel_calls += 1
            if out.shape != qv.shape:
                raise RuntimeError("JR H3 Streaming: backend returned unexpected head-major shape")
            return out

        if self.use_history:
            history_k, history_v = self.cache.history(layer, start_head, heads, device=k.device, dtype=k.dtype)
            cond = delegate(q[:, :, :aa], k[:, :, :aa], v[:, :, :aa])
            keys = torch.cat([k[:, :, :aa], *history_k, k[:, :, aa:]], dim=2)
            values = torch.cat([v[:, :, :aa], *history_v, v[:, :, aa:]], dim=2)
            media = delegate(q[:, :, aa:], keys, values)
            result = torch.cat((cond, media), dim=2)
        else:
            result = delegate(q, k, v)
        return result if skip_output_reshape else result.transpose(1, 2).reshape(1, q.shape[2], -1)

    def clear(self):
        self.cache.clear()
        self.phase = self.layout = None
        self.head_counts = {}
        self.clean = self.active = False


class CleanCommitSampler:
    """Run clean forward inside native loaded-model lifetime, discard prediction."""

    def __init__(self, sampler, runtime):
        self.sampler, self.runtime = sampler, runtime

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        output = self.sampler.sample(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        if denoise_mask is not None:
            output = torch.where(denoise_mask == 0, latent_image, output)
        r = self.runtime
        if r.clean_enabled:
            begin = time.perf_counter()
            r.clean = True
            try:
                if r.collect:
                    r.cache.begin_clean_commit(r.phase.phase_index)
                # output is in the native internal AV carry domain here, not
                # VAE latent space. Native MiniMaxH3.forward reverses that carry.
                # A clone ensures cache-only evaluation cannot alter the output.
                prediction = model_wrap(output.clone(), sigmas.new_zeros((1,)),
                                        model_options=extra_args["model_options"], seed=extra_args.get("seed"))
                del prediction
                if r.collect:
                    r.cache.commit()
                r.commit_seconds += time.perf_counter() - begin
                trim_start = time.perf_counter()
                r.cache.trim()
                r.trim_seconds += time.perf_counter() - trim_start
            except BaseException:
                r.cache.rollback()
                raise
            finally:
                r.clean = False
        return output
