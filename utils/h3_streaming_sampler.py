"""One-request experimental H3 streaming orchestration; production path untouched."""

import logging
import time

import torch

from .h3_stream_attention import CleanCommitSampler, StreamingRuntime
from .h3_stream_cache import JR_H3_CleanAVKVCache, select_layers
from .h3_stream_plan import validate_plan

MODES = ("Geometry Only", "Micro Chunk", "Clean Commit", "Clean KV", "Streaming Attention", "Sparse KV")
LOGGER = logging.getLogger(__name__)


def validate_inputs(model, positive, noise, sampler, sigmas, latent, plan):
    from comfy.k_diffusion.sampling import sample_euler
    from comfy.latent_formats import MiniMaxH3AV
    from comfy.model_base import MiniMaxH3
    from comfy.model_sampling import CONST, ModelSamplingAV
    from comfy.samplers import KSAMPLER

    from .h3_av_latent_builder import build_h3_av_latent
    from .h3_progressive_sampler import _resolve_noise_factory

    validate_plan(plan)
    if not isinstance(model.model, MiniMaxH3):
        raise ValueError("JR H3 Streaming: connect native MiniMax H3 MODEL")
    sampling = model.get_model_object("model_sampling")
    if not isinstance(sampling, ModelSamplingAV) or not isinstance(sampling, CONST):
        raise ValueError("JR H3 Streaming: native AV flow sigma contract required")
    if (sampling.multiplier != 1000 or not isinstance(model.get_model_object("latent_format"), MiniMaxH3AV)
            or not 0 < sampling.audio_scale < float("inf")):
        raise ValueError("JR H3 Streaming: unsupported H3 timestep/latent-format/audio-scale contract")
    if type(sampler) is not KSAMPLER or sampler.sampler_function is not sample_euler or sampler.inpaint_options:
        raise ValueError("JR H3 Streaming: only standard Euler is supported initially")
    if sampler.extra_options.get("s_churn", 0) != 0 or set(sampler.extra_options) - {"s_churn", "s_tmin", "s_tmax", "s_noise"}:
        raise ValueError("JR H3 Streaming: custom Euler options are unsupported")
    if not isinstance(sigmas, torch.Tensor) or not sigmas.is_floating_point() or sigmas.ndim != 1 or len(sigmas) < 2:
        raise ValueError("JR H3 Streaming: invalid sigmas")
    s = sigmas.detach().float().cpu()
    if not bool(torch.isfinite(s).all()) or not 0 < s[0] <= 1 or s[-1] != 0 or not bool((s[:-1] > s[1:]).all()):
        raise ValueError("JR H3 Streaming: finite decreasing sigmas in [0,1] ending at zero required")
    if len(positive) != 1 or not isinstance(positive[0][1], dict):
        raise ValueError("JR H3 Streaming: one positive conditioning entry required")
    forbidden = {"area", "mask", "control", "hooks", "start_percent", "end_percent"} & positive[0][1].keys()
    if forbidden:
        raise ValueError(f"JR H3 Streaming: unsupported regional/scheduled conditioning {sorted(forbidden)}")
    _resolve_noise_factory(noise)  # official provider identity, including native path-loader copies
    if type(noise.seed) is not int or not 0 <= noise.seed < 2**64:
        raise ValueError("JR H3 Streaming: invalid noise seed")
    samples = latent.get("samples")
    if not getattr(samples, "is_nested", False) or len(samples.unbind()) != 2:
        raise ValueError("JR H3 Streaming: expected official two-stream AV LATENT")
    video, audio = samples.unbind()
    build_h3_av_latent({"samples": video}, {"samples": audio})
    if video.shape[0] != 1 or video.shape[2] != plan.video_latent_count or audio.shape[-1] != plan.audio_latent_count:
        raise ValueError("JR H3 Streaming: canonical plan requires batch 1, 37 video latents / 207 audio latents (124 frames)")
    for kf in positive[0][1].get("minimax_keyframes", ()):
        index = kf.get("resolved_frame_index")
        ref = kf.get("latent")
        if type(index) is not int or not 0 <= index < plan.native_frame_count:
            raise ValueError("JR H3 Streaming: keyframe indices must be resolved on the complete 124-frame timeline")
        if ref is not None and ref.shape[-2:] != video.shape[-2:]:
            raise ValueError("JR H3 Streaming: keyframe conditioning must match final latent spatial resolution")
    masks = None
    mask = latent.get("noise_mask")
    if mask is not None:
        if not getattr(mask, "is_nested", False) or len(mask.unbind()) != 2:
            raise ValueError("JR H3 Streaming: mask must contain both AV streams")
        try:
            masks = tuple(torch.broadcast_to(m, t.shape) for m, t in zip(mask.unbind(), (video, audio)))
        except RuntimeError as exc:
            raise ValueError("JR H3 Streaming: invalid AV mask shapes") from exc
        if any(not bool(torch.isfinite(m).all()) or not bool(((m == 0) | (m == 1)).all()) for m in masks):
            raise ValueError("JR H3 Streaming: first experiment supports binary AV masks only")
    return video, audio, masks


def patch_runtime(model, runtime):
    from comfy.patcher_extension import get_all_wrappers

    from .h3_temporal_transport import CONFIG_KEY, WRAPPER_KEY, TemporalTransport

    opts = model.model_options.get("transformer_options", {})
    if opts.get("sol_morton") or opts.get("patches_replace") or model.get_attachment("jr_h3_adaptive_cache"):
        raise ValueError("JR H3 Streaming: disable Morton, Adaptive Cache and block replacements")
    previous = opts.get("optimized_attention_override")
    tst = previous if isinstance(previous, TemporalTransport) else None
    allowed = [tst.forward] if tst else []
    wrappers = list(get_all_wrappers("diffusion_model", opts))
    wrappers.extend(w for group in model.wrappers.get("diffusion_model", {}).values() for w in group)
    for wrapper in wrappers:
        if wrapper not in allowed:
            raise ValueError("JR H3 Streaming: unknown diffusion wrapper / Adaptive Cache is unsupported")
    if opts.get(CONFIG_KEY) and tst is None:
        raise ValueError("JR H3 Streaming: TST dispatcher was replaced upstream")
    for key, patch in model.object_patches.items():
        if ".attn.forward" in key and not getattr(patch, "_uses_optimized_attention", False):
            raise ValueError(f"JR H3 Streaming: unsupported attention object patch {key}")
        if key.endswith("diffusion_model.forward") or key.endswith("diffusion_model._forward"):
            raise ValueError("JR H3 Streaming: compiled/replaced model forward is unsupported")
    patched = model.clone()
    if tst:
        patched.remove_wrappers_with_key("diffusion_model", WRAPPER_KEY)
        runtime.tst_strength = tst.strength
        previous = tst.previous
    runtime.previous = previous
    target = patched.model_options.setdefault("transformer_options", {})
    target["optimized_attention_override"] = runtime
    patched.add_wrapper_with_key("diffusion_model", "jr_h3_streaming", runtime.forward)
    return patched


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def sample_streaming(*, model, positive, vae, noise, sampler, sigmas, latent_image, stream_plan,
                     streaming_mode="Geometry Only", retention="sink_plus_recent_2", layer_policy="every_4",
                     custom_layers="", cache_device="cpu", max_kv_mib=8192, audio_reset_interval_requests=1):
    from comfy.model_management import intermediate_device
    from comfy.nested_tensor import NestedTensor
    from comfy.samplers import CFGGuider

    from .h3_vae_compat import inspect_h3_video_vae

    if streaming_mode not in MODES or type(audio_reset_interval_requests) is not int or audio_reset_interval_requests < 1:
        raise ValueError("JR H3 Streaming: invalid mode/audio reset interval")
    video, audio, masks = validate_inputs(model, positive, noise, sampler, sigmas, latent_image, stream_plan)
    capabilities = inspect_h3_video_vae(vae)
    if not capabilities["decode_probe_candidate"]:
        raise ValueError("JR H3 Streaming: connect an H3 video VAE capable of final decode; VAE is not executed here")
    diffusion = model.get_model_object("diffusion_model")
    total_layers, heads = len(diffusion.blocks), diffusion.blocks[0].attn.heads
    chosen = select_layers(layer_policy if streaming_mode == "Sparse KV" else "all", total_layers, custom_layers)
    if type(max_kv_mib) is not int or not 64 <= max_kv_mib <= 262144:
        raise ValueError("JR H3 Streaming: max_kv_mib must be 64..262144")
    cache = JR_H3_CleanAVKVCache(chosen, heads, retention=retention, storage=cache_device, max_bytes=max_kv_mib * 1024**2)
    collect = streaming_mode in ("Clean KV", "Streaming Attention", "Sparse KV")
    runtime = StreamingRuntime(stream_plan, cache, total_layers, heads, None,
                               use_history=streaming_mode in ("Streaming Attention", "Sparse KV"), collect=collect,
                               clean_enabled=streaming_mode not in ("Geometry Only", "Micro Chunk"),
                               schedule=tuple(float(s) for s in sigmas))
    # Exact conservative peak including staged commit and head-group merge copy.
    # This guards cache allocations, not the model/activation working set.
    rows = ((video.shape[3] + 1) // 2) * ((video.shape[4] + 1) // 2)
    phase_tokens = [p.video_latent_count * rows + 2 * p.audio_latent_count for p in stream_plan.phases]
    recent = 2 if retention == "sink_plus_recent_2" else 1
    sink = 0 if retention == "previous_only" else stream_plan.phases[0].video_latent_count * rows
    max_tokens = sink + (recent + 2) * max(phase_tokens)
    estimated_bytes = max_tokens * heads * diffusion.blocks[0].attn.head_dim * 4 * len(chosen)
    if collect and estimated_bytes > cache.max_bytes:
        raise ValueError(f"JR H3 Streaming: conservative KV+staging bound {estimated_bytes / 1024**2:.0f} MiB exceeds "
                         f"budget {max_kv_mib} MiB; reduce layer count/resolution/retention or explicitly raise the {cache_device} budget")
    patched = model.clone() if streaming_mode == "Geometry Only" else patch_runtime(model, runtime)
    old_shapes = model.model.latent_shapes
    guider = CFGGuider(patched)
    guider.set_conds(positive, [])
    device = model.load_device
    begin = time.perf_counter()
    phase_stats = []
    try:
        # One full-request CPU noise field; phase seeds are not independently reset.
        all_noise = noise.generate_noise(latent_image)
        nv, na = all_noise.unbind()
        if nv.shape != video.shape or na.shape != audio.shape:
            raise ValueError("JR H3 Streaming: noise AV shape mismatch")
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if streaming_mode == "Geometry Only":
            result = guider.sample(all_noise, latent_image["samples"], sampler, sigmas,
                                   denoise_mask=latent_image.get("noise_mask"), seed=noise.seed)
            # Match SamplerCustomAdvanced's node boundary before allocator cleanup.
            output = dict(latent_image, samples=result.to(intermediate_device()))
            return output, "JR H3 Streaming: Geometry Only — ordinary native sampling, no streaming patches or KV.\n" + stream_plan.status()
        out_video, out_audio = torch.empty_like(video, device="cpu"), torch.empty_like(audio, device="cpu")
        wrapped_sampler = CleanCommitSampler(sampler, runtime)
        for phase in stream_plan.phases:
            runtime.phase = phase
            vs = slice(phase.video_latent_start, phase.video_latent_stop)
            au = slice(phase.audio_latent_start, phase.audio_latent_stop)
            part = NestedTensor((video[:, :, vs].clone(), audio[..., au].clone()))
            part_noise = NestedTensor((nv[:, :, vs].clone(), na[..., au].clone()))
            part_mask = None if masks is None else NestedTensor((masks[0][:, :, vs].clone(), masks[1][..., au].clone()))
            _sync(device)
            phase_start = time.perf_counter()
            result = guider.sample(part_noise, part, wrapped_sampler, sigmas, denoise_mask=part_mask, seed=noise.seed)
            rv, ra = result.unbind()
            if rv.shape != part.unbind()[0].shape or ra.shape != part.unbind()[1].shape:
                raise RuntimeError("JR H3 Streaming: sampled AV shape mismatch")
            if not bool(torch.isfinite(rv).all()) or not bool(torch.isfinite(ra).all()):
                raise RuntimeError("JR H3 Streaming: non-finite sampled output")
            if masks is not None:
                rv = torch.where(masks[0][:, :, vs].to(rv.device) == 0, video[:, :, vs].to(rv), rv)
                ra = torch.where(masks[1][..., au].to(ra.device) == 0, audio[..., au].to(ra), ra)
            out_video[:, :, vs].copy_(rv)
            out_audio[..., au].copy_(ra)
            _sync(device)
            phase_stats.append(dict(phase=phase.phase_index, seconds=time.perf_counter() - phase_start, **cache.metrics()))
            del result, rv, ra, part, part_noise, part_mask
        metrics = cache.metrics()
        elapsed = time.perf_counter() - begin
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        status = ["JR H3 Streaming Sampler (Experimental)", f"Mode: {streaming_mode}; phases: {len(phase_stats)}",
                  "Geometry: 124 frames / 37 video latents / 207 audio ticks; single request",
                  f"Denoise forwards: {runtime.denoise_forwards}; clean forwards: {runtime.clean_forwards}",
                  f"Cached layers: {metrics['cached_layers']}/{total_layers}; BF16 storage: {cache_device}",
                  f"KV: {metrics['kv_mib']:.3f} MiB; history video={metrics['video_history_tokens']}, audio={metrics['audio_history_tokens']}",
                  f"Retention: {retention}; commits: {metrics['retained_commits']}; audio reset every {audio_reset_interval_requests} requests (continuation disabled)",
                  f"KV layer-byte reduction vs all: {100 * (1 - len(chosen) / total_layers):.1f}% (same geometry)",
                  f"Total: {elapsed:.3f}s; clean-forward+commit host: {runtime.commit_seconds:.3f}s; trim host: {runtime.trim_seconds:.6f}s",
                  f"Attention: {runtime.kernel_calls} calls; host dispatch only: {runtime.attention_host_seconds:.3f}s (not GPU kernel time)",
                  f"Peak CUDA allocated: {peak / 1024**2:.1f} MiB; reserved: {reserved / 1024**2:.1f} MiB",
                  f"TST: {runtime.tst_strength:g} on noisy current video; bypassed on clean forwards",
                  "Unified delegate preserved. Rectangular media attention is ineligible for Sol sparse: dense Sage/native fallback.",
                  "Clean forward uses native sigma=0 (DiT clamps internally to 1e-6); prediction discarded; VAE not executed."]
        for stats in phase_stats:
            status.append(f"Phase {stats['phase']}: {stats['seconds']:.3f}s; KV {stats['kv_mib']:.3f} MiB; history {stats['history_tokens']} tokens")
        if not runtime.use_history:
            status.append("Diagnostic mode: no persistent history supplied to attention; not a continuity-quality reference.")
        LOGGER.info("JR H3 Streaming finished: %s, %.3fs; cleanup discards all KV", streaming_mode, elapsed)
        return dict(latent_image, samples=NestedTensor((out_video, out_audio))), "\n".join(status)
    finally:
        runtime.clear()
        model.model.latent_shapes = old_shapes
        # Some ComfyUI versions skip their post-outer_sample condition cleanup
        # on exceptions. Release additional-model references owned by this guider.
        if hasattr(guider, "loaded_models"):
            from comfy.sampler_helpers import cleanup_models
            try:
                cleanup_models(getattr(guider, "conds", {}), guider.loaded_models)
            except Exception:
                LOGGER.exception("JR H3 Streaming: additional-model cleanup failed")
            finally:
                del guider.loaded_models
        if hasattr(guider, "inner_model"):
            del guider.inner_model
        guider.conds = {}
        patched.remove_wrappers_with_key("diffusion_model", "jr_h3_streaming")
        # The clone can be retained by Comfy's model manager. Drop its runtime
        # override rather than leaving an empty execution object on a loaded model.
        options = patched.model_options.get("transformer_options", {})
        if options.get("optimized_attention_override") is runtime:
            options["optimized_attention_override"] = runtime.previous
