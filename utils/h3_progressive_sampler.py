"""JR-native progressive H3 Euler orchestration; no external sampler source vendored.

The low-stage callback exposes internal x and x0 BEFORE its Euler update. We lift
only video x0 in the VAE domain, reconstruct video noise at the boundary, and
advance audio with the original Euler derivative. A fresh native guider resumes
with zero additional noise after inverse_noise_scaling at the shared sigma.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import torch

from .h3_av_latent_builder import build_h3_av_latent
from .h3_av_latent_split import split_h3_av_latent
from .h3_neural_latent_upscaler import get_h3_spatial_contract, upscale_h3_video_to_size
from .h3_temporal_chunk_sampler import _append_noise_factory, _registered_noise_factory

PREFIX = "JR H3 Progressive Sampler: "
MAX_SEED = 0xffffffffffffffff


@dataclass(frozen=True)
class ProgressivePlan:
    low_h: int
    low_w: int
    target_h: int
    target_w: int
    transition_step: int
    total_steps: int

    @property
    def identity(self):
        return (self.low_h, self.low_w) == (self.target_h, self.target_w)


def _error(message):
    return ValueError(PREFIX + message)


def plan_progressive(sigmas, transition_step, lowres_scale, target_h, target_w):
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise _error("sigmas must be a one-dimensional floating tensor.")
    if sigmas.device.type == "meta" or sigmas.layout != torch.strided:
        raise _error("sigmas must be materialized and strided.")
    values = sigmas.detach().cpu().double()
    if (len(values) < 3 or not bool(torch.isfinite(values).all()) or
            bool((values < 0).any()) or bool((values > 1).any()) or
            bool((values[1:] >= values[:-1]).any()) or values[-1] != 0):
        raise _error("Use at least two Euler steps with finite, strictly decreasing sigmas in [0,1], ending at zero.")
    if not math.isclose(float(values[0]), 1.0, abs_tol=1e-5):
        raise _error("Progressive sampling requires a full denoise schedule starting at sigma 1 (denoise=1).")
    if type(transition_step) is not int or not 1 <= transition_step < len(values) - 1:
        raise _error("transition_step must be >= 1 and smaller than the total number of steps.")
    if not math.isfinite(lowres_scale) or not 0.25 <= lowres_scale <= 1.0:
        raise _error("lowres_scale must be finite and between 0.25 and 1.0.")
    contract = get_h3_spatial_contract()
    low = []
    for target, alignment in ((target_h, contract.latent_alignment_h), (target_w, contract.latent_alignment_w)):
        if target <= 0 or target % alignment:
            raise _error("Target H/W must align with the native H3 spatial patch grid.")
        # Ceiling prevents alignment from accidentally exceeding the upscaler's 4x limit.
        low.append(min(target, max(alignment, math.ceil(target * lowres_scale / alignment) * alignment)))
    return ProgressivePlan(*low, target_h, target_w, transition_step, len(values) - 1)


def _resolve_noise_factory(noise):
    """Match live registered providers as well as direct imports of the built-ins.

    ComfyUI's path-based loader can execute nodes_custom_sampler.py a second
    time. Its official NOISE classes then differ by identity from package imports.
    Reuse JR's registry probing instead of accepting class names or duck types.
    """
    from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise

    random_factories, empty_factories = [], []
    _append_noise_factory(random_factories, _registered_noise_factory("RandomNoise"), 0)
    _append_noise_factory(empty_factories, _registered_noise_factory("DisableNoise"))
    _append_noise_factory(random_factories, Noise_RandomNoise, 0)
    _append_noise_factory(empty_factories, Noise_EmptyNoise)
    for provider_type, factory in random_factories:
        if type(noise) is provider_type:
            return factory
    for provider_type, factory in empty_factories:
        if type(noise) is provider_type:
            return lambda _seed, empty_factory=factory: empty_factory()
    provider_name = f"{type(noise).__module__}.{type(noise).__qualname__}"
    raise _error("Use official RandomNoise or DisableNoise; custom NOISE has no reproducible substream contract. "
                 f"Received unregistered provider: {provider_name}")


def _validate_inputs(model, positive, noise, sampler, latent_image, *, guided=False):
    from comfy.k_diffusion.sampling import sample_euler
    from comfy.latent_formats import MiniMaxH3AV
    from comfy.model_base import MiniMaxH3
    from comfy.model_sampling import CONST, ModelSamplingAV
    from comfy.samplers import KSAMPLER

    if not isinstance(getattr(model, "model", None), MiniMaxH3):
        raise _error("Connect a native MiniMax H3 MODEL.")
    sampling = model.get_model_object("model_sampling")
    latent_format = model.get_model_object("latent_format")
    if not isinstance(sampling, (ModelSamplingAV,)) or not isinstance(sampling, CONST):
        raise _error("MODEL must use the native H3 AV flow sampling contract.")
    if not isinstance(latent_format, MiniMaxH3AV):
        raise _error("MODEL must use the native H3 AV latent format.")
    if not math.isfinite(sampling.audio_scale) or sampling.audio_scale <= 0:
        raise _error("Invalid H3 audio sigma shift.")
    if type(sampler) is not KSAMPLER or sampler.sampler_function is not sample_euler:
        raise _error("Only standard Euler is supported; select euler in KSamplerSelect.")
    options = sampler.extra_options
    if set(options) - {"s_churn", "s_tmin", "s_tmax", "s_noise"} or options.get("s_churn", 0) != 0:
        raise _error("Euler must have s_churn=0 and no custom sampler options.")
    if sampler.inpaint_options:
        raise _error("Custom sampler inpaint options are not supported.")
    noise_factory = _resolve_noise_factory(noise)
    if type(noise.seed) is not int or not 0 <= noise.seed <= MAX_SEED:
        raise _error("NOISE seed must be an unsigned 64-bit integer.")
    video_latent, audio_latent = split_h3_av_latent(latent_image)
    build_h3_av_latent(video_latent, audio_latent)  # shape, timeline, dtype/device and finite checks
    video, audio = video_latent["samples"], audio_latent["samples"]
    if video.shape[0] != 1:
        raise _error("The initial prototype supports batch size 1.")
    if video.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise _error("AV tensors must use fp32, fp16 or bf16.")
    if not guided and latent_image.get("noise_mask") is not None:
        raise _error("noise_mask is not supported. Audio-driven, hard-prefix and inpainting workflows must use legacy sampling.")
    # Guided validates the exact locked-prefix/empty-suffix contract before VAE
    # work. The original T2VA node remains strictly empty and unmasked.
    if not guided and (bool(torch.count_nonzero(video)) or bool(torch.count_nonzero(audio))):
        raise _error("Connect an empty target-resolution AV latent; encoded, sampled or continuation latents are unsupported.")
    if not isinstance(positive, (list, tuple)) or not positive:
        raise _error("positive must contain native H3 text conditioning.")
    for entry in positive:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not isinstance(entry[1], dict):
            raise _error("Invalid CONDITIONING entry.")
        for key in ("minimax_keyframes", "mask", "area", "control", "concat_latent_image", "hooks", "gligen"):
            if guided and key == "minimax_keyframes":
                continue
            if entry[1].get(key) is not None:
                raise _error(f"Conditioning {key} is unsupported during spatial transition; use the legacy sampler.")
    batch_index = latent_image.get("batch_index")
    if batch_index is not None and (not isinstance(batch_index, (list, tuple)) or len(batch_index) != 1 or
                                    type(batch_index[0]) is not int or not 0 <= batch_index[0] <= 10000):
        raise _error("batch_index must contain one integer in [0,10000].")
    return video, audio, sampling, latent_format, noise_factory


def _cpu_copy(nested):
    from comfy.nested_tensor import NestedTensor
    if type(nested) is not NestedTensor or len(nested.unbind()) != 2:
        raise RuntimeError(PREFIX + "Native sampler callback must expose H3 AV NestedTensor views.")
    return NestedTensor([t.detach().to(device="cpu", dtype=torch.float32).clone() for t in nested.unbind()])


def _run_stage(model, positive, noise_tensor, latent, sampler, sigmas, seed, callback, denoise_mask=None):
    from comfy import utils
    from comfy_extras.nodes_custom_sampler import Guider_Basic

    guider = Guider_Basic(model)
    guider.set_conds(positive)
    return guider.sample(noise_tensor, latent, sampler, sigmas, denoise_mask=denoise_mask,
                         callback=callback, disable_pbar=not utils.PROGRESS_BAR_ENABLED, seed=seed)


def _reset_cache(model):
    getter = getattr(model, "get_attachment", None)
    runtime = getter("jr_h3_adaptive_cache") if getter else None
    if runtime is not None:
        runtime.reset("progressive spatial stage boundary", keep_stats=False)


def transition_state(x, x0, sampling, latent_format, plan, transition_noise, sigma_from, sigma_to, *, lifter=None):
    """Return a VAE-domain resume latent. All boundary tensors are CPU fp32.

    Audio advances with its already-scaled native Euler state. Dividing by
    audio_scale happens only when returning to the public AV latent domain.
    """
    from comfy.nested_tensor import NestedTensor

    lifter = lifter or upscale_h3_video_to_size
    video_x0, audio_x0 = x0.unbind()
    _, audio_x = x.unbind()
    clean_video = latent_format.process_out(video_x0)
    lifted = lifter(clean_video, plan.target_h, plan.target_w)
    expected = (*video_x0.shape[:-2], plan.target_h, plan.target_w)
    if not isinstance(lifted, torch.Tensor) or tuple(lifted.shape) != expected or not bool(torch.isfinite(lifted).all()):
        raise RuntimeError(PREFIX + "Neural lift returned an invalid video x0.")
    lifted = latent_format.process_in(lifted.to(device="cpu", dtype=torch.float32))
    # Re-noise lifted x0 at sigma_from, then complete this Euler interval.
    video_x = sampling.noise_scaling(sigma_from, transition_noise, lifted)
    video_next = video_x + (video_x - lifted) / sigma_from * (sigma_to - sigma_from)
    audio_next = audio_x + (audio_x - audio_x0) / sigma_from * (sigma_to - sigma_from)
    video_resume = sampling.inverse_noise_scaling(sigma_to, video_next)
    audio_resume = sampling.inverse_noise_scaling(sigma_to, audio_next) / sampling.audio_scale
    return latent_format.process_out(NestedTensor((video_resume, audio_resume)))


@torch.inference_mode()
def sample_h3_progressive(*, model, positive, noise, sampler, sigmas, latent_image,
                          transition_step=3, lowres_scale=0.5, transition_seed_offset=1,
                          aggressive_memory_cleanup=False, _guided=False, vae=None):
    from comfy import model_management, utils
    from comfy.nested_tensor import NestedTensor

    started = time.perf_counter()
    video, audio, sampling, latent_format, noise_factory = _validate_inputs(
        model, positive, noise, sampler, latent_image, guided=_guided)
    plan = plan_progressive(sigmas, transition_step, lowres_scale, *video.shape[-2:])
    if type(transition_seed_offset) is not int or not 0 <= transition_seed_offset <= MAX_SEED:
        raise _error("transition_seed_offset must be an unsigned 64-bit integer.")
    guidance = None
    if _guided:
        from .h3_progressive_guidance import prepare_guidance
        guidance = prepare_guidance(positive, latent_image, video, audio, plan, vae)
    audio_locked = guidance is not None and guidance.audio_locked
    schedule = sigmas.detach().clone()
    if getattr(model, "model_options", {}).get("transformer_options", {}).get("jr_h3_tst_config"):
        # Isolated clone: both native guider calls share the original time axis,
        # never persist this prompt's schedule on the caller's MODEL.
        model = model.clone()
        model.model_options["transformer_options"]["jr_h3_tst_schedule"] = tuple(schedule.cpu().tolist())
    transition_seed = (noise.seed + transition_seed_offset) & MAX_SEED
    progress = utils.ProgressBar(plan.total_steps)
    counts = [0, 0]
    captured = {}

    def callback_for(stage, expected, offset):
        def callback(step, x0, x, total):
            if step != counts[stage] or total != expected or step >= expected:
                raise RuntimeError(PREFIX + "Unexpected Euler callback sequence; remove sampler-changing wrappers.")
            counts[stage] += 1
            if stage == 0 and not plan.identity and step == expected - 1:
                captured["x0"], captured["x"] = _cpu_copy(x0), _cpu_copy(x)
            progress.update_absolute(offset + step + 1, plan.total_steps)
        return callback

    _reset_cache(model)
    try:
        initial = dict(latent_image)
        # Public generation area remains zero. Guided copies/resizes only clean
        # locked context; the caller's high-resolution master is never modified.
        initial["samples"] = NestedTensor((
            guidance.low_video(video, plan.low_h, plan.low_w) if guidance else
            torch.zeros((*video.shape[:-2], plan.low_h, plan.low_w), dtype=video.dtype, device="cpu"),
            audio.detach().cpu().clone() if audio_locked else torch.zeros_like(audio, device="cpu"),
        ))
        low_steps = plan.total_steps if plan.identity else plan.transition_step
        stage_options = {"denoise_mask": guidance.mask_for(initial["samples"])} if guidance else {}
        low_output = _run_stage(model, guidance.low_positive if guidance else positive,
                                noise.generate_noise(initial), initial["samples"], sampler,
                                schedule[:low_steps + 1], noise.seed, callback_for(0, low_steps, 0), **stage_options)
        stage_options.clear()
        if counts[0] != low_steps:
            raise RuntimeError(PREFIX + "Low stage did not execute the expected Euler steps.")
        low_seconds = time.perf_counter() - started
        lift_seconds = 0.0
        if plan.identity:
            samples = low_output
        else:
            del low_output, initial
            _reset_cache(model)
            if aggressive_memory_cleanup:
                model_management.soft_empty_cache()
            lift_started = time.perf_counter()
            # Official noise policy, with an independent deterministic video seed.
            transition_provider = noise_factory(transition_seed)
            transition_template = dict(latent_image)
            transition_template["samples"] = torch.zeros_like(video, device="cpu", dtype=torch.float32)
            transition_noise = transition_provider.generate_noise(transition_template)
            resume = transition_state(
                captured["x"], captured["x0"], sampling, latent_format, plan, transition_noise,
                schedule[transition_step - 1].cpu(), schedule[transition_step].cpu(),
            )
            if audio_locked:
                # The inpaint anchor is CLEAN public audio, never the inverse-scaled
                # noisy boundary state. H3 will apply its own sigma/audio scaling.
                resume = NestedTensor((guidance.restore_prefix(resume.unbind()[0], video),
                                       audio.detach().to(device="cpu", dtype=torch.float32).clone()))
            captured.clear()
            del transition_noise, transition_template
            if any(not bool(torch.isfinite(t).all()) for t in resume.unbind()):
                raise RuntimeError(PREFIX + "Non-finite state at the resolution transition.")
            lift_seconds = time.perf_counter() - lift_started
            remaining = plan.total_steps - transition_step
            samples = _run_stage(
                model, positive, NestedTensor([torch.zeros_like(t) for t in resume.unbind()]), resume,
                sampler, schedule[transition_step:], noise.seed, callback_for(1, remaining, transition_step),
                **({"denoise_mask": guidance.mask_for(resume)} if guidance else {}),
            )
            if counts[1] != remaining:
                raise RuntimeError(PREFIX + "High stage did not execute the expected Euler steps.")
        output_video, output_audio = samples.unbind()
        if tuple(output_video.shape) != tuple(video.shape) or tuple(output_audio.shape) != tuple(audio.shape):
            raise RuntimeError(PREFIX + "Native sampler changed the target AV shapes.")
        if any(not bool(torch.isfinite(t).all()) for t in samples.unbind()):
            raise RuntimeError(PREFIX + "Native sampler produced NaN or Inf.")
        if audio_locked:
            # Avoid round-trip normalization drift, including fp16/bf16 sources.
            samples = NestedTensor((guidance.restore_prefix(output_video.to(dtype=video.dtype), video),
                                    audio.detach().clone()))
            if not bool(torch.isfinite(samples.unbind()[0]).all()):
                raise RuntimeError(PREFIX + "Video output overflowed the original AV dtype.")
        output = dict(latent_image)
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        output["samples"] = samples.to(model_management.intermediate_device())
        elapsed = time.perf_counter() - started
        status = (f"Experimental JR H3 Progressive Sampler | {'native identity baseline' if plan.identity else 'JR neural x0 lift'}\n"
                  f"video latent: {plan.low_w}x{plan.low_h} -> {plan.target_w}x{plan.target_h}; audio shape unchanged\n"
                  f"Euler evaluations: {counts[0]} low + {counts[1]} high = {sum(counts)}; same sigma schedule\n"
                  f"seed: {noise.seed}; transition seed: {transition_seed}; audio scale: {sampling.audio_scale:g}\n"
                  f"wall time: {elapsed:.2f}s (low: {low_seconds:.2f}s, transition: {lift_seconds:.2f}s)\n"
                  "Fixed-seed repeatability is per configuration; different resolutions can change audio predictions.\n"
                  "Prototype: compare with cache/Sol/compile off first. Legacy dual sampling remains available.")
        if guidance:
            status += "\n" + guidance.description
        tst_config = getattr(model, "model_options", {}).get("transformer_options", {}).get("jr_h3_tst_config")
        if tst_config:
            status += f"\nTST experimental: strength={tst_config[1]:g}; shared full sigma schedule; existing attention backend retained."
        logging.info("%s", status)
        return output, status
    finally:
        captured.clear()
        _reset_cache(model)
