"""H3 video-only regeneration using native ComfyUI sampling and JR upscaling."""
from __future__ import annotations

import logging
import math

import torch

from . import h3_acceleration_adapters as acceleration
from .h3_av_latent_split import split_h3_av_latent
from .h3_neural_latent_upscaler import upscale_h3_video_latent
from .h3_regenerate_windows import VideoWindowReader, encode_window, plan_windows
from .h3_temporal_chunk_sampler import derive_chunk_seed

ATTENTION_MODES = ("inherit", "auto", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn_qk_int8_pv_fp8_cuda",
                   "sageattn_qk_int8_pv_fp16_triton")


def regenerate_sigmas(denoise, steps, sigma_schedule="low_denoise_shift6"):
    if not math.isfinite(float(denoise)) or not .15 <= denoise <= .35:
        raise ValueError("Regenerate denoise must be between 0.15 and 0.35")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 50:
        raise ValueError("Regenerate steps must be an integer from 1 to 50")
    if sigma_schedule == "release_original":
        if steps == 3:
            return torch.tensor([.909375, 6 / 7, 2 / 3, 0.], dtype=torch.float32)
        if steps == 2:
            return torch.tensor([.7, 42 / 85, 0.], dtype=torch.float32)
        raise ValueError("The original release schedule supports exactly 2 or 3 steps")
    if sigma_schedule != "low_denoise_shift6":
        raise ValueError("Unknown Regenerate sigma schedule")
    raw = denoise / (6 - 5 * denoise)
    values = [denoise]
    for i in range(1, steps):
        x = raw * (1 - i / steps)
        values.append(6 * x / (1 + 5 * x))
    return torch.tensor(values + [0.], dtype=torch.float32)


class VideoOnlyNoise:
    def __init__(self, seed, video_start=0):
        self.seed = seed
        self.video_start = video_start

    def generate_noise(self, latent):
        from comfy.nested_tensor import NestedTensor
        video, audio = latent["samples"].unbind()
        noise = torch.empty(video.shape, dtype=torch.float32, device="cpu")
        generator = torch.Generator(device="cpu")
        # Each absolute token has its own deterministic stream. Window partitioning
        # cannot change noise in the overlap; the sampler itself always gets seed.
        for t in range(video.shape[2]):
            generator.manual_seed(derive_chunk_seed(self.seed, self.video_start + t))
            noise[:, :, t] = torch.randn(noise[:, :, t].shape, generator=generator)
        return NestedTensor((noise, torch.zeros(audio.shape, dtype=torch.float32)))


def select_attention(model, attention):
    if attention not in ATTENTION_MODES:
        raise ValueError(f"Unsupported Regenerate attention: {attention}")
    if attention == "inherit":
        return model
    options = model.model_options.get("transformer_options", {})
    if options.get("optimized_attention_override") is not None or options.get("patches_replace", {}).get("dit"):
        raise ValueError("MODEL already has an attention patch. Select inherit and set Sage 2++ in Unified.")
    return acceleration.apply_sage(model, sage_attention=attention, allow_compile=False)


def regenerate(model, clip, source_mode, denoise, steps, seed, prompt, attention,
               resize_mode, scale, target_megapixels, upscaler_model,
               latent=None, video=None, vae=None, audio_vae=None, sigma_schedule="low_denoise_shift6",
               chunk_mode="off"):
    from comfy.model_management import throw_exception_if_processing_interrupted
    from comfy.nested_tensor import NestedTensor
    from comfy.samplers import ksampler
    from comfy.utils import ProgressBar
    from comfy_extras.nodes_custom_sampler import BasicGuider, SamplerCustomAdvanced
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    sigmas = regenerate_sigmas(denoise, steps, sigma_schedule)
    acceleration.ensure_minimax_h3_model(model)
    selected = select_attention(model, attention)
    if resize_mode not in ("keep", "scale", "megapixels"):
        raise ValueError("Unknown Regenerate resize mode")
    reader, original_audio, raw_audio, original_video = None, None, None, None
    if source_mode == "video":
        if latent is not None:
            raise ValueError("Connect only VIDEO when source_mode is video")
        if video is None or vae is None:
            raise ValueError("VIDEO input requires video and video VAE; audio VAE is not used")
        reader = VideoWindowReader(video)
        raw_audio, video_t = reader.audio, reader.video_t
        source = {"jr_h3_regenerate_video": {
            "source_frames_24fps": reader.count, "encoded_frames": (video_t - 2) // 5 * 17 + 5,
            "frame_rate": 24}}
    elif source_mode == "H3 latent":
        if video is not None:
            raise ValueError("Connect only H3 LATENT when source_mode is H3 latent")
        if latent is None:
            raise ValueError("Connect an H3 LATENT")
        source = latent
        if isinstance(source["samples"], NestedTensor):
            video_latent, audio_latent = split_h3_av_latent(source)
            original_video, original_audio = video_latent["samples"], audio_latent["samples"]
        else:
            original_video = source["samples"]
        if original_video.ndim != 5 or original_video.shape[1] != 24:
            raise ValueError("H3 video latent must have shape [1,24,T,H,W]")
        if original_video.shape[0] != 1 or any(n % 2 for n in original_video.shape[-2:]):
            raise ValueError("H3 requires batch 1 and even latent spatial dimensions")
        video_t = original_video.shape[2]
    else:
        raise ValueError("Unknown Regenerate source mode")
    windows = plan_windows(video_t, chunk_mode)
    # SigmaShift preserves upstream LoRA curves, Unified FFN and attention patches.
    selected = MiniMaxH3SigmaShift.execute(selected, 6., 3.)[0]
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    guider = BasicGuider.execute(selected, positive)[0]
    progress = ProgressBar(len(windows))
    output = None
    upscale_status = "Input latent size retained"
    for index, window in enumerate(windows):
        throw_exception_if_processing_interrupted()
        logging.info("H3 Regenerate window %d/%d: tokens [%d, %d), fixed seed %d, locked prefix %d",
                     index + 1, len(windows), window.start, window.end, seed, window.prefix)
        if reader is not None:
            visual = encode_window(reader, window, vae)
        else:
            visual = original_video[:, :, window.start:window.end].detach().cpu().clone()
        if visual.shape[2] != window.end - window.start:
            raise ValueError("Video VAE returned an unexpected H3 temporal grid")
        if resize_mode != "keep":
            upscaled, upscale_status = upscale_h3_video_latent(
                {"samples": visual}, resize_mode, scale, target_megapixels, model_name=upscaler_model)
            visual = upscaled["samples"].cpu()
            del upscaled
        if output is None:
            output = torch.empty((1, 24, video_t, *visual.shape[-2:]), dtype=visual.dtype, device="cpu")
        mask = torch.ones_like(visual)
        if window.prefix:
            visual[:, :, :window.prefix] = output[:, :, window.start:window.start + window.prefix]
            mask[:, :, :window.prefix] = 0
        # Native H3 is a joint AV model. Bounded neutral audio tokens satisfy its
        # interface; source audio is NEVER supplied, sliced, encoded or sampled.
        dummy_audio = torch.zeros((1, 32, 2, round(window.frames * 40 / 24)), dtype=visual.dtype)
        working = {"samples": NestedTensor((visual, dummy_audio)),
                   "noise_mask": NestedTensor((mask, torch.zeros_like(dummy_audio)))}
        sampled = SamplerCustomAdvanced.execute(
            VideoOnlyNoise(seed, window.start), guider, ksampler("euler"), sigmas, working)[0]
        refined = sampled["samples"].unbind()[0]
        keep = window.start + window.prefix
        output[:, :, keep:window.end].copy_(refined[:, :, window.prefix:].detach().cpu())
        del refined, sampled, working, mask, visual, dummy_audio
        progress.update_absolute(index + 1)
    result = dict(source)
    # Old masks/batch indices refer to source dimensions. Do not propagate them.
    result.pop("noise_mask", None)
    result.pop("batch_index", None)
    result["samples"] = NestedTensor((output, original_audio)) if original_audio is not None else output
    result["jr_h3_regenerate_windows"] = {
        "mode": chunk_mode, "count": len(windows), "seed": seed,
        "max_window_frames": max(w.frames for w in windows), "overlap_frames": 39 if len(windows) > 1 else 0}
    marker = selected.model_options.get("jr_h3_regenerate_lora")
    status = (f"H3 Regenerate: Euler / CFG 1 / {steps} steps / start sigma {sigmas[0].item():.6f}\n"
              f"schedule: {sigma_schedule} (release_original ignores denoise)\n"
              f"sigmas: {sigmas.tolist()}\nattention: {attention}; upstream FFN preserved\n"
              f"Regenerate adapter: {marker or 'not loaded by JR loader; verify input MODEL'}\n"
              f"windows: {len(windows)} / {chunk_mode}; fixed seed {seed}; global-coordinate noise; hard video prefix\n"
              f"audio: original bypass, never passed to sampler\n{upscale_status}")
    return result, status, raw_audio
