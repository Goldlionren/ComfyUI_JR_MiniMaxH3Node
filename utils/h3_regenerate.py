"""H3 video-only regeneration using native ComfyUI sampling and JR upscaling."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import h3_acceleration_adapters as acceleration
from .h3_av_latent_split import split_h3_av_latent
from .h3_neural_latent_upscaler import upscale_h3_video_latent

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
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, latent):
        from comfy.nested_tensor import NestedTensor
        video, audio = latent["samples"].unbind()
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        return NestedTensor((torch.randn(video.shape, dtype=torch.float32, generator=generator),
                             torch.zeros(audio.shape, dtype=torch.float32)))


def encode_input_video(video, vae, audio_vae):
    from comfy.nested_tensor import NestedTensor
    from comfy_extras.nodes_minimax_h3 import _encode_ref_audio, align_frame_count

    if video is None or vae is None or audio_vae is None:
        raise ValueError("VIDEO input requires video, video VAE and audio VAE")
    components = video.get_components()
    images = components.images
    fps = float(components.frame_rate)
    if not math.isfinite(fps) or fps <= 0 or images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("VIDEO must have decoded frames and a positive frame rate")
    count = max(1, round(len(images) * 24 / fps))
    indices = (torch.arange(count, dtype=torch.float64) * fps / 24).floor().long().clamp(max=len(images) - 1)
    images = images.index_select(0, indices.to(images.device))[..., :3]
    # Keep source detail; only align the native 16x VAE / 2x DiT spatial grid.
    height, width = (max(32, round(n / 32) * 32) for n in images.shape[1:3])
    if images.shape[1:3] != (height, width):
        images = F.interpolate(images.movedim(-1, 1), size=(height, width), mode="bicubic",
                               align_corners=False).movedim(1, -1).clamp(0, 1)
    frame_count = align_frame_count(max(5, count))
    images = torch.cat((images, images[-1:].expand(frame_count - count, -1, -1, -1)), dim=0)
    visual = vae.encode(images)
    sample_rate = getattr(audio_vae, "audio_sample_rate", 32000)
    source_audio = components.audio
    if source_audio is None:
        audio = {"waveform": torch.zeros(1, 2, round(frame_count / 24 * sample_rate)),
                 "sample_rate": sample_rate}
    else:
        audio = dict(source_audio)
        waveform = audio["waveform"]
        target = round(frame_count / 24 * audio["sample_rate"])
        audio["waveform"] = F.pad(waveform[..., :target], (0, max(0, target - waveform.shape[-1])))
    sound, _ = _encode_ref_audio(audio_vae, audio)
    return {"samples": NestedTensor((visual, sound)),
            "jr_h3_regenerate_video": {"source_frames_24fps": count, "encoded_frames": frame_count,
                                       "frame_rate": 24, "source_audio": source_audio}}


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
               latent=None, video=None, vae=None, audio_vae=None, sigma_schedule="low_denoise_shift6"):
    from comfy.nested_tensor import NestedTensor
    from comfy.samplers import ksampler
    from comfy_extras.nodes_custom_sampler import BasicGuider, SamplerCustomAdvanced
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    sigmas = regenerate_sigmas(denoise, steps, sigma_schedule)
    acceleration.ensure_minimax_h3_model(model)
    selected = select_attention(model, attention)
    if source_mode == "video":
        if latent is not None:
            raise ValueError("Connect only VIDEO when source_mode is video")
        source = encode_input_video(video, vae, audio_vae)
    elif source_mode == "H3 latent":
        if video is not None:
            raise ValueError("Connect only H3 LATENT when source_mode is H3 latent")
        source = latent
    else:
        raise ValueError("Unknown Regenerate source mode")
    video_latent, audio_latent = split_h3_av_latent(source)
    original_video, original_audio = video_latent["samples"], audio_latent["samples"]
    if original_video.shape[0] != 1:
        raise ValueError("Native H3 regeneration currently requires batch size 1")
    if any(n % 2 for n in original_video.shape[-2:]):
        raise ValueError("H3 video latent spatial dimensions must be even (32-pixel image grid)")
    if resize_mode == "keep":
        upscaled, upscale_status = video_latent, "Input latent size retained"
    elif resize_mode in ("scale", "megapixels"):
        upscaled, upscale_status = upscale_h3_video_latent(
            video_latent, resize_mode, scale, target_megapixels, model_name=upscaler_model)
    else:
        raise ValueError("Unknown Regenerate resize mode")
    visual = upscaled["samples"]
    # Preserve every upstream FFN/attention patch. SigmaShift only clones the
    # patcher and updates the native AV schedule, just like the existing flow.
    selected = MiniMaxH3SigmaShift.execute(selected, 6., 3.)[0]
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    working = dict(source)
    working["samples"] = NestedTensor((visual, original_audio))
    working["noise_mask"] = NestedTensor((torch.ones_like(visual), torch.zeros_like(original_audio)))
    guider = BasicGuider.execute(selected, positive)[0]
    sampled = SamplerCustomAdvanced.execute(VideoOnlyNoise(seed), guider, ksampler("euler"), sigmas, working)[0]
    result = dict(source)
    output_video = sampled["samples"].unbind()[0]
    # Native inpainting freezes audio throughout denoising. Reattach the exact
    # input tensor as well: normalization/casting must not drift its output bits.
    result["samples"] = NestedTensor((output_video, original_audio))
    result["noise_mask"] = working["noise_mask"]
    marker = selected.model_options.get("jr_h3_regenerate_lora")
    status = (f"H3 Regenerate: Euler / CFG 1 / {steps} steps / start sigma {sigmas[0].item():.6f}\n"
              f"schedule: {sigma_schedule} (release_original ignores denoise)\n"
              f"sigmas: {sigmas.tolist()}\nattention: {attention}; upstream FFN preserved\n"
              f"Regenerate adapter: {marker or 'not loaded by JR loader; verify input MODEL'}\n"
              f"audio: exact original tensor, locked at every step\n{upscale_status}")
    return result, status
