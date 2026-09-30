"""Native ComfyUI handoff of completed H3 drafts to LTX-2.5 refinement."""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import torch

MARKER = "jr_h3_ltx_bridge"
FPS = 24
SIGMAS = (0.909375, 0.725, 0.421875, 0.0)
ADAPTER_SHA256 = "170199a390c40ac97f5895bc9c8cc29817e74fb9193c858a85d8c0f1f30724ac"


def video_tensor(latent, channels):
    if not isinstance(latent, dict) or not isinstance(latent.get("samples"), torch.Tensor):
        raise ValueError("Connect a plain video LATENT, split the AV latent first.")
    samples = latent["samples"]
    if samples.is_nested or samples.ndim != 5 or samples.shape[0] != 1 or samples.shape[1] != channels:
        raise ValueError(f"Expected batch-one [1,{channels},T,H,W] video latent.")
    if not samples.is_floating_point() or samples.device.type == "meta" or samples.layout != torch.strided:
        raise ValueError("Video latent must be a materialized floating tensor.")
    if min(samples.shape) <= 0 or not bool(torch.isfinite(samples).all()):
        raise ValueError("Video latent is empty or contains nonfinite values.")
    if any(latent.get(key) is not None for key in ("noise_mask", "batch_index")):
        raise ValueError("H3→LTX accepts a completed clean draft; masks/batch indices must not be carried across models.")
    return samples


def h3_frame_count(tokens):
    if tokens < 2 or (tokens - 2) % 5:
        raise ValueError("H3 draft must use the native 17k+5 frame grid (T=5k+2).")
    return (tokens - 2) // 5 * 17 + 5


def bridge_metadata(samples):
    frames = h3_frame_count(samples.shape[2])
    if samples.shape[-2] % 2 or samples.shape[-1] % 2:
        raise ValueError("Upscaled H3 spatial dimensions must correspond to pixel multiples of 32.")
    padded = ((frames - 1 + 7) // 8) * 8 + 1
    return {"version": 1, "source_frames": frames, "padded_frames": padded, "fps": FPS,
            "height": samples.shape[-2]*16, "width": samples.shape[-1]*16,
            "adapter_sha256": ADAPTER_SHA256, "normalization": "ltx_checkpoint_normalized"}


def validate_bridge(latent):
    video = video_tensor(latent, 128)
    meta = latent.get(MARKER)
    if not isinstance(meta, dict) or meta.get("version") != 1 or meta.get("fps") != FPS:
        raise ValueError("Connect the output of JR H3 → LTX Latent Adapter; timeline metadata is missing.")
    frames, padded = meta["source_frames"], meta["padded_frames"]
    if type(frames) is not int or frames < 5 or (frames-5) % 17:
        raise ValueError("Invalid H3 source frame count.")
    expected_padded = ((frames-1+7)//8)*8+1
    expected = (1, 128, (expected_padded-1)//8+1, meta["height"]//32, meta["width"]//32)
    if padded != expected_padded or tuple(video.shape) != expected:
        raise ValueError("Adapted latent no longer matches its frame/spatial metadata.")
    return video, dict(meta)


def validate_audio(audio, frames):
    if not isinstance(audio, dict) or not isinstance(audio.get("waveform"), torch.Tensor):
        raise ValueError("Connect decoded original H3 AUDIO.")
    wave, rate = audio["waveform"], audio.get("sample_rate")
    if type(rate) is not int or rate <= 0 or wave.ndim != 3 or wave.shape[0] != 1 or wave.shape[1] not in (1, 2):
        raise ValueError("Expected batch-one mono/stereo AUDIO with a positive sample rate.")
    if not wave.is_floating_point() or not bool(torch.isfinite(wave).all()):
        raise ValueError("Audio must be finite floating PCM.")
    # Native H3 audio uses 40 Hz latent rounding. Permit one such interval,
    # never quietly accept a different clip or many seconds of missing audio.
    if abs(wave.shape[-1]/rate - frames/FPS) > 1/40 + 0.005:
        raise ValueError("H3 audio duration does not match the draft; use audio from the same Stage1 request.")
    return wave, rate


def load_adapter(path):
    from .h3_ltx_frozen.adapter import H3ToLTXAdapter
    path = Path(path)
    config = path.with_name("config.json")
    if path.name != "model.safetensors" or not config.is_file():
        raise ValueError("Install the released model.safetensors and config.json together in h3_ltx_adapters.")
    specification = json.loads(config.read_text(encoding="utf-8"))
    model_config = specification.get("model_config", {})
    dimensions = {"model_type": "tiny", "in_channels": 384, "out_channels": 128,
                  "width": 752, "num_blocks": 22, "expansion": 2, "groups": 16}
    if specification.get("model_sha256") != ADAPTER_SHA256 or any(model_config.get(k) != v for k, v in dimensions.items()):
        raise ValueError("Adapter config differs from the tested released architecture.")
    with path.open("rb") as stream:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
    if digest != ADAPTER_SHA256:
        raise ValueError("H3→LTX checkpoint hash differs from the tested released adapter.")
    return H3ToLTXAdapter.from_pretrained(path.parent, device="cpu", dtype=torch.bfloat16)


def convert_video(latent, adapter):
    import comfy.model_management as mm
    video = video_tensor(latent, 24)
    meta = bridge_metadata(video)
    device = mm.get_torch_device()
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("H3→LTX adapter currently requires a BF16-capable CUDA GPU.")
    # Full-volume GroupNorm prevents exact temporal chunking. Reserve workspace
    # for the frozen Conv3D model rather than silently altering its math.
    volume = ((meta["padded_frames"]-1)//8+1)*(meta["height"]//32)*(meta["width"]//32)
    mm.free_memory(2*1024**3 + volume*752*2*16, device)
    try:
        adapter.model.to(device)
        adapter.device = device
        with torch.inference_mode():
            output = adapter.convert(video, pixel_frames=meta["source_frames"],
                                     pixel_height=meta["height"], pixel_width=meta["width"],
                                     input_normalization="normalized").cpu()
    finally:
        adapter.model.cpu()
        adapter.device = torch.device("cpu")
    result = {"samples": output, MARKER: meta}
    validate_bridge(result)
    return result, meta


def refine_sigmas(steps=3, denoise=1.0, scheduler="sol_h3_original", ltx_model=None):
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError("steps must be an integer from 1 to 100.")
    if not isinstance(denoise, (int, float)) or not math.isfinite(denoise) or not 0 <= denoise <= 1:
        raise ValueError("denoise must be finite and between 0 and 1.")
    if scheduler == "sol_h3_original":
        if steps != 3 or denoise != 1:
            raise ValueError("sol_h3_original is the old fixed recipe: set steps=3 and denoise=1, or select simple for adjustable refinement.")
        return torch.tensor(SIGMAS, dtype=torch.float32)
    import comfy.samplers
    from comfy_extras.nodes_custom_sampler import BasicScheduler
    if scheduler not in comfy.samplers.SCHEDULER_NAMES:
        raise ValueError("Unknown native scheduler.")
    if ltx_model is None:
        raise ValueError("Connect ltx_model to the same LTX MODEL used by BasicGuider to calculate native denoise sigmas.")
    return BasicScheduler.execute(ltx_model, scheduler, steps, denoise)[0]


def prepare_refine(video_latent, original_audio, audio_vae,
                   steps=3, denoise=1.0, scheduler="sol_h3_original", ltx_model=None):
    from comfy.nested_tensor import NestedTensor
    from comfy_extras.nodes_audio import VAEEncodeAudio
    from comfy_extras.nodes_lt import LTXVConcatAVLatent
    from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio
    sigmas = refine_sigmas(steps, denoise, scheduler, ltx_model)
    video, meta = validate_bridge(video_latent)
    validate_audio(original_audio, meta["source_frames"])
    if audio_vae.latent_channels != 8 or not hasattr(audio_vae.first_stage_model, "num_of_latents_from_frames"):
        raise ValueError("Connect the LTX-2.5 audio VAE, not the H3 audio VAE.")
    encoded = VAEEncodeAudio.execute(audio_vae, original_audio)[0]["samples"]
    target = LTXVEmptyLatentAudio.execute(meta["padded_frames"], FPS, 1, audio_vae)[0]["samples"]
    fitted, _ = LTXVConcatAVLatent.fit_audio(target, encoded, None)
    if not bool(torch.isfinite(fitted).all()):
        raise ValueError("LTX audio encoding produced nonfinite values.")
    latent = {"samples": NestedTensor((video.cpu().float(), fitted.cpu().float())), MARKER: meta}
    report = (f"JR H3→LTX: {meta['source_frames']} source frames -> {meta['padded_frames']} internal frames; "
              f"scheduler={scheduler}, denoise={denoise}, updates={max(0, len(sigmas)-1)}, sigmas={sigmas.tolist()}; original H3 PCM retained for final mux. "
              "Native ComfyUI experiment; model/attention selection is supplied by the workflow.")
    logging.info(report)
    return latent, sigmas, original_audio, report


def finish_media(images, original_audio, bridge_latent):
    _, meta = validate_bridge(bridge_latent)
    wave, rate = validate_audio(original_audio, meta["source_frames"])
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("Expected decoded IMAGE batch [frames,height,width,channels].")
    if (images.shape[0] != meta["padded_frames"] or images.shape[1:3] != (meta["height"], meta["width"])):
        raise ValueError("Decoded video geometry/frame count differs from the adapted latent.")
    frames = meta["source_frames"]
    audio = dict(original_audio)
    # Preserve every original sample up to the source-video duration. No resample,
    # generated Stage2 audio replacement or padding is hidden in the final mux.
    audio["waveform"] = wave[..., :round(frames/FPS*rate)]
    return images[:frames], audio, float(FPS), json.dumps(meta, ensure_ascii=False)
