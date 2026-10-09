"""Convert the decoded H3 endpoint to a reusable image-keyframe latent.

The legacy keyframe output is an explicit one-encode baseline, not lossless.
The appended context output retains the original decoder window without encode.
Only the reviewed 17-frame / 5-token decoder contract permits a 7-token tail.
"""

import torch

from .h3_keyframe_latent import validate_keyframe_latent
from .h3_vae_compat import inspect_h3_video_vae


def _error(message):
    return ValueError(f"JR H3 Tail Frame Latent: {message}")


def validate_tail_video_latent(latent):
    z = latent.get("samples") if isinstance(latent, dict) else None
    if (not isinstance(z, torch.Tensor) or z.is_nested or z.layout != torch.strided or
            z.device.type == "meta" or z.ndim != 5 or tuple(z.shape[:2]) != (1, 24) or
            min(z.shape[2:]) < 1 or not z.is_floating_point()):
        raise _error("Connect Split AV Latent's video output [1,24,T,H,W], not the AV pair or audio.")
    if z.shape[2] != 1 and z.shape[2] % 5 != 2:
        raise _error("Expected a complete H3 video (T=5k+2), or a single image keyframe (T=1).")
    if not bool(torch.isfinite(z).all()):
        raise _error("Input contains NaN or Inf.")
    return z


def tail_frame_count(tokens):
    return 1 if tokens == 1 else ((tokens - 2) // 5) * 17 + 5


def _has_reviewed_tail_contract(vae):
    stage = getattr(vae, "first_stage_model", None)
    expected = {"clip_length": 17, "tokens_chunk_size": 5, "token_overlap": 2,
                "token_drop": 3, "frame_pre_padding": 3, "vae_ratio_t": 4, "frame_overlap": 5}
    return all(getattr(stage, name, None) == value for name, value in expected.items())


def _last_image(pixels, height, width, frames, *, supplied=False):
    if isinstance(pixels, torch.Tensor) and pixels.ndim == 5 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if (not isinstance(pixels, torch.Tensor) or pixels.is_nested or pixels.layout != torch.strided or
            pixels.device.type == "meta" or pixels.ndim != 4 or not pixels.is_floating_point() or
            tuple(pixels.shape[1:]) != (height, width, 3) or pixels.shape[0] < 1):
        raise _error(f"Expected nonempty RGB IMAGE frames [N,{height},{width},3].")
    count = pixels.shape[0]
    if (count > frames if supplied else count != frames):
        raise _error(f"Decoded frame count {count} does not match the H3 source ({frames}).")
    image = pixels[-1:].detach().clone().contiguous()
    if not bool(torch.isfinite(image).all()) or bool((image < 0).any()) or bool((image > 1).any()):
        raise _error("Tail IMAGE must contain finite RGB values in [0,1].")
    return image, count


def extract_h3_tail_frame_latent(video_latent, vae, decode_mode="tail_context", decoded_frames=None):
    z = validate_tail_video_latent(video_latent)
    if decode_mode not in {"tail_context", "full_video"}:
        raise _error("decode_mode must be tail_context or full_video.")
    capabilities = inspect_h3_video_vae(vae)
    metadata_ok = capabilities["latent_channels"] == 24 and capabilities["spatial_compression"] == 16
    if not (metadata_ok or capabilities["known_trt_wrapper"]):
        raise _error("Use an H3 video VAE (24 channels, spatial compression 16).")
    reuse_single = z.shape[2] == 1 and decoded_frames is None
    if not reuse_single and not capabilities["can_encode"]:
        raise _error("A video tail needs an encoder. Use the native H3 VAE; a decode-only TRT engine is insufficient.")
    height, width = z.shape[-2] * 16, z.shape[-1] * 16
    source_frames = tail_frame_count(z.shape[2])
    use_tail = decode_mode == "tail_context" and _has_reviewed_tail_contract(vae)
    tokens = min(7, z.shape[2]) if use_tail else z.shape[2]
    # Preserve the original decoder window, NOT the re-encoded image keyframe.
    # Own the compact storage and isolate it from potentially mutating VAE code.
    context = {"samples": z[:, :, -tokens:].detach().clone().contiguous()}
    if decoded_frames is not None:
        image, count = _last_image(decoded_frames, height, width, source_frames, supplied=True)
        source = f"provided IMAGE endpoint ({count} supplied frames; source has {source_frames})"
    else:
        if not capabilities["can_decode"]:
            raise _error("Decoder unavailable. Connect decoded_frames or use a complete H3 video VAE.")
        # Owned copy: do not let a VAE mutate the sampling output or retain its full storage.
        decoded = vae.decode(context["samples"].clone())
        image, _ = _last_image(decoded, height, width, tail_frame_count(tokens))
        del decoded
        source = f"decoded {tokens}/{z.shape[2]} latent time slices; endpoint frame {source_frames - 1}"
        if decode_mode == "tail_context" and not use_tail:
            source += "; unrecognized temporal contract: full decode fallback"
    if reuse_single:
        samples = z.detach().clone()
    else:
        # Pixels are authoritative here: video tokens are not image-keyframe tokens.
        samples = vae.encode(image.clone())
        validate_keyframe_latent({"samples": samples}, "encoded tail")
        if tuple(samples.shape[-2:]) != tuple(z.shape[-2:]):
            raise _error("Encoder changed spatial size; use a matching H3 video VAE.")
        samples = samples.detach().clone()
    # This is a new image latent. Video masks, audio, temporal guides and batch indices do not apply.
    status = (f"{source}\n{width}x{height}; single-frame H3 latent [1,24,1,{height // 16},{width // 16}]\n"
              f"VAE encode calls: {0 if reuse_single else 1}; no resize, no detail enhancement. "
              "Connect both IMAGE and LATENT to the next Director Builder.\n"
              f"tail_context_latent: original {tokens} tokens / {tail_frame_count(tokens)} frames; "
              "no re-encode in this output. Decode then select the last IMAGE; not a Director keyframe.")
    if decoded_frames is not None:
        status += ("\nWarning: context retains the original native endpoint, not the supplied IMAGE endpoint. "
                   "Trimmed or externally modified frames may not match its final decoded frame.")
    return {"samples": samples}, image, status, context
