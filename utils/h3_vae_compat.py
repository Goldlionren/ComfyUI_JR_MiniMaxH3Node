"""Read-only VAE capabilities and opt-in final-decode probes; no TRT dependency.

Does not synthesize missing VAE metadata, change shared VAE objects, or claim
Guided/tiled compatibility from a VAE socket type. TRT integration remains opt-in.
"""

from __future__ import annotations


def inspect_h3_video_vae(vae):
    issues = []
    stage = getattr(vae, "first_stage_model", None)
    known_trt = type(vae).__name__ == "ComfyTRTVAE" and type(stage).__name__ == "MiniMaxH3TRTVAE"
    channels = getattr(vae, "latent_channels", None)
    ratio = None
    try:
        getter = getattr(vae, "spacial_compression_decode", None)
        ratio = getter() if callable(getter) else None
    except (AttributeError, TypeError, IndexError):
        issues.append("VAE spatial compression metadata is incomplete")
    if channels != 24:
        issues.append("H3 video VAE must explicitly declare latent_channels=24")
    if ratio != 16:
        issues.append("H3 video VAE must explicitly declare spatial compression=16")
    encode = callable(getattr(vae, "encode", None))
    decode = callable(getattr(vae, "decode", None))
    if known_trt:
        encode = encode and getattr(stage, "encoder_runner", None) is not None
        decode = decode and getattr(stage, "decoder_runner", None) is not None
    if not encode:
        issues.append("Encoder is unavailable; Guided keyframe round-trip requires both roles")
    if not decode:
        issues.append("Decoder is unavailable")
    metadata_ok = channels == 24 and ratio == 16
    return dict(provider=f"{type(vae).__module__}.{type(vae).__name__}", known_trt_wrapper=known_trt,
                latent_channels=channels, spatial_compression=ratio, can_encode=encode, can_decode=decode,
                guided_ready=metadata_ok and encode and decode,
                decode_probe_candidate=decode and (metadata_ok or known_trt),
                engine_tested=False, issues=issues)


def check_trt_decoder_profile(vae, video):
    """Explicit probe only: load an existing engine and check BEFORE inference.

    First experiment supports the reviewed 256px runtime tile and 16x16 engine.
    Smaller canvases / 32x32 engines are rejected, not padded or reconfigured.
    """
    stage = vae.first_stage_model
    if getattr(stage, "tile_size", None) != 256 or min(video.shape[-2:]) < 16:
        raise ValueError("JR H3 VAE: TRT decode probe requires 256px tiles and canvas >=256px on both axes.")
    runner = stage.decoder_runner
    if runner is None:
        raise ValueError("JR H3 VAE: decoder runner missing.")
    runner.load_to_gpu()
    engine = runner.engine
    expected = (1, 24, 7, 16, 16)
    profile = tuple(tuple(s) for s in engine.get_tensor_profile_shape("latent_tile", 0))
    if profile != (expected, expected, expected):
        raise ValueError(f"JR H3 VAE: unvalidated TRT decoder profile {profile}; expected fixed {expected}.")
    for name in ("latent_tile", "pixel_tile"):
        if str(engine.get_tensor_dtype(name)).lower().split(".")[-1] not in ("half", "float16"):
            raise ValueError("JR H3 VAE: the reviewed TRT runner requires fp16 input/output bindings.")
    # The upstream runner allocates this exact output buffer; don't infer with a
    # mismatched engine even if its input profile happens to match.
    if not runner.context.set_input_shape("latent_tile", expected):
        raise ValueError("JR H3 VAE: TRT rejected the decoder input shape.")
    if tuple(runner.context.get_tensor_shape("pixel_tile")) != (1, 3, 28, 256, 256):
        raise ValueError("JR H3 VAE: unexpected TRT decoder output binding shape.")


def decode_h3_video_checked(vae, latent):
    """Final-decode validation entry; returns IMAGE, never feeds audio to a VAE."""
    import torch
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN
    from comfy.nested_tensor import NestedTensor

    report = inspect_h3_video_vae(vae)
    if not report["decode_probe_candidate"]:
        raise ValueError("JR H3 VAE: " + "; ".join(report["issues"]))
    video = latent.get("samples") if isinstance(latent, dict) else latent
    if isinstance(video, NestedTensor):
        parts = video.unbind()
        if len(parts) != 2:
            raise ValueError("JR H3 VAE: expected an AV pair.")
        video = parts[0]
    if (not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[:2] != (1, 24) or
            not video.is_floating_point() or not bool(torch.isfinite(video).all()) or min(video.shape[2:]) < 1):
        raise ValueError("JR H3 VAE: expected finite batch-1 [1,24,T,H,W] video latent.")
    if report["known_trt_wrapper"]:
        check_trt_decoder_profile(vae, video)
    with torch.inference_mode():
        pixels = vae.decode(video.clone())
    if isinstance(pixels, torch.Tensor) and pixels.ndim == 5 and pixels.shape[0] == 1:
        pixels = pixels[0]
    frames = sum(FRAME_PER_TOKEN[i % 5] for i in range(video.shape[2]))
    expected = (frames, video.shape[-2] * 16, video.shape[-1] * 16, 3)
    if (not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != expected or
            not pixels.is_floating_point() or not bool(torch.isfinite(pixels).all())):
        raise ValueError(f"JR H3 VAE: decode must return finite IMAGE with shape {expected}.")
    return pixels
