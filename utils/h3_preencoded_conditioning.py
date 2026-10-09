"""Native H3 conditioning payloads with explicit pre-encoded frame anchors.

Only the pre-encoded path uses this adapter. Legacy IMAGE-only requests keep
delegating to the official nodes. No VAE/CLIP/model objects are patched.
"""

import logging
import math

import torch

from .h3_keyframe_latent import CROP_KEY, MASTER_KEY, resize_keyframe_latent, validate_keyframe_latent


def _anchor_image(media, vae):
    if media.payload is not None:
        return media.payload
    logging.info("[JR H3] No keyframe IMAGE supplied; decoding the master once for vision only (no re-encode).")
    pixels = vae.decode(validate_keyframe_latent(media.keyframe_latent).detach().clone())
    if isinstance(pixels, torch.Tensor) and pixels.ndim == 5 and pixels.shape[:2] == (1, 1):
        pixels = pixels[0]
    if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 4 or pixels.shape[0] != 1 or
            pixels.shape[-1] != 3 or not bool(torch.isfinite(pixels).all())):
        raise ValueError("H3 keyframe VAE decode must return exactly one finite RGB IMAGE.")
    return pixels


def _anchor_block(role, media, image, prepared, vae, frame_count):
    crop = "disabled" if role == "first_frame" else "center"
    block = {"resolved_frame_index": 0 if role == "first_frame" else frame_count - 1}
    if media.keyframe_latent is None:
        block["latent"] = vae.encode(image)
    else:
        master = validate_keyframe_latent(media.keyframe_latent, role).detach().clone()
        block.update({"latent": resize_keyframe_latent(master, prepared.width, prepared.height, crop),
                      MASTER_KEY: master, CROP_KEY: crop})
        logging.info("[JR H3] %s: master %sx%s -> canvas %sx%s; VAE encode skipped.", role,
                     master.shape[-1] * 16, master.shape[-2] * 16, prepared.width, prepared.height)
    return block


def condition_preencoded_anchors(prepared, pipe, clip, vae, audio_vae, native, ref_image_size):
    import node_helpers

    if prepared.ref_audios and audio_vae is None:
        raise ValueError("audio_vae is required when the Director PIP contains Reference or Driving Audio.")
    # Reuse native allocation, frame alignment, image transforms and audio encoding.
    latent, frame_count = native._empty_av_latent(prepared.width, prepared.height, prepared.length)
    pictures = [r for r in pipe.reference_registry if r.family == "Picture"]
    images, ref_items, keyframes, refs = [], [], [], []
    for record in pictures:
        media = pipe.media_for_item(record.item_id)
        image = _anchor_image(media, vae) if media.keyframe_latent is not None else media.payload
        if record.role in {"first_frame", "last_frame"}:
            crop = "disabled" if record.role == "first_frame" else "center"
            image = native._resize(image[:1], prepared.width, prepared.height, crop)
            keyframes.append(_anchor_block(record.role, media, image, prepared, vae, frame_count))
        else:
            h, w = image.shape[1:3]
            scale = min(1.0, math.sqrt(prepared.width * prepared.height / (w * h))) if ref_image_size == "match" else min(1.0, native.REF_IMAGE_SHORT_EDGE / min(h, w))
            tw, th = (max(32, round(v * scale / 32) * 32) for v in (w, h))
            image = native._resize(image[:1], tw, th, "disabled")
            refs.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": vae.encode(image)})
        images.append(image)
        ref_items.append({"type": "image", "data": image})

    for _, video in prepared.ref_videos:
        vh, vw = video.shape[1:3]
        cw, ch = native.adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw, ch = (max(32, round(v / 32) * 32) for v in (vw, vh))
        n = min(video.shape[0], frame_count)
        n -= (n - 5) % 17
        if n < 5:
            raise ValueError("H3 reference video requires at least 5 frames.")
        frames = native._resize(video[:n], cw, ch, "disabled")
        indices = list(range(0, n, native.FPS // 2))
        ref_items.append({"type": "video", "data": frames[indices], "timestamps": [i / 2 for i in range(len(indices))]})
        z = vae.encode(frames)
        refs.append({"kind": "video", "latent_t": z.shape[2], "latent_h": ch // 16,
                     "latent_w": cw // 16, "latent": z, "ref_audio_t": 0, "audio_latent": None})
    for _, audio in prepared.ref_audios:
        ref_items.append({"type": "audio"})
        z, rt = native._encode_ref_audio(audio_vae, audio)
        refs.append({"kind": "audio", "ref_audio_t": rt, "audio_latent": z})

    tokens = (clip.tokenize(prepared.prompt, images=images) if prepared.mode == "Image to Video"
              else clip.tokenize(prepared.prompt, minimax_ref_items=ref_items))
    positive = clip.encode_from_tokens_scheduled(tokens)
    values = {"minimax_keyframes": keyframes}
    if refs:
        values["minimax_refs"] = refs
    return node_helpers.conditioning_set_values(positive, values), latent
