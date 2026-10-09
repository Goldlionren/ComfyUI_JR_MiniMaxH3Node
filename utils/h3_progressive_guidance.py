"""Stage-local H3 guides, full audio locks and sequential hard-prefix support.

Independent references have their own spatial grid. Keyframes share the target
grid. Legacy keyframes are re-encoded at the low canvas; explicitly supplied
clean masters use independent spatial copies. Never resize noisy state.
"""

from dataclasses import dataclass

import torch

from .h3_keyframe_latent import CROP_KEY, MASTER_KEY, resize_keyframe_latent, validate_keyframe_latent
from .h3_progressive_sampler import _error


def _tensor(value, name, *, audio=False):
    prefix = (1, 32, 2) if audio else (1, 24)
    ndim = 4 if audio else 5
    if (not isinstance(value, torch.Tensor) or value.ndim != ndim or
            tuple(value.shape[:len(prefix)]) != prefix or any(s <= 0 for s in value.shape) or
            not value.is_floating_point() or value.layout != torch.strided or value.device.type == "meta" or
            not bool(torch.isfinite(value).all())):
        raise _error(f"Invalid {name}: expected finite {'[1,32,2,T]' if audio else '[1,24,T,H,W]'} latent.")
    if not audio and (value.shape[-2] % 2 or value.shape[-1] % 2):
        raise _error(f"{name} must align to the H3 2x2 patch grid.")
    return value


def _blocks(metadata, key):
    values = metadata.get(key)
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)) or any(not isinstance(v, dict) for v in values):
        raise _error(f"{key} must be a sequence of native H3 guide dictionaries.")
    return values


def _audio_lock(latent_image, video, audio):
    from comfy.nested_tensor import NestedTensor

    mask = latent_image.get("noise_mask")
    if mask is None:
        if bool(torch.count_nonzero(video)):
            raise _error("Connect an empty target video or a fully masked JR 12-token hard prefix.")
        if bool(torch.count_nonzero(audio)):
            raise _error("Nonempty audio requires a fully locked audio mask from JR Audio Driven Latent Builder.")
        return False, 0
    if type(mask) is not NestedTensor or len(mask.unbind()) != 2:
        raise _error("noise_mask must be an H3 AV NestedTensor (video=1, audio=0 or 1).")
    vm, am = mask.unbind()
    for value, target in ((vm, video), (am, audio)):
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape or
                value.layout != torch.strided or value.device.type == "meta" or not bool(torch.isfinite(value).all())):
            raise _error("Guided noise_mask must have exact AV stream shapes and finite values.")
    locked = bool((am == 0).all())
    if not locked and (not bool((am == 1).all()) or bool(torch.count_nonzero(audio))):
        raise _error("Audio noise_mask must be all 0 (locked) or all 1 with empty audio; partial/soft locks are unsupported.")
    prefix_steps = 0
    if not bool((vm == 1).all()):
        # Share the disk-backed driver's temporal contract, not an arbitrary
        # inpainting mask. A fresh generation suffix must remain empty.
        from .h3_sequential_audio import HARD_CONTEXT_LATENT_STEPS

        prefix_steps = HARD_CONTEXT_LATENT_STEPS
        if (video.shape[2] <= prefix_steps or not locked or
                not bool((vm[:, :, :prefix_steps] == 0).all()) or
                not bool((vm[:, :, prefix_steps:] == 1).all())):
            raise _error("Video noise_mask must be all 1, or a JR 12-token hard prefix (0) followed by "
                         "an unlocked suffix (1), with fully locked audio. Spatial/soft/other video masks are unsupported.")
    if bool(torch.count_nonzero(video[:, :, prefix_steps:])):
        raise _error("Connect an empty target video generation area; only the locked JR hard prefix may be nonempty.")
    return locked, prefix_steps


@dataclass
class ProgressiveGuidance:
    low_positive: list
    audio_locked: bool
    has_mask: bool
    description: str
    prefix_steps: int = 0

    def mask_for(self, samples):
        from comfy.nested_tensor import NestedTensor

        if not self.has_mask:
            return None
        video, audio = samples.unbind()
        video_mask = torch.ones_like(video)
        video_mask[:, :, :self.prefix_steps] = 0
        return NestedTensor((video_mask,
                             torch.zeros_like(audio) if self.audio_locked else torch.ones_like(audio)))

    def low_video(self, source, height, width):
        """Resize only CLEAN context, spatially per token; never the noisy state."""
        output = torch.zeros((*source.shape[:-2], height, width), dtype=source.dtype, device="cpu")
        if self.prefix_steps:
            prefix = source[:, :, :self.prefix_steps].detach().to(device="cpu", dtype=torch.float32)
            if prefix.shape[-2:] != (height, width):
                # Treat each temporal slice as an image: no mixing across time.
                slices = prefix[0].movedim(1, 0)
                prefix = torch.nn.functional.interpolate(slices, size=(height, width), mode="area")
                prefix = prefix.movedim(0, 1).unsqueeze(0)
            output[:, :, :self.prefix_steps] = prefix.to(dtype=source.dtype)
        return output

    def restore_prefix(self, output, source):
        """Restore the original CLEAN high-resolution anchor, with owned storage."""
        if not self.prefix_steps:
            return output
        output = output.clone()
        output[:, :, :self.prefix_steps] = source[:, :, :self.prefix_steps].to(output)
        return output


def prepare_guidance(positive, latent_image, video, audio, plan, vae=None):
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE

    locked, prefix_steps = _audio_lock(latent_image, video, audio)
    frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(video.shape[2]))
    keyframe_count = reference_count = 0
    needs_vae = False
    needs_resize = False
    # Validate everything before any expensive VAE work.
    for _, metadata in positive:
        for kf in _blocks(metadata, "minimax_keyframes"):
            keyframe_count += 1
            index = kf.get("resolved_frame_index")
            if type(index) is not int or not 0 <= index < frame_count:
                raise _error("Keyframe resolved_frame_index is outside the target timeline.")
            if kf.get("latent") is None and kf.get("audio_latent") is None:
                raise _error("Keyframe must contain video and/or audio latent.")
            if kf.get("latent") is not None:
                z = _tensor(kf["latent"], "keyframe video")
                if z.shape[-2:] != video.shape[-2:]:
                    raise _error("High-resolution keyframe H/W must match the final target canvas.")
                vt = z.shape[2]
                if vt != 1 and (vt < 2 or (vt - 2) % 5):
                    raise _error("Keyframe clip has an unsupported H3 temporal length.")
                duration = sum(FRAME_PER_TOKEN[k % 5] for k in range(vt))
                if index + duration > frame_count:
                    raise _error("Keyframe clip extends beyond the target timeline.")
                if MASTER_KEY in kf:
                    validate_keyframe_latent({"samples": kf[MASTER_KEY]}, "pre-encoded keyframe master")
                    if vt != 1 or kf.get(CROP_KEY) not in {"disabled", "center"}:
                        raise _error("Pre-encoded master requires a single-frame keyframe and an explicit crop mode.")
                else:
                    needs_vae |= not plan.identity
                needs_resize |= not plan.identity
            if kf.get("audio_latent") is not None:
                z = _tensor(kf["audio_latent"], "keyframe audio", audio=True)
                if z.shape[-1] > audio.shape[-1] - FRAME_RESCALE * index:
                    raise _error("Keyframe audio extends beyond the target timeline.")
        for ref in _blocks(metadata, "minimax_refs"):
            reference_count += 1
            kind = ref.get("kind")
            if kind not in ("image", "video", "video_audio", "audio"):
                raise _error("Unknown minimax_refs kind.")
            if kind != "audio":
                z = _tensor(ref.get("latent"), "reference video")
                if (ref.get("latent_h"), ref.get("latent_w")) != tuple(z.shape[-2:]):
                    raise _error("Reference spatial metadata does not match its latent.")
                if (kind == "image" and z.shape[2] != 1) or (kind != "image" and ref.get("latent_t") != z.shape[2]):
                    raise _error("Reference temporal metadata does not match its latent.")
            if kind != "image":
                rt = ref.get("ref_audio_t")
                if type(rt) is not int or rt < 0 or (kind in ("audio", "video_audio") and rt == 0):
                    raise _error("Invalid reference audio length metadata.")
                if rt:
                    z = _tensor(ref.get("audio_latent"), "reference audio", audio=True)
                    if rt != z.shape[-1]:
                        raise _error("Reference audio metadata does not match its latent.")
    if needs_vae:
        from .h3_vae_compat import inspect_h3_video_vae
        capability = inspect_h3_video_vae(vae)
        if not capability["guided_ready"]:
            raise _error("Connect the same H3 VIDEO VAE used to encode keyframes to the optional vae input (not the audio VAE). "
                         + "; ".join(capability["issues"]))

    low_positive = positive
    if needs_resize:
        from comfy.utils import common_upscale

        low_positive, converted = [], {}
        for text, metadata in positive:
            low_metadata = dict(metadata)
            if metadata.get("minimax_keyframes") is not None:
                low_keyframes = []
                for original in metadata["minimax_keyframes"]:
                    kf = dict(original)
                    z = kf.get("latent")
                    if z is not None:
                        if MASTER_KEY in kf:
                            kf["latent"] = resize_keyframe_latent(
                                kf[MASTER_KEY], plan.low_w * 16, plan.low_h * 16, kf[CROP_KEY]
                            ).to(device=z.device, dtype=z.dtype)
                            low_keyframes.append(kf)
                            continue
                        if id(z) not in converted:
                            pixels = vae.decode(z)
                            # Native video VAE returns [B,frames,H,W,C]; the
                            # VAEDecode node normally flattens this to IMAGE.
                            if isinstance(pixels, torch.Tensor) and pixels.ndim == 5 and pixels.shape[0] == 1:
                                pixels = pixels[0]
                            if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 4 or pixels.shape[-1] != 3 or
                                    not bool(torch.isfinite(pixels).all())):
                                raise _error("H3 keyframe VAE decode must return finite [frames,H,W,3] images.")
                            expected_frames = sum(FRAME_PER_TOKEN[k % 5] for k in range(z.shape[2]))
                            if pixels.shape[0] != expected_frames:
                                raise _error("VAE decode changed the keyframe's pixel-frame duration.")
                            pixels = common_upscale(pixels.movedim(-1, 1), plan.low_w * 16, plan.low_h * 16,
                                                    "area", "disabled").movedim(1, -1)
                            low_z = _tensor(vae.encode(pixels), "re-encoded low-resolution keyframe")
                            if tuple(low_z.shape) != (*z.shape[:-2], plan.low_h, plan.low_w):
                                raise _error("VAE re-encoding changed keyframe temporal length or returned the wrong canvas.")
                            converted[id(z)] = low_z.to(device=z.device, dtype=z.dtype)
                        kf["latent"] = converted[id(z)]
                    low_keyframes.append(kf)
                low_metadata["minimax_keyframes"] = low_keyframes
            low_positive.append([text, low_metadata])
    return ProgressiveGuidance(low_positive, locked, latent_image.get("noise_mask") is not None,
                               f"Guides: {reference_count} independent refs, {keyframe_count} keyframe blocks; "
                               f"audio {'LOCKED (original latent preserved)' if locked else 'generated'}"
                               + (f"; hard prefix LOCKED: {prefix_steps} latent tokens / 39 frames; "
                                  "original high-resolution context restored" if prefix_steps else ""),
                               prefix_steps=prefix_steps)
