"""Endpoint semantics use native temporal decoding, not a T=-1 assumption."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE, ViT3DDecoder
from comfy.nested_tensor import NestedTensor
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_tail_frame_latent import JR_H3_TailFrameLatent
from ComfyUI_JR_MiniMaxH3Node.utils.h3_tail_frame_latent import extract_h3_tail_frame_latent
from test_h3_keyframe_latent import build, condition


def stage(*, real_decoder=False):
    # Native normalization, temporal frame plan, padding and blending; no checkpoint load.
    obj = MiniMaxH3VideoVAE.__new__(MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(obj)
    for key, value in dict(clip_length=17, tokens_chunk_size=5, token_overlap=2,
                           token_drop=3, frame_pre_padding=3, vae_ratio_t=4,
                           frame_overlap=5, vae_ratio=16).items():
        setattr(obj, key, value)
    obj.latents_mean = torch.zeros(24)
    obj.latents_std = torch.ones(24)
    obj.pixel_mean = torch.full((1, 3, 1, 1, 1), 0.5)
    obj.pixel_std = torch.full((1, 3, 1, 1, 1), 0.1)
    if real_decoder:
        with torch.random.fork_rng():
            torch.manual_seed(772)
            obj.decoder = ViT3DDecoder(num_layers=1, heads=2, dim_head=16)
            for name, param in obj.decoder.named_parameters():
                torch.nn.init.normal_(param, std=0.05)
        obj._adaptive_decode = lambda z: obj.decoder(z)
    else:
        obj.decoder = SimpleNamespace(out_channels=3)

        def context_decode(z):
            # Each output depends on its whole window, like the decoder's temporal attention.
            data = z[:, :3] * 0.3 + z[:, :3].mean(dim=2, keepdim=True) * 0.7
            return data.repeat_interleave(4, 2).repeat_interleave(16, 3).repeat_interleave(16, 4)

        obj._adaptive_decode = context_decode
    return obj


class VAE:
    latent_channels = 24

    def __init__(self, *, real_decoder=False, output_dtype=torch.float32):
        self.first_stage_model = stage(real_decoder=real_decoder)
        self.decode_calls, self.encode_calls = [], []
        self.output_dtype = output_dtype

    def spacial_compression_decode(self):
        return 16

    def decode(self, z):
        self.decode_calls.append(z.clone())
        return self.first_stage_model.decode(z).movedim(1, -1)[0]

    def encode(self, image):
        self.encode_calls.append(image.clone())
        z = F.interpolate(image.movedim(-1, 1), scale_factor=1 / 16, mode="area")
        return z.unsqueeze(2).repeat(1, 8, 1, 1, 1).to(self.output_dtype)


def video(tokens=12, dtype=torch.float32):
    return {"samples": torch.linspace(-1, 1, 24 * tokens * 4).reshape(1, 24, tokens, 2, 2).to(dtype)}


@pytest.mark.parametrize("tokens", [2, 7, 12, 37, 52, 72, 107])
def test_tail_matches_full_native_temporal_decode(tokens):
    vae = VAE()
    source = video(tokens)
    before = source["samples"].clone()
    expected = vae.decode(before)[-1:].clone()
    vae.decode_calls.clear()
    latent, image, status, context = extract_h3_tail_frame_latent(source, vae)
    assert torch.equal(context["samples"], before[:, :, -min(7, tokens):])
    assert torch.equal(vae.decode(context["samples"])[-1:], image)
    assert context["samples"].data_ptr() != source["samples"].data_ptr()
    assert context["samples"].untyped_storage().nbytes() == context["samples"].numel() * context["samples"].element_size()
    assert torch.equal(image, expected)
    assert vae.decode_calls[0].shape[2] == min(7, tokens)
    assert len(vae.encode_calls) == 1 and torch.equal(vae.encode_calls[0], image)
    assert latent["samples"].shape == (1, 24, 1, 2, 2)
    assert "VAE encode calls: 1" in status
    assert torch.equal(source["samples"], before)
    assert image.untyped_storage().nbytes() == image.numel() * image.element_size()


def test_real_small_vit_decoder_requires_window_not_last_token():
    vae = VAE(real_decoder=True)
    with torch.inference_mode():
        z = video()["samples"]
        full = vae.decode(z)[-1:]
        naive = vae.decode(z[:, :, -1:])
        _, image, _, context = extract_h3_tail_frame_latent({"samples": z}, vae)
        torch.testing.assert_close(vae.decode(context["samples"])[-1:], image, rtol=0, atol=0)
    torch.testing.assert_close(image, full, rtol=0, atol=0)
    assert not torch.allclose(image, naive, rtol=0, atol=1e-6)


def test_full_decode_and_unknown_contract_fallback():
    source = video()
    vae = VAE()
    _, image, _, context = extract_h3_tail_frame_latent(source, vae, "full_video")
    assert torch.equal(context["samples"], source["samples"])
    assert torch.equal(vae.decode(context["samples"])[-1:], image)
    assert vae.decode_calls[0].shape[2] == 12
    del vae.first_stage_model.frame_overlap
    # Opaque wrapper with a usable decode; the optimization must not guess its chunking.
    vae.decode = lambda z: torch.zeros(39, 32, 32, 3)
    _, _, status, context = extract_h3_tail_frame_latent(source, vae)
    assert torch.equal(context["samples"], source["samples"])
    assert "full decode fallback" in status


@pytest.mark.parametrize("rank5", [False, True])
@pytest.mark.parametrize("count", [1, 30, 39])
def test_supplied_frames_reuse_trimmed_endpoint_without_decoding(rank5, count):
    vae = VAE()
    frames = torch.linspace(0, 1, count * 32 * 32 * 3).reshape(count, 32, 32, 3)
    source = video()
    latent, image, status, context = extract_h3_tail_frame_latent(source, vae, decoded_frames=frames[None] if rank5 else frames)
    assert torch.equal(context["samples"], source["samples"][:, :, -7:])
    assert "not the supplied IMAGE endpoint" in status
    assert not vae.decode_calls and len(vae.encode_calls) == 1
    assert torch.equal(image, frames[-1:]) and "provided IMAGE" in status
    assert latent["samples"].shape[2] == 1
    image.zero_()
    assert frames[-1:].sum() > 0


def test_single_image_latent_is_not_reencoded():
    source, vae = video(1), VAE()
    latent, _, status, context = extract_h3_tail_frame_latent(source, vae)
    assert torch.equal(context["samples"], source["samples"])
    assert context["samples"].data_ptr() != latent["samples"].data_ptr()
    assert not vae.encode_calls and "VAE encode calls: 0" in status
    assert torch.equal(latent["samples"], source["samples"])
    assert latent["samples"].data_ptr() != source["samples"].data_ptr()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_new_latent_owns_samples_and_uses_encoder_dtype(dtype):
    vae, source = VAE(output_dtype=dtype), video(dtype=dtype)
    source.update(noise_mask=torch.ones(1), batch_index=[10], minimax_keyframes=["old"], label="video")
    result, image, _, context = extract_h3_tail_frame_latent(source, vae)
    assert set(context) == {"samples"}
    assert context["samples"].dtype == source["samples"].dtype
    assert context["samples"].device == source["samples"].device
    assert set(result) == {"samples"}
    assert result["samples"].dtype == dtype and result["samples"].device.type == "cpu"
    assert image.dtype == torch.float32
    assert "noise_mask" in source


@pytest.mark.parametrize("value", [None, {}, {"samples": torch.zeros(1, 32, 2, 4)},
    {"samples": torch.zeros(2, 24, 7, 2, 2)}, {"samples": torch.zeros(1, 24, 6, 2, 2)},
    {"samples": torch.zeros(1, 24, 0, 2, 2)}, {"samples": torch.zeros(1, 24, 7, 2, 2, dtype=torch.int64)},
    {"samples": torch.full((1, 24, 7, 2, 2), float("inf"))},
    {"samples": NestedTensor((torch.zeros(1, 24, 7, 2, 2), torch.zeros(1, 32, 2, 37)))}])
def test_invalid_video_rejected_before_vae(value):
    vae = VAE()
    with pytest.raises(ValueError):
        extract_h3_tail_frame_latent(value, vae)
    assert not vae.decode_calls and not vae.encode_calls


@pytest.mark.parametrize("frames", [torch.zeros(0, 32, 32, 3), torch.zeros(40, 32, 32, 3),
    torch.zeros(1, 16, 32, 3), torch.zeros(1, 32, 32, 4), torch.ones(1, 32, 32, 3) * 2,
    torch.full((1, 32, 32, 3), float("nan"))])
def test_invalid_frames_rejected_before_encode(frames):
    vae = VAE()
    with pytest.raises(ValueError):
        extract_h3_tail_frame_latent(video(), vae, decoded_frames=frames)
    assert not vae.encode_calls


def test_vae_capabilities_and_bad_encoder_output():
    vae = VAE()
    vae.encode = None
    with pytest.raises(ValueError, match="encoder"):
        extract_h3_tail_frame_latent(video(), vae)
    vae = VAE()
    vae.latent_channels = 16
    with pytest.raises(ValueError, match="H3 video VAE"):
        extract_h3_tail_frame_latent(video(), vae)
    vae = VAE()
    vae.encode = lambda image: torch.zeros(1, 24, 2, 2, 2)
    with pytest.raises(ValueError, match="single-frame"):
        extract_h3_tail_frame_latent(video(), vae)


def test_tail_feeds_director_first_and_last_without_more_vae_calls():
    tail, image, _, _ = JR_H3_TailFrameLatent().extract(video(), VAE())
    pipe = build(first_frame=image, first_latent=tail, last_frame=image, last_latent=tail)
    (positive, _), vae, _ = condition(pipe, width=32, height=32)
    assert not vae.encode_calls and not vae.decode_calls
    blocks = positive[0][1]["minimax_keyframes"]
    assert [b["resolved_frame_index"] for b in blocks] == [0, 4]
    assert all(torch.equal(b["latent"], tail["samples"]) for b in blocks)


def test_schema_and_invalid_mode():
    schema = JR_H3_TailFrameLatent.INPUT_TYPES()
    assert set(schema["required"]) == {"video_latent", "vae", "decode_mode"}
    assert set(schema["optional"]) == {"decoded_frames"}
    assert JR_H3_TailFrameLatent.RETURN_TYPES == ("LATENT", "IMAGE", "STRING", "LATENT")
    assert JR_H3_TailFrameLatent.RETURN_NAMES == ("tail_latent", "tail_image", "status", "tail_context_latent")
    with pytest.raises(ValueError, match="decode_mode"):
        extract_h3_tail_frame_latent(video(), VAE(), "guess")


def test_context_isolated_from_mutating_decoder_and_encoder():
    vae, source = VAE(), video()
    before = source["samples"].clone()
    original_decode = vae.decode

    def decode(z):
        result = original_decode(z)
        z.zero_()
        return result

    vae.decode = decode
    _, _, _, context = extract_h3_tail_frame_latent(source, vae)
    assert torch.equal(source["samples"], before)
    assert torch.equal(context["samples"], before[:, :, -7:])
    context["samples"].zero_()
    assert torch.equal(source["samples"], before)


@pytest.mark.skipif(not os.environ.get("JR_H3_TAIL_TEST_VAE") or not torch.cuda.is_available(),
                    reason="Opt-in installed H3 video VAE checkpoint and CUDA required")
def test_installed_vae_context_endpoint_equivalence():
    """Real weights, synthetic non-content latent; no diffusion or media files written."""
    import comfy.sd
    import comfy.utils

    state = comfy.utils.load_torch_file(os.environ["JR_H3_TAIL_TEST_VAE"], safe_load=True)
    vae = comfy.sd.VAE(sd=state)
    del state
    generator = torch.Generator(device="cpu").manual_seed(9157)
    source = {"samples": torch.randn(1, 24, 12, 16, 16, generator=generator) * .3}
    before = source["samples"].clone()
    with torch.inference_mode():
        full = vae.decode(source["samples"].clone()).reshape(-1, 256, 256, 3)[-1:].clone()
        _, image, _, context = extract_h3_tail_frame_latent(source, vae)
        decoded = vae.decode(context["samples"].clone()).reshape(-1, 256, 256, 3)[-1:]
    assert torch.equal(context["samples"], before[:, :, -7:])
    assert torch.equal(source["samples"], before)
    assert torch.isfinite(decoded).all()
    torch.testing.assert_close(decoded, image, rtol=0, atol=1e-5)
    torch.testing.assert_close(full, image, rtol=0, atol=1e-5)
    print(f"Real H3 VAE: context vs tail max_abs={(decoded - image).abs().max().item():.8g}; "
          f"full vs tail max_abs={(full - image).abs().max().item():.8g}")
