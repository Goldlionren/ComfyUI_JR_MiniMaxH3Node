from types import SimpleNamespace

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from ComfyUI_JR_MiniMaxH3Node.utils.h3_vae_compat import (
    check_trt_decoder_profile,
    decode_h3_video_checked,
    inspect_h3_video_vae,
)


class NativeVAE:
    latent_channels = 24
    def spacial_compression_decode(self):
        return 16
    def encode(self, pixels):
        raise AssertionError("Final decode must not invoke encoder")
    def decode(self, video):
        from comfy.ldm.minimax.model import FRAME_PER_TOKEN
        frames = sum(FRAME_PER_TOKEN[i % 5] for i in range(video.shape[2]))
        video.zero_()  # deliberately hostile fixture: probe must isolate input
        return torch.zeros(1, frames, video.shape[-2] * 16, video.shape[-1] * 16, 3)


class MiniMaxH3TRTVAE:
    tile_size = 256
    decoder_runner = object()
    encoder_runner = None


class ComfyTRTVAE(NativeVAE):
    latent_channels = None
    first_stage_model = MiniMaxH3TRTVAE()
    def spacial_compression_decode(self):
        raise AttributeError("upscale_ratio")


def test_partial_trt_contract_is_not_silently_repaired():
    vae = ComfyTRTVAE()
    before = vars(vae).copy()
    report = inspect_h3_video_vae(vae)
    assert report["known_trt_wrapper"] and report["decode_probe_candidate"]
    assert not report["guided_ready"] and not report["can_encode"] and not report["engine_tested"]
    assert vars(vae) == before
    assert "compression metadata" in "; ".join(report["issues"])
    assert inspect_h3_video_vae(NativeVAE())["guided_ready"]


@pytest.mark.parametrize("frames", [1, 2, 7])
def test_final_decode_extracts_video_and_preserves_av(frames):
    video = torch.ones(1, 24, frames, 4, 6)
    audio = torch.randn(1, 32, 2, 8)
    saved = audio.clone()
    latent = {"samples": NestedTensor([video, audio]), "metadata": object()}
    pixels = decode_h3_video_checked(NativeVAE(), latent)
    assert pixels.shape[-3:] == (64, 96, 3)
    assert torch.equal(video, torch.ones_like(video)) and torch.equal(audio, saved)
    assert latent["samples"].unbind()[0] is video


@pytest.mark.parametrize("bad", [None, torch.zeros(1, 32, 2, 8), torch.zeros(2, 24, 1, 2, 2),
                                  torch.full((1, 24, 1, 2, 2), float("nan"))])
def test_invalid_video_rejected(bad):
    with pytest.raises(ValueError, match="video latent"):
        decode_h3_video_checked(NativeVAE(), bad)


def test_wrong_vae_and_wrong_duration_rejected():
    vae = NativeVAE()
    vae.latent_channels = 16
    with pytest.raises(ValueError, match="latent_channels"):
        decode_h3_video_checked(vae, torch.zeros(1, 24, 1, 4, 4))
    vae.latent_channels = 24
    vae.decode = lambda z: torch.zeros(1, 64, 64, 3)
    with pytest.raises(ValueError, match="shape"):
        decode_h3_video_checked(vae, torch.zeros(1, 24, 2, 4, 4))


def fake_runner(tile=16, dtype="DataType.HALF", output=(1, 3, 28, 256, 256)):
    shape = (1, 24, 7, tile, tile)
    return SimpleNamespace(load_to_gpu=lambda: None,
                           engine=SimpleNamespace(get_tensor_profile_shape=lambda name, index: (shape,) * 3,
                                                  get_tensor_dtype=lambda name: dtype),
                           context=SimpleNamespace(set_input_shape=lambda name, shape: True,
                                                   get_tensor_shape=lambda name: output))


def test_trt_profile_binding_guards_without_loading_trt():
    vae = ComfyTRTVAE()
    vae.first_stage_model = MiniMaxH3TRTVAE()
    z = torch.zeros(1, 24, 7, 16, 24)
    vae.first_stage_model.decoder_runner = fake_runner()
    check_trt_decoder_profile(vae, z)
    for runner, pattern in [(fake_runner(tile=32), "profile"), (fake_runner(dtype="DataType.FLOAT"), "fp16"),
                            (fake_runner(output=(1, 3, 17, 256, 256)), "output binding")]:
        vae.first_stage_model.decoder_runner = runner
        with pytest.raises(ValueError, match=pattern):
            check_trt_decoder_profile(vae, z)
    vae.first_stage_model.decoder_runner = fake_runner()
    with pytest.raises(ValueError, match="canvas"):
        check_trt_decoder_profile(vae, z[..., :8, :])
