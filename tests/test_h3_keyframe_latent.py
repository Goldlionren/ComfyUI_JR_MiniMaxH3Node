"""Pre-encoded Director anchors: no VAE round trip and independent canvas copies."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from comfy.nested_tensor import NestedTensor
from comfy_extras import nodes_minimax_h3 as native
from ComfyUI_JR_MiniMaxH3Node.nodes.director_pipe_io import JR_H3_DirectorPipeBuilder, JR_H3_DirectorPipeUnpack
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_directed_video_conditioning import JR_H3_DirectedVideoConditioning
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_progressive_guided_sampler import JR_H3_ProgressiveGuidedSampler
from ComfyUI_JR_MiniMaxH3Node.utils import h3_progressive_sampler as progressive
from ComfyUI_JR_MiniMaxH3Node.utils.director_pipe_adapter import pipe_to_optimizer_context
from ComfyUI_JR_MiniMaxH3Node.utils.h3_keyframe_latent import (
    CROP_KEY,
    MASTER_KEY,
    resize_keyframe_latent,
    validate_keyframe_latent,
)
from ComfyUI_JR_MiniMaxH3Node.utils.h3_progressive_guidance import prepare_guidance
from test_h3_progressive_sampler import inputs, nearest, tiny_patcher


def master(h=90, w=90, dtype=torch.float32):
    return {"samples": torch.linspace(-1, 1, 24 * h * w).reshape(1, 24, 1, h, w).to(dtype), "label": "master"}


class Clip:
    def __init__(self):
        self.presentation = None

    def tokenize(self, text, **kwargs):
        self.presentation = kwargs
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 2, 128), {"test_text": tokens}]]


class VAE:
    def __init__(self, allow_encode=False):
        self.allow_encode = allow_encode
        self.encode_calls, self.decode_calls = [], []

    def encode(self, image):
        assert self.allow_encode, "Unexpected VAE encode for pre-encoded keyframe"
        self.encode_calls.append(image.clone())
        z = F.interpolate(image.movedim(-1, 1), scale_factor=1 / 16, mode="area")
        if image.shape[0] == 1:
            return z.unsqueeze(2).repeat(1, 8, 1, 1, 1)
        return torch.zeros(1, 24, native.temporal_shape(image.shape[0])[1], z.shape[-2], z.shape[-1])

    def decode(self, z):
        self.decode_calls.append(z.clone())
        return F.interpolate(z[:, :3, 0], scale_factor=16, mode="nearest").movedim(1, -1).clamp(0, 1)


def build(**kwargs):
    return JR_H3_DirectorPipeBuilder.build("A person speaks.", duration_seconds=5 / 24, **kwargs)[0]


def condition(pipe, *, width=192, height=128, vae=None, audio_vae=None):
    vae, clip = vae or VAE(), Clip()
    result = JR_H3_DirectedVideoConditioning().condition(
        clip, vae, pipe, dimension_source="Prefer Node", width=width, height=height, length=5,
        audio_vae=audio_vae,
    )
    return result, vae, clip


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("crop", ["disabled", "center"])
def test_canvas_copies_preserve_master_storage_dtype_and_repeatability(dtype, crop):
    z = master(dtype=dtype)["samples"]
    saved = z.clone()
    for width, height in [(640, 640), (896, 896), (768, 512), (1440, 1440)]:
        out = resize_keyframe_latent(z, width, height, crop)
        assert out.shape == (1, 24, 1, height // 16, width // 16)
        assert out.dtype == dtype and out.device == z.device
        assert out.data_ptr() != z.data_ptr()
        assert torch.equal(out, resize_keyframe_latent(z, width, height, crop))
        out.zero_()
        assert torch.equal(z, saved)


def test_center_crop_uses_latent_geometry_not_stretch():
    z = torch.arange(8.).reshape(1, 1, 1, 1, 8).expand(1, 24, 1, 4, 8)
    out = resize_keyframe_latent(z, 64, 64, "center")
    assert torch.equal(out, z[..., 2:6])


@pytest.mark.parametrize("value", [None, {}, {"samples": torch.zeros(1, 16, 1, 4, 4)},
    {"samples": torch.zeros(1, 24, 2, 4, 4)}, {"samples": torch.zeros(2, 24, 1, 4, 4)},
    {"samples": torch.zeros(1, 24, 1, 4, 4, dtype=torch.int64)},
    {"samples": torch.full((1, 24, 1, 4, 4), float("nan"))},
    {"samples": torch.zeros(1, 24, 1, 4, 4), "noise_mask": torch.ones(1)},
    {"samples": NestedTensor((torch.zeros(1, 24, 1, 4, 4), torch.zeros(1, 32, 2, 2)))}])
def test_invalid_anchor_contract_is_rejected(value):
    with pytest.raises(ValueError):
        validate_keyframe_latent(value)


def test_pipe_roundtrip_does_not_serialize_latents_and_builder_snapshots_input():
    source = master()
    image = torch.ones(1, 32, 32, 3)
    pipe = build(first_frame=image, first_latent=source, last_latent=master())
    snapshot = pipe.media_for_item("runtime-first-frame").keyframe_latent["samples"].clone()
    source["samples"].zero_()
    assert torch.equal(pipe.media_for_item("runtime-first-frame").keyframe_latent["samples"], snapshot)
    assert "samples" not in json.dumps(pipe.to_persisted())
    assert "keyframe_latent" not in json.dumps(pipe.to_persisted())
    derived = pipe.derive(reviewed_prompt="Reviewed")
    out = JR_H3_DirectorPipeUnpack().unpack(derived)
    assert out[9] is image and out[10] is None
    assert out[17]["label"] == "master" and out[18]["samples"].shape == (1, 24, 1, 90, 90)
    assert derived.runtime_media is pipe.runtime_media


@pytest.mark.parametrize("which", ["first", "last", "both"])
def test_real_native_allocation_frame_positions_and_no_encode_decode_with_paired_images(which):
    image = torch.ones(1, 48, 64, 3)
    kwargs = {}
    for role in (["first", "last"] if which == "both" else [which]):
        kwargs.update({role + "_frame": image, role + "_latent": master()})
    pipe = build(**kwargs)
    (positive, av), vae, clip = condition(pipe)
    blocks = positive[0][1]["minimax_keyframes"]
    assert [k["resolved_frame_index"] for k in blocks] == ({"first": [0], "last": [4], "both": [0, 4]}[which])
    assert isinstance(av["samples"], NestedTensor)
    assert av["samples"].unbind()[0].shape == (1, 24, 2, 8, 12)
    for k in blocks:
        assert k["latent"].shape == (1, 24, 1, 8, 12)
        assert k[MASTER_KEY].shape == (1, 24, 1, 90, 90)
    assert not vae.encode_calls and not vae.decode_calls
    assert len(clip.presentation["images"]) == len(blocks)


def test_latent_only_decodes_for_vision_without_reencoding_and_optimizer_explains_missing_image():
    pipe = build(first_latent=master())
    (_, _), vae, _ = condition(pipe)
    assert len(vae.decode_calls) == 1 and not vae.encode_calls
    with pytest.raises(ValueError, match="matching IMAGE"):
        pipe_to_optimizer_context(pipe, 64)
    paired = build(first_frame=torch.ones(1, 32, 32, 3), first_latent=master())
    context = pipe_to_optimizer_context(paired, 64)
    assert context.has_first_frame and len(context.encoded_images) == 1


def test_mixed_image_and_latent_only_encodes_image_side():
    (positive, _), vae, _ = condition(build(first_latent=master(), last_frame=torch.zeros(1, 32, 32, 3)), vae=VAE(True))
    assert len(vae.encode_calls) == 1 and len(vae.decode_calls) == 1
    assert MASTER_KEY in positive[0][1]["minimax_keyframes"][0]
    assert MASTER_KEY not in positive[0][1]["minimax_keyframes"][1]


def test_reference_and_audio_combination_preserves_real_anchor_semantics():
    class AudioVAE:
        def encode(self, waveform):
            return torch.zeros(1, 32, 2, 8)
    pipe = build(first_frame=torch.zeros(1, 32, 32, 3), first_latent=master(),
                 reference_images=torch.ones(1, 32, 32, 3),
                 driving_audio={"waveform": torch.zeros(1, 2, 1000), "sample_rate": 32000})
    with pytest.raises(ValueError, match="audio_vae"):
        condition(pipe)
    (positive, _), vae, clip = condition(pipe, vae=VAE(True), audio_vae=AudioVAE())
    assert len(vae.encode_calls) == 1 and not vae.decode_calls
    metadata = positive[0][1]
    assert metadata["minimax_keyframes"][0]["resolved_frame_index"] == 0
    assert [r["kind"] for r in metadata["minimax_refs"]] == ["image", "audio"]
    assert [r["type"] for r in clip.presentation["minimax_ref_items"]] == ["image", "image", "audio"]


def test_progressive_uses_master_not_previous_canvas_and_never_calls_vae():
    (positive, latent), vae, _ = condition(build(first_frame=torch.zeros(1, 32, 32, 3), first_latent=master()))
    video, audio = latent["samples"].unbind()
    plan = SimpleNamespace(identity=False, low_w=6, low_h=4)
    original = positive[0][1]["minimax_keyframes"][0]
    saved = original["latent"].clone()
    result = prepare_guidance(positive, latent, video, audio, plan, vae=None)
    actual = result.low_positive[0][1]["minimax_keyframes"][0]["latent"]
    expected = resize_keyframe_latent(original[MASTER_KEY], 96, 64, original[CROP_KEY])
    assert torch.equal(actual, expected) and torch.equal(original["latent"], saved)
    assert result.low_positive is not positive and not vae.encode_calls
    # Corrupt only the stage copy: low-stage conversion must still use the master.
    original["latent"].zero_()
    repeat = prepare_guidance(positive, latent, video, audio, plan, vae=None)
    assert torch.equal(repeat.low_positive[0][1]["minimax_keyframes"][0]["latent"], expected)


@pytest.mark.parametrize("with_refs", [False, True])
@pytest.mark.parametrize("with_tst", [False, True])
def test_preencoded_conditioning_runs_native_h3_progressive_and_repeats(monkeypatch, with_refs, with_tst):
    pipe = build(first_frame=torch.zeros(1, 32, 32, 3), first_latent=master(12, 16),
                 last_frame=torch.ones(1, 32, 32, 3), last_latent=master(12, 16),
                 reference_images=torch.zeros(1, 32, 32, 3) if with_refs else None)
    (positive, latent), _, _ = condition(pipe, vae=VAE(with_refs))
    patcher = tiny_patcher(monkeypatch)
    if with_tst:
        from ComfyUI_JR_MiniMaxH3Node.utils.h3_temporal_transport import apply_temporal_transport
        patcher = apply_temporal_transport(patcher, strength=.2)
    from ComfyUI_JR_MiniMaxH3Node.utils.h3_audio_driven_latent_builder import build_h3_audio_driven_latent
    source_audio = torch.linspace(-.5, .5, 512).reshape(1, 32, 2, 8)
    latent, _ = build_h3_audio_driven_latent(latent, {"samples": source_audio})
    # Keep native sampling/model/packing real; replace only the expensive learned spatial lift.
    monkeypatch.setattr(progressive, "upscale_h3_video_to_size", nearest)
    args = inputs(model=patcher, positive=positive, latent_image=latent, lowres_scale=.5)
    output, _ = JR_H3_ProgressiveGuidedSampler().sample(**args, vae=None)
    repeated, _ = JR_H3_ProgressiveGuidedSampler().sample(**args, vae=None)
    for a, b in zip(output["samples"].unbind(), repeated["samples"].unbind()):
        assert torch.isfinite(a).all()
        assert torch.equal(a, b)
    assert torch.equal(output["samples"].unbind()[1], source_audio)


def test_reference_video_path_preserves_native_frame_trimming_and_encoder_calls():
    from test_director_pipe_io import FakeVideo
    video = FakeVideo(frames=torch.ones(22, 32, 64, 3), width=64, height=32, duration=22 / 24)
    pipe = build(first_latent=master(), reference_video=video)
    (positive, _), vae, clip = condition(pipe, vae=VAE(True))
    refs = positive[0][1]["minimax_refs"]
    assert len(vae.encode_calls) == 1 and vae.encode_calls[0].shape[0] == 5
    assert refs[0]["kind"] == "video" and refs[0]["latent_t"] == 2
    assert [r["type"] for r in clip.presentation["minimax_ref_items"]] == ["image", "video"]


def test_mixed_progressive_keeps_legacy_vae_path_only_for_unmarked_keyframe():
    from test_h3_progressive_guided_sampler import GeometryVAE
    (positive, latent), _, _ = condition(build(first_latent=master(), last_frame=torch.ones(1, 32, 32, 3)), vae=VAE(True))
    vae = GeometryVAE()
    result = prepare_guidance(positive, latent, *latent["samples"].unbind(),
                              SimpleNamespace(identity=False, low_w=6, low_h=4), vae=vae)
    assert len(vae.calls) == 1
    assert all(k["latent"].shape[-2:] == (4, 6) for k in result.low_positive[0][1]["minimax_keyframes"])


def test_progressive_rejects_malformed_master_policy():
    (positive, latent), _, _ = condition(build(first_latent=master()))
    block = positive[0][1]["minimax_keyframes"][0]
    block[CROP_KEY] = "bad"
    with pytest.raises(ValueError, match="crop mode"):
        prepare_guidance(positive, latent, *latent["samples"].unbind(), SimpleNamespace(identity=False))


def test_preencoded_payload_cannot_be_attached_to_reference_item():
    from ComfyUI_JR_MiniMaxH3Node.utils.director_pipe import validate_director_pipe
    pipe = build(reference_images=torch.ones(1, 32, 32, 3))
    altered = replace(pipe, runtime_media=(replace(pipe.runtime_media[0], keyframe_latent=master()),))
    with pytest.raises(ValueError, match="first/last"):
        validate_director_pipe(altered)
