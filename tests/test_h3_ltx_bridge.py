import copy

import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.utils import h3_ltx_bridge as bridge
from ComfyUI_JR_MiniMaxH3Node.utils.h3_ltx_frozen.adapter import expected_shapes
from ComfyUI_JR_MiniMaxH3Node.utils.h3_ltx_frozen.geometry import align_h3_to_ltx


def draft(tokens=37):
    return {"samples": torch.randn(1, 24, tokens, 4, 6)}


def adapted(tokens=52):
    meta = bridge.bridge_metadata(draft(tokens)["samples"])
    samples = torch.randn(1, 128, (meta['padded_frames']-1)//8+1, 2, 3)
    return {"samples": samples, bridge.MARKER: meta}


def audio(frames):
    return {"waveform": torch.randn(1, 2, round(frames/24*32000)), "sample_rate": 32000}


@pytest.mark.parametrize('tokens,frames,padded', [(2,5,9), (17,56,57), (37,124,129), (52,175,177), (107,362,369)])
def test_native_timeline_matches_frozen_adapter(tokens, frames, padded):
    latent = draft(tokens)
    meta = bridge.bridge_metadata(latent['samples'])
    assert meta['source_frames'] == frames and meta['padded_frames'] == padded
    shapes = expected_shapes(frames, 64, 96)
    assert list(latent['samples'].shape[1:]) == shapes['h3']
    packed = align_h3_to_ltx(latent['samples'], pixel_frames=frames, target_height=2, target_width=3)
    assert list(packed.shape[1:]) == shapes['aligned_h3']
    assert torch.isfinite(packed).all()


@pytest.mark.parametrize('tokens', [1,3,5,36,51])
def test_reject_ambiguous_or_non_native_timeline(tokens):
    with pytest.raises(ValueError, match='frame grid'):
        bridge.h3_frame_count(tokens)


@pytest.mark.parametrize('key', ['noise_mask', 'batch_index'])
def test_masks_and_batch_mapping_cannot_cross_models(key):
    latent = draft()
    latent[key] = torch.ones(1)
    with pytest.raises(ValueError, match='completed clean draft'):
        bridge.video_tensor(latent, 24)


def test_bad_tensor_and_nonfinite_rejected():
    with pytest.raises(ValueError, match='batch-one'):
        bridge.video_tensor({'samples': torch.zeros(1,128,3,2,2)},24)
    x = draft()
    x['samples'][0,0,0,0,0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        bridge.video_tensor(x,24)


def test_changed_bridge_shape_rejected():
    latent = adapted()
    latent['samples'] = latent['samples'][:,:,:-1]
    with pytest.raises(ValueError, match='metadata'):
        bridge.validate_bridge(latent)


def test_finish_removes_only_padding_and_preserves_pcm():
    latent = adapted()
    original = audio(175)
    wave_before = original['waveform'].clone()
    images = torch.rand(177,64,96,3)
    output, sound, fps, _ = bridge.finish_media(images,original,latent)
    assert output.shape[0] == 175 and fps == 24
    assert torch.equal(output, images[:175])
    assert torch.equal(sound['waveform'], wave_before)
    assert torch.equal(original['waveform'], wave_before)
    assert images.shape[0] == 177


def test_short_decode_and_wrong_audio_are_not_silently_accepted():
    latent = adapted()
    with pytest.raises(ValueError, match='frame count'):
        bridge.finish_media(torch.rand(169,64,96,3),audio(175),latent)
    with pytest.raises(ValueError, match='duration'):
        bridge.validate_audio(audio(124),175)


def test_refine_setup_encodes_audio_but_returns_original(monkeypatch):
    from types import SimpleNamespace

    from comfy_extras.nodes_audio import VAEEncodeAudio
    from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio
    latent = adapted()
    original = audio(175)
    calls = []
    encoded = torch.ones(1,8,292,16)
    target = torch.zeros(1,8,295,16)
    def encode(vae, value):
        calls.append(value)
        return ({'samples': encoded},)
    monkeypatch.setattr(VAEEncodeAudio, 'execute', encode)
    monkeypatch.setattr(LTXVEmptyLatentAudio, 'execute', lambda *a: ({'samples':target},))
    fake = SimpleNamespace(latent_channels=8,first_stage_model=SimpleNamespace(num_of_latents_from_frames=lambda *a:295))
    av, sigmas, preserved, _ = bridge.prepare_refine(latent, original, fake)
    assert preserved is original and calls[0] is original
    video, audio_latent = av['samples'].unbind()
    assert torch.equal(video,latent['samples'])
    assert torch.equal(audio_latent[:,:,:292],encoded)
    assert not audio_latent[:,:,292:].any()
    assert torch.equal(sigmas,torch.tensor(bridge.SIGMAS))
    assert 'noise_mask' not in av  # joint AV updates, final generated audio is discarded


def test_setup_rejects_h3_audio_vae_before_encode():
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='LTX-2.5 audio VAE'):
        bridge.prepare_refine(adapted(),audio(175),SimpleNamespace(latent_channels=32))


def test_bridge_registration():
    from ComfyUI_JR_MiniMaxH3Node import NODE_CLASS_MAPPINGS
    for name in ['JR_H3ToLTXLatentAdapter','JR_H3LTXRefineSetup','JR_H3LTXFinishMedia']:
        assert name in NODE_CLASS_MAPPINGS


def test_metadata_is_not_mutated():
    latent = adapted()
    before = copy.deepcopy(latent[bridge.MARKER])
    _, meta = bridge.validate_bridge(latent)
    meta['source_frames'] = 0
    assert latent[bridge.MARKER] == before


@pytest.mark.parametrize('steps,denoise', [(3,0.25),(6,0.2),(1,0.1),(10,1.0)])
def test_adjustable_schedule_matches_native_tail_and_step_count(steps, denoise):
    from types import SimpleNamespace

    from comfy.model_sampling import ModelSamplingDiscreteFlow
    from comfy.samplers import calculate_sigmas
    sampling = ModelSamplingDiscreteFlow()
    model = SimpleNamespace(get_model_object=lambda key: sampling)
    sigmas = bridge.refine_sigmas(steps,denoise,'simple',model)
    expected = calculate_sigmas(sampling,'simple',int(steps/denoise))[-(steps+1):]
    assert torch.equal(sigmas,expected)
    assert len(sigmas) == steps+1 and sigmas[-1] == 0
    assert bool((sigmas[:-1] > sigmas[1:]).all())
    if denoise <= 0.25:
        assert sigmas[0] < bridge.SIGMAS[0]


def test_zero_denoise_and_missing_model():
    from comfy.samplers import CFGGuider
    sigmas = bridge.refine_sigmas(3,0,'simple',object())
    assert len(sigmas) == 0
    latent = adapted()
    result = CFGGuider.sample(None,None,latent['samples'],None,sigmas)
    assert result is latent['samples']
    with pytest.raises(ValueError,match='Connect ltx_model'):
        bridge.refine_sigmas(3,0.25,'simple')


@pytest.mark.parametrize('steps,denoise', [(0,0.25),(101,0.25),(3,-0.1),(3,1.1),(3,float('nan'))])
def test_invalid_adjustable_schedule_rejected(steps,denoise):
    with pytest.raises(ValueError):
        bridge.refine_sigmas(steps,denoise,'simple',object())


def test_legacy_recipe_is_exact_and_cannot_silently_ignore_controls():
    assert torch.equal(bridge.refine_sigmas(),torch.tensor(bridge.SIGMAS))
    with pytest.raises(ValueError,match='old fixed recipe'):
        bridge.refine_sigmas(3,0.25,'sol_h3_original')


def test_visible_controls_default_to_gentle_native_refinement():
    from ComfyUI_JR_MiniMaxH3Node.nodes.h3_ltx_bridge import JR_H3LTXRefineSetup
    options = JR_H3LTXRefineSetup.INPUT_TYPES()['optional']
    assert options['steps'][1]['default'] == 3
    assert options['denoise'][1]['default'] == 0.25
    assert options['scheduler'][1]['default'] == 'simple'
    assert options['ltx_model'][0] == 'MODEL'
