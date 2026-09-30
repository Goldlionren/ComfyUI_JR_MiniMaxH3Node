"""Opt-in actual released adapter and native LTX audio VAE; no mocked GPU math."""
import os
from pathlib import Path

import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.utils import h3_ltx_bridge as bridge

pytestmark = pytest.mark.skipif(os.environ.get('JR_H3_LTX_GPU') != '1', reason='opt-in released-weight GPU tests')


@pytest.mark.parametrize('tokens,frames', [(37,124),(52,175)])
def test_released_adapter_and_audio_vae(tokens, frames):
    import comfy.sd
    import comfy.utils
    model_file = Path(os.environ['JR_H3_LTX_ADAPTER'])
    torch.manual_seed(935)
    source = {'samples':torch.randn(1,24,tokens,64,64)}
    adapter = bridge.load_adapter(model_file)
    converted, meta = bridge.convert_video(source, adapter)
    assert converted['samples'].shape == (1,128,(meta['padded_frames']-1)//8+1,32,32)
    assert torch.isfinite(converted['samples']).all()
    assert next(adapter.model.parameters()).device.type == 'cpu'
    wave = torch.sin(torch.arange(round(frames/24*32000))*0.02).view(1,1,-1).repeat(1,2,1)*0.01
    original = {'waveform':wave, 'sample_rate':32000}
    state, metadata = comfy.utils.load_torch_file(os.environ['JR_H3_LTX_AUDIO_VAE'],return_metadata=True)
    vae = comfy.sd.VAE(sd=state,metadata=metadata)
    av, sigmas, retained, _ = bridge.prepare_refine(converted, original, vae)
    assert retained is original
    video, audio = av['samples'].unbind()
    assert torch.isfinite(audio).all() and torch.isfinite(video).all()
    assert sigmas.numel() == 4
    assert audio.shape[2] == vae.first_stage_model.num_of_latents_from_frames(meta['padded_frames'],24)
