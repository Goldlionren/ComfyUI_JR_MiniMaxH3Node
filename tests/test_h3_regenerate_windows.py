"""Window coverage, bounded decoding and real native H3 prefix regressions."""
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from comfy_api.input_impl import VideoFromComponents, VideoFromFile
from comfy_api.util import VideoComponents
from ComfyUI_JR_MiniMaxH3Node.utils import h3_regenerate as regen
from ComfyUI_JR_MiniMaxH3Node.utils.h3_regenerate_lora import patch_regenerate_bundle
from ComfyUI_JR_MiniMaxH3Node.utils.h3_regenerate_windows import (
    VideoWindowReader,
    plan_windows,
    read_original_audio,
)
from test_h3_progressive_sampler import tiny_patcher
from test_h3_regenerate import inputs, small_bundle


@pytest.mark.parametrize("mode,limit", [("off", 302), ("5s", 37), ("8s", 57), ("10s", 72)])
@pytest.mark.parametrize("total", [2, 32, 37, 42, 57, 72, 107, 302])
def test_windows_cover_once_with_aligned_overlap(mode, limit, total):
    windows = plan_windows(total, mode)
    cursor = 0
    for w in windows:
        assert w.start % 5 == 0
        assert (w.end - w.start - 2) % 5 == 0
        assert w.end - w.start <= limit
        assert w.start + w.prefix == cursor
        cursor = w.end
    assert cursor == total


def test_noise_is_independent_of_window_partition_and_preserves_global_rng():
    state = torch.random.get_rng_state().clone()
    def noise(start, length):
        x = {"samples": NestedTensor((torch.zeros(1, 24, length, 4, 4), torch.zeros(1, 32, 2, 8)))}
        return regen.VideoOnlyNoise(2**64 - 1, start).generate_noise(x).unbind()[0]
    whole = noise(0, 72)
    assert torch.equal(whole[:, :, 25:62], noise(25, 37))
    assert torch.equal(whole[:, :, 45:], noise(45, 27))
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("fps", [24, 30, Fraction(30000, 1001)])
def test_file_reader_overlap_matches_global_frames_and_audio(tmp_path, fps):
    import av
    import numpy as np
    path = tmp_path / "source.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for n in range(round(9 * fps)):
            pixels = np.full((32, 32, 3), n % 240, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    video = VideoFromFile(str(path)).as_trimmed(1., 7.)
    reader = VideoWindowReader(video)
    components = video.get_components()
    expected = components.images
    windows = plan_windows(reader.video_t, "5s")
    for window in windows:
        images = reader.read(window)
        start = window.start // 5 * 17
        indices = (torch.arange(start, start + window.frames).clamp(max=reader.count - 1).double()
                   * float(fps) / 24).floor().long().clamp(max=len(expected) - 1)
        torch.testing.assert_close(images, expected[indices], rtol=0, atol=0)
    assert reader.audio is None


def test_components_audio_identity_and_bounded_encode_sample(monkeypatch):
    from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
    model = tiny_patcher(monkeypatch)
    audio = {"waveform": torch.randn(1, 2, 15 * 32000), "sample_rate": 32000}
    video = VideoFromComponents(VideoComponents(
        images=torch.rand(360, 32, 32, 3), audio=audio, frame_rate=Fraction(24)))
    encoded, sampled_lengths, prefixes = [], [], []
    def encode(images):
        encoded.append(len(images))
        return torch.zeros(1, 24, (len(images) - 5) // 17 * 5 + 2, 2, 2)
    def sample(noise, guider, sampler, sigmas, latent):
        visual, sound = latent["samples"].unbind()
        mask = latent["noise_mask"].unbind()[0]
        assert not sound.count_nonzero()
        assert noise.seed == 92
        sampled_lengths.append(visual.shape[2])
        if len(sampled_lengths) > 1:
            assert torch.equal(visual[:, :, :12], torch.ones_like(visual[:, :, :12]))
            assert not mask[:, :, :12].count_nonzero()
            prefixes.append(True)
        return ({"samples": NestedTensor((visual + 1, sound))},)
    monkeypatch.setattr(SamplerCustomAdvanced, "execute", sample)
    args = inputs(model)
    args.pop("latent")
    result, _, bypass = regen.regenerate(**(args | {
        "source_mode": "video", "video": video, "vae": SimpleNamespace(encode=encode),
        "audio_vae": SimpleNamespace(encode=lambda *_: pytest.fail("Audio VAE must never be called")),
        "chunk_mode": "5s"}))
    assert bypass is audio
    assert len(encoded) == len(sampled_lengths) > 1
    assert max(encoded) == 124 and max(sampled_lengths) == 37 and prefixes
    assert result["samples"].shape[2] == 107
    assert torch.equal(result["samples"], torch.ones_like(result["samples"]))
    assert result["jr_h3_regenerate_video"]["source_frames_24fps"] == 360


def test_real_native_multiwindow_prefix_audio_bypass_and_repeatability(monkeypatch):
    from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
    model = tiny_patcher(monkeypatch)
    model = patch_regenerate_bundle(model, small_bundle(model))
    args = inputs(model)
    video = torch.randn(1, 24, 67, 2, 2)
    audio = torch.randn(1, 32, 2, 377)
    args["latent"] = {"samples": NestedTensor((video, audio))}
    before = audio.clone()
    real_execute = SamplerCustomAdvanced.execute
    calls = []
    def checked(noise, guider, sampler, sigmas, latent):
        visual, sound = latent["samples"].unbind()
        assert not sound.count_nonzero() and sound.shape[-1] <= 207
        result = real_execute(noise, guider, sampler, sigmas, latent)
        refined = result[0]["samples"].unbind()[0]
        if noise.video_start:
            # Native masking plus exact previous-window prefix should survive all steps.
            torch.testing.assert_close(refined[:, :, :12], visual[:, :, :12], rtol=1e-4, atol=1e-4)
        calls.append(noise.video_start)
        return result
    monkeypatch.setattr(SamplerCustomAdvanced, "execute", checked)
    first, _, bypass = regen.regenerate(**args, chunk_mode="5s")
    second, _, _ = regen.regenerate(**args, chunk_mode="5s")
    assert calls == [0, 25, 50, 0, 25, 50]
    assert bypass is None and first["samples"].unbind()[1] is audio
    assert torch.equal(before, audio)
    assert torch.equal(first["samples"].unbind()[0], second["samples"].unbind()[0])
    assert first["samples"].unbind()[0].device.type == "cpu"
    model.unpatch_model()


def test_file_audio_bypass_preserves_pcm_samples_rate_and_channels(tmp_path):
    import av
    import numpy as np
    path = tmp_path / "original.wav"
    rate = 44100
    original = np.random.default_rng(12).uniform(-.9, .9, (2, rate * 3)).astype(np.float32)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("pcm_f32le", rate=rate)
        stream.layout = "stereo"
        for start in range(0, original.shape[1], 1024):
            frame = av.AudioFrame.from_ndarray(original[:, start:start + 1024].copy(),
                                               format="fltp", layout="stereo")
            frame.sample_rate = rate
            frame.pts = start
            frame.time_base = Fraction(1, rate)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    full = read_original_audio(VideoFromFile(str(path)))
    assert full["sample_rate"] == rate
    assert torch.equal(full["waveform"][0], torch.from_numpy(original))
    trimmed = read_original_audio(VideoFromFile(str(path), start_time=.25, duration=1.5))
    assert torch.equal(trimmed["waveform"][0], torch.from_numpy(original[:, rate // 4:rate * 7 // 4]))
