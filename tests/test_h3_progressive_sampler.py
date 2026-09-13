"""Numerical, contract and native-runtime coverage without downloading H3 weights."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from comfy.latent_formats import MiniMaxH3AV
from comfy.model_base import MiniMaxH3, ModelType, model_sampling
from comfy.nested_tensor import NestedTensor
from comfy.samplers import ksampler
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_progressive_sampler import JR_H3_ProgressiveSampler
from ComfyUI_JR_MiniMaxH3Node.utils import h3_progressive_sampler as progressive
from ComfyUI_JR_MiniMaxH3Node.utils.h3_neural_latent_upscaler import upscale_h3_video_to_size


def latent(dtype=torch.float32):
    return {"samples": NestedTensor((torch.zeros(1, 24, 2, 8, 12, dtype=dtype),
                                     torch.zeros(1, 32, 2, 8, dtype=dtype))),
            "batch_index": [0], "metadata": {"marker": "preserved"}}


def sampling(shift=12, audio_shift=3):
    value = model_sampling(SimpleNamespace(sampling_settings={"shift": shift, "audio_shift": audio_shift}), ModelType.FLOW_AV)
    value.set_parameters(shift=shift, audio_shift=audio_shift)
    return value


def fake_patcher():
    base = MiniMaxH3.__new__(MiniMaxH3)
    torch.nn.Module.__init__(base)
    # Deliberately different from the effective patch: catch accidental use of base.model_sampling.
    base.model_sampling = sampling(12, 3)
    objects = {"model_sampling": sampling(6, 3), "latent_format": MiniMaxH3AV()}
    return SimpleNamespace(model=base, get_model_object=objects.__getitem__, get_attachment=lambda key: None)


def inputs(**changes):
    values = dict(model=fake_patcher(), positive=[[torch.zeros(1, 2, 128), {}]],
                  noise=Noise_RandomNoise(123), sampler=ksampler("euler"),
                  sigmas=torch.tensor([1., .8, .5, .2, 0.]), latent_image=latent(),
                  transition_step=2, lowres_scale=.5, transition_seed_offset=1)
    values.update(changes)
    return values


def nearest(clean, height, width):
    return F.interpolate(clean, size=(clean.shape[2], height, width), mode="nearest")


@pytest.mark.parametrize("sigmas", [[1., .5, .6, 0.], [1., .5, .5, 0.], [1., 0.], [1., float('nan'), 0.],
                                   [1., float('inf'), 0.], [1., -.1, 0.], [1.1, .5, 0.], [.8, .4, 0.], [1., .5, .1]])
def test_invalid_schedules(sigmas):
    with pytest.raises(ValueError, match="JR H3 Progressive"):
        progressive.plan_progressive(torch.tensor(sigmas), 1, .5, 8, 12)


@pytest.mark.parametrize("step", [0, 4, -1, True, 1.5])
def test_invalid_transition(step):
    with pytest.raises(ValueError, match="transition_step"):
        progressive.plan_progressive(inputs()["sigmas"], step, .5, 8, 12)


@pytest.mark.parametrize("scale", [.24, 1.1, float('nan')])
def test_invalid_scale(scale):
    with pytest.raises(ValueError, match="lowres_scale"):
        progressive.plan_progressive(inputs()["sigmas"], 2, scale, 8, 12)


def test_alignment_and_identity():
    plan = progressive.plan_progressive(inputs()["sigmas"], 2, .25, 10, 18)
    assert (plan.low_h, plan.low_w) == (4, 6)
    assert not plan.identity
    assert progressive.plan_progressive(inputs()["sigmas"], 2, 1., 10, 18).identity
    with pytest.raises(ValueError, match="patch grid"):
        progressive.plan_progressive(inputs()["sigmas"], 2, .5, 9, 18)


@pytest.mark.parametrize("name", ["heun", "euler_ancestral", "res_multistep", "dpmpp_2m"])
def test_reject_stateful_or_other_samplers(name):
    with pytest.raises(ValueError, match="standard Euler"):
        progressive.sample_h3_progressive(**inputs(sampler=ksampler(name)))


def test_reject_churn_and_generic_noise():
    with pytest.raises(ValueError, match="s_churn"):
        progressive.sample_h3_progressive(**inputs(sampler=ksampler("euler", {"s_churn": .1})))
    with pytest.raises(ValueError, match="official RandomNoise"):
        progressive.sample_h3_progressive(**inputs(noise=SimpleNamespace(seed=1)))


@pytest.mark.parametrize("key", ["minimax_keyframes", "mask", "area", "control", "hooks"])
def test_conditioning_rejected_before_sampling(key):
    with pytest.raises(ValueError, match=key):
        progressive.sample_h3_progressive(**inputs(positive=[[torch.zeros(1, 2, 128), {key: []}]]))


def test_reject_mask_and_nonempty_av():
    value = latent()
    value["noise_mask"] = NestedTensor([torch.ones_like(t) for t in value["samples"].unbind()])
    with pytest.raises(ValueError, match="noise_mask"):
        progressive.sample_h3_progressive(**inputs(latent_image=value))
    for stream in (0, 1):
        value = latent()
        value["samples"].unbind()[stream].fill_(1)
        with pytest.raises(ValueError, match="empty target"):
            progressive.sample_h3_progressive(**inputs(latent_image=value))


@pytest.mark.parametrize("audio_scale", [1., 2., 4.])
def test_transition_numerical_oracle_and_x0_domain(audio_scale):
    ms = sampling(audio_scale * 3, 3)
    fmt = MiniMaxH3AV()
    fmt.scale_factor = 2.0  # Nonidentity domain catches missing process_in/out.
    plan = progressive.plan_progressive(inputs()["sigmas"], 2, .5, 8, 12)
    video = torch.full((1, 24, 2, 4, 6), 6.)
    ax = torch.full((1, 32, 2, 8), 7.)
    a0 = torch.full_like(ax, 3.)
    x0 = NestedTensor((video, a0))
    x = NestedTensor((video * 10, ax))
    noise = torch.full((1, 24, 2, 8, 12), 2.)
    seen = []

    def lift(clean, h, w):
        seen.append(clean.clone())
        return nearest(clean, h, w)

    resume = progressive.transition_state(x, x0, ms, fmt, plan, noise, torch.tensor(.8), torch.tensor(.5), lifter=lift)
    assert torch.equal(seen[0], video / 2)
    rv, ra = resume.unbind()
    # Public->internal conversion + zero-noise resume must recover exactly sigma=.5 state.
    internal_v = ms.noise_scaling(torch.tensor(.5), torch.zeros_like(rv), fmt.process_in(rv))
    internal_a = ms.noise_scaling(torch.tensor(.5), torch.zeros_like(ra), fmt.process_in(ra) * audio_scale)
    torch.testing.assert_close(internal_v, .5 * nearest(video, 8, 12) + .5 * noise)
    torch.testing.assert_close(internal_a, ax + (ax - a0) / .8 * (.5 - .8))
    assert torch.equal(x.unbind()[1], ax)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_exact_neural_size_reuses_backend_and_preserves_dtype(dtype):
    source = torch.ones((1, 24, 2, 4, 6), dtype=dtype)
    seen = []

    def runner(value, plan):
        seen.append(plan)
        return nearest(value.float(), plan.output_h, plan.output_w)

    result = upscale_h3_video_to_size(source, 10, 14, neural_runner=runner)
    assert result.shape == (1, 24, 2, 10, 14)
    assert result.dtype == dtype and result.device == source.device
    assert seen[0].effective_scale == pytest.approx((140 / 24)**.5)
    assert upscale_h3_video_to_size(source, 4, 6, neural_runner=runner) is source
    with pytest.raises(ValueError, match="patch grid"):
        upscale_h3_video_to_size(source, 9, 14, neural_runner=runner)
    with pytest.raises(ValueError, match="1x to 4x"):
        upscale_h3_video_to_size(source, 18, 24, neural_runner=runner)


def fake_native_stage(records):
    """Uses ComfyUI's actual Euler/CONST math while replacing only denoiser/model loading."""
    from comfy.k_diffusion.sampling import sample_euler
    from comfy.utils import pack_latents, unpack_latents

    def stage(model, positive, noise, latent_value, sampler, sigmas, seed, callback):
        ms = model.get_model_object("model_sampling")
        fmt = model.get_model_object("latent_format")
        tensors = latent_value.unbind()
        internal = NestedTensor((fmt.process_in(tensors[0]), fmt.process_in(tensors[1]) * ms.audio_scale))
        flat, shapes = pack_latents(internal.unbind())
        packed_noise, _ = pack_latents(noise.unbind())
        x = ms.noise_scaling(sigmas[0], packed_noise.float(), flat.float())
        records.append((sigmas.clone(), _cpu(noise), shapes))

        def wrapped(event):
            callback(event['i'], NestedTensor(unpack_latents(event['denoised'], shapes)),
                     NestedTensor(unpack_latents(event['x'], shapes)), len(sigmas) - 1)

        # A state-dependent denoiser makes repeatability and resume errors observable.
        out = sample_euler(lambda state, sigma, **kwargs: state * .2 + .1,
                           x, sigmas, callback=wrapped, disable=True)
        out = ms.inverse_noise_scaling(sigmas[-1], out)
        v, a = unpack_latents(out, shapes)
        return fmt.process_out(NestedTensor((v, a / ms.audio_scale)))
    return stage


def _cpu(nested):
    return NestedTensor([t.clone().cpu() for t in nested.unbind()])


def test_same_schedule_zero_resume_noise_reproducible_metadata_and_no_extra_steps(monkeypatch):
    records = []
    monkeypatch.setattr(progressive, "_run_stage", fake_native_stage(records))
    monkeypatch.setattr(progressive, "upscale_h3_video_to_size", nearest)
    resets = []
    monkeypatch.setattr(progressive, "_reset_cache", lambda model: resets.append("reset"))
    args = inputs()
    before = _cpu(args["latent_image"]["samples"])
    output, status = progressive.sample_h3_progressive(**args)
    repeated, _ = progressive.sample_h3_progressive(**args)
    assert torch.equal(records[0][0], args["sigmas"][:3])
    assert torch.equal(records[1][0], args["sigmas"][2:])
    assert all(torch.count_nonzero(t) == 0 for t in records[1][1].unbind())
    assert output["metadata"] is args["latent_image"]["metadata"]
    assert output["batch_index"] == [0]
    assert "2 low + 2 high = 4" in status and "audio scale: 2" in status
    for original, current, result, repeat in zip(before.unbind(), args["latent_image"]["samples"].unbind(),
                                                 output["samples"].unbind(), repeated["samples"].unbind()):
        assert torch.equal(original, current)
        assert torch.equal(result, repeat)
    assert len(resets) == 6


def test_scale_one_is_exact_native_baseline_and_does_not_load_upscaler(monkeypatch):
    records = []
    stage = fake_native_stage(records)
    monkeypatch.setattr(progressive, "_run_stage", stage)
    monkeypatch.setattr(progressive, "upscale_h3_video_to_size", lambda *args: pytest.fail("identity loaded upscaler"))
    args = inputs(lowres_scale=1.)
    expected = stage(args["model"], args["positive"], args["noise"].generate_noise(args["latent_image"]),
                     args["latent_image"]["samples"], args["sampler"], args["sigmas"], args["noise"].seed, lambda *a: None)
    output, status = progressive.sample_h3_progressive(**args)
    for result, native in zip(output["samples"].unbind(), expected.unbind()):
        assert torch.equal(result, native)
    assert "native identity baseline" in status


def test_disabled_noise_keeps_both_stages_and_transition_deterministic(monkeypatch):
    records = []
    monkeypatch.setattr(progressive, "_run_stage", fake_native_stage(records))
    monkeypatch.setattr(progressive, "upscale_h3_video_to_size", nearest)
    result, _ = progressive.sample_h3_progressive(**inputs(noise=Noise_EmptyNoise()))
    repeat, _ = progressive.sample_h3_progressive(**inputs(noise=Noise_EmptyNoise(), transition_seed_offset=99))
    assert all(torch.count_nonzero(t) == 0 for record in records for t in record[1].unbind())
    for first, second in zip(result["samples"].unbind(), repeat["samples"].unbind()):
        assert torch.equal(first, second)


@pytest.fixture
def live_noise_nodes(monkeypatch):
    """Load the same built-in file under ComfyUI's real path-based module identity."""
    import importlib.util
    import sys
    from pathlib import Path

    import comfy_extras.nodes_custom_sampler as canonical

    import nodes as comfy_nodes

    path = str(Path(canonical.__file__).resolve())
    alias = str(Path(path).with_suffix(""))
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias, module)
    spec.loader.exec_module(module)
    mappings = dict(getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}))
    mappings.update(RandomNoise=module.RandomNoise, DisableNoise=module.DisableNoise)
    monkeypatch.setattr(comfy_nodes, "NODE_CLASS_MAPPINGS", mappings, raising=False)
    return comfy_nodes.NODE_CLASS_MAPPINGS


@pytest.mark.parametrize("node_id", ["RandomNoise", "DisableNoise"])
@pytest.mark.parametrize("scale", [.5, 1.])
def test_real_node_loader_noise_identity_and_transition_policy(monkeypatch, live_noise_nodes, node_id, scale):
    output = live_noise_nodes[node_id].execute(123) if node_id == "RandomNoise" else live_noise_nodes[node_id].execute()
    provider = output.result[0]
    assert type(provider) not in (Noise_RandomNoise, Noise_EmptyNoise)
    records = []
    monkeypatch.setattr(progressive, "_run_stage", fake_native_stage(records))
    monkeypatch.setattr(progressive, "upscale_h3_video_to_size", nearest)
    observed = []
    original_transition = progressive.transition_state

    def transition(*args, **kwargs):
        observed.append(args[5].clone())
        return original_transition(*args, **kwargs)

    monkeypatch.setattr(progressive, "transition_state", transition)
    result, _ = progressive.sample_h3_progressive(**inputs(noise=provider, lowres_scale=scale))
    repeated, _ = progressive.sample_h3_progressive(**inputs(noise=provider, lowres_scale=scale))
    for first, second in zip(result["samples"].unbind(), repeated["samples"].unbind()):
        assert torch.equal(first, second)
    if scale < 1:
        template = {"samples": torch.zeros(1, 24, 2, 8, 12), "batch_index": [0]}
        expected_provider = (live_noise_nodes[node_id].execute(124) if node_id == "RandomNoise"
                             else live_noise_nodes[node_id].execute()).result[0]
        assert torch.equal(observed[0], expected_provider.generate_noise(template))
        assert bool(torch.count_nonzero(observed[0])) == (node_id == "RandomNoise")
        assert all(torch.count_nonzero(t) == 0 for t in records[1][1].unbind())
    else:
        assert not observed


def test_unregistered_noise_impostor_still_rejected(live_noise_nodes):
    # Matching class name/attributes does not establish a registered noise contract.
    impostor = type("Noise_RandomNoise", (), {"seed": 123, "generate_noise": lambda self, value: value})()
    with pytest.raises(ValueError, match="custom NOISE"):
        progressive.sample_h3_progressive(**inputs(noise=impostor))


def test_missing_callbacks_and_invalid_lift_fail_closed(monkeypatch):
    monkeypatch.setattr(progressive, "_run_stage", lambda *args: args[3])
    with pytest.raises(RuntimeError, match="expected Euler"):
        progressive.sample_h3_progressive(**inputs())
    monkeypatch.setattr(progressive, "_run_stage", fake_native_stage([]))
    monkeypatch.setattr(progressive, "upscale_h3_video_to_size", lambda value, h, w: value)
    with pytest.raises(RuntimeError, match="invalid video x0"):
        progressive.sample_h3_progressive(**inputs())


def test_node_registration_and_schema():
    import ComfyUI_JR_MiniMaxH3Node as package
    assert package.NODE_CLASS_MAPPINGS["JR_H3_ProgressiveSampler"] is JR_H3_ProgressiveSampler
    assert JR_H3_ProgressiveSampler.RETURN_TYPES == ("LATENT", "STRING")
    assert JR_H3_ProgressiveSampler.INPUT_TYPES()["required"]["lowres_scale"][1]["default"] == .5


def tiny_patcher(monkeypatch):
    """Real miniature H3 model; verifies runtime contracts, not generation quality."""
    from comfy.ldm.minimax import model as h3_model
    from comfy.ldm.modules.attention import attention_pytorch
    from comfy.model_patcher import CoreModelPatcher
    from comfy.supported_models import MiniMaxH3 as H3Config

    # Use the real CPU-compatible native backend when this Windows install defaults to xformers.
    monkeypatch.setattr(h3_model, "optimized_attention", attention_pytorch)

    torch.manual_seed(42)
    config = H3Config(dict(hidden_size=128, num_layers=1, token_refiner_num_layers=0,
                           num_attention_heads=1, attention_head_dim=128, ffn_hidden_size=256,
                           text_dim=128, timestep_input_dim=32, time_embed_hidden_size=128,
                           time_embed_dim=64, dtype=torch.float32))
    base = MiniMaxH3(config, device=torch.device("cpu"))
    with torch.no_grad():
        for parameter in base.diffusion_model.parameters():
            parameter.uniform_(-.02, .02)
        base.diffusion_model.rope.inv_freq.copy_(torch.linspace(.01, 1, 16))
    patcher = CoreModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    effective = sampling(6, 3)
    patcher.add_object_patch("model_sampling", effective)
    patcher.model_options["transformer_options"].update(minimax_h3_sigma_shift_video=6., minimax_h3_sigma_shift_audio=3.)
    return patcher


def test_real_native_guider_with_tiny_h3_and_patched_audio_shift(monkeypatch):
    patcher = tiny_patcher(monkeypatch)
    effective = patcher.get_model_object("model_sampling")
    monkeypatch.setattr(progressive, "upscale_h3_video_to_size", nearest)
    args = inputs(model=patcher)
    output, status = progressive.sample_h3_progressive(**args)
    repeat, _ = progressive.sample_h3_progressive(**args)
    for actual, second in zip(output["samples"].unbind(), repeat["samples"].unbind()):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, second, rtol=0, atol=0)
    assert "audio scale: 2" in status
    # Native model loading can leave an object patch installed until model unload.
    assert patcher.get_model_object("model_sampling") is effective
    assert effective.audio_scale == 2
