from dataclasses import replace

import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.utils.h3_cache_config import build_preset_config
from ComfyUI_JR_MiniMaxH3Node.utils.h3_cache_metrics import metric_sample
from ComfyUI_JR_MiniMaxH3Node.utils.h3_cache_runtime import H3AdaptiveCacheRuntime

CUDA_AVAILABLE = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device required for cache placement test")


def _runtime(cache_device: str):
    config = replace(
        build_preset_config("balanced", cache_device=cache_device, audio_content="Speech"),
        warmup_steps=0,
        start_percent=0.0,
        end_percent=1.0,
    )
    return H3AdaptiveCacheRuntime(config, 50, audio_required=True)


def _cuda_streams(dtype=torch.float32):
    return (
        torch.ones((1, 2, 2, 2), device="cuda", dtype=dtype),
        torch.ones((1, 2, 4), device="cuda", dtype=dtype),
    )


class _PassExecutor:
    class_obj = object()

    def __init__(self):
        self.calls = 0

    def __call__(self, x, *_args, **_kwargs):
        self.calls += 1
        return [x[0].clone(), x[1].clone()]


def _run_cuda_step(runtime, executor, streams, timestep):
    context = torch.ones((1, 1, 4), device="cuda")
    return runtime.diffusion_wrapper(
        executor, list(streams), torch.tensor([timestep], device="cuda"), context, {}, {"seed": 1}
    )


@needs_cuda
def test_metric_sample_is_small_graph_free_fp32_and_stays_on_compute_device():
    source = torch.ones(64, device="cuda", dtype=torch.float16, requires_grad=True)
    sample = metric_sample(source, 8)
    assert sample.device == source.device
    assert sample.dtype == torch.float32
    assert sample.numel() == 8
    assert not sample.requires_grad and sample.grad_fn is None
    assert sample.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


@needs_cuda
def test_cpu_cache_keeps_audio_and_video_metrics_on_cuda_and_residual_on_cpu():
    runtime = _runtime("CPU")
    video, audio = _cuda_streams()
    runtime._remember_inputs(video, audio)
    residual = runtime._store_residual(video)
    video_metric, audio_metric = runtime._previous_input_metrics
    assert video_metric.device == video.device
    assert audio_metric.device == audio.device
    assert residual.device.type == "cpu"
    assert runtime.stats.metric_migrations == 0
    assert runtime.stats.residual_to_cpu == 1


@needs_cuda
def test_gpu_cache_keeps_metrics_and_residual_on_cuda():
    runtime = _runtime("GPU")
    video, audio = _cuda_streams()
    runtime._remember_inputs(video, audio)
    residual = runtime._store_residual(audio)
    assert all(sample.device.type == "cuda" for sample in runtime._previous_input_metrics)
    assert residual.device.type == "cuda"
    assert runtime.stats.cpu_gpu_transfers == 0
    assert runtime.stats.metric_migrations == 0


@needs_cuda
def test_auto_to_cpu_only_moves_residual(monkeypatch):
    runtime = _runtime("Auto")
    video, audio = _cuda_streams()
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (0, 32 * 1024**3))
    runtime._remember_inputs(video, audio)
    residual = runtime._store_residual(video)
    assert runtime._resolved_device == "CPU"
    assert residual.device.type == "cpu"
    assert all(sample.device.type == "cuda" for sample in runtime._previous_input_metrics)
    assert runtime.stats.metric_migrations == 0


@needs_cuda
@pytest.mark.parametrize("cache_device", ["CPU", "GPU"])
def test_full_cuda_wrapper_completes_for_cpu_and_gpu_cache(cache_device):
    runtime = _runtime(cache_device)
    executor = _PassExecutor()
    streams = _cuda_streams()
    _run_cuda_step(runtime, executor, streams, 900)
    outputs = _run_cuda_step(runtime, executor, streams, 800)
    assert executor.calls == 1
    assert all(output.device.type == "cuda" for output in outputs)
    assert all(sample.device.type == "cuda" for sample in runtime._previous_input_metrics)
    assert runtime.stats.metric_migrations == 0
    if cache_device == "CPU":
        assert all(cached.device.type == "cpu" for cached in runtime._previous_outputs)
        assert runtime.stats.residual_to_gpu == 2
    else:
        assert all(cached.device.type == "cuda" for cached in runtime._previous_outputs)
        assert runtime.stats.cpu_gpu_transfers == 0


@needs_cuda
def test_full_cuda_wrapper_completes_for_auto_to_cpu(monkeypatch):
    runtime = _runtime("Auto")
    executor = _PassExecutor()
    streams = _cuda_streams()
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (0, 32 * 1024**3))
    _run_cuda_step(runtime, executor, streams, 900)
    outputs = _run_cuda_step(runtime, executor, streams, 800)
    assert runtime._resolved_device == "CPU"
    assert executor.calls == 1
    assert all(output.device.type == "cuda" for output in outputs)
    assert all(sample.device.type == "cuda" for sample in runtime._previous_input_metrics)
    assert runtime.stats.metric_migrations == 0


@needs_cuda
@pytest.mark.parametrize("stream_index", [0, 1], ids=["video", "audio"])
def test_cpu_residual_hit_restores_stream_device_and_dtype(stream_index):
    runtime = _runtime("CPU")
    streams = _cuda_streams(dtype=torch.bfloat16)
    current = streams[stream_index]
    residual = runtime._store_residual(torch.ones_like(current, dtype=torch.float32))
    restored = runtime._restore_residual(residual, current)
    output = current + restored
    assert residual.device.type == "cpu"
    assert restored.device == current.device and restored.dtype == current.dtype
    assert output.device == current.device and output.dtype == current.dtype
    assert runtime.stats.residual_to_cpu == 1
    assert runtime.stats.residual_to_gpu == 1


@needs_cuda
def test_reset_releases_all_metric_state_without_cpu_samples():
    runtime = _runtime("CPU")
    video, audio = _cuda_streams()
    runtime._remember_inputs(video, audio)
    runtime._probe_video = runtime._store_metric(video, 2)
    runtime._probe_audio = runtime._store_metric(audio, 2)
    assert all(sample.device.type == "cuda" for sample in runtime._previous_input_metrics)
    runtime.reset("test")
    assert runtime._previous_input_metrics is None
    assert runtime._previous_output_metrics is None
    assert runtime._probe_video is None and runtime._probe_audio is None
    assert runtime._metric_device is None


@needs_cuda
def test_full_refresh_metrics_do_not_restore_cpu_residuals_without_hits():
    runtime = _runtime("CPU")
    video, audio = _cuda_streams()
    for value in (1.0, 1.1, 1.2, 1.3):
        outputs = (video * value, audio * value)
        runtime.remember_full_forward(video * value, audio * value, outputs)
    assert runtime.stats.residual_to_cpu == 8
    assert runtime.stats.residual_to_gpu == 0
    assert runtime.stats.metric_migrations == 0
    assert runtime.stats.cpu_gpu_transfers == runtime.stats.residual_to_cpu
