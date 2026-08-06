# Changelog

## 0.3.2

- Fixed excessive cache invalidation caused by using transient conditioning/reference tensor storage addresses in the sampling signature.
- Equivalent tensors recreated by ComfyUI now preserve cache history; seed, model, tensor structure, reference structure, packed layout, and timestep restarts still invalidate it.
- Initial state creation is no longer counted as a reset, and cleanup now starts the next workflow with fresh per-sampling statistics.
- Added regression coverage proving storage-address changes can hit while semantic/structural changes still reset safely.

## 0.3.1

- Fixed CPU and Auto residual-cache modes keeping sampled metric history on CPU, which caused CUDA/CPU device mismatch during real MiniMax H3 sampling.
- Split small fp32 metric history from large residual storage. Metric snapshots now remain graph-free on the active compute device regardless of `cache_device`.
- Added explicit metric-device validation and separate `residual_to_cpu`, `residual_to_gpu`, and `metric_migrations` statistics.
- Removed non-hit CPU round trips for output metrics and transient middle-block entry state.
- Added CUDA coverage for CPU, GPU, and Auto-to-CPU placement, independent audio/video restoration, cleanup, and transfer accounting.

## 0.3.0

- Added `JR H3 Cache Config Router`, an independent OpenAI-compatible scene classifier that maps semantic output to immutable local cache presets.
- Added `JR H3 Adaptive Cache` with Auto, Visual Fast, Dialogue Safe, Action Safe, Balanced, and Off modes.
- Added separate video/audio relative-delta metrics, full-step fast path, block-probe middle cache, tail refresh, forced refresh, cache-device selection, invalidation, statistics, and cache-conflict detection.
- Added Mock H3 path tests, deterministic Router/config tests, metric and state-reset tests, algorithm documentation, and a manual benchmark result tool.
- Kept Prompt Optimizer inputs, outputs, system prompt, response parsing, and workflow behavior unchanged.
