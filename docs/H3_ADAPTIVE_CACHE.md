# JR H3 Adaptive Cache

## Scope and verified H3 integration

This node targets ComfyUI's native `comfy.ldm.minimax.model.MiniMaxH3Model`. The inspected implementation defaults to 50 `DiTBlock` objects and packs one sequence as text/reference rows followed by target audio and target video. Its final layer separates the target streams, then audio is unpacked and video is unpatchified.

The plugin does not replace or edit that implementation. It clones the input ModelPatcher and uses:

- a keyed `DIFFUSION_MODEL` wrapper for full-step decisions;
- native `patches_replace["dit"][("double_block", index)]` callbacks for Block probing;
- an attachment for state ownership;
- a keyed ModelPatcher cleanup callback for reset and summary logging.

The H3 core advances its Dynamic VRAM/prefetch queue before every Block callback and drains the queue after the loop. A skipped replacement therefore does not bypass the core prefetch bookkeeping. SageAttention/FlashAttention remains inside every Block that is actually executed.

## Modes

### Visual Fast

The wrapper samples video and audio independently in fp32 and measures relative delta. Full-step output can be reused only when video and required audio are both below their thresholds, the denoise window is active, warmup is complete, and the consecutive-hit limit is not exhausted. With `audio_content=None`, only video vetoes a hit.

### Dialogue Safe

Default Balanced preset layout is F1-M47-B2 for a 50-Block model. Block 0 executes, the entry state of Block 1 is compared independently over target audio/video rows, Blocks 1–47 are replaced by one cached aggregate middle residual on a hit, and Blocks 48–49 execute. The default limit permits at most one consecutive Block hit.

### Action Safe

Default layout is F2-M46-B2: Blocks 0–1 and 48–49 execute. Its window and thresholds are more conservative than Visual Fast. Speech/Singing configurations continue to evaluate audio; choosing Action never disables voice protection.

### Balanced

Balanced has three paths:

1. **Fast Path:** both-stream input change is below `fast_path_threshold`; reuse full-step output.
2. **Probe Path:** change is below `probe_path_threshold`; execute front Blocks, compare target audio/video probe rows, reuse the aggregate middle residual only if both pass, then execute tail Blocks.
3. **Full Path:** high change, a veto, missing cache, forced refresh, warmup, or an inactive window runs every Block.

Counters distinguish full-step hits, Block hits, true full forwards, forced refreshes, and audio/video vetoes.

The summary also reports count/minimum/average/maximum for input and front-Block probe deltas. Dialogue/Action veto counters refer to the actual Block probe only; they are not duplicated by the earlier input estimate.

### Auto and Off

Auto first honors a valid `profile_hint`. Otherwise Speech/Singing selects Dialogue Safe; Music/Ambient/None selects Visual Fast; Auto selects Balanced. Off returns the original ModelPatcher without cloning or adding callbacks.

## Metric and cache device

The first implementation uses sampled relative delta:

```text
mean(abs(current - previous)) / (mean(abs(previous)) + epsilon)
```

Sampling remains on the active compute device, calculation and saved metric history use fp32, and only the scalar score synchronizes. Each strided snapshot is detached and cloned so it keeps neither an autograd graph nor the backing storage of the full tensor. BF16, FP16, FP32, non-contiguous and tiny tensors are supported. Audio and video use separate strides and scores.

`cache_device` controls only large full-step and aggregate middle residuals. It never moves input, output, audio, video, or block-probe metric history to CPU. A metric-device mismatch raises an explicit internal-state error instead of silently adding a per-step transfer.

GPU residual cache minimizes latency. CPU residual cache stores a whole full-step stream or aggregate middle residual per transfer rather than transferring each Block. A CPU residual is restored explicitly to the hit target's device and dtype immediately before use. Auto compares estimated cache bytes with free CUDA memory after preserving `gpu_reserve_mb`; query failure safely selects CPU. Runtime summaries report `residual_to_cpu`, `residual_to_gpu`, and `metric_migrations` separately; a normal run has zero metric migrations.

## Invalidation and forced refresh

State belongs to one cloned ModelPatcher. It resets when model identity, seed, conditioning/reference structure, packed layout segments, tensor shape, dtype, device, batch, video/audio length or presence changes, or when timestep order restarts. It deliberately does not use tensor storage addresses: ComfyUI may recreate equivalent context and reference tensors between denoise steps. Forward exceptions and ModelPatcher cleanup also reset state. Hit streak limits force a real refresh even if metrics remain low.

Reference conditioning is part of the H3 payload/conditioning identity. Cache content never crosses ModelPatcher clones or sampling cleanup.

Runtime summary statistics are per sampling workflow. Initial state creation is not a reset, and cleanup clears counters after logging them. A high `resets` count therefore indicates a real structural change or timestep restart rather than normal tensor allocation churn.

## Router boundary

The Router makes a second independent chat-completions request. Its LLM output contains semantic enums only. Python validates and locally reviews those fields, then selects a versioned preset. The immutable config contains no API key, full prompt, or raw response. Connected Router config replaces all manual widget values; it is never merged with them.

## Conflicts and combinations

Do not stack this node with EasyCache, TeaCache, First Block Cache, CacheDiT, another Block replacement cache, or a second JR H3 Adaptive Cache. It may be used with attention backends, quantization, Dynamic VRAM, CPU offload, and downstream RTX/video nodes.

Diffusion timestep is not the final video's timeline. A mode applies to denoise computation, not to named seconds in the generated clip.

## Calibration status

Preset values are deterministic values in this implementation's relative-delta scale; they are not copied from another cache's scale. Version 0.3.3 recalibrated the active presets with native 25-step H3 measurements after the earlier initial thresholds proved lower than the calmest observed input/probe deltas. Benchmark representative seeds, prompts, reference media, resolutions, frame counts, quantization, and attention backends before production use. Conservative remains the recommended starting point for strict quality evaluation.
