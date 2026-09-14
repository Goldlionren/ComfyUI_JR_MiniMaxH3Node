# JR TaoMate-inspired streaming — experimental branch

This is a parallel path, not an upgrade/replacement of Hard AV Latent Prefix. The user's existing ver2.5 16G workflow is unchanged. No production deployment, merge, GitHub push, Registry release or model/engine installation is included.

## Public nodes and modes

`JR_H3_TaoMateChunkPlanner` emits an immutable `JR_H3_STREAM_PLAN` and human-readable status. `JR_H3_StreamingSampler` accepts MODEL, positive, external video VAE, NOISE, Euler SAMPLER, SIGMAS, AV LATENT and the plan; it returns AV LATENT and status.

| Mode | Execution | Persistent history supplied to attention |
|---|---|---|
| Geometry Only (default) | Ordinary full native sampling, no streaming patch | None |
| Micro Chunk | Four phases, shared global noise/positions | None; diagnostic only |
| Clean Commit | Above plus an extra clean forward per phase | None; diagnostic only |
| Clean KV | Clean forwards transactionally capture all layers | None; collection diagnostic only |
| Streaming Attention | Clean commits and historical AV attention | All layers; correctness reference |
| Sparse KV | Same streaming algorithm | Selected layers only |

All modes currently require one canonical 124-frame request. `layer_policy` is active only in Sparse KV: all, every_2, every_4, or a comma-separated custom list of unique in-range indices. Unselected layers still execute conditioning/current-media attention but receive no persistent history.

## Exact geometry

| Phase | Groups | Native frames | Video latents | Audio ticks (40 Hz) |
|---|---:|---|---|---|
| 0 | 2 | 0–39 | 0–12 | 0–65 |
| 1 | 2 | 39–73 | 12–22 | 65–122 |
| 2 | 2 | 73–107 | 22–32 | 122–178 |
| 3 | 1 | 107–124 | 32–37 | 178–207 |

Intervals are half-open. Audio uses exact integer round-half-even of frame × 40/24. Native duration is **5.166667 seconds**, not exactly five seconds. Do not use a 120-frame or 243-frame latent and expect automatic resampling. No implicit trimming/stretching is performed.

Positions are taken from one native full-request PackedLayout, including independent reference blocks and first/last conditioning. Stereo audio slices preserve channel-major order. A phase starting at video latent 12 does not restart the five-token temporal cycle. Final VAE decode consumes the assembled 37 latents, not four independently decoded clips.

## Clean state, cache and backend

- Full request noise is generated once and sliced by AV ranges. One runtime and guider are reused across phases; there are no extra ComfyUI queue jobs.
- After native Euler completes a phase at sigma zero, a sampler adapter performs another model evaluation while models are loaded. It receives internal sampler-domain AV (including native audio carry), not raw VAE latents. Native MiniMaxH3 reverses the carry. The prediction is discarded; generated output is not replaced. H3 internally clamps sigma to 1e-6, so this is **near-clean**, not an exact numerical port of TaoMate's timestep=1 runtime.
- Q/K normalization and RoPE are completed before interception. Only target video/audio K/V from clean forwards is staged, in owned compact BF16 storage. Text, conditioning/reference rows and noisy forwards are never committed.
- A commit validates all selected layers and all head groups before atomic publication. Missing/duplicated heads, inconsistent rows and exceptions cause failure/rollback. Trimming cannot run during a transaction.
- Retention choices: previous_only; first commit's video sink + one recent complete AV commit; video sink + two recent complete AV commits. No duplicate sink rows while the first commit is still recent. Old sink audio is dropped. A reset operation exists; `audio_reset_interval_requests=1` is reserved metadata because cross-request continuation is not exposed.
- Conditioning Q sees conditioning K/V only. Current media Q sees conditioning + retained clean AV + current AV K/V. This changes the attention graph; it is not ordinary bidirectional H3 with a longer K buffer.
- Existing Unified attention delegate remains in the call chain. KJ head chunks and FFN remain active. Sol's current sparse kernel cannot handle rectangular current-media Q vs longer K, so that branch falls back through Sol to Sage/native. Status states this. Sol speedups from the legacy workflow must not be assumed here.
- TST reuses JR's pooled-Q transform and layer/sigma weighting for noisy current video. It is explicitly bypassed for clean forwards. No TST output/block cache is reused.
- State, hooks and KV belong to one node execution. Finally blocks clear cache/phase/attention references, remove clone-only wrappers and restore native latent shape state, including on failure. Original input MODEL options and wrappers are not overwritten.

## Memory controls

Default KV storage is **CPU**, with an 8192 MiB budget; CUDA storage is opt-in. This budget is for KV/staging, not total model RAM/VRAM. CPU storage transfers the required head slice back for attention and may be slower. No pinned-memory allocator or asynchronous prefetch is implemented yet.

BF16 K+V bytes per cached layer are `tokens × heads × head_dim × 4`. The preflight bound includes retained history, staging and head-group merge storage. If it exceeds `max_kv_mib`, sampling fails before loading model work with an actionable message. Do not blindly raise this limit on a 12/16GB GPU. At full H3 width, dense KV can consume tens of GiB; bounded history does not mean small history.

Status reports actual retained KV bytes/MiB, AV row counts, retained commits, cached layer count, phase timings, denoise/clean forward counts, peak CUDA allocated/reserved and attention host dispatch time. Host dispatch time is **not** isolated GPU kernel time. Sparse layer-byte reduction is relative to the same geometry/full layer cache, not whole-workflow VRAM reduction.

## First manual ComfyUI test

Use a separate **development** ComfyUI instance with this feature checkout. Do not replace the production node directory just to make these nodes visible.

1. Duplicate the user's latest workflow; keep the original untouched. For the duplicate set the actual upstream conditioning/Director duration to produce **124 native frames**, 37 video latents and 207 audio ticks. Verify the AV shape/status, not just a duration widget. Start at **512×288**, rather than the high-resolution production preset, to establish full-layer KV reference within the default CPU budget.
2. Keep the already tested Progressive scale 0.6, neural spatial upscale, TaoMate LoRA and its existing three-step sigma setup. Set the **final** latent resolution to 512×288 for this first correctness run. The toy sigma schedule in `tools/smoke_h3_streaming.py` is not a recommended LoRA schedule.
3. On the experimental second-stage branch replace the TemporalChunkSampler with `JR_H3_StreamingSampler`. Supply the same MODEL after TaoMate LoRA/Unified, same positive, external video VAE, official RandomNoise/DisableNoise, Euler, sigmas and upscaled AV latent. Connect the new Planner's `stream_plan` to it. Split its output with JR Split AV Latent and reuse the existing video/audio decode and output nodes. External TRT decode remains external.
4. Disable Adaptive Cache, Morton and compiled forward replacements. Keep Sage/LowVRAM/FFN. Start TST off. Run Geometry Only once as the same-node ordinary baseline, then Streaming Attention/all layers/CPU/previous_only. Next compare sink_plus_recent_2. Save status and output from each.
5. With fixed seed/conditions/LoRA/schedule compare TST off vs 0.2; only then test Sparse KV/every_4. Evaluate identity, lip sync, motion and continuity near frame boundaries **39/73/107**. Repeat each setting to check reproducibility. Test first+last anchors and driven audio as separate cases before combining guides.
6. Raise resolution after the correctness reference is accepted. Increase CPU KV budget only after checking available RAM; measure CPU transfers and CUDA peak. Try GPU sparse KV only with adequate free VRAM.

Legacy Hard AV Prefix presets do not accept every canonical length; compare legacy and streaming on a common supported setup only after deliberately defining the duration mapping. Do not silently pad one side and report a like-for-like speedup. The initial exact A/B is Geometry Only versus Streaming Attention on the same 124-frame latent.

## Unsupported / unverified

- Cross-request continuation, nonzero global offsets, arbitrary/custom geometry, exact 5-second delivery, incremental preview/decode, Hard Prefix+KV hybrid and audio teacher are not implemented. Timeline fields exist but altered plans are rejected.
- Only batch 1, native H3 AV flow, standard Euler without churn, finite decreasing sigma schedule ending at zero, one positive conditioning entry and binary broadcastable AV masks are supported. Regional/scheduled/control-hook conditioning, mismatched keyframe resolution, unknown diffusion wrappers, Adaptive Cache and Morton fail explicitly.
- Full checkpoint/TaoMate LoRA quality and performance remain **user acceptance work**, not established by random tiny-model tests. Windows CUDA offload variants and third-party replacements beyond the installed stack are unverified. No runtime type/filename check can prove a loaded LoRA was trained for streaming.
- CPU FP32 attention caches are downcast to BF16 intentionally; full cache is the reference for this implementation, not bit-identical ordinary H3 or official TaoMate. Native near-clean sigma, sparse missing-history layers, conditioning isolation and the absence of upstream latent renormalization are numerical/quality differences to evaluate.
- No Hopper FA3, vLLM, W8A8, quantized KV, multi-GPU/distributed runtime, new dependencies, server, model download or engine compilation.

See [design audit](H3_STREAMING_AUDIT.md) and [engineering verification](H3_STREAMING_VERIFICATION.md).
