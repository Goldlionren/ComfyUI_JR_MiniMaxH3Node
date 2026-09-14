# JR TaoMate-inspired streaming — experimental branch

This is a parallel path, not an upgrade/replacement of Hard AV Latent Prefix. The user's existing ver2.5 16G workflow is unchanged. Production test deployment is separately recorded in the task-directory `PRODUCTION_TEST.md` outside the repository. No main-branch merge, GitHub push, Registry release or model/engine installation is included.

## Public nodes and modes

`JR_H3_TaoMateChunkPlanner` emits an immutable `JR_H3_STREAM_PLAN` and human-readable status. Appended outputs `native_frames` and `audio_ticks` expose exact sizes; the original two socket indices are unchanged. `JR_H3_StreamingSampler` accepts MODEL, positive, external video VAE, NOISE, Euler SAMPLER, SIGMAS, AV LATENT and the plan; it returns AV LATENT and status.

| Mode | Execution | Persistent history supplied to attention |
|---|---|---|
| Geometry Only (default) | Ordinary full native sampling, no streaming patch | None |
| Micro Chunk | 4/8/12 phases, shared global noise/positions | None; diagnostic only |
| Clean Commit | Above plus an extra clean forward per phase | None; diagnostic only |
| Clean KV | Clean forwards transactionally capture all layers | None; collection diagnostic only |
| Streaming Attention | Clean commits and historical AV attention | All layers; correctness reference |
| Sparse KV | Same streaming algorithm | Selected layers only |

All modes require one complete AV latent matching the selected preset. `layer_policy` is active only in Sparse KV: all, every_2, every_4, or a comma-separated custom list of unique in-range indices. Unselected layers still execute conditioning/current-media attention but receive no persistent history. Streaming Attention always uses all layers, regardless of this widget; tooltip/status make the effective policy explicit.

## Exact geometry

| Phase | Groups | Native frames | Video latents | Audio ticks (40 Hz) |
|---|---:|---|---|---|
| 0 | 2 | 0–39 | 0–12 | 0–65 |
| 1 | 2 | 39–73 | 12–22 | 65–122 |
| 2 | 2 | 73–107 | 22–32 | 122–178 |
| 3 | 1 | 107–124 | 32–37 | 178–207 |

Intervals are half-open. Audio uses exact integer round-half-even of frame × 40/24. Native duration is **5.166667 seconds**, not exactly five seconds. No implicit trimming/stretching is performed.

### Full first pass, windowed second pass (10s/15s experimental)

| Planner preset | Native frames at 24fps | Actual seconds | Video latents | Audio ticks | Windows / micro-phases |
|---|---:|---:|---:|---:|---:|
| TaoMate 5s Canonical (unchanged default) | 124 | 5.166667 | 37 | 207 | 1 / 4 |
| JR 10s Experimental | 243 | 10.125000 | 72 | 405 | 2 / 8 |
| JR 15s Experimental | 362 | 15.083333 | 107 | 603 | 3 / 12 |

Keep the first Progressive/Guided pass full-length with one full-story prompt. Its output is spatially upscaled as before, then refined by Streaming/Sparse KV. Only the second sampler is temporally windowed. The user's approximately 0.4MP first-pass target is a tested starting point, not a hard minimum and not a minimum for Progressive's internal low-resolution stage.

Select the matching planner preset AND change the first-pass duration/conditioning to the full length. The native duration nodes align nominal 5/10/15 seconds to these H3 lengths. Alternatively connect `native_frames` to a native **frame-count** input, not a seconds input. `audio_ticks` counts audio latents, not waveform samples or frames. A planner selection does not extend an existing 5-second latent; mismatches fail with expected and actual AV dimensions before sampling.

Each logical window contains the `(2,2,2,1)` micro-phase pattern. Only the beginning of the whole timeline owns H3's affine prefix. Window frame ranges are `0:124`, `124:243`, `243:362`; video ranges `0:37`, `37:72`, `72:107`; audio ranges `0:207`, `207:405`, `405:603`. Do not concatenate three independent 124-frame clips. There is no hard-prefix overlap or independent sampler/session reset at window boundaries: clean history continues under the selected bounded retention. These windows are grouping/accounting boundaries; the actual model working unit remains a micro-phase, not a whole 5-second block.

Positions are taken from one native full-request PackedLayout, including independent reference blocks and first/last conditioning. Stereo audio slices preserve channel-major order. A phase starting at video latent 12 or crossing a window boundary does not restart the five-token temporal cycle. Final VAE decode consumes the assembled full latent, not independently decoded clips. The existing TemporalChunkSampler/Hard Prefix implementation remains unchanged; this is not its overlap algorithm.

### Sigma audit and current decision

The planner is **geometry only**; it never chooses or changes SIGMAS. Every micro-phase runs the complete supplied second-pass schedule, then a separate near-clean cache forward. This is not a progressive resolution transition splitting one schedule across temporal windows.

Reviewed local TaoMate-H3 revision `ccc1a70adbf7f552a84a0cd7eeac0a6f3d461cad`, `src/taomate_h3/denoise_schedule.py`, `model/pipeline.py`, `model/denoise.py` and `streaming/runtime.py`:

- Upstream samples states `(0,16,33,49)` from 50 linearly spaced states shifted by video=12, audio=3. Video SIGMAS are approximately `[1, 0.9611651, 0.8533333, 0]`; audio SIGMAS `[1, 0.8608695, 0.5925926, 0]`.
- The user's tested second-pass `BasicScheduler(simple, steps=3, denoise=0.3)` with shift_video=12 instead yields approximately `[0.8372093, 0.75, 0.5714286, 0]` in the inspected ComfyUI revision. `denoise=0.3` does **not** mean initial sigma=0.3. These are different state selections, despite both doing three Euler evaluations.
- Keep the user's refinement schedule unchanged for this release. Blindly replacing it with the upstream full-noise schedule starting at 1 can erase the first-pass anchors. If evaluated later, expose a separate explicit SIGMAS producer, preserve the existing input, and A/B test full-noise generation separately from latent refinement. Do not silently clamp/select a supposed distilled refinement subset.
- Native ModelSamplingAV already carries audio onto the video sigma domain using the shift ratio (12/3=4); do not independently replace packed audio's schedule or its latent scaling in JR. The sampler now reports the supplied video SIGMAS and native audio scale.
- Upstream's clean timestep=1 is in its `t=1-sigma` convention, not Comfy's raw sigma. JR uses native sigma=0 for the extra forward, with H3's internal clamp. Upstream additionally renormalizes generated clean video against an anchor; JR deliberately does not alter the output this way. Matching sigma values alone would not make the runtimes numerically equivalent.

No TaoMate schedule node, renormalization or sigma algorithm change is included in the duration extension.

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

The 10s/15s presets keep the same largest micro-phase. For equal spatial size, layer policy and retention, the conservative KV+staging bound is unchanged. Full-length first-pass sampling, source/output/noise buffers, conditioning and final decode still scale with duration. This is not a constant-memory whole workflow. Status separately reports peak **post-trim** retained KV and the conservative KV+staging bound; neither measures peak process RAM.

## First manual ComfyUI test

Use a separate development instance or the explicitly authorized, backed-up production test deployment. Keep stable workflows recoverable.

1. Duplicate the user's latest workflow; keep the original untouched. For the duplicate set the actual upstream conditioning/Director duration to produce **124 native frames**, 37 video latents and 207 audio ticks. Verify the AV shape/status, not just a duration widget. Start at **512×288**, rather than the high-resolution production preset, to establish full-layer KV reference within the default CPU budget.
2. Keep the already tested Progressive scale 0.6, neural spatial upscale, TaoMate LoRA and its existing three-step sigma setup. Set the **final** latent resolution to 512×288 for this first correctness run. The toy sigma schedule in `tools/smoke_h3_streaming.py` is not a recommended LoRA schedule.
3. On the experimental second-stage branch replace the TemporalChunkSampler with `JR_H3_StreamingSampler`. Supply the same MODEL after TaoMate LoRA/Unified, same positive, external video VAE, official RandomNoise/DisableNoise, Euler, sigmas and upscaled AV latent. Connect the new Planner's `stream_plan` to it. Split its output with JR Split AV Latent and reuse the existing video/audio decode and output nodes. External TRT decode remains external.
4. Disable Adaptive Cache, Morton and compiled forward replacements. Keep Sage/LowVRAM/FFN. Start TST off. Run Geometry Only once as the same-node ordinary baseline, then Streaming Attention/all layers/CPU/previous_only. Next compare sink_plus_recent_2. Save status and output from each.
5. With fixed seed/conditions/LoRA/schedule compare TST off vs 0.2; only then test Sparse KV/every_4. Evaluate identity, lip sync, motion and continuity near frame boundaries **39/73/107**. Repeat each setting to check reproducibility. Test first+last anchors and driven audio as separate cases before combining guides.
6. Raise resolution after the correctness reference is accepted. Increase CPU KV budget only after checking available RAM; measure CPU transfers and CUDA peak. Try GPU sparse KV only with adequate free VRAM.

Legacy Hard AV Prefix presets do not accept every canonical length; compare legacy and streaming on a common supported setup only after deliberately defining the duration mapping. Do not silently pad one side and report a like-for-like speedup. The initial exact A/B is Geometry Only versus Streaming Attention on the same 124-frame latent.

For longer acceptance tests, first retain the accepted 5s first-pass resolution, sigma settings, Sparse KV/every_2 and previous_only. Change total first-pass duration and planner to 10s, then 15s. Check the full story, seams around frame 124/243 plus internal micro-phase boundaries, driven audio and the final-frame anchor. Compare fixed seeds/settings; every_4 is a speed-first option with user-observed mild background drift. Do not promise seamless output based only on shape/position tests.

## Unsupported / unverified

- Cross-request/session continuation, nonzero external offsets, arbitrary/custom geometry, exact-duration trimming, incremental preview/decode, Hard Prefix+KV hybrid and audio teacher are not implemented. 10s/15s windows belong to one node execution and one full timeline; altered plans are rejected.
- Only batch 1, native H3 AV flow, standard Euler without churn, finite decreasing sigma schedule ending at zero, one positive conditioning entry and binary broadcastable AV masks are supported. Regional/scheduled/control-hook conditioning, mismatched keyframe resolution, unknown diffusion wrappers, Adaptive Cache and Morton fail explicitly.
- Full checkpoint/TaoMate LoRA quality and performance remain **user acceptance work**, not established by random tiny-model tests. Windows CUDA offload variants and third-party replacements beyond the installed stack are unverified. No runtime type/filename check can prove a loaded LoRA was trained for streaming.
- CPU FP32 attention caches are downcast to BF16 intentionally; full cache is the reference for this implementation, not bit-identical ordinary H3 or official TaoMate. Native near-clean sigma, sparse missing-history layers, conditioning isolation and the absence of upstream latent renormalization are numerical/quality differences to evaluate.
- No Hopper FA3, vLLM, W8A8, quantized KV, multi-GPU/distributed runtime, new dependencies, server, model download or engine compilation.

See [design audit](H3_STREAMING_AUDIT.md) and [engineering verification](H3_STREAMING_VERIFICATION.md).
