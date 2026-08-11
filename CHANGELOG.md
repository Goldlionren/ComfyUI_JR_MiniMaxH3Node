# Changelog

## Unreleased

## 0.7.0

- Added immutable PIPE v2 prompt stages so Prompt Optimizer derives `optimized_prompt` and Prompt Review derives `reviewed_prompt` without losing timeline, registry or runtime media.
- Added `JR_H3_DirectedVideoConditioning`, a unified PIPE consumer that delegates directly to the installed ComfyUI MiniMax H3 Image-to-Video or Reference-to-Video conditioning implementation and returns native `CONDITIONING` plus AV `LATENT`.
- Added deterministic `reviewed > optimized > director` prompt precedence, Auto I2V/Ref2V routing, explicit override conflicts, fixed-24-fps duration conversion, Prefer Pipe/Prefer Node dimensions and native reference-count enforcement.
- Extended Director state schema v1 additively with a Last Frame point role while keeping existing v0.6.0 workflow JSON valid; runtime PIPE schema is now v2 and remains non-serializable.
- Added real-media adapters for reference IMAGE batches, 24 fps video frame decoding and AUDIO waveform loading/trimming. Video/audio remain labels only during LLM optimization and are decoded only at conditioning time.
- Updated the full Director example, README and focused architecture/pipeline/node-reference documentation, plus PIP stage, Review, routing, native delegation, ordering and end-to-end regression coverage.

## 0.6.0

- Added `JR_H3_DirectorDesk`, a large in-node SHOT/VISUAL/AUDIO timeline editor with Global Direction, Inspector editing, drag/resize/snap, split/duplicate/delete/reorder, stacked overlapping reference lanes, media preview/import and persistent user sizing.
- Added schema-versioned JSON-only Director state, deterministic timeline validation/compiler/reference registry, and immutable `JR_H3_DIRECTOR_PIPE` runtime objects whose compiled prompt exactly matches the STRING output.
- Added safe ComfyUI asset descriptors, bounded media probing, input/temp/output path containment, lazy IMAGE decoding, missing/corrupt asset errors and video/audio descriptor-only runtime handling.
- Appended optional `pip` to the existing Prompt Optimizer without removing or reordering legacy inputs. PIP mode reuses the existing H3 router, JPEG conversion, validator and one-shot format repair while rejecting legacy-media conflicts.
- Added Director Desk architecture, usage/reference documentation, an importable example workflow, frontend contract tests, compiler/PIP/state tests and optimizer PIP integration coverage.
- Recorded clean-room product research for ComfyTV, ComfyUI-qwenmultiangle, ComfyUI-mesh2motion and DaSiWa without vendoring their code, styles, assets, project systems or timeline libraries.

### Documentation

- Reconciled the root README and focused guides with the current ten-node implementation.
- Added `docs/NODE_REFERENCE.md` as the canonical input/default/range/output reference.
- Clarified Router-to-Cache wiring, model-structure compatibility, cache hit expectations, optional dependency boundaries and example-workflow limitations.
- Documented current Prompt Review timeout/size behavior and current Enhanced Video Combine preview/output contracts.

### Fixes already present on `main` after package version 0.5.0

- Resolution Scale Calculator now exposes divisor values as string combo options while accepting numeric values saved by older workflows.
- Prompt Review defaults to 3600 seconds, normalizes invalid legacy values, and no longer shrinks a user-resized node after new inference output.
- Enhanced Video Combine publishes video assets through the ComfyUI `gifs` UI payload for Node 2.0 compatibility, so MP4 files are not routed through Pillow image decoding.
- Enhanced Video Combine treats Windows FFmpeg stdin `EPIPE/EINVAL` as an encoder failure and continues fallback; exact 4352×2880 H.264 fallback to `libx264` was locally verified.

## 0.5.0

- Added `JR_H3_UnifiedAcceleration`, a V1 MODEL orchestration node that applies installed KJNodes Sage, H3 Low VRAM Attention, H3 Chunk FeedForward, and Sol-Attn in the fixed verified order.
- Added true per-layer bypass switches, complete current Sol controls, optional `tau_profile`, explicit non-H3/dependency/API-drift errors, and lazy runtime dependency resolution.
- Added CPU/mock coverage for patch order, switches, parameter forwarding, return normalization, missing dependencies, import safety, and existing-node registration.
- Documented the validated default profile, prior hardware validation points, upstream commit audit, and no-vendoring license strategy.
- Documented user-performed GPU acceptance: RTX 4080 SUPER 16GB at ~0.8MP/15s to ~2.4MP in ~8 minutes, and RTX 5090 32GB at 1.5MP/15s to ~2.4MP in ~11 minutes.
- Recorded user-observed runtime equivalence with the original four-node chain, resolution strategy, OOM improvement, and explicit Turbo LoRA attribution without claiming strict benchmark percentages or hardware maxima.

## 0.4.3

- Fixed a builder/validator contract conflict by requiring the Ref2VA `subject_definitions:` heading to occupy its own line in the minimum output skeleton.
- Canonicalize common section-heading wrappers/inline bodies, `<Subject N>: ...` definitions, and unambiguous visible/audio cross-taxonomy retention values inside the existing single repair pass.
- Verified a real local `qwen3.6-27b-abliterated-Q4_K_M.gguf` Ref2VA request succeeds with the corrected initial contract.

## 0.4.2

- Shield protected literals with immutable local sentinels during the single format-repair request, then restore the exact original text before full validation.
- Recognize whitespace-only mutations such as `介绍一下 MiniMax H3` as the location of the protected `介绍一下MiniMax H3` literal without weakening validator equality.
- Reject repaired output when a sentinel is removed, duplicated, or returned with an inconsistent count.

## 0.4.1

- Added exactly one low-temperature, format-only repair request after an initial H3 validation failure, followed by the unchanged full validator.
- Repair prompts preserve user literals and content, omit reasoning extensions, and cannot trigger more than one repair attempt.
- Added `repaired=0`/`repaired=1` success status while preserving final Return Original and Stop Workflow behavior.

## 0.4.0

- Rebuilt Prompt Optimizer as a local H3-oriented prompt/context preprocessor with separate JR Director and clean-room MiniMax H3 interoperability layers.
- Added deterministic Auto, T2VA, I2VA, FL2VA, L2VA, and Ref2VA routing; explicit first/last-frame semantics; and a collision-free Reference Registry.
- Added current base/ref section contracts, keyframe alignment rules, reference/retention validation, exact multilingual dialogue/text preservation, and golden prompt fixtures.
- Preserved the node ID, three outputs, legacy widget prefix, OpenAI-compatible URL/model behavior, one-time reasoning compatibility retry, image processing, and fail modes.
- Recorded upstream SHA, file hashes, license decision, and metadata-only clean-room distribution strategy without bundling official guide prose.

## 0.3.3

- Recalibrated all four active cache presets against real 25-step native H3 input/probe relative-delta measurements; the original thresholds were below even the calmest observed steps and made Router-selected profiles inert.
- Dialogue Safe now produces guarded block hits while retaining audio/video vetoes, no full-step reuse, and one-hit forced refresh protection.
- Added per-workflow input/probe score count/min/average/max summaries so future model, quantization, and resolution calibration is evidence-based.
- Safe profiles no longer count input deltas as block vetoes before their actual front-block probe, eliminating misleading duplicate veto statistics.

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
