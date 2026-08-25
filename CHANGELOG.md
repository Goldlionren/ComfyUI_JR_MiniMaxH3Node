# Changelog

## Unreleased

- Added `JR_H3_DirectorPipeBuilder` to construct an immutable `JR_H3_DIRECTOR_PIPE` from standard prompt, IMAGE batch, VIDEO and AUDIO inputs without serializing runtime media into workflow JSON.
- Added `JR_H3_DirectorPipeUnpack` to pass PIPE values through while exposing prompt stages, timeline metadata, inferred dimensions, registry JSON and index-selected standard media outputs.
- Extended Directed Video Conditioning to consume standard ComfyUI VIDEO objects from Builder PIPE values through bounded trimming and the existing native 24 fps validation path.

## 0.13.0

- Added `JR_MiniMaxH3NeuralLatentUpscaler`, a single-purpose 3D neural spatial upscaler for plain MiniMax H3 video LATENT tensors between Split and Builder.
- Added deterministic linear-scale and decoded pixel-space megapixel planning using the installed native H3 VAE compression and DiT spatial patch grids, preserving B/C/T, dtype, device and unrelated LATENT metadata.
- Added strict checkpoint signature loading from ComfyUI's `latent_upscale_models` folder, canonical H3 normalization from the installed ComfyUI implementation, temporal chunking for long latent sequences, and model-specific ComfyUI offload after inference.
- Added explicit no-checkpoint/no-network/no-interpolation-fallback behavior, license provenance, synthetic neural-checkpoint tests and Split → Upscaler → Builder compatibility coverage.

## 0.12.0

- Added `JR_H3_SplitAVLatent` to split the official MiniMax H3 two-stream `NestedTensor` LATENT into standard video and audio LATENT mappings through the public `unbind()` API.
- Added fail-closed official-type, stream-count, tensor-shape, batch and finite-value validation while preserving the exact input Tensor objects without clone, cast, device transfer or eager contiguous conversion.
- Documented Builder/Split round trips, native Save Latent compatibility, cross-workflow save/load wiring and the requirement to keep audio latent out of spatial video-latent upscaling chains.

## 0.11.2

- Fixed official core `RandomNoise` and `DisableNoise` being rejected when ComfyUI loaded `nodes_custom_sampler.py` under its path-derived runtime module identity while the sampler imported the same source through its package identity.
- Resolve standard NOISE types and factories from ComfyUI's authoritative live `NODE_CLASS_MAPPINGS`, while retaining exact-type checks and fail-closed behavior for genuine generic/custom NOISE providers.
- Added regression coverage for runtime-registry RandomNoise/DisableNoise identities and for seeded custom providers that must remain unsupported.

## 0.11.1

- Fixed identical stochastic noise being reused by same-shaped H3 temporal chunks when the official fixed-seed `RandomNoise` object was called repeatedly.
- Added stable uint64 chunk seed derivation from the base seed and absolute global frame start while preserving the original native NOISE object and seed for single-chunk sampling.
- Kept official `DisableNoise` native for multi-chunk sampling and made unsupported generic/custom NOISE objects fail closed before partial sampling because ComfyUI exposes no common clone, offset or substream contract.
- Added native behavior reproduction, distinctness, determinism, temporal-identity, remainder, single-chunk and generic-NOISE regression coverage.

## 0.11.0

- Added `JR_H3_TemporalChunkSampler`, a sequential H3 AV temporal sampler with the native Advanced Sampler input contract.
- Added deterministic 5-video-token / 17-frame chunk planning and separate 24 fps video to 40 Hz audio boundary mapping, including the official final ±1 audio tick tolerance.
- Delegated every chunk to ComfyUI's current `SamplerCustomAdvanced`, then copied it directly into CPU-preallocated full-length video/audio buffers before releasing the chunk; no retained output list, final `cat`, parallel chunk execution, decoder, overlap or core monkey patching.
- Added fail-closed H3 NestedTensor validation, explicit phase-1 `noise_mask` rejection, optional post-chunk `soft_empty_cache`, lifecycle tests, documentation and node registration.

## 0.10.0

- Added `JR_MiniMaxH3AVLatentBuilder` to assemble separately encoded H3 video `[B,24,T,H,W]` and audio `[B,32,2,T]` latents into ComfyUI's official two-stream `NestedTensor` LATENT.
- Added fail-closed rank/channel/batch/device/dtype/finite-value validation and official H3 `17k+5` frame-grid versus 40 Hz audio temporal checks with clear prefixed errors.
- Added focused success/failure tests, node registration, status diagnostics, node reference documentation and a dedicated workflow guide without adding encoding, file I/O or dependencies.

## 0.9.1

- Fixed Director Shot/Visual/Audio move and edge-resize gestures being cancelled when timeline preview rendering replaced the original pointer target.
- Increased the visible resize hit area and moved drag tracking to window capture listeners so start/end handles remain responsive throughout redraws.
- Replaced per-field Inspector blur commits with an explicit item-level `Save` / `Cancel` draft, atomic validation and one undo transaction; unsaved changes now block navigation and timeline actions instead of disappearing.

## 0.9.0

- Added `JR_H3_HybridLoader`, a single-MODEL FL2VA-base loader with header-first, selected-only REF2VA AdaLN overlays.
- Preserved the current ComfyUI native FL `load_torch_file`/AIMDO/mmap path and FL metadata while avoiding a full REF state dict or second REF MODEL.
- Added deterministic Recommended/All/Custom/Pure/Advanced profiles, family-level custom FL overrides, quant sibling co-travel and fail-closed selected-family shape/dtype/layout checks.
- Added cached patcher reconstruction for Dynamic VRAM/multi-GPU paths, real installed BF16/INT8/pruned-INT8 header accounting, synthetic selective-read/provenance/short-circuit tests, and Scott Mudge MIT attribution.

## 0.8.3

- Expanded protected-dialogue detection to accept half-width/full-width colons, straight/full-width double quotes, and curly double quotes in either IME orientation.
- Added Chinese pleading verbs and English `beg` / `plead` forms to the explicit speech-hint vocabulary while keeping ordinary quoted visible text out of dialogue blocks.
- Added Director PIPE regression coverage proving reversed curly-quoted Chinese dialogue is preserved byte-exactly through Prompt Optimizer formatting.

## 0.8.2

- Added deterministic normalization for the known semantic reference aliases `reference_label`, `visible_retention`, and `audio_retention` before strict validation.
- Added conflict detection so canonical and aliased reference fields can never silently overwrite different values; unrelated unknown fields remain rejected.
- Clarified the model contract that every reference object must use the canonical `label` and `retention` property names.

## 0.8.1

- Tightened Prompt Optimizer into a closed-world faithful rewrite: Director facts and directly visible reference facts are authoritative, creative profiles cannot invent narrative/action/emotion details, and speculative or alternative semantic prose is rejected before deterministic formatting.
- Expanded protected-dialogue detection to vocalized lines such as Chinese `呻吟` / `低语` and English `murmurs` / `moans`, preventing valid Director dialogue from falling back merely because it did not use “说”.
- Made the single semantic repair repeat the authoritative source request and dialogue mapping, so it removes unsupported inventions instead of repairing against the candidate alone.

## 0.8.0

- Replaced free-form final H3 text generation with a strict semantic JSON contract and one optional low-temperature structured repair.
- Added deterministic official Base-family and Ref2VA formatters pinned to MiniMax-H3 commit `8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea`.
- Made Python authoritative for section names/order, alignment preambles, Shot numbering/timestamps, protected dialogue literals/language tags/stable speaker IDs, reference order and retention taxonomy.
- Strengthened final validation for unknown headings, dialogue syntax and section ownership while preserving legacy inputs, four outputs and immutable PIPE derivation.
- Recorded the fixed source paths and SHA256 values without redistributing upstream guide prose because its documentation license terms are not safely generalizable.

## 0.7.1

- Fixed Director Inspector edits being discarded when the user clicked or dragged a timeline item before the browser emitted the input's blur/change event.
- Replaced the ambiguous Director ordering arrows with `Earlier` / `Later` for Shots and `Lane ↑` / `Lane ↓` for Visual/Audio display lanes, including contextual labels and tooltips.

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
