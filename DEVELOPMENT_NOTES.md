# Development notes

## JR Cut Audio editor (2026-09-17)

Added an independent V1 `JR_CutAudio` node and scoped DOM editor. The node/inputs/outputs/datatypes/lifecycle/frontend skills informed keeping serialized input state separate from an unsaved draft, explicitly locking output sample ranges and keeping playback volume outside the tensor path. Existing node sockets, sampling, acceleration and media workflows are unchanged. Upload uses native ComfyUI `/upload/image` with `type=input`, `subfolder=jr_cut_audio`; no CDN, remote audio fetching, additional models or codec package installation.

The CPU service uses installed PyAV to decode to float32 while retaining sample rate/channels. Waveform extrema span all channels without averaging away opposite-phase stereo. Cut and execution verify a source SHA-256, round seconds half-up to `[start_sample,end_sample)`, and slice identical PCM. Download is IEEE float32 WAV, original download preserves source bytes. No normalization/fades/MP3 recompression. Source paths are restricted to input; preview paths are validated temp tokens. File/PCM/duration caps, bounded JSON and one non-queued editor worker limit resource use; cancelled HTTP requests retain the slot until the worker exits. Codec timeout checks are cooperative, not a hard process timeout. Temp files may accumulate across distinct selections and are not silently deleted.

Validation: **1085 CPU tests passed, 16 skipped** (15 existing CUDA/opt-in skips plus unavailable libvorbis fixture encoder). Actual WAV/MP3/FLAC/M4A/AIFF decoding passed; WAV downloads equal AUDIO tensors sample-for-sample. HTTP tests cover downloads, byte ranges, invalid payloads and source identity; cancellation and mono endpoint tests passed. Two Node.js tests passed for range validation and Comfy extension registration/serialization/restore/removal; Ruff passed. Import/workflow smoke: 30 nodes, 25 workflows, the same four tolerated legacy link warnings.

Browser QA used a localhost-only isolated editor with synthetic stereo audio: seconds inputs, Cut lock, output `[1,2,156000]` at 24 kHz for 6.5s, playback state, volume control, restore, unlock and dragging waveform edges/playhead were verified. This does not constitute actual ComfyUI canvas/upload/multi-node workflow acceptance; those remain the next user test after an authorized deployment. The harness itself has no native upload endpoint. Production ComfyUI, GitHub, Registry and package version were not changed.

## Explicit neural-upscaler model dropdown (2026-09-17)

User approved a model_name dropdown on the existing JR Neural Latent Upscaler, not a separate Loader or bundled checkpoint. Added an optional final COMBO with default auto, preserving required widget order, existing ports and omitted-argument API workflows. Input/lifecycle skills informed append-only schema compatibility, native input-key cache invalidation when the selection changes, and fail-closed explicit names. Listing reads filenames only; schema remains available with auto when no model folder exists. Actual upscale retains the existing missing-model error.

Named selection must be in ComfyUI's registered H3-upscaler file list and bypasses dtype-based ranking; no arbitrary path loading or fallback to another checkpoint. Internal loading, normalization, network, memory management and output dtype/device remain unchanged. Status distinguishes actual model from requested selection; identity-size execution explicitly reports no checkpoint loaded. Progressive's exact-size helper still uses default auto. Same-name checkpoint replacement is not a new hot-reload feature; restart to clear existing path-keyed weight caches.

Validation: 1058 CPU tests passed, 15 CUDA/opt-in tests skipped; Ruff passed; smoke registered 29 nodes and checked 25 workflows with the four known legacy stale-link warnings. New regressions cover list filtering/sorting/subfolders, missing folders, explicit-vs-auto ranking, missing/invalid paths, actual synthetic SafeTensors loading with two different miniature networks, legacy omitted-parameter equivalence, identity no-load behavior and accurate status. No inference using the full downloaded checkpoint or user media was performed. No changes to production, model files, workflow files, version, GitHub or Registry in this turn.

## Progressive Guided in disk-backed infinite MV (2026-09-17)

The user's latest submitted infinite-MV workflow completed its first clip, then failed on chunk 2 because Guided rejected the nonempty video before inspecting the sequential hard-prefix mask. This was an explicit unsupported-input guard, not evidence of an attention backend, audio shift or OOM failure. User authorized development, not production deployment or GitHub publication in this turn.

Extended the existing Guided node without new slots/widgets: accept exactly JR Sequential's 12-token / 39-frame fully locked video prefix, fully unlocked empty suffix and fully locked audio. Share the driver's prefix-length constant; do not change the sequential manifest, checkpoints, seed policy, queue or trim logic. Empty unmasked T2VA remains strict; arbitrary sampled latents, soft/spatial masks, partial audio locks and incomplete schedules still fail closed before VAE/model work. Lifecycle/datatype/input skills informed early validation, immutable input tensors/metadata and stable node sockets.

Low stage area-resizes only clean prefix slices in CPU fp32 and casts back to input dtype; time is never resized. At the sigma boundary, keep the existing neural x0 lift and native resume path but reinsert original high-resolution CLEAN prefix/audio anchors, not inverse-scaled noisy context. Both stages retain masks at every native inpaint call. Installed H3 injects visual anchors using VISUAL_COND_TIMESTEP (0.999), not ordinary `(1-sigma)*anchor`; its native audio shift correction is preserved. High-stage noise remains zero as in the existing progressive implementation, including its tiny visual conditioning augmentation. Final exact prefix/audio restoration avoids normalization/dtype drift; it is not a replacement for per-step conditioning. Full-model visual continuity is still an experiment, not guaranteed by identical prefix values.

Regression: **1049 passed, 15 skipped** on CPU using installed ComfyUI dependencies. New tests use real miniature H3, PackedLayout/Guider/Euler/inpaint; inspect all model-step anchor injections and high-stage originals, fp32/fp16/bf16, fixed-configuration repeatability, scale=1 native generated-suffix equality, and confirm changing the prefix changes generated suffix predictions. Three disk-backed chunks use the actual 8s preset (57 video tokens / 320 audio ticks), checkpoint/load/guide/commit, independent refs, TST off/on and RandomNoise/DisableNoise; verify 153-frame starts and 39-frame trimming. Learned super-resolution and media codecs are replaced in these synthetic tests; no user media or full diffusion weights were used. Smoke: 29 registered nodes, 25 workflows, four existing tolerated legacy stale-link warnings. CUDA/real-VAE tests remain skipped in this CPU run.

Keep RandomNoise.noise_seed connected to SequentialAudioChunkDriver.chunk_seed; the inspected submitted graph used an unrelated unconnected seed widget. Wiring guidance and A/B checklist are in docs/H3_PROGRESSIVE_SAMPLER.md. Start a separate run_id for quality comparisons and test at least two joins before auto-queueing. This turn leaves production files, live browser graph, job cache, running service, version, GitHub and Registry untouched.

## Original tail-context output and duration-matched audio (2026-09-15)

User A/B isolated darkening/stripes to single-frame VAE encode/decode before neural upscale or second-pass sampling. This does not establish that keyframe latents are intrinsically non-decodable, nor identify encoder versus decoder fault. User requested retaining the original multi-token window responsible for the normal endpoint IMAGE. Added append-only output index 3 `tail_context_latent`; old indices 0/1/2 and their behavior remain unchanged. Datatype/output/input skills informed storage isolation, unchanged slots and no new duration widget. Native tail mode retains at most 7 tokens (22 frames at T=7); full mode or unknown temporal contract retains the full input. Context samples bypass encode, although execution still encodes once for the legacy keyframe output. Original metadata/masks are not transplanted onto a fresh local refinement timeline. Supplied/trimmed IMAGE endpoints do not retarget the context; status and documentation warn explicitly.

Empty Audio now follows the same canonical grid as AV Builder: T=1 -> 2 audio ticks; T=7 -> 22 frames -> 37 ticks, retaining source dtype/device. No changes to AV Builder video tolerance, diffusion samplers, neural weights, Director or production files. Short-context sampling tests use native miniature H3, Euler and TST off/on, with deterministic replacement upscale and synthetic decoder; they test structure and repeatability, not image quality.

Real installed `minimax_h3_video_vae_fp16.safetensors` CUDA check passed on a fixed synthetic `[1,24,12,16,16]` latent: returned T=7 context vs tail IMAGE max absolute pixel difference 0; full native decode endpoint vs tail IMAGE difference 0. No diffusion, user prompts, image files or production changes were involved. Actual upscaled/second-pass image quality, single-frame reconstruction and reuse of a refined context as a new first-frame guide remain separate acceptance items.

Full CPU regression: 1026 passed, 15 CUDA/opt-in cases skipped; the installed-VAE test was run separately on CUDA and passed. No production synchronization, GitHub push, version change or Registry action in this development turn.

## Empty audio for standalone tail refinement (2026-09-15)

User reported little visible loss when feeding the converted tail latent into the next first_latent, then requested empty audio to pair the spatially upscaled tail with the existing AV Builder for second-pass refinement. This observation is material-specific, not proof of lossless multi-round behavior.

The official empty-video UI rounds requests to at least five frames, but native H3 PackedLayout/DiT and single-image VAE can execute a T=1 target. The new node constructs `[1,32,2,2]` zeros with `video.new_zeros`; `round(40/24)=2` is a standalone local timeline convention, not a crop at the old video's end. Datatype/input-output skills informed a one-input additive node and explicit Builder single-frame rule, with no changes to legacy video tolerance, stream order or casting. Empty audio remains an unmasked sampled stream; discard its output. The resulting split video latent remains T=1 and can be reused without another encode. Conditioning migration is not performed implicitly.

Validation against local ComfyUI 0.35.0: 999 CPU tests passed, 14 CUDA-only tests skipped; Ruff and 29-node / 25-workflow smoke passed, with the four known legacy stale-link warnings. New coverage includes real miniature H3 plus native SamplerCustomAdvanced/Euler with partial-denoise sigmas, fixed-seed repeatability, audio schedule handling, TST off/on at 0.2, split and single-image decode. Only the learned upscale in the integration test is replaced with a deterministic spatial lift; production-weight quality is not inferred from this test. This change has not yet been deployed to production or published to GitHub.

## Tail endpoint baseline (2026-09-15)

The user requested testing real video-tail-to-next-first-latent continuation before further master-resize work. Native H3 decodes a complete terminal 7-token window; the last compressed token alone uses a different attention context and its single-image decode is not the video's terminal frame. The new node therefore explicitly decodes the endpoint and encodes that IMAGE once, or reuses an already decoded endpoint. This is a correctness baseline, not fulfillment of the long-term zero-re-encode objective. No pixel/latent resize or invented detail enhancement is included. The original IMAGE workflow, existing samplers and Director semantics are unchanged.

Input/output and datatype skills informed a separate additive node, paired IMAGE/LATENT outputs, owned single-frame storage, shape validation and removal of inapplicable video masks/metadata. Native temporal decoding plus a small actual ViT decoder test cover the endpoint-context distinction without loading production weights. Full-checkpoint visual acceptance and latent resize / mixed Ref2V semantic confounders remain unresolved. No production deployment or GitHub publication is implied by these local changes.

Local validation against installed ComfyUI 0.35.0: 965 tests passed, 13 CUDA-only tests skipped in the CPU-isolated run; Ruff passed; 28-node / 25-workflow registration smoke passed (four pre-existing legacy stale-link warnings). This does not claim production-weight or TRT-engine visual validation.

## Director pre-encoded keyframe masters (2026-09-15)

Hermes uses IMAGE as wallpaper/vision and the corresponding clean H3 LATENT as reusable first/last-frame state. Input/output skills require append-only optional sockets and unchanged legacy indices; datatype handling keeps masters runtime-only, snapshots their sample tensors, validates `[1,24,1,H,W]`, and does not treat a video's final compressed time slice as an exact decoded terminal frame.

`RuntimeMedia.keyframe_latent` is an additive optional runtime field; persisted Director JSON and PIPE v2 compatibility remain unchanged. The old IMAGE-only native delegation remains intact. An opt-in adapter is necessary because native conditioning has no pre-encoded keyframe arguments; it builds native keyframes/references without VAE/CLIP monkey patches. In mixed Ref2V requests this new path retains real temporal anchors, rather than silently demoting first/last to independent references. Target audio locking remains a separate builder operation.

Resize clean master copies with native center/stretch geometry and float32 area resampling; no learned upscaling or noisy-state conversion occurs here. Original dtype/device are restored. Progressive Guided's explicit master marker selects the same transform from the master, while unmarked legacy keyframes still decode/re-encode. All-preencoded paired-image requests perform zero keyframe VAE operations. Latent-only conditioning decodes once for vision; the optimizer instead requires an IMAGE since it owns no VAE. Independent refs still use normal encoding. Initially tested on ComfyUI 0.35.0; the 2026-10-09 integration aligns GitHub CI with the validated ComfyUI 0.39.0 production source (`926d828`). Visual multi-round acceptance remains separate.

## H3 Hybrid Loader (2026-08-17)

Version 0.9.0 adds a V1 `JR_H3_HybridLoader` with two `diffusion_models` selectors and one MODEL output. Per the local ComfyUI node Skills, imports remain side-effect free, path selection is delegated to `folder_paths`, stock MODEL construction is retained, and `cached_patcher_init` contains every setting needed for Dynamic VRAM delegate/multi-GPU reconstruction.

The installed ComfyUI commit `0f1fa67ad8a68b62c65ebc97a7bf485df2459c3a` routes `comfy.utils.load_torch_file` through AIMDO `ModelMMAP` when enabled and through safetensors `safe_open` otherwise. Hybrid FL therefore uses that public current path with `return_metadata=True`; the JR code never imports or reimplements private AIMDO APIs. REF uses a bounded header parser plus selected-only `safe_open.get_tensor` calls and owned CPU clones. The resulting state dict is handed to `comfy.sd.load_diffusion_model_state_dict`, and no second REF MODEL is created.

Scott Mudge's MIT loader at commit `a44c69b02242e41fbd01e22abe2a492adc853038` supplied the experimental profile semantics and family provenance concept. Its dual-safe_open full-state construction was deliberately replaced. Current BF16, INT8 ConvRot and pruned INT8 headers demonstrate three materially different AdaLN representations; the resolver compares only selected complete families and fails closed across incompatible representations instead of requiring identical global key sets.

## Director PIPE through conditioning (2026-08-11)

Version 0.8.0 keeps the eleven stable V1 nodes and adds the deterministic official H3 prompt formatter. Runtime PIPE schema v2 carries immutable compiled/optimized/reviewed prompt stages and validated video/audio handles; persisted Director state stays schema v1 and v0.6.0 JSON remains valid. Prompt Optimizer and Prompt Review append PIPE outputs while preserving their historical STRING output indices.

The conditioning node calls the installed ComfyUI `MiniMaxH3ImageToVideo` or `MiniMaxH3ReferenceToVideo` implementation instead of copying or monkey-patching upstream code. Current native behavior fixes output at 24 fps, supports 9 images/3 videos/3 video soundtracks/3 standalone audios, lacks a distinct Driving Audio port and cannot preserve hard first/last anchors in Ref2V. These are documented capability boundaries, not emulated features.

## Director Desk and JR_H3_DIRECTOR_PIPE (2026-08-10)

Version 0.6.0 adds the tenth stable V1 node, `JR_H3_DirectorDesk`. The architecture decision is recorded in `docs/DIRECTOR_DESK_ARCHITECTURE.md`: lightweight schema-versioned workflow state is separate from the pure compiler and immutable runtime PIP. A hidden V1 STRING widget carries JSON to Python while `node.properties.jr_h3_director_state` remains the frontend persistence source; no tensor, decoded media, base64 or binary is serialized.

The frontend Skill rules directly affect this phase: the DOM editor imports only stable `scripts/app`/`scripts/api`, uses a WeakMap per node instance, commits graph state once per user transaction, restores after configure/load, cleans media/listeners on removal and never resets node size during execution. Shot and Driving Audio overlap are validation errors; Visual and Reference Audio overlap use deterministic display-only lane stacking. Lane order never controls reference labels.

The datatype Skill confirms that `JR_H3_DIRECTOR_PIPE` is a normal custom connection type. The PIP is a frozen dataclass/tuple graph and is never JSON STRING data. Prompt Optimizer receives it through one optional socket appended after all old optional inputs. With no PIP, all old behavior and status formats remain unchanged; with PIP, existing mode routing, image conversion, validator, single repair and fail modes remain authoritative.

Media import reuses ComfyUI `/upload/image`; the JR probe route and execution resolver validate only relative input/temp/output descriptors. Pillow and ffprobe are lazy execution/route dependencies, subprocess calls use argv with timeout/no shell, and import does not inspect media or run FFmpeg.

Six Luna workers independently covered architecture/PIP, frontend lifecycle/reference projects, media/security, optimizer integration, tests/failures and product UX before implementation. DaSiWa is GPL-3.0; qwenmultiangle metadata claims MIT but its audited tree lacked a LICENSE file; no third-party source was copied. Two fresh Luna reviewers are required after the implementation tests.

## Documentation reconciliation (2026-08-10)

This pass changes documentation only. No Python, JavaScript, JSON workflow, test, dependency or package-metadata file is modified.

The public documentation was reconciled against the current `main` implementation and now treats these facts as authoritative:

- Nine stable V1 node IDs are registered by root `__init__.py`; package version remains `0.5.0`.
- Prompt Optimizer defaults to `max_tokens=1800` and `timeout_seconds=180`. The model returns semantic JSON; schema failures receive at most one `temperature=0.1` structured repair before deterministic official formatting and final validation. Existing Return Original/Stop Workflow semantics remain unchanged.
- Prompt Review defaults to 3600 seconds, validates 60..86400, normalizes invalid legacy UI values to 3600, and preserves user-enlarged node dimensions.
- Router connects to Adaptive Cache through `cache_config` only. `selected_profile`/`analysis` are diagnostic outputs, and connected config replaces all manual cache widgets.
- Adaptive Cache checks the loaded native MiniMax H3 model class/structure rather than safetensors filenames. Profile selection is not a hit or speedup guarantee.
- Adaptive metric history stays on the active compute device; only large residuals obey CPU/GPU/Auto.
- Unified Acceleration order remains Sage -> Low VRAM Attention -> Chunk FFN -> Sol, with true bypass switches and lazy optional dependencies.
- Resolution divisor is a string combo `"8"/"16"/"32"` with numeric legacy-workflow validation compatibility.
- RTX uses `nvvfx.VideoSuperRes` plus the binding's `QualityLevel` values; VSR availability does not guarantee Denoise/Deblur enum availability.
- Enhanced Video Combine returns video UI assets in `gifs` for Node 2.0 compatibility, returns PNG exports in `images`, increments filenames across repeated runs, cache-busts previews, and falls back after Windows FFmpeg pipe `EPIPE/EINVAL`. Exact `4352×2880` H.264 fallback to `libx264` was locally smoke-tested.
- The complete current input/default/range contract is centralized in `docs/NODE_REFERENCE.md` to reduce duplication drift.

Historical sections below describe the development phases at their recorded dates. When a historical value conflicts with `README.md` or `docs/NODE_REFERENCE.md`, the current reference documents take precedence.

## H3 Unified Acceleration (2026-08-09)

### Goal and source audit

Added `JR_H3_UnifiedAcceleration`, a V1 Python `MODEL → MODEL` orchestration node for the fixed Sage → MiniMax H3 Low VRAM Attention → MiniMax H3 Chunk FeedForward → Sol-Attn topology. It does not integrate Turbo LoRA, ReservedVRAMSetter, SigmaShift, cache, sampler, VAE, RTX, or video output.

The requested `JR_MiniMax_H3_T2VA_FL2VA加速放大 (ver4.1).json` was not present after recursive searches of the available local workflow locations. The closest source-of-truth artifact audited read-only was `F:\ComfyUI-aki-v3\ComfyUI\user\default\workflows\JR_MiniMax_H3_T2VA加速放大 (ver4.0) .json`. Its MODEL links confirm `88 → 89 → 90 → 86`; commits are KJ `60cd6bc1870db94c6eeb05fbe455147a8e91c4e9` and Sol `0e334dc981cfe3b0ed926ee13ad43f64914b7f5b`. The ver4.0 outer subgraph uses `chunks=3` (overriding the internal widget value 2), while this task explicitly requires the validated Unified-node default `ffn_chunks=4`; the task default is implemented and the discrepancy is recorded rather than hidden. Exact ver4.1 JSON replacement by Codex remains NOT RUN because that artifact was unavailable; this is separate from the completed user GPU acceptance below.

Installed/current upstream audit:

- ComfyUI-KJNodes installed `60cd6bc1870db94c6eeb05fbe455147a8e91c4e9`; official main returned the same SHA on 2026-08-09. Its working tree had a pre-existing untracked `config.json`, which was not modified.
- ComfyUI-SolAttn_triton installed `842c4eaa7d91dbaef3fee3ccdbf36a39521e82fc`; official main returned the same SHA. Its reference-to-current changes are in kernel files; the node API is unchanged.

### Skills applied

Read the local basics, inputs, outputs, datatypes, lifecycle, packaging, and migration Skill files. This phase uses V1 `INPUT_TYPES`/`FUNCTION`/`RETURN_TYPES`, stable global node IDs, exact MODEL output arity, an optional `forceInput` STRING, root registration, and import-safe execution-time dependency validation. No frontend Skill was needed because the node has no custom JavaScript or UI output.

### Architecture and compatibility decisions

`nodes/h3_unified_acceleration.py` contains only the V1 panel/orchestration and one compact success log. `utils/h3_acceleration_adapters.py` centralizes the upstream node IDs, Sage modes, runtime registry lookup, keyword-signature validation, error context, H3 structure check, and normalization of direct MODEL, `(MODEL,)`, and `io.NodeOutput(MODEL)` results.

All upstream resolution is lazy. Global disable returns the exact original model before model/dependency validation. Each subsystem switch is a true bypass, so disabled Sage/Sol never resolves that dependency. No CUDA, Triton, Sage, KJNodes, Sol, or ComfyUI registry import occurs when the JR package is imported.

Sage precedes Sol because Sage installs `optimized_attention_override`; Sol clones afterward and captures it as `previous`, so ineligible/dense fallback delegates to Sage. KJ Low VRAM publishes `sol_take_forward` and marks its optimized-attention forward for composition. The wrapper calls the real upstream nodes in the verified order and does not recreate these details. FFN has an explicit enable because calling upstream with `chunks=1` is not equivalent to a true disabled layer and would still bind the dependency/API.

No upstream source is vendored. KJ's audited root license is GPL-3.0. Sol has no explicit license file, packaging metadata, or source header in either audited commit, so its status is recorded as “No explicit license confirmed” and its source is not redistributed.

### Luna Max work

- Task A independently audited installed/reference/current KJ and Sol APIs, clone/object-patch/model-options behavior, return styles, composition, commits, and licenses.
- Task B searched for and audited the available workflow JSON, verified the link topology and parameters, and identified the missing ver4.1 artifact plus ver4.0 value/order discrepancies.
- Task C independently designed the CPU/mock test matrix for ordering, switches, forwarding, normalization, dependency errors, drift, non-H3 handling, lazy imports, and registration regression.
- Task D independently reviewed the finished implementation and reported Critical 0 / High 0 / Medium 3 / Low 3. Medium M1 was handled by strengthening the H3 preflight to require the real attention and FFN block structure; M3 was handled by reserving “API drift” for signature binding and reporting execution-time TypeError with the normal upstream failure context. M2 was reduced with a full-chain clone/composition mock that preserves the Sage previous marker, LowVRAM `sol_take_forward`/attention object patch, and FFN object patch through Sol. Its remaining real-kernel/GPU aspect is honestly NOT RUN. Low findings concern future NodeOutput shapes and additional end-to-end dependency/GPU breadth; they do not change the current audited APIs.

### Validation and limitations

Targeted CPU/mock tests cover exact order, true bypasses, all Sol parameters, Sage/LowVRAM/FFN forwarding and bounds, `tau_profile` None/empty/multiline states, MODEL/tuple/NodeOutput normalization, missing dependencies, import errors, API drift, non-H3 models, and root registration. Full-suite, lint, compile, production import, and GPU statuses are recorded from their final commands rather than inferred.

User-performed real GPU acceptance is now complete: RTX 4080 SUPER 16GB, about 0.8MP native, 15 seconds, about 2.4MP after JR RTX upscale, about 8 minutes total — USER-VALIDATED PASS; RTX 5090 32GB, 1.5MP native, 15 seconds, about 2.4MP after JR RTX upscale, about 11 minutes total — USER-VALIDATED PASS. The 5090 workload uses substantially higher native resolution, so the two total times are not an apples-to-apples GPU benchmark. The user also compared the Unified wrapper with the equivalent four-node KJNodes/Sol-Attn chain and observed no meaningful runtime regression; no precise percentage is claimed.

The user observed OOM on comparable high-resolution/long-video targets before using the current Turbo + Unified Acceleration + VRAM optimization workflow. The two configurations above completed, but they are validated working points rather than hardware maxima or a universal no-OOM guarantee. Native generation below roughly 0.6MP is not recommended by the user as the main high-quality starting point when substantial post-upscaling is required; this is workflow experience, not an official MiniMax limit. Codex automated regression and import tests remain distinct from these user-performed GPU tests. Per-toggle real-GPU experiments, exact peak VRAM, strict cold/warm timing, and exact ver4.1 JSON replacement remain NOT RUN/NOT MEASURED unless separately executed.

## Local ComfyUI skill rules applied

The local skills under `C:\Users\Admin\.agents\skills\comfyui-custom-node-skills` were read before implementation: basics, inputs, outputs, datatypes, lifecycle, packaging, migration, and frontend. The frontend skill is used by Enhanced Video Combine and Prompt Review & Continue.

- Phase 1 deliberately uses the V1 Python node API requested by the task: `INPUT_TYPES`, `FUNCTION`, `RETURN_TYPES`, tuple results, and root `NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS`.
- Node IDs use the globally unique `JR_H3_` prefix and should remain stable after release.
- Execution parameters match input IDs; optional values have defaults. Every data result matches the declared output count and order.
- IMAGE values are tensors shaped `[B,H,W,C]`. Tensor existence is checked with `is not None`, not truthiness. Last Frame preserves the batch dimension.
- Output-writing nodes use `OUTPUT_NODE = True`; video encoding uses `IS_CHANGED` so queuing creates a fresh output.
- V1 UI results use `{"ui": ..., "result": (...)}`. Enhanced Video Combine publishes complete `gifs` and `images` asset descriptors and exposes `WEB_DIRECTORY = "./js"` for its DOM preview widget.
- Frontend extensions import only the stable `scripts/app` and `scripts/api` modules, preserve existing node lifecycle callbacks, prevent DOM interactions from reaching the canvas, and release the video element when a node is removed.
- Prompt Review & Continue accepts a legacy multiline STRING plus optional Director PIPE, uses a non-serialized DOM editor, `UNIQUE_ID`, and `IS_CHANGED = NaN`. Its WebSocket event is sent only to the executing client ID; a bounded thread-safe state store and short interruptible waits prevent stale reviews and allow ComfyUI Stop to cancel execution.
- Custom POST/GET routes are registered once per PromptServer instance. Route handlers never perform long synchronous waits or log submitted review text.
- Validation that depends on actual tensors, CUDA, FFmpeg, HTTP, or optional SDKs occurs only during execution.
- Imports do not contact HTTP services, run FFmpeg, initialize CUDA, or import `nvvfx`.
- ComfyUI already supplies torch, NumPy, and Pillow, so they are not duplicated in ordinary requirements.
- V3 migration is optional future work. The skill does not identify a requirement that forces V3 for this suite.
- Adaptive Cache remains V1 at the node boundary but uses the current ModelPatcher clone, keyed diffusion wrapper, keyed cleanup callback, and native `patches_replace["dit"]` Block hook. Its state is attached to one cloned patcher, never stored in an unprotected global.
- Adaptive Cache treats sampled fp32 metric history as lifecycle state on the active tensor device. Only large reusable residuals obey CPU/GPU/Auto placement; cleanup and invalidation release both classes without retaining graphs or full-tensor backing storage.
- Sampling invalidation uses semantic/structural signatures, not transient tensor `data_ptr()` values. ComfyUI may reconstruct equivalent conditioning tensors during one denoise run; cleanup and timestep restart provide the run boundary while seed, model, layout, reference structure, shape, dtype, device, and batch protect correctness.
- Cache thresholds must be calibrated in this implementation's own relative-delta scale. v0.3.3 uses native 25-step H3 measurements and logs input/probe count/min/average/max; profile selection alone never bypasses audio/video vetoes or forced-refresh limits.
- The production MiniMax H3 implementation was inspected read-only: `MiniMaxH3Model` defaults to 50 joint packed audio/video blocks, target audio then target video are the final packed segments, and the core prefetch queue advances outside Block replacement callbacks. Skipped blocks therefore still receive balanced prefetch pop/cleanup calls.
- `JR_H3_CACHE_CONFIG` is an immutable Python object. Router results override every manual cache widget; Prompt Optimizer retains its first three historical STRING output indices and appends one Director PIPE output.

## Licensing decision

The task described DaSiWa as Apache-2.0, but the required shallow clone resolved to commit `a297af20318dfb7d8bdd2295a920172437551036`, whose root `LICENSE` is GPL-3.0. No DaSiWa source was copied or ported into this Apache-2.0 project. The three corresponding nodes were independently written from the task's functional specification, with upstream names/docs consulted only to understand expected behavior. See `THIRD_PARTY_NOTICES.md`.

The OpenAI request layer is independent. The H3 prompt constraint strategy was reorganized and rewritten after reviewing the signerzwb reference, as recorded in `THIRD_PARTY_NOTICES.md`.

## MiniMax H3 official-prompt integration (2026-08-08)

### Task background and provenance

This phase adds a local MiniMax H3-oriented Prompt/Context Preprocessor while keeping the historical node ID, three outputs, and saved-workflow widget positions stable. The audited upstream is [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3), branch `main`, pinned to commit `8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea` (retrieved 2026-08-08). The source paths and hashes are recorded in [`resources/minimax_h3_spec/UPSTREAM.json`](resources/minimax_h3_spec/UPSTREAM.json); the local directory contains metadata only.

The upstream GitHub repository had no root `LICENSE` file. Its README links to the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE), whose territory limits/exclusions cover the US, EU, UK, and Korea. The fixed project decision is therefore to redistribute no official guide prose or examples and to ship only clean-room metadata and interoperability facts. This is not an endorsement and does not grant rights beyond that license.

### Architecture and compatibility

The old implementation was a single Chinese image-to-video prompt template. The new path separates a JR Creative Director layer (`JR_DIRECTOR_PROFILES`) from a clean-room official-format layer. The Director supplies profile direction and continuity priorities; the official layer supplies published section names, label syntax, ordering, timing/alignment facts, and retention taxonomies. JR profile names are local names, not MiniMax format names.

`utils/h3_prompt_modes.py` resolves Auto and validates the five generation modes; `utils/h3_reference_registry.py` assigns deterministic labels; `utils/h3_prompt_validator.py` validates section order, shots, references, timing, and preserved literals; and `utils/h3_prompt_builder.py` composes the system/context and user prompts. The node in `nodes/h3_prompt_optimizer_official.py` registers first/last anchors and reference slots, sends IMAGE payloads, validates the returned text, and keeps Return Original/Stop Workflow failure behavior. `nodes/h3_openai_prompt_optimizer.py` remains the historical import path.

Legacy required widgets remain the exact prefix and `h3_input_mode` plus `reference_instructions` are appended. Legacy optional `api_key` and `ref_image_1` through `ref_image_9` remain the exact prefix; `first_frame` and `last_frame` are appended. Existing callers may omit all new optimize arguments. API roots, `/v1`, and full endpoint paths normalize to one `/v1` segment. A 400 caused by optional reasoning fields retries once without those fields; other HTTP failures are not retried.

### Boundaries and limitations

The node is not MiniMax's hosted proprietary H3-Context-IR and does not reproduce or replace it. It uploads connected IMAGE inputs to the configured OpenAI-compatible service. Video and Audio labels may be declared in `reference_instructions` for downstream context, but this node does not upload or universally understand binary video/audio; backend and downstream support decide whether those references are usable. Ref2VA prompts may require a `max_tokens` value above the default 1800 for complex descriptions.

The optional local-LLM integration smoke reached the configured local service at `http://127.0.0.1:10000`, selected `Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf`, generated a 952-character T2VA prompt, and passed strict validation. An initial response omitted `[Shot 1]`; adding a clean-room minimum syntax skeleton to the system contract made the repeat deterministic smoke pass without weakening validation. The default pytest suite remains offline and uses mocked HTTP where network behavior is tested.

Version 0.4.1 adds one constrained format-repair pass after initial validation failure. The repair payload uses temperature 0.1, contains the candidate, exact validation errors, authoritative contract, and protected literals, but no images or optional reasoning fields. It is validated by the unchanged full validator and cannot recursively repair. Final Return Original and Stop Workflow semantics remain unchanged.

Version 0.4.2 addresses a real local-model failure where both initial and repair responses changed `介绍一下MiniMax H3` to `介绍一下 MiniMax H3`. The repair layer now locates exact or whitespace-only literal variants, replaces them with counted immutable sentinels, and restores the original text locally before the unchanged full validation pass. Sentinel removal or duplication remains a hard failure.

Version 0.4.3 fixes a discovered contradiction: the clean-room minimum skeleton previously showed Ref2VA `subject_definitions:` content inline while the validator correctly required that first heading to stand alone. Real local outputs and their single repair therefore repeated the same invalid structure. The skeleton now shows the standalone heading and `<Subject N> is ...` definition form. The same one repair pass may deterministically canonicalize section wrappers/inline bodies, colon-style subject definitions, and clear visible/audio retention-taxonomy crossovers before the unchanged final validator. A real local qwen3.6-27b Ref2VA request passed the corrected initial contract with `repaired=0`.

### Agent split for this phase

- Luna A — upstream resource metadata and hashes.
- Luna B — input modes and reference registry.
- Luna C — validator and prompt fixtures.
- Luna D — regression coverage and user/developer documentation (this section).
- Main — audit, license decision, architecture, builder/node integration, and deployment.
