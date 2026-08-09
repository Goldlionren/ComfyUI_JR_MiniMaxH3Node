# Development notes

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
- Prompt Review & Continue uses a force-connected multiline STRING input, a non-serialized DOM editor, `UNIQUE_ID`, and `IS_CHANGED = NaN`. Its WebSocket event is sent only to the executing client ID; a bounded thread-safe state store and short interruptible waits prevent stale reviews and allow ComfyUI Stop to cancel execution.
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
- `JR_H3_CACHE_CONFIG` is an immutable Python object. Router results override every manual cache widget; Prompt Optimizer remains unchanged at three outputs and retains its independent system prompt.

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
