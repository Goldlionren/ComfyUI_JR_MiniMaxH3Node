# Development notes

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

### Agent split for this phase

- Luna A — upstream resource metadata and hashes.
- Luna B — input modes and reference registry.
- Luna C — validator and prompt fixtures.
- Luna D — regression coverage and user/developer documentation (this section).
- Main — audit, license decision, architecture, builder/node integration, and deployment.
