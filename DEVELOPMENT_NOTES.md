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
