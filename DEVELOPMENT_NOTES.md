# Development notes

## Local ComfyUI skill rules applied

The local skills under `C:\Users\Admin\.agents\skills\comfyui-custom-node-skills` were read before implementation: basics, inputs, outputs, datatypes, lifecycle, packaging, and migration. The frontend skill was not needed because standard ComfyUI output preview payloads are sufficient.

- Phase 1 deliberately uses the V1 Python node API requested by the task: `INPUT_TYPES`, `FUNCTION`, `RETURN_TYPES`, tuple results, and root `NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS`.
- Node IDs use the globally unique `JR_H3_` prefix and should remain stable after release.
- Execution parameters match input IDs; optional values have defaults. Every data result matches the declared output count and order.
- IMAGE values are tensors shaped `[B,H,W,C]`. Tensor existence is checked with `is not None`, not truthiness. Last Frame preserves the batch dimension.
- Output-writing nodes use `OUTPUT_NODE = True`; video encoding uses `IS_CHANGED` so queuing creates a fresh output.
- Validation that depends on actual tensors, CUDA, FFmpeg, HTTP, or optional SDKs occurs only during execution.
- Imports do not contact HTTP services, run FFmpeg, initialize CUDA, or import `nvvfx`.
- ComfyUI already supplies torch, NumPy, and Pillow, so they are not duplicated in ordinary requirements.
- V3 migration is optional future work. The skill does not identify a requirement that forces V3 for this suite.

## Licensing decision

The task described DaSiWa as Apache-2.0, but the required shallow clone resolved to commit `a297af20318dfb7d8bdd2295a920172437551036`, whose root `LICENSE` is GPL-3.0. No DaSiWa source was copied or ported into this Apache-2.0 project. The three corresponding nodes were independently written from the task's functional specification, with upstream names/docs consulted only to understand expected behavior. See `THIRD_PARTY_NOTICES.md`.

The signerzwb prompt optimizer was treated as behavioral reference only. Its source, comments, prompts, and long strings were not copied.
