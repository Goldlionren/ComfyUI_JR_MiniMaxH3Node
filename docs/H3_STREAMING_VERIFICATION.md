# Streaming engineering verification — 2026-09-14

## User acceptance and main integration

Following the 10s/15s extension at `5cd1792`, the user reported testing without issues and explicitly approved merging into main and publishing to GitHub. The engineering and deployment records below describe earlier checkpoints; their no-merge/no-push restrictions applied before that approval. Keep the experimental labels, legacy workflows and first-pass full-story generation unchanged. This acceptance does not establish identical quality on all hardware, masks or prompts; Sparse KV/every_4 has a user-observed mild background-drift trade-off.

## 10s/15s second-pass extension

Implementation baseline: `d13a0a4`, clean `feature/taomate-streaming`. Added duration presets only to the experimental planner/runtime; the full-story first pass, existing Hard Prefix, neural upscale and Unified algorithms are unchanged. Planner socket indices 0/1 remain unchanged; exact length outputs are appended. See `H3_STREAMING_SAMPLER.md` for duration setup and the TaoMate sigma audit. No upstream sigma/renormalization implementation was copied or enabled.

- Targeted plan/cache/runtime: **53 passed**. Full suite: **908 passed, 1 skipped**, 27.65s; skip remains the non-CUDA dependency case. Ruff for changed Python files and `git diff --check` passed. Import/workflow smoke passed: 27 nodes, 25 workflows, same four tolerated legacy stale-link records.
- Native tests cover 124/243/362 frames, no duplicate prefix/gaps, exact audio rounding, full-timeline RoPE slices including window boundaries and last-frame conditioning, three bounded retention policies, unchanged per-phase SIGMAS, exactly one noise draw per execution, repeatability, locked audio/final video slice, metadata/mask preservation, and failure during the second window followed by a successful retry.
- Independent RTX 5090 test: **15s, 50-layer reduced-width random H3**, not full checkpoint weights. Installed Sage FP8++, LowVRAM head groups, Chunk FFN and Sol; full CPU KV with TST off/on 0.2, plus GPU Sparse KV/every_4 (13/50 layers). All settings repeated with AV max differences `[0,0]` and bit-identical locked audio. Each execution completed 36 denoise / 12 clean forwards. Sol reported no errors; its sparse kernel was ineligible and dense delegation remained active.
- The same three RTX configurations also passed on a two-layer reduced model for both 10s and 15s. The 10s case completed 24 denoise / 8 clean forwards per execution; repeats had AV max differences `[0,0]`, locked audio unchanged and no Sol errors. Reports live in the task directory as `smoke-long-10s.json`, `smoke-long-15s.json` and `smoke-long-15s-50layers.json`.
- In that reduced model, peak post-trim KV was 72.266 MiB for full CPU cache and 18.789 MiB for GPU sparse cache; bounds were 144.531 / 37.578 MiB. Those are not full-H3 memory or speed estimates. One/two-phase audio rounding variations can slightly change retained-byte peaks without unbounded growth.
- The initial sandbox GPU attempt was blocked by writes to the normal Triton compilation cache; moving its environment path to a Chinese-named directory exposed an installed Triton Unicode decoding issue. The unchanged GPU test passed in a normal-permission independent process using the existing cache. No production debug flags, dependencies or kernels were modified to work around these test-environment issues.

Reproduce (in addition to the suite commands below):

```powershell
& 'F:\ComfyUI-aki-v3\python\python.exe' tools/smoke_h3_streaming.py --comfy-root F:/ComfyUI-aki-v3/ComfyUI --unified-plugins-root F:/ComfyUI-aki-v3/ComfyUI/custom_nodes --layers 50 --preset 'JR 15s Experimental' --report '../smoke-long-15s-50layers.json'
```

This verifies execution structure, not full-checkpoint 10s/15s quality, real peak process RAM, TRT long-decode performance or seamless continuity. User acceptance still needs first-pass full duration and matching planner, then realistic-resolution Sparse KV/every_2 or every_4. Full first-pass/noise/source/output/final-decode allocations remain duration-dependent. Production deployment and restart state are recorded separately outside the repository; GitHub is not pushed.

## Branch / checkpoints

The original audit below predates the user-authorized production test deployment. See the post-audit validation at the end for the current real-checkpoint test results and output-boundary fix.

`feature/taomate-streaming`, based on synchronized main `c9b1526405d9013476e61941f6c81e1b23b9635b`. Main and the production directory remain on their original implementation. The source implementation checkpoint is `8a10404`; later commits on this branch contain test harness, smoke tooling and documentation. Resolve final documentation HEAD with `git rev-parse feature/taomate-streaming` (a document cannot contain its own commit hash).

Logical commits: audited geometry planner (`e25a436`); transactional bounded cache (`e3efa39`); runtime/clean commit/history attention/sparse integration (`8a10404`); verification and documentation follow separately. No remote feature push or merge is performed.

## Files / architecture

Added: `nodes/h3_taomate_chunk_planner.py`, `nodes/h3_streaming_sampler.py`; `utils/h3_stream_plan.py`, `utils/h3_stream_cache.py`, `utils/h3_stream_attention.py`, `utils/h3_streaming_sampler.py`; corresponding three `tests/test_h3_stream*.py` modules; `tools/smoke_h3_streaming.py`; these three streaming documents.

Modified: root `__init__.py` (two additive registrations), `tests/test_import_registration.py` (27-node expectation), `tests/conftest.py` and two old resource/validator test imports (native ComfyUI namespace/server import context). No protected production sampler/upscaler/Unified/TST implementation or saved workflow is changed.

The immutable plan feeds one execution-owned runtime. The runtime slices full PackedLayout positions, delegates native phase Euler evaluation, intercepts post-RoPE QKV, commits only extra clean-forward target AV, and trims a transactional BF16 cache. The two public nodes, exact geometry, clean semantics, retention, streaming/sparse attention, acceleration integration and first workflow are detailed in `H3_STREAMING_SAMPLER.md`.

## Commands

From the feature package directory, using installed Python 3.13 and the configured ComfyUI/test-tool import paths:

```powershell
$env:PYTHONPATH='F:\ComfyUI-aki-v3\ComfyUI;F:\AI\custom_nodes\ComfyUI_JR_MiniMaxH3Node\.test-tools'
& 'F:\ComfyUI-aki-v3\python\python.exe' -m pytest -q --basetemp='<fresh task-local test directory>'
& 'F:\ComfyUI-aki-v3\python\python.exe' -m ruff check .
& 'F:\ComfyUI-aki-v3\python\python.exe' tools/ci_smoke.py --comfy-root F:/ComfyUI-aki-v3/ComfyUI
& 'F:\ComfyUI-aki-v3\python\python.exe' tools/smoke_h3_streaming.py --comfy-root F:/ComfyUI-aki-v3/ComfyUI --unified-plugins-root F:/ComfyUI-aki-v3/ComfyUI/custom_nodes --layers 50
git diff --check
```

## Results and limitations

- Final full suite after cleanup hardening: **893 passed, 1 skipped**, 22.89s. Runtime/cache subset: **27 passed**. Geometry plus cache plus execution suite: **38 passed**. Skip is the non-CUDA RTX dependency path on a CUDA machine.
- Ruff passed. Import/workflow smoke: **27 nodes, 25 workflows**, sequential-video and strict ComfyTV links passed. Four pre-existing tolerated legacy stale-link records remain unchanged.
- Cache unit tests run 100 commits for each retention policy and assert a stable byte plateau; rollback, staged-head/layer completeness, owned storage, audio reset and cleanup are covered. This is bounded-cache evidence, not a long full-checkpoint streaming VRAM benchmark; continuation is disabled.
- Real native model/guider tests: all six modes, full 50 layers, sparse 13/50, no NaN/Inf, correct AV reconstruction, global video/audio positions, native baseline parity, clean-forward output invariance, locked audio, references+tail anchor+TST, repeated seeds, injected post-stage failure followed by a successful job, masks/dtypes/invalid inputs.
- RTX 5090 smoke uses **50 layers but a reduced 512-hidden/4-head random H3**, 8×8 spatial video latents, not full H3 weights. Installed Sage FP8++, LowVRAM head_chunks=4, FFN and Sol are exercised. All three settings repeated with AV max differences `[0.0, 0.0]`; locked audio is bit-exact. Each request reports 12 denoise forwards and 4 clean forwards.

| RTX structural case | Final retained KV | Peak CUDA allocated | Notes |
|---|---:|---:|---|
| Full 50-layer, CPU KV, TST off | 58.789 MiB | 325.4 MiB | 50/50 layers |
| Full 50-layer, CPU KV, TST 0.2 | 58.789 MiB | 325.5 MiB | clean pass bypasses TST |
| Sparse every_4, GPU KV, TST 0.2 | 15.285 MiB | 350.3 MiB | 13/50 layers, 74% KV byte reduction |

The sparse GPU case has a higher GPU peak than CPU full-cache because storage location differs; do **not** claim 74% total VRAM reduction. Retained full KV reached 72.070 MiB at phase 2 then fell to 58.789 MiB at phase 3. Sol stats across these runs: sparse=0, dense_fallback=3200, outside_range=9600, errors=0. Therefore actual delegation works, but Sol sparse speedup is not demonstrated. Toy timings were 4.295/5.888/4.177s for the listed first runs; these are not comparable full-model speed/quality benchmarks and include warmup/environment effects.

Local ComfyUI is `19e1058f4c445ef74047e77a23f9ca7684c1e4b6`. During full-suite setup its new `utils.mime_types` exposed two ambiguous legacy test imports; test-only qualification/path setup fixes them. The offline test server context registers routes but starts no listener. Production files were not patched to fix the test harness. GitHub CPU CI on the older pinned core is not claimed as run for this unpushed branch.

## Outstanding quality/performance acceptance

Full real checkpoint + TaoMate LoRA, realistic resolutions, 4080S/5060Ti/3060, long continuation, real decode of generated output and subjective phase-boundary continuity are not validated here. CPU cache transfer overhead, large full-width KV, near-clean epsilon, conditioning isolation, sparse approximation and caller LoRA choice remain explicit risks. This is a testable experimental branch, not production approval or an official TaoMate numerical port.

## Rollback

No production rollback is needed: production was not deployed. Main development checkout at `F:\AI\custom_nodes\ComfyUI_JR_MiniMaxH3Node` remains main at the checkpoint. To inspect the stable baseline in the feature worktree without deleting feature commits (only when the worktree is clean):

```powershell
git -C 'C:/Users/Admin/Documents/Comfyui本地开发/taomate-streaming/ComfyUI_JR_MiniMaxH3Node' switch --detach c9b1526405d9013476e61941f6c81e1b23b9635b
```

To return, use `git switch feature/taomate-streaming` in that directory. Do not reset, force-push, or overwrite the production plugin.

Skills used: ComfyUI basics/datatypes/lifecycle informed additive stable V1 registration, immutable plan/custom socket, preserved AV metadata and execution-local transactional cleanup. These constraints did not require a migration of existing nodes.

## Post-audit production validation — 2026-09-14

The user subsequently authorized a backed-up production test deployment and debugging of its first real-checkpoint failure. The original main branch, original workflows, dependencies, models and TRT engines were not changed; no GitHub push was made.

The uncorrected Geometry Only path reproduced a CUDA illegal-address error at ComfyUI's node-final `reset_cast_buffers()` / offload-stream synchronization, followed by a cleanup failure that terminated `prompt_worker`. Disabling second-stage Sol alone did not fix it. A fresh process with `CUDA_LAUNCH_BLOCKING=1` completed the same Sol-off workflow, supporting an asynchronous lifecycle issue rather than proving a particular attention kernel was defective.

Unlike native `SamplerCustomAdvanced`, Geometry Only returned the sampler's GPU AV result directly. It now calls `result.to(comfy.model_management.intermediate_device())` before leaving the node. This preserves NestedTensor streams and metadata and follows the native output handoff. No global synchronization/debug setting is kept, and no denoising, KV or attention math was changed. The precise internal allocator/kernel race was not independently localized; the missing native handoff and its corrective effect were tested.

Validation after this change:

- Full suite: **894 passed, 1 skipped**; streaming/plan/cache subset: **39 passed**; Ruff and diff whitespace checks pass.
- Real Hybrid/pruned INT8 H3, first-stage Turbo LoRA + Progressive scale 0.6, second-stage TaoMate 3-step LoRA, 124 frames at 512×288. TST off; Sage FP8++, LowVRAM 4 head groups, FFN and Sol enabled. External TRT final decode.
- `CUDA_LAUNCH_BLOCKING` absent; original production launch flags retained. Geometry Only completed with audio/video output, 35.95s for the whole cold workflow.
- Streaming Attention completed in the same process: 4 phases, 12 denoise forwards, 4 clean forwards, 50/50 CPU KV layers, previous_only, 8192 MiB budget. Node time 25.212s; retained KV peak across phase reports 2540.234 MiB, final retained KV 1063.672 MiB. These are retained-cache measurements, not KV-plus-staging peak or total process memory.
- Switching back from Streaming to Geometry Only also completed. The seed was incremented by one to force actual resampling rather than count a cached result as a repeat test; this checks lifecycle reuse, not fixed-seed identity.
- ffprobe confirms both first Geometry and Streaming outputs contain **124 video frames, 512×288, 24fps and a 32kHz audio track**. Whole-workflow timing is not comparable between cold Geometry and Streaming with cached first-stage outputs. PyTorch CUDA allocator counters do not include all dynamic-VRAM/aimdo allocations; they must not be presented as total model VRAM.

These runs establish real-workflow execution for the tested case, not subjective quality, exact reproducibility, a speedup, or full-model acceptance of Sparse KV/TST/combined anchors/audio-drive. User visual review remains required. The local deployment record and full original plugin backup live outside the repository in the task's `taomate-streaming` directory.
