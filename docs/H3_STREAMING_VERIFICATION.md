# Streaming engineering verification — 2026-09-14

## Branch / checkpoints

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
