# H3 Unified Acceleration v2 (Experimental)

Node ID: `JR_H3_UnifiedAccelerationV2`. The original `JR_H3_UnifiedAcceleration` keeps its input order, defaults and implementation. Existing workflows are not automatically migrated.

Add the v2 node explicitly, connect the MODEL after loading/LoRA/sigma configuration and before sampling. Start from a MODEL without other attention, block replacement or Unified patches. Keep v2 as the final attention patch. The node composes Sage → KJ LowVRAM → KJ FFN → selected sparse backend.

## Backend selection

| Setting | Behavior |
|---|---|
| `legacy_kijai` | Conservative default. Requires the existing `SolAttnPatch` node. |
| `core` | ComfyUI `BlockSparseAttention` with comfy-kitchen Sol. Explicit unsupported configurations fail before sampling; never switches to legacy after an error. |
| `auto` | Checks explicit configuration constraints and Core availability before applying patches. Selects legacy only for known missing capability or legacy-only configuration; logs the reason. Unknown API drift/runtime errors propagate. |
| `disabled` | Adds no sparse patch; the selected Sage and KJ memory passes still apply. |

`enable_sol_attn=false` overrides sparse selection. `enable=false` returns the input MODEL unchanged without dependency checks. Disabling this node does not remove upstream patches. With `enable=true`, existing attention/block/FFN patches are rejected rather than silently replaced.

Core mode does not require Kijai Sol. Enabled Sage/LowVRAM/FFN still require their KJ nodes. GPU libraries are imported at execution time, not while importing the JR module.

## Parameters and compatibility

- Shared fields retain their names. `dense_blocks` uses legacy negative indexing and clipping, then converts to absolute indices before Core: on 50 layers, `-1` becomes `49`. Invalid syntax raises a descriptive error.
- New-node `min_tokens` default: 12288. The old node remains 4096. Set a lower value explicitly for small diagnostic runs; short sequences need not be faster sparse.
- `extra_tokens`: 0/64/128/192/256, default 256, passed explicitly to Core. Ignored on legacy/disabled because those backends do not support it. This is a quality/performance candidate, not a quality guarantee.
- Core accepts the default legacy quantization flags as unused UI compatibility fields. An explicit `int8_qk=false`, `int8_pv=false`, `morton=true`, `use_tma=true`, or nonempty `tau_profile` requests behavior Core cannot preserve: explicit core errors and auto selects legacy. `morton_curve` is irrelevant when Morton is disabled.
- Active TST means `enable_tst=true` and `tst_strength>0`. It requires legacy or sparse disabled. Strength zero is a no-op. Input MODELs already containing TST are rejected; enable it on this node instead. Existing TST validation and mathematics remain unchanged.
- Core + `allow_compile=true` is not validated and is rejected; auto chooses legacy. External compilation/wrappers are not certified by this first release.
- Core requires the native H3 model and the `attention=` block interface. Older KJ LowVRAM block forwards without this argument are rejected.
- This release rejects repeated v2 and preinstalled attention/block/FFN patch stacks. Place a single v2 on each clean MODEL branch. Its sampling callback also detects later replacement of the installed attention stack.

## What logs prove

`requested` and `resolved` report the selected backend. They do not claim that every call was sparse. The report runs when patches are created and on each model `on_pre_run`, including reuse of a cached node output. `verbose=true` enables upstream per-shape path messages.

Core may execute H3 chunked sparse, generic sparse using existing QKV, or dense attention. The H3 producer requires eligible BF16 CUDA activations, head dimension 128 and RoPE, in addition to token/block/sigma policy. Model weight format alone does not prove activation dtype. A small sequence, dense layer or out-of-window sigma intentionally uses the selected dense backend.

`start_percent/end_percent` map through the model's sigma schedule. They are not percentages of the current truncated Stage2 step count. Core also bootstraps QKV statistics with two producer passes on first use after cleanup; include that cost when benchmarking short requests.

## Validation and rollout

The 2026-09-29 development tests cover policy/legacy/TST/registration and a small native H3 model on RTX 5090. GPU tests trace actual Core chunked calls, Sage fallback for token/block/sigma exclusions, chunk sizes and repeat-after-cleanup behavior. Randomly initialized small-model tests prove execution contracts, not generated media quality or full-model performance.

Run opt-in integration tests with `JR_H3_GPU_INTEGRATION=1` and `JR_H3_KJ_ROOT` pointing to the installed KJNodes checkout, using the actual ComfyUI Python and import paths.

For production comparisons, keep Sage, LowVRAM and FFN fixed between sparse disabled, legacy and Core. Record Stage1/Stage2 separately and use identical Stage1 latents for isolated Stage2 comparisons. Compare full warm requests, not graph-cache hits. Check multiple scenes/seeds and audio/reference integrity before changing a production workflow. Legacy remains the new-node default until that quality/performance gate is passed.
