# JR MiniMax H3 Hybrid Loader

`JR_H3_HybridLoader` 输出一个标准 ComfyUI `MODEL`。输入是 `diffusion_models` 下的 FL2VA/REF2VA checkpoint selector，不是两个已经实例化的 MODEL。

## Architecture

```text
FL checkpoint
  -> ComfyUI load_torch_file(return_metadata=True)
  -> native AIMDO/mmap when enabled, stock fallback otherwise
  -> full authoritative FL state dict

REF checkpoint
  -> bounded safetensors header scan
  -> deterministic HybridPlan / tensor-family compatibility
  -> get_tensor(selected keys only)
  -> owned CPU copies
  -> close REF

FL state dict + selected replacements
  -> comfy.sd.load_diffusion_model_state_dict(..., metadata=FL metadata)
  -> one MODEL
```

The loader never calls `load_torch_file` for REF in Hybrid mode and never constructs a second REF model/state dict. Unselected FL tensors are not cloned by the plugin. Pure FL and Pure REF short-circuit directly to `comfy.sd.load_diffusion_model`, without opening the other selector.

If both hybrid selectors resolve to the same checkpoint, the node emits a warning and performs a stock FL load. There is no meaningful cross-checkpoint overlay in that case.

## Profiles

| Profile | REF source | FL source |
| --- | --- | --- |
| Recommended | blocks 25–49 `adaln_proj.linear` families | blocks 0–24, Final AdaLN, output heads, all other weights |
| All Block AdaLN | blocks 0–49 AdaLN | Final AdaLN and all other weights |
| All Block AdaLN + Final | blocks 0–49 plus Final AdaLN | all other weights |
| Custom Range | selected inclusive block range; optional Final | all remaining families |
| Pure FL | none | stock-load the FL checkpoint only |
| Pure REF | stock-load the REF checkpoint only | none |
| Advanced Custom | `custom_ref` prefix/glob minus family-level `custom_fl` overrides | all remaining families |

Recommended 25–49 originates from experimental analysis in Scott Mudge's MIT project, not from MiniMax. It is a research profile, not a universal quality guarantee.

## Tensor families and quantization

The resolver discovers the direct parameter members that actually exist under a selected module stem. A selected quantized weight therefore co-travels with its present `weight_scale`, `.comfy_quant`, bias and other recognized weight/bias metadata. Selected FL and REF families must have identical member suffixes, shapes and dtypes. Unsupported cross-quant overlays fail before full FL or selected REF tensor reads. Non-selected global key differences are allowed.

Verified locally by header inspection:

- BF16 H3 pairs: compatible selected families.
- current INT8 ConvRot pairs: compatible `weight` + `weight_scale` + `.comfy_quant` + bias families.
- current pruned INT8 ConvRot pairs: compatible pruned AdaLN family representation.
- INT8 and pruned INT8 cross-pairs: rejected as incompatible.
- W4A8/NVFP4: no matching local H3 checkpoint was installed, so support is NOT RUN and remains fail-closed.

Do not mix checkpoint formats merely because filenames look related. The header is authoritative.

## Memory boundary

Only selected REF tensors become owned CPU copies, but AdaLN size depends on the checkpoint representation. On the locally installed files, Recommended selected bytes were about 12.1 GiB for BF16, 6.1 GiB for ordinary INT8 ConvRot, and 41.5 MiB for pruned INT8 ConvRot. These are deterministic header byte counts, not process RSS measurements. The node logs exact family/tensor/byte counts and does not claim a fixed saving percentage.

The FL state dict still has the normal memory behavior of the installed ComfyUI loader. Apply Turbo LoRA/Unified Acceleration after this loader, not before it.

## Lifecycle

The resulting patcher receives a complete `cached_patcher_init` factory so Dynamic VRAM delegate creation and multi-GPU deepclone can recreate the same plan. The factory stores only paths and node settings, never tensor payloads. Importing the plugin does not open checkpoints, read headers, initialize CUDA or access the network.

## Attribution

Preset semantics, block-range/custom selection concepts and quant-sibling provenance philosophy are adapted from Scott Mudge's MIT-licensed `ComfyUI_MinimaxH3HybridLoader`, audited at commit `a44c69b02242e41fbd01e22abe2a492adc853038`. The JR selective-reader architecture is different: native full FL plus selected-only owned REF overlay. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
