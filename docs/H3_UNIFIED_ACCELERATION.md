# H3 Unified Acceleration

`JR_H3_UnifiedAcceleration`（显示名 `H3 Unified Acceleration`）是 V1 Python ComfyUI 节点，分类为 `JR MiniMax H3/Optimization`，输入和输出均为 `MODEL`。

## 固定 Patch 顺序

```text
1. PathchSageAttentionKJ
2. MiniMaxLowVRAMAttention
3. MiniMaxChunkFeedForward
4. SolAttnPatch
```

上游 Node ID 中 `Pathch` 的拼写来自 KJNodes 注册表，兼容层必须按此真实 ID 解析。Sol 必须最后执行，以便把 Sage override 保存为 previous backend，并与 KJ Low VRAM 的 `optimized_attention`/`sol_take_forward` 路径组合。

## 参数

| 参数 | 默认值 | 范围/选项 |
| --- | --- | --- |
| `enable` | `true` | 全局 bypass |
| `sage_attention` | `sageattn_qk_int8_pv_fp8_cuda++` | `disabled`, `auto`, fp16 CUDA/Triton, fp8 CUDA/++, `sageattn3`, `sageattn3_per_block_mean` |
| `allow_compile` | `false` | boolean |
| `enable_low_vram_attention` | `true` | boolean |
| `head_chunks` | `4` | 1–56 |
| `enable_low_vram_ffn` | `true` | boolean |
| `ffn_chunks` | `4` | 1–64 |
| `ffn_seq_threshold` | `4096` | 256–262144, step 256 |
| `enable_sol_attn` | `true` | boolean |
| `tau` | `1.3` | 0–4, step 0.05 |
| `start_percent` | `0.2` | 0–1 |
| `end_percent` | `0.9` | 0–1 |
| `min_tokens` | `4096` | 0–1048576, step 512 |
| `int8_qk` | `true` | boolean |
| `int8_pv` | `true` | boolean |
| `sink_conditioning` | `exact_kv_and_rows` | `exact_kv`, `exact_kv_and_rows`, `off` |
| `morton` | `false` | boolean |
| `morton_curve` | `2d_frame` | `3d`, `2d_frame` |
| `verbose` | `false` | boolean |
| `use_tma` | `false` | boolean |
| `dense_blocks` | empty | block selection string |
| `tau_profile` | unconnected / `None` | optional force-input STRING; empty and multiline text remain distinct |

## 依赖与失败行为

- KJNodes 提供 Sage、MiniMax H3 Low VRAM Attention 和 Chunk FeedForward。
- SageAttention/sageattn3 由 KJNodes 在选中相应 Sage mode 时导入。
- ComfyUI-SolAttn_triton 提供 Sol-Attn；Triton kernel 可用性由上游执行阶段验证。
- JR package import 不解析这些依赖，也不初始化 CUDA/Triton。
- 关闭某一层不会解析或调用该层。
- 启用层缺失、签名漂移、返回值异常或运行失败都会带层名和上游 Node ID 明确报错；不会 silent fallback。
- 非 H3 MODEL 在应用任何 patch 前被拒绝，避免 `blocks[0]` 等不可读错误。

## 不包含的功能

Turbo LoRA、ReservedVRAMSetter、MiniMaxH3SigmaShift、EasyCache/JR Adaptive Cache、Sampler、VAE、RTX Upscaler 与视频合成都不属于此节点。

## 性能记录

Sol-Attn 是实验性实现，第一次使用会编译 Triton kernels。记录性能时至少区分 cold/warm run，并记录 GPU、模型、分辨率、帧数/时长、steps、采样时间、总时间、peak VRAM（无法测量时写 `NOT MEASURED`）、Sage mode、head/FFN chunks 与 Sol 参数。
