# H3 Unified Acceleration

> 本页以当前 `JR_H3_UnifiedAcceleration` 和运行时适配器为准。该节点是外部 KJ/Sol 节点的编排层，不包含或复制其 kernels。

## 当前兼容性摘要

- Node ID：`JR_H3_UnifiedAcceleration`；显示名称：`H3 Unified Acceleration`。
- V1 Python API，输入/输出均为 `MODEL`。
- 模型检查基于已加载的 MiniMax H3 结构（`rope_freqs`、`_forward`、Blocks 的 attention/FFN 字段），不按 safetensors 文件名白名单判断。
- `enable=false` 在任何模型或依赖检查前原样返回输入 MODEL。
- 每个子系统开关是真 bypass；关闭层不会解析相应外部依赖。
- 上游节点返回的直接 MODEL、`(MODEL,)` 和单输出 `io.NodeOutput` 会统一归一化。
- 依赖缺失、签名漂移、非 H3 模型和异常返回都会明确报错，不做静默 fallback。

`JR_H3_UnifiedAcceleration`（显示名 `H3 Unified Acceleration`）是 V1 Python ComfyUI 节点，分类为 `JR MiniMax H3/Optimization`，输入和输出均为 `MODEL`。

## 固定 Patch 顺序

```text
1. PathchSageAttentionKJ
2. MiniMaxLowVRAMAttention
3. MiniMaxChunkFeedForward
4. SolAttnPatch
```

上游 Node ID 中 `Pathch` 的拼写来自 KJNodes 注册表，兼容层必须按此真实 ID 解析。Sol 必须最后执行，以便把 Sage override 保存为 previous backend，并与 KJ Low VRAM 的 `optimized_attention`/`sol_take_forward` 路径组合。

Sol-Attn 不会在 SageAttention 之后再次执行完整 attention。适用调用由 Sol sparse path 接管；不适用或 fallback 调用交给 previous dense backend，也就是此配置中的 SageAttention。

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

这一组值称为 **Validated H3 Acceleration Profile**。它们有意相对保守，优先保证画质和稳定性，不应宣传为所有硬件上的最佳设置。高级用户可以针对自己的 GPU、分辨率和内容调整 Sol sparsity、head/FFN chunking 与 sampling window。

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

工作流中使用的 Turbo LoRA 来自 [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) 和 [Larryvrh/ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)，不是 JR Unified 节点的一部分。

## 用户 GPU 验收

| GPU | VRAM | Native H3 | Duration | Final RTX Output | Approx. Workflow Time | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RTX 4080 SUPER | 16GB | ~0.8MP | 15s | ~2.4MP | ~8 min | USER-VALIDATED PASS |
| RTX 5090 | 32GB | 1.5MP | 15s | ~2.4MP | ~11 min | USER-VALIDATED PASS |

这些是用户完成的真实验收，不是 Codex 自动测试。两次 workload 不同：5090 使用了明显更高的 1.5MP 原生分辨率，因此 11 分钟与 4080S 的 8 分钟不能作为跨 GPU 同负载速度比较。两个配置是 validated working points，不是最大分辨率。

用户还确认 Unified wrapper 与等价的四节点 KJNodes/Sol-Attn chain 生成时间基本一致，未观察到有意义的 runtime regression；没有严格 benchmark，因此不提供百分比结论。用户观察到采用 Turbo + Unified Acceleration + VRAM optimization 前，相同目标的高分辨率/长视频配置曾 OOM，而上述两个配置完成运行；这不是对其他系统的无 OOM 保证。

## Resolution 与 post-processing

用户 workflow 经验是不把低于约 0.6MP 的原生 H3 输出作为需要大幅放大时的主要高画质起点；这不是 MiniMax 官方硬限制。推荐从已验证的约 0.8MP（4080S）或 1.5MP（5090）工作点开始：

```text
MiniMax H3 native generation
    → VAE Decode
    → JR H3 Resolution Scale Calculator
    → JR H3 RTX Upscaler & Refiner
    → JR H3 Enhanced Video Combine
```

## 性能记录

Sol-Attn 是实验性实现，第一次使用会编译 Triton kernels。记录性能时至少区分 cold/warm run，并记录 GPU、模型、分辨率、帧数/时长、steps、采样时间、总时间、peak VRAM（无法测量时写 `NOT MEASURED`）、Sage mode、head/FFN chunks 与 Sol 参数。
