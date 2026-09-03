# H3 Temporal Chunk Sampler

`JR_H3_TemporalChunkSampler` 顺序采样普通 MiniMax H3 生成的 AV latent。节点提供两个生产模式：

- `Hard AV Latent Prefix`（默认、推荐）
- `Legacy Independent Chunks`（保留升级前的末帧 AddGuide 路径）

Node ID：`JR_H3_TemporalChunkSampler`

显示名称：`JR MiniMax H3 Temporal Chunk Sampler`

分类：`JR MiniMax H3/Sampling`

## 接线与设置

```text
MODEL ───────────────────────┐
original positive ───────────┤
MiniMax H3 video VAE ────────┤
NOISE ───────────────────────┤
SAMPLER ─────────────────────┼-> JR MiniMax H3 Temporal Chunk Sampler -> output: H3 AV LATENT
SIGMAS ──────────────────────┤                                      └-> status: STRING
normal H3 AV LATENT ─────────┘
```

Hard 模式必须设置：

```text
continuity_mode = Hard AV Latent Prefix
hard_chunk_preset = 5.875s / 141 frames / 235 ticks   # 放大/低显存推荐
```

选择 Hard 时前端只显示 `hard_chunk_preset` 下拉菜单并隐藏 `chunk_duration_seconds`；选择 Legacy 时反向显示。Hard 后端完全忽略旧的自由时长数值，不受 FLOAT 两位小数显示限制。

`vae` 插口为了旧工作流兼容仍然保留；Hard 模式不执行 VAE decode/re-encode。不要把 Audio Driven latent 接入本节点的 Hard 模式；长音频驱动请使用 Sequential Audio 套件。

## Hard AV Latent Prefix

四档都固定保留同一段 39-frame / 12-video-T / 65-audio-T hard prefix：

| `hard_chunk_preset` | local video T | local audio T | fresh frames | fresh video T | fresh audio T |
|---|---:|---:|---:|---:|---:|
| 5.875s / 141 frames / 235 ticks | 42 | 235 | 102 | 30 | 170 |
| 8.000s / 192 frames / 320 ticks | 57 | 320 | 153 | 45 | 255 |
| 10.125s / 243 frames / 405 ticks | 72 | 405 | 204 | 60 | 340 |
| 14.375s / 345 frames / 575 ticks | 102 | 575 | 306 | 90 | 510 |

窗口从全局 raw frame 0 开始，并按所选 fresh frames 前进。例如 5.875s 档是 `0,102,204...`，14.375s 档是 `0,306,612...`。Chunk 1 走正常原生采样，不需要前缀。Chunk 2+：

1. 只保留上一段已采样结果的最后 12 video T 与 65 audio T。
2. 将两段 sampled tail 覆盖到当前段 AV latent 的开头。
3. 建立官方双流 `NestedTensor` mask：video 前 12 T、audio 前 65 T 为 0，其余为所选档位对应的 fresh T，并全部设为 1。
4. 从 original positive 新建 Basic Guider，委托原生 `SamplerCustomAdvanced`。
5. 原生采样完成后重新写回上一段 sampled tail，并校验两个锁定前缀逐位一致；这会消除原生 float32/H3 latent in/out 往返造成的末位数值漂移，不改变原生去噪过程。
6. Chunk 1 全量写入 CPU 全局缓冲；Chunk 2+ 只把 local video `[12:]`、audio `[65:]` 写入 fresh 全局范围。

最终不使用 `torch.cat`，不会同时保留全部 sampled chunks。输出仍是完整 CPU-backed H3 AV `LATENT`：video `[B,24,T,H,W]`、audio `[B,32,2,T]`，重叠只出现一次。

全局时间线不需要被 fresh stride 整除。最后不足一个完整窗口时，节点保持相同的 local preset 尺寸，在末尾用零 latent 补齐后交给原生采样器，并且只把真实存在的 fresh video/audio 范围写回全局 CPU 缓冲；补齐区结果全部丢弃。因此任意合法 H3 总长度都不会因为短尾块产生 gap 或 duplicate，显存峰值也不会超过所选 local window。`status` 的每段范围会报告 `tail_pad v=... a=...`。

若 video/audio 在官方允许的 ±1 tick 边界上需要不同数量的 local windows，节点仍会 fail closed，避免两条流失去共同分块时间轴。

Hard 模式不添加 `MiniMaxH3AddGuide`，多块 positive 也不得已有 `minimax_keyframes`。输入 `noise_mask` 只允许不存在、为 `None`，或为形状与双流完全一致的全 1 mask；非平凡/未知 mask（包括 Audio Driven 常见的锁定 audio mask）会明确报错。

## Legacy Independent Chunks

该选项保持升级前的生产行为：按 `chunk_duration_seconds` 在 H3 5-token / 17-frame 周期上规划无 latent overlap 的分块；Chunk 2+ 解码上一段最终 RGB 帧，并通过官方 `MiniMaxH3AddGuide(frame_idx=0)` 建立当前段 positive。每段仍重新建立 Basic Guider并调用原生 `SamplerCustomAdvanced`。

Legacy 模式允许原有任意正数近似 chunk 时长，尾块过小时仍按旧逻辑并入前一块。它继续拒绝任何 `noise_mask`，且每个非末段会产生一次额外完整 video VAE decode。

## Noise、内存与输出

- 单块原样使用输入 NOISE。
- 多块官方 RandomNoise 使用 base seed 与绝对 raw `frame_start` 派生确定性 uint64 子 seed；Hard 起点由所选 fresh stride 决定。
- 多块官方 DisableNoise 保持全零；generic/custom NOISE 因无公共 substream API 而拒绝。
- 每段结果直接写入预分配 CPU 缓冲。只保留下一段需要的小型 sampled AV tail（Hard）或一张 CPU RGB 末帧（Legacy）。
- `aggressive_memory_cleanup=false` 默认依赖引用释放和 ComfyUI allocator；开启后每段执行 GC 与 `soft_empty_cache()`，通常更慢。
- 节点没有 hidden-state/KV carry 或 global model position offset，也不声称与整段单次采样数值等价。

`status` 会报告所选 mode、固定 window/prefix/stride、chunk raw/global 范围、noise mode、前缀数量、CPU 回写以及是否使用 continuation guide。

0.17 A/B/C 实验说明保存在 [H3 Temporal Chunk Sampler 0.17 archive](H3_TEMPORAL_CHUNK_SAMPLER_0.17_ARCHIVE.md)，仅作历史参考，不会恢复到节点 UI。
