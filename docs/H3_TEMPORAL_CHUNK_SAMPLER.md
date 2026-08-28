# H3 Temporal Chunk Sampler

`JR_H3_TemporalChunkSampler` 顺序采样 MiniMax H3 AV latent，并用上一段解码后的最终像素帧约束下一段起始状态。

Node ID：`JR_H3_TemporalChunkSampler`

显示名称：`JR MiniMax H3 Temporal Chunk Sampler`

分类：`JR MiniMax H3/Sampling`

## 接线

```text
MODEL ───────────────────────┐
original positive ───────────┤
MiniMax H3 video VAE ────────┤
NOISE ───────────────────────┤
SAMPLER ─────────────────────┼-> JR MiniMax H3 Temporal Chunk Sampler -> output: H3 AV LATENT
SIGMAS ──────────────────────┤                                      └-> status: STRING
Directed Conditioning latent ┘
```

不要在外面先建立 GUIDER。节点需要原始 `positive`，因为每一段的 conditioning 不同，并且必须在应用当前段 guide 后重新建立 Basic Guider。

## 每段执行顺序

```text
Chunk 1
  original positive
    -> official BasicGuider (new instance)
    -> official SamplerCustomAdvanced
    -> decode complete Chunk 1 video latent
    -> retain only its final RGB frame on CPU

Chunk 2+
  original positive + previous decoded final frame
    -> official MiniMaxH3AddGuide(frame_idx=0)
    -> chunk-specific positive
    -> official BasicGuider (new instance)
    -> official SamplerCustomAdvanced
    -> decode complete current video latent when another chunk follows
    -> retain only its final RGB frame on CPU
```

上游 `positive` 不会被原地修改。每一段得到一个新的 Basic Guider；上一段只保留一张 CPU RGB 末帧，完整采样结果直接写入预分配的全长 CPU video/audio 缓冲区。

## 时间规划

输入必须是官方 H3 AV `NestedTensor`：

- video：`[B,24,T_video,H,W]`
- audio：`[B,32,2,T_audio]`
- `T_video = 5k + 2`
- video/audio batch、dtype、device 相同
- `T_audio` 与 24 fps video / 40 Hz audio latent 的共享时间线一致，允许官方编码边界的 ±1 tick

切点对齐完整的 5-video-token / 17-pixel-frame 周期。每个 video 边界先换算为全局 24 fps frame boundary，再换算 40 Hz audio boundary：

```text
audio_boundary = round(frame_boundary * 40 / 24)
```

尾块过小时会并入前一块，因此 `chunk_duration_seconds` 是近似目标而不是硬上限。

## Noise

- 单块原样使用输入 NOISE。
- 多块官方 RandomNoise 使用 base seed 与绝对 `frame_start` 派生确定性 uint64 子 seed。
- 多块官方 DisableNoise 保持全零。
- generic/custom NOISE 因没有公共 clone/offset/substream API 而明确拒绝。

## 约束与边界

- 当前算法是无 latent overlap 的顺序分段；旧 0.17 B/C overlap 实验不再出现在节点 UI。
- 多块输入的 original positive 不得已经含有 `minimax_keyframes`。否则旧的绝对帧 guide 会与每段局部 frame-0 guide 冲突，节点会明确报错。使用 Reference-to-Video conditioning 作为 original positive。
- 为取得真实最终像素帧，每个非末段都必须完整执行一次 video VAE decode。最终输出仍是 latent，因此下游若再解码完整结果，会产生额外 VAE 计算；这是保持当前 `LATENT` 输出兼容性的明确代价。
- 节点没有 hidden-state/KV carry，也没有 global model position offset。连续性来自官方 image guide，而不是声称存在模型原生 recurrent state。
- 不支持 `noise_mask`，不会猜测 packed H3 AV 双流 mask 的时间映射。
- `aggressive_memory_cleanup=false` 默认依赖引用释放和 ComfyUI allocator；打开后每段额外执行 GC 和 `soft_empty_cache()`，通常更慢。
- 输入 latent 和 original positive 均不被原地修改。

## 输出

`output` 是完整 CPU-backed H3 AV `LATENT`，可继续连接现有 H3 VAE Decode / video workflow。`status` 会报告 chunk 范围、noise mode、guide 数量以及每段重新建立 Basic Guider 的事实。

0.17 A/B/C 实验说明保存在 [H3 Temporal Chunk Sampler 0.17 archive](H3_TEMPORAL_CHUNK_SAMPLER_0.17_ARCHIVE.md)，仅作历史参考。
