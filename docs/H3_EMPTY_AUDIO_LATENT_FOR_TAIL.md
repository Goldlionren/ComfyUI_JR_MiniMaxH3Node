# Empty Audio Latent for Tail（实验功能）

节点名称：**JR MiniMax H3 Empty Audio Latent for Tail (Experimental)**

节点 ID：`JR_H3_EmptyAudioLatentForTail`。

## 新版：尾部上下文接线

```text
Tail Frame Latent.tail_context_latent -> Neural Latent Upscaler -> AV Latent Builder.video_latent
                                               └-> Empty Audio Latent for Tail -> AV Latent Builder.audio_latent
AV Latent Builder -> 原生 SamplerCustomAdvanced -> Split AV Latent.video_latent
                  -> VAE Decode -> 取最后一张 IMAGE -> 高清壁纸
```

先跳过放大和二采，直接解码 `tail_context_latent` 并取最后一张，与 `tail_image`
比较。使用同一个原生 VAE、`decode_mode=tail_context`，不接 `decoded_frames`。
通常输出 22 张图，原生 Image From Batch 可设 `batch_index=21, length=1`。
其他长度或 full_video 模式须按实际解码帧数选择最后一张，不应固定索引 21。

节点 ID、唯一输入 `video_latent` 和两个输出不变。现在自动匹配干净的
`[1,24,T,H,W]`，T 为 1 或 5k+2；拒绝 AV、音频、非浮点、非有限数值及 noise_mask。

| video latent T | 解码帧数 | 空音频 ticks |
|---|---|---|
| 1（旧单帧） | 1 | 2 |
| 2 | 5 | 8 |
| 7（通常的尾部窗口） | 22 | 37 |
| 12 | 39 | 65 |

音频形状为 `[1,32,2,round(frames*40/24)]`，全零、dtype/device 与输入一致。
22 帧约 0.917 秒，37 ticks 为 0.925 秒，属于离散网格舍入。完整视频也按相同
公式匹配，但本节点不负责裁短视频。请连接放大后、与 AV Builder 相同的视频。

7-token 二采输出仍是短视频，不能直接作为 Director `first_latent` / `last_latent`。
高清壁纸到下一轮单帧引导 latent 的转换仍需单独验证。旧单帧 encode/decode
已实测变暗和条纹，以下只记录兼容接口，不代表推荐继续用单帧重建高清壁纸。

## 旧版单帧接线（保留兼容，重建画质未通过）

```text
Tail Frame Latent.tail_latent -> Neural Latent Upscaler -> video_latent ───────┐
                                                         │                 │
                                                         └-> Empty Audio   │
                                                             Latent for Tail
                                                                 │         │
                                                                 v         v
                                                          AV Latent Builder
                                                                 │
                                                     原生二采 SamplerCustomAdvanced
                                                                 │
                                                           Split AV Latent
                                                                 │ video_latent
                                            ┌────────────────────┴─────────────────────┐
                                            v                                          v
                                    VAE Decode -> 高清壁纸                    保留 / Save Latent
                                            │                                          │
                                    下一轮 first_frame                       下一轮 first_latent
```

旧单帧用法的唯一输入是**放大之后**的单帧 `video_latent`。
此时仍自动输出全零 `[1,32,2,2]` 音频 latent，dtype/device 与输入一致，无需连接音频 VAE、
输入时长、输入原视频的尾帧时间戳或配置 seed。AV Latent Builder 的两个输入分别接
放大后的单帧视频和这个空音频。

## 对齐的含义

- 视频保持 `[1,24,1,H,W]`，不复制成 5 帧，不插值时间轴。
- 单独的尾帧二采从局部时间零开始；它不是原视频音频的末尾切片。
- 根据 H3 的 24 fps / 40 audio ticks/s 约定，`round(1 × 40 / 24) = 2`。
  一帧约 41.7ms，两音频 tick 为 50ms，这是离散时间网格的舍入，不是精确等长。
- AV Builder 对单帧严格要求音频 `T=2`；原有完整视频 `T_video=5k+2` 的规则及
  ±1 tick 容差不变。仍检查 batch、dtype、device 和有限数值。
- 这是空 latent 初始化，不是编码过的静音，也不锁定音频；它仍参与联合模型采样，
  不能声称对图像数学上完全没有影响。单帧二采输出的音频不用于播放，直接弃用。

## 测试注意

使用原生 `SamplerCustomAdvanced` 的普通二采路径，先沿用已有二采低 denoise 设置。
不要把这份已经有内容的 latent 接到要求空目标的 JR Progressive / Progressive Guided。
本节点不改 conditioning：原长视频的首尾帧锚点、时间索引、音频引导和分辨率不能
自动迁移到局部短片或单帧。先用无时间锚点的条件验证；需要关键帧引导时另行适配。
此处不承诺 Temporal Chunk Sampler 对不足一秒窗口的支持。

旧 T=1 二采输出结构仍为单帧，兼容 Save/Load Latent，但重建画质未通过验收。
新 T=7 二采输出应完整保留，Decode 后取最后一张作为壁纸；不要强行截成一个
latent 时间片，也不要直接接 Director 的单帧输入。

官方空视频节点最短会生成 5 帧，所以这里是基于当前 H3 模型可接受的单帧结构增加的
实验入口，不是宣称官方单帧工作流已获质量认证。小型真实 H3 模型、原生采样器和
TST 开关测试验证了结构与重复性；真实权重画质、硬件 attention 后端和多轮质量仍待实测。

新增测试包括 22 帧 / 37 ticks 的原生小型 H3 二采，及 opt-in 原生 VAE 真实权重
上下文端点一致性检查。小模型测试中的放大器使用确定性替代，不验证放大权重画质。
