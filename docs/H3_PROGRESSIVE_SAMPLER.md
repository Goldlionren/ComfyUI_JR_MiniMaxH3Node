# JR H3 Progressive Sampler（本地实验版）

## 2026-09-13 用户验收基线

用户已在 RTX 5090 生产环境实际运行 Progressive Guided，并确认可与 JR Unified Acceleration 一起使用。当前配置已达到用户可用的阶段性稳定状态；实验标记仍保留，表示跨模型／跨素材的画质和加速收益尚未系统验证，而不是运行失败。

用户报告：scale=0.5 相比其传统双采工作流约快 5 秒，速度基本打平；0.65 画质可接受但提示词遵从下降；0.6 在当前素材上兼顾画质与遵从，可作为该测试场景的推荐起点。未据此修改节点默认值，也不宣称所有 seed、时长或引导模式都以 0.6 最优。后续实验应以本提交作为可回退的质量基线，并固定 Unified 设置。

本次基线的自动验证：815 tests passed / 1 skipped；真实 H3 视频 VAE、神经放大权重和小型原生 H3 组合冒烟通过。用户画质验收与这些接口回归测试是两类不同证据。

Node ID：`JR_H3_ProgressiveSampler`。输出 `LATENT, STRING`。此原型保持原有 dual sampling、Temporal Chunk Sampler 和全部旧工作流可用。

## 快速测试

导入 `examples/JR_H3_Progressive_T2VA_Experimental.json`，确认模型、LoRA、CLIP 和 VAE 名称在本机可用。示例从现有 T2VA turbo 工作流派生，默认 Euler、turbo 8 steps、固定 seed、关闭 Unified Acceleration。

1. 首先设置 `lowres_scale=1.0`，跑原生全分辨率 Euler 基线。
2. 保持提示词、seed、模型、LoRA、sigma shift、scheduler 和总步数不变，改为 `lowres_scale=0.5`、`transition_step=3`。
3. 对比耗时、显存、细节、运动连续性、语音/音乐和音画同步。也可试 `lowres_scale=0.75`。

`latent_image` 必须是**最终目标分辨率**的空 H3 AV latent。首次测试采用 T2VA，不连接 first/last frame。8 步 / transition_step=3 表示前 3 次 denoiser evaluation 使用低分辨率，后 5 次使用高分辨率。1.0 会完全绕过 Neural Upscaler，作为原生 A/B 基线。

Neural Upscaler 使用现有 `models/latent_upscale_models/minimax_h3_latent_upscaler_3d_*.safetensors`。不自动下载权重，不新增依赖。重启 ComfyUI 后搜索 `JR MiniMax H3 Progressive Sampler (Experimental)`。

## 输入

| 参数 | 说明 |
| --- | --- |
| model / positive | 原生 H3 MODEL 和 CONDITIONING；MODEL 上的 sigma shift 会被保留 |
| noise | 官方 RandomNoise 或 DisableNoise；固定 seed 用于可重复测试 |
| sampler | KSamplerSelect 的 `euler`；禁止 churn、多步、SDE、ancestral、自定义 sampler |
| sigmas | 同一条完整 schedule，至少 2 步；从 1 开始、严格下降到 0，使用 denoise=1 |
| latent_image | batch=1、空的官方二流 AV NestedTensor，目标 H/W 按原生 2x2 patch grid 对齐 |
| transition_step | 低分辨率 evaluation 数，1 到总步数减 1；默认 3 |
| lowres_scale | 0.25–1，默认 0.5；按原生 patch grid 向上对齐 |
| transition_seed_offset | 独立视频切换噪声 seed = (NOISE seed + offset) mod 2^64，默认 1 |
| aggressive_memory_cleanup | 默认 false；可在阶段间调用 ComfyUI soft_empty_cache |

## 数学与 AV 契约

低分辨率阶段运行 `sigmas[:k+1]`。回调捕获最后一次 evaluation 的内部 `x` 和预测 `x0`，该回调位于 Euler 更新之前。公开的非零 sigma 截断 LATENT **不送进** Neural Upscaler。

仅 video x0 经 `latent_format.process_out` 转换为 VAE latent 域，使用 JR Neural Upscaler 放大到精确目标 H/W，再经 `process_in` 返回内部域。video 在 sigma[k-1] 用固定派生 seed 重建噪声状态，然后完成同一个 Euler 区间。audio 直接使用捕获的原生 Euler 状态和预测完成该区间，不空间缩放、不重新加噪。

接续状态经过 `inverse_noise_scaling(sigma[k])`，audio 再除以有效 `model_sampling.audio_scale`，返回公开 AV latent 域。新原生 Guider 以全零附加噪声运行 `sigmas[k:]`，原生输入转换恰好还原边界状态。总 evaluation 数不增加；代码检查两阶段回调数量与顺序。

保留 video `[1,24,T,H,W]`、audio `[1,32,2,Taudio]` 顺序、timeline、外层 batch_index 和自定义 metadata。返回方式与原生 sampler 一致，删除已失效的 downscale ratio hints。Neural backend 使用现有 dtype/device、normalization 和模型卸载逻辑；boundary 保存为 CPU fp32。通过输入 MODEL 读取有效 sampling patch，避免使用未 patch 的默认音频 shift。

固定 seed 保证同一配置在同一运行环境中可重复；不同 lowres_scale 会改变初始 AV 随机数消费和模型联合预测，因此不同配置间 audio 不保证相同。

## 当前限制

以下三项为原始 `JR_H3_ProgressiveSampler` 的限制；新版 Guided 节点的支持范围见下节。

- 拒绝 noise_mask（包括全一 mask）、非零 video/audio 初始 latent、batch>1。
- 拒绝 keyframes/AddGuide、conditioning mask/area/control/hooks。支持无 guide 的 T2VA；独立 minimax_refs 保留原结构，但完整 Ref2VA 质量尚未验证。
- 不支持音频驱动、hard prefix、无限 MV 或 temporal chunk 内嵌。使用旧采样路径完成这些任务。
- Adaptive Cache 在开始、分辨率边界和 finally 中显式 reset，原生 Guider 同时保留自己的 cleanup。首次质量测试请关闭 cache、Sol-Attn 和 compile；Sage/Low VRAM/FFN 等需分别做 A/B。
- 低分辨率 H3 推理以及预测 x0 的 learned lift 仍属于实验分布；速度和质量需要真实权重测试，不能由单元测试证明。

## 验证范围

测试包含 schedule/shape/mask/sampler 拒绝、边界 Euler 数值、非平凡 latent normalization 与 audio shift、固定 seed、零附加噪声、metadata 不变、scale=1 原生基线、checkpoint backend 的精确尺寸及 dtype。

小型随机 H3 的原生集成测试使用真实 H3 forward、PackedLayout、CoreModelPatcher、Guider 和 Euler；只用确定性插值替代大尺寸 learned upscaler，以便回归测试无需下载权重。此项证明运行时协议，不证明视频生成质量。

正式 benchmark 建议记录：完整耗时、各阶段 GPU 同步计时、峰值 allocated/reserved VRAM、系统 RAM、实际 evaluations、视频空间/时间质量、音频质量与 AV sync。对照包含全分辨率单 schedule、现有 dual sampling、progressive；分别报告 shipped 参数和相同 Euler/NFE 的比较。

## Progressive Guided Sampler：参考图、首尾帧、音频驱动

新增独立 Node ID：`JR_H3_ProgressiveGuidedSampler`，显示名称 `JR MiniMax H3 Progressive Guided Sampler (Experimental)`。
保留原 T2VA 节点的严格行为。输入与输出同原节点，另有可选 `vae: VAE`：存在视频 keyframe 且 scale<1 时必须接**编码引导所用的同一个 H3 视频 VAE**，不能接音频 VAE。

| 用法 | 接线 | 示例 |
| --- | --- | --- |
| 独立参考图 | 原生 MiniMaxH3ReferenceToVideo 的 positive / LATENT → Guided；使用 Ref2VA 模型与匹配 LoRA | `JR_H3_Progressive_Ref2VA_Experimental.json` |
| 图生视频 / 首尾帧 | 原生 MiniMaxH3ImageToVideo 的 positive / LATENT → Guided；video VAE 同时接 Guided.vae；使用 FL2VA 模型 | `JR_H3_Progressive_FirstLast_Experimental.json`，只要首帧时断开 last_frame |
| 音频驱动 + 首帧 | LoadAudio → H3 VAEEncodeAudio → JR Audio Driven Latent Builder → Guided.latent_image；positive 来自 ImageToVideo | `JR_H3_Progressive_AudioDrive_Experimental.json`，可断开首帧或再接尾帧 |

三类引导可以组合：Ref2VA positive 后串原生 MiniMaxH3AddGuide 添加 keyframe，AV LATENT 经 Audio Driven Latent Builder 锁音频，再送入 Guided。采样接口支持组合，不代表每个 checkpoint 都对组合任务有同等训练支持。

示例放在 `examples/`，媒体选择故意留空，请选择自己的文件。默认 duration=5 秒，实际按 H3 网格对齐到 124 帧，约 5.17 秒。音频驱动不自动采用整首音频长度；Builder 按目标时长裁切或在 latent 域补零。输出音频 latent 保持输入完全一致，但经过有损音频 VAE 的解码音频不等于原始 PCM；需要原声时将原始／已裁切 AUDIO 接到 CreateVideo.audio。

### 引导在切换处如何保留

- `minimax_refs` 使用独立空间网格，两阶段保留原始 block、顺序、尺寸、音视频 tensor 和 metadata，不将参考图缩成目标低分辨率。
- `minimax_keyframes` 使用目标空间网格。低阶段将干净 keyframe latent 经 video VAE 解码、像素域 area 缩小、同一 VAE 重编码；保留帧索引、时间长度、audio_latent 和其他 metadata。高阶段直接使用原始 conditioning，不用重编码结果覆盖它。重复引用只编码一次。scale=1 不加载／调用 VAE。
- 接受无 mask 的空 AV，或形状与 AV 完全一致的官方 NestedTensor mask：video 全 1；audio 全 0（锁定）或全 1（必须为空 audio）。拒绝局部／软音频 mask、video mask、hard-prefix、已采样 video、batch>1、conditioning mask/area/control/hooks。
- audio 锁定时，低阶段原生 inpaint 注入原始干净 audio；切换后恢复相同干净 audio 作为高阶段锚点，不使用 noisy resume audio。最终再次保留原始音频数值与 dtype，外层 noise_mask、batch_index 和 metadata 不变。仍使用有效 MODEL 的 AV sigma shift。
- 两阶段使用独立原生 Guider，PackedLayout 根据实际 target grid 重建。沿用开始／切换／finally 的 Adaptive Cache reset。拒绝隐式替换 Euler 的 wrapper。

### 验收与已知风险

首尾帧重编码有一次 VAE 重建损失，低分辨率布局和 x0 神经放大也可能影响身份、肢体与运动；不是数学等价的全分辨率去噪。原始高分辨率引导的恢复不能保证消除低阶段的构图偏差。独立 refs 的 token 成本不会随目标画布缩小，参考图多时加速收益可能很小。

分别做 Ref2VA、首帧、首尾帧、音频驱动及组合测试，每项固定提示词、输入文件、seed、MODEL/LoRA、shift、scheduler、总步数、分辨率和时长，比较 scale=1 / 0.75 / 0.5，再比较旧 dual sampling。优先观察人脸身份、首尾帧贴合、身体扭曲、运动衔接、口型与音画同步。Sage、Sol、Cache、compile 应分别切换后再比较；不要把不同分辨率和 shift 的耗时直接归因于加速开关。

自动测试覆盖真实小型 H3 + PackedLayout + 原生 inpaint/Euler 的三类引导及组合、dtype、完整 audio lock、固定 seed、原始 metadata 不变、scale=1 与原生视频结果一致、无效 mask 与 guide 拒绝。可运行 `tools/smoke_h3_progressive.py --comfy-root <ComfyUI> --video-vae <installed H3 video VAE>`，使用本机真实 VAE / learned upscaler 与小型 H3 验证完整路径，不下载权重、不提交生产任务。此测试不代表完整模型的画质验收。
