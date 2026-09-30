# H3 → LTX Bridge（实验版）

目标：完成低分辨率 H3 草稿，在 latent 空间放大并转入 LTX-2.5，以三步细化生成高分辨率视频；最终使用原 H3 音频。

## 与现有 Progressive Sampler 的区别

Stage1 使用普通 H3 完整去噪（sigma 走到 0），再 Split AV。不要把渐进采样中途尚未完成的噪声状态交给适配器。可以使用已完成的 H3 参考生成结果，但参考保持质量要另行验收。当前支持 batch=1、24fps、H3 原生 17k+5 帧网格；不接收残留 noise_mask 或跨批索引。

推荐起步：640×640、175 帧、H3 Turbo 4 步；JR Neural Latent Upscaler 设 scale=1.6，得到 1024×1024。再接以下节点：

1. **JR H3 → LTX Latent Adapter**：输入放大后的纯 H3 video LATENT。
2. **JR H3 → LTX Refine Setup**：输入转换后的 LTX video LATENT、原 H3 解码音频、LTX-2.5 audio VAE，以及与 BasicGuider 相同的 LTX MODEL；输出联合 AV latent、可调 SIGMAS、原音频。
3. **SamplerCustomAdvanced**：LTX-2.5 MODEL、LTX 文本 CONDITIONING（24fps）、BasicGuider、Euler、RandomNoise，与 Setup 输出的 latent/sigmas。
4. **LTXVSeparateAVLatent → 匹配的 LTX Conv VAE Decode**：只解码细化后的视频。
5. **JR H3 → LTX Finish Media**：输入解码图片、原 H3 音频、步骤 1 的 video LATENT，移除内部补帧，再交给 Enhanced Video Combine。

辅助 **JR LTX Bridge Text Encoder Loader** 使用 ComfyUI 原生加载器，从 text_encoders 中加载 Gemma，并从 diffusion_models 中所选的 LTX transformer 读取对应 connectors。应与 Stage2 模型选择一致，避免把 H3 Qwen conditioning 接给 LTX。

## 模型位置

```text
ComfyUI/models/
  h3_ltx_adapters/H3-to-LTX-Latent-Adapter/
    config.json
    model.safetensors
  vae/
    ltx-2.5-video-vae-conv-bf16.safetensors
    ltx-2.5-audio-vae-bf16.safetensors
```

适配器来源：[Efficient-Large-Model/H3-to-LTX-Latent-Adapter](https://huggingface.co/Efficient-Large-Model/H3-to-LTX-Latent-Adapter)，固定 revision `cfcd8a7cc36c135142287584728f7fe362df0867`。运行前校验公开权重 SHA-256。不会在节点执行时联网下载。冻结推理模块来源及版本记录在 `utils/h3_ltx_frozen/PROVENANCE.md`。

Conv VAE 来源：[Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)，官方固定文件 SHA-256 `685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d`。普通 `ltx-2.5-video-vae-bf16.safetensors` 是另一种 diffusion decoder，不是本例选择的快速 Conv decoder。

## 时间轴、音频与归一化

原生 ComfyUI H3 latent 已做逐通道归一化，直接输入适配器；不得再次按 raw H3 归一化。转换结果也是 LTX checkpoint normalized latent，可直接用于原生 LTX 采样和解码。

适配器先使用冻结的 H3 时刻坐标、线性插值与 nearest temporal packing，再做 2× pixel-unshuffle 和训练过的 Conv3D 网络。pixel-unshuffle 是两种 VAE 压缩率之间的布局转换，不会再次放大视频像素尺寸。不能用普通 interpolate 或通道 reshape 替代网络。

H3 的 175 帧需要 LTX 内部 177 帧；124 帧内部需要 129 帧。最后仅删除额外尾帧，保留原始时长。不会像固定官方演示一样把 124 帧截成 121 帧。原 H3 PCM 编码到 LTX audio latent 参与联合细化，Stage2 生成的音频不用于最终合成。Finish Media 保持原采样率与原音频样本（最多截去原视频时长以外的编码余量），不做隐式时间拉伸。

## 本地实验配置与官方配方的差异

本例复用现有 `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` 和匹配 Gemma4。官方演示是 BF16 dev transformer + distilled LoRA（0.8），因此不是完全相同的权重/精度实验，不能套用官方性能或质量结论。

使用固定通用提示词 `4K, refined, high quality, cinematic detail, clean textures, natural motion.`，由本机匹配的 ComfyUI encoder/connectors 编码并复用图缓存；不是官方离线 INT8 dev connector 上下文缓存。

旧版三步 sigma 为 `0.909375, 0.725, 0.421875, 0`，它不是一个可直接换算成标准 denoise 的单值。当前新增可见 `steps=3`、`denoise=0.25`、`scheduler=simple`，使用 ComfyUI 原生 BasicScheduler：按 `int(steps/denoise)` 计算完整调度，取最后 `steps+1` 个 sigma。因此 denoise=0.25 不等于起始 sigma=0.25，实际值依赖所接模型，状态输出和日志会列出实际序列。降低 denoise 用来减弱 Stage2 改写，步数只控制 LTX 更新次数。

保守测试可从 0.15–0.25 开始，固定同一草稿与种子；denoise=0 跳过 LTX 去噪，可用于区分“适配/解码造成的变化”和“细化造成的变化”，不是直接还原 H3 原始像素。选择 `sol_h3_original` 并设置 steps=3、denoise=1 可精确重现旧固定调度。该模式 denoise=1 仅作为旧配方开关，不代表旧起始 sigma 为 1。旧 API 请求完全省略新增字段时仍保留原固定配方。旧 UI 工作流加载新控件后，需要补接 `ltx_model`；新的 adjustable 示例已接好。重启 ComfyUI 并刷新前端才会加载新节点定义。

继续使用 BasicGuider（CFG=1）、Euler。Core BlockSparseAttention 可用于 LTX 的视频自注意力；音频、跨注意力及不匹配 head dimension 的调用保持 dense。当前本机 LTX 未上报 block index，因此 `dense_blocks` 必须留空，不宣称复现官方“第 0 层 dense”的逐层策略。用于纯兼容/质量对照时可直接绕过此 sparse 节点。

先比较同一份 Stage1 草稿在“适配后直接解码”和“三步细化后解码”的输出，再比较原 H3→H3 渐进路线。不要通过更换 Stage1 种子来测 Stage2 差异。记录每阶段耗时和完整请求耗时，未完成解码前的时间不应当作最终视频生成时间。

## 资源与异常

本机 RTX 5090 使用 `--reserve-vram 3` 启动，完成三次连续全流程：首次 41.74 秒，换种子重跑 32.06 秒，前端载入工作流再运行 32.44 秒（均含视频解码和合成，后两次复用文本/模型缓存，采样重新执行）。这是中性茶壶提示词的单例结果，不能直接与双参考人物工作流比较。未预留时，曾在一次 H3/LTX 切换后的 H3 sparse 采样中发生停滞；预留 3 GB 后这三次未复现，根因尚未确认。秋叶启动器测试时建议同样预留约 3 GB，避免同时叠加两个显存预留机制。

普通可编辑工作流在 `examples/H3_to_LTX_640_to_1024_175f.json`；同名 `.api.json` 供 API 使用。`latent_test` 版本仅保存 latent，不产出最终视频。

适配器按完整时空体计算 GroupNorm，当前不做改变数学结果的 temporal chunking。使用 CUDA BF16，通过 ComfyUI 释放所需显存，适配结束或失败后模型回到 CPU。缺权重、配置/hash 不匹配、时间轴错误或 CUDA 错误会明确报错，不回退成普通图像放大。

已有 Unified v2、原 H3 节点和工作流均保留。该功能是独立实验入口。
