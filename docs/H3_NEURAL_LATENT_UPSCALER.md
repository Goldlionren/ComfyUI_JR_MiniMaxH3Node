# JR MiniMax H3 Neural Latent Upscaler

## 职责与接线

该节点只对 Split 后的普通 H3 video LATENT 做 neural 空间放大：

```text
H3 AV LATENT -> Split AV Latent
                  ├-> video -> Neural Latent Upscaler ┐
                  └-> audio (unchanged) ---------------├-> AV Latent Builder -> Pass-2 sampler
```

输入必须是 floating、finite、strided、materialized `[B,24,T,H,W]` Tensor。输出只改变 H/W，B/C/T 不变。完整 AV `NestedTensor` 会被明确拒绝并提示先使用 Split；audio 不能进入本节点。

本节点不接管 H3 diffusion MODEL、conditioning、NOISE、SIGMAS、GUIDER、sampler、scheduler、denoise、re-noise 或 VAE。

## Neural backend 与 checkpoint

JR backend 是独立编写的 3D residual + scale-conditioning + temporal-convolution inference implementation。空间 resize 发生在 learned feature volume 中，前后都有 checkpoint-trained 3D blocks；不存在 raw latent nearest/bilinear/bicubic fallback。

节点自动扫描：

```text
ComfyUI/models/latent_upscale_models/
```

候选文件必须是 `.safetensors` / `.pth` / `.pt`，且文件名包含 H3 与 upscaler 标识。多文件时优先匹配输入 dtype，再优先 SafeTensors，最后按文件名确定性选择。缺少模型、签名不兼容、非 24-channel、包含未支持 attention block 或 state-dict 不完整都会 fail closed。节点不会下载模型。

参考 checkpoint：`LBH-123-AI/Minimax_h3_latent_Upscaler`。其 Hugging Face 模型卡声明 Apache-2.0；checkpoint 不随本仓库分发。配套 GitHub custom-node 仓库在 2026-08-25 审计时没有 LICENSE 文件，因此 JR 没有复制或 vendor 其 Python 源码、权重、注释或 UI。模型使用者仍应自行复核当前模型页面条款。

模型卡：https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler

参考实现页面：https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler

## 尺寸规则

节点在运行时检查当前 ComfyUI 原生类的构造合同：

- `MiniMaxH3VideoVAE.space_down` 的默认乘积为 16，即 latent H/W 对应 pixel H/W 的 16x 压缩。
- `MiniMaxH3Model.patch_size` 的空间部分为 2x2。
- 因而合法输出 latent H/W 对齐 2，decode 后 pixel H/W 对齐 32。

`scale` 模式把 latent/pixel 宽高各乘指定线性倍率，再在合法网格中选择面积、宽高比和目标尺寸综合误差最小的结果。`1.5x` 表示面积约 `1.5² = 2.25x`。

`megapixels` 模式使用十进制 `1 MP = 1,000,000 pixels`：先按输入 aspect ratio 求理想 pixel W/H，再映射到最接近的合法 32-pixel grid。目标如果构成 downscale 或需要超过 checkpoint 训练范围的 4x 线性放大会明确拒绝。

## dtype、device、metadata 与 VRAM

- 支持 fp32、fp16、bf16 输入；checkpoint 可用不同计算 dtype，但输出恢复输入 dtype/device。
- 返回新的 LATENT dictionary，只替换 `samples`，其他 metadata 原样保留；输入容器不被原地修改。
- T 从不插值。T 大于 24 时使用带 temporal context 的顺序 chunk，以限制 3D 中间激活峰值。
- checkpoint 由 ComfyUI `CoreModelPatcher` 管理；推理结束调用 model-specific unload，只卸载该 upscaler 及 clone，不全局卸载 H3 diffusion model。

## 已知边界

- Neural 模型节省的是低分辨率首轮采样时间，不会降低高分辨率 Pass-2 sampler 本身的显存峰值。
- 长序列 temporal chunking 保留局部上下文，但不声明与整段单次 3D forward 逐位等价；应通过真实视频 A/B 检查边界和运动一致性。
- 本节点不 resize `minimax_refs` / keyframes / conditioning。需要依赖空间绝对位置的二采 conditioning 时，应由独立、理解 H3 conditioning 的节点处理，不能假装本节点已完成。
- 本机开发环境没有兼容 H3 upscaler checkpoint，因此真实 691 MB checkpoint、完整 GPU Pass-2 和画质 A/B 不属于单元测试结果。
