# JR H3 VEDA Attention（独立实验节点）

Node ID：`JR_H3_VedaAttention`，输出 `MODEL, STRING`。它与 Sol-H3 / Unified V2 平行存在，不修改两者实现、默认值或旧工作流。

## 接线

从加载器/LoRA 后的干净 H3 MODEL 建立独立分支：

```text
H3 loader → LoRA / sigma 配置 ┬→ 原 Unified V2 / Core Sol-H3 → 原采样器
                            └→ JR H3 VEDA Attention       → 对照采样器
```

每次比较运行一条采样分支。不要把 VEDA 串在 Unified V2 前后。已安装 Sol/Core、其他 attention override、TST 或 block/forward replacement 时节点会明确报错；采样前再次检查下游覆盖。

原生 `MiniMaxH3SigmaShift`（显示为 `ModelSamplingMiniMaxH3`）可放在 VEDA 前后：它只更换 `model_sampling` 并配置 video/audio shift，不替换注意力。`LoRA → VEDA → SigmaShift → BasicGuider → SamplerCustomAdvanced` 是允许的接线。注意力、block/forward 补丁仍不能在 VEDA 之后被覆盖。

节点只克隆自身的 MODEL，不改变输入分支。enable=false 在模型/依赖检查前直接透传。关闭节点不会移除输入已有的补丁。

## 依赖与参数

- ComfyUI >=0.38.0；安装官方 [Veda-on-ComfyUI](https://github.com/veda-sparse/Veda-on-ComfyUI)（Registry ID：veda-sparse-attention），重启 ComfyUI。
- predictor 放入 models/veda；首版使用 `minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors`。[下载来源](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview)。
- LowVRAM Attention / FFN 默认启用，来自 KJNodes。没有 KJNodes 时可关闭两项；VEDA 自身仍有按 head 控制工作集的代码。
- sage_attention 默认 disabled：dense 回退使用 ComfyUI 的现有全局注意力。显式选择 Sage 时需要 KJNodes 和对应 Sage 库，compile 固定关闭。
- generated_sparsity/reference_sparsity 默认 90%。整数表示固定 tile 数；参考人物对照可先用 reference_sparsity=0%。
- full_attention_layers/full_attention_steps 为 0 起的完整注意力保留列表。verbose 打开官方 VEDA 运行统计。
- 依赖只在启用节点时解析；不自动安装或下载模型，包导入阶段不导入 kernels。官方节点的 prepare/cleanup、kernel 自检、运行回退和进度文字保留；通过 V3 class clone 传递当前 JR 节点 ID，不修改上游注册类。

STRING 输出只是配置状态，不能证明发生了 sparse 计算。请看节点采样中的 Veda running / done 文字、verbose sparse/dense 调用和回退原因。

## 首版边界

不认证 TST、Adaptive Cache 或 Streaming Sparse KV 组合。Progressive/Guided、Hard Prefix 和混合权重的实际质量需单独验收。VEDA 接收完整 QKV，不继承 Core Sol-H3 producer 的显存优势。

2026-10-09 发布的专用 R2VA predictor 使用 fixed-tile metadata；当前官方 loader 必读缺失的 keep_ratio，因此本节点给出清晰错误，不填造预算或修改上游包。现有 T2VA predictor 的官方模型卡允许 R2VA/FL2VA 使用；这不表示专用 R2VA 文件已适配。

只添加了独立适配与保护检查，未复制 VEDA kernels 或 predictor 推理。详情和证据见 [可行性研究](H3_VEDA_FEASIBILITY.md)。

## 验证与对照

接口回归覆盖关闭透传、缺依赖、冲突拒绝、输入分支隔离、调用顺序、V3 hidden 隔离、返回值归一化、R2VA metadata 错误及下游覆盖。原 Unified/Sol 回归继续运行。已使用生产中安装的官方 VEDA 完成真实导入和接口绑定检查；尚未进行完整 predictor GPU 画质/速度验收，不能宣称已比 Core Sol-H3 更快。

正式对照固定 seed、模型/LoRA、conditioning、sigma、步数、分辨率、帧数、FFN 和解码路线；比较 Sage dense / 原 Core Sol-H3 / 新 VEDA，记录暖跑总时间、阶段耗时、显存/RAM、fallback 和媒体质量。先 5 秒，再测最新 ver3.2 的 10–30 秒实际负载。

2026-10-09 本地验证：完整回归 1003 passed / 7 skipped；Ruff、compileall、注册与工作流 smoke 通过（33 个注册节点）。跳过项为 opt-in GPU/权重集成及非 CUDA 路径检查。Git 对照确认原 Unified V1/V2、Sol-H3 后端/适配器和 examples 无差异。

2026-10-09 部署：开发目录是指向 F:/AI/custom_nodes/ComfyUI_JR_MiniMaxH3Node 的 junction，该套已同步；另已部署到 F:/ComfyUI-aki-v3/ComfyUI/custom_nodes/ComfyUI_JR_MiniMaxH3Node。只添加 nodes/h3_veda_attention.py、utils/h3_veda_attention.py 和三项注册；旧 Sol-H3 文件哈希、已有注册保持不变。生产 Python 导入、关闭透传、官方 VEDA 的 execute 签名和 V3 hidden 绑定通过；没有运行生成。备份与记录位于工作区 veda-production-deploy-20261009/081421-268170。用户自行进行生成测试。

2026-10-09 后续合并与修复：开发路径已改为独立 Git 仓库，GitHub main `9c960a9` 与 CutAudio、尾帧、Director latent 等本地成果完整合并，注册 36 个节点；合并记录位于工作区 jr-reconcile-20261009。实际工作流发现 SigmaShift 位于 VEDA 之后时，旧保护逻辑将正常的 `model_sampling` 替换误报为注意力覆盖；已将采样调度器排除在该对象补丁比较之外，保留对注意力/forward 覆盖的拒绝，并增加原生小模型及 SigmaShift 前后顺序回归。
