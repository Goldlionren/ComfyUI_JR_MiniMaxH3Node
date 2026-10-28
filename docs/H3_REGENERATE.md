# JR MiniMax H3 Regenerate

以原生 ComfyUI MODEL / ModelPatcher 接入 Open-H3-Regenerate-2K。两个节点可以独立使用：
**JR MiniMax H3 Regenerate LoRA** 输出普通 MODEL，可直接接现有二采；
**JR MiniMax H3 Regenerate** 接 VIDEO 或原生 H3 LATENT。VIDEO 模式输出视频 LATENT + 原音频 AUDIO；AV LATENT 模式保留整条音频 latent。

## 推荐接线

```text
原生 BF16 H3 Loader
  → JR MiniMax H3 Regenerate LoRA
  → JR Unified Acceleration（可选；Sage 2++ / FFN）
  → JR MiniMax H3 Regenerate（attention = inherit）
  → 视频 LATENT → 原生视频 VAE Decode → CreateVideo
  → original_audio ────────────────────→ CreateVideo.audio
```

使用 [完整 VIDEO 示例](../examples/JR_H3_Regenerate_Video.json)，API 版本见同名 .api.json。
默认截取最多 15 秒，按 5s 档分窗，并在输出端裁掉 H3 网格补帧；更改片段长度时，同时修改前后两个 Video Slice。
H3 latent 模式：断开 video，改 source_mode 为 H3 latent，连接已有 AV LATENT 到 latent。
AV 模式输出仍经过 Split AV Latent 再分别解码。该模式不需要 VAE 输入。它接受 ComfyUI NestedTensor 的 video [1,24,T,H,W]、
audio [1,32,2,A]；不把 Diffusers 的归一化 .pt 文件当成 ComfyUI LATENT。

VIDEO 模式只需要 H3 video VAE。旧 audio_vae 插口保留以便加载旧工作流，但不使用；
请断开它并按新示例重接，VIDEO 输出不再是 AV NestedTensor。
普通视频 LATENT 也可接 latent 输入，返回普通视频 LATENT，original_audio 为 None；
AV 输入则返回 AV NestedTensor，音频 tensor 保持同一对象。无音轨 VIDEO 的 AUDIO 输出为 None。

视频按全局帧索引以源平均帧率重采样到 24 fps，空间对齐到 32 像素，尾部复制末帧补齐
17k+5 网格。不主动把 768p 降为 576p。文件 VIDEO 支持原生 LoadVideo / Video Slice；
CreateVideo 的帧本来已在内存里，节点不会再次展开整条副本。

## 分窗与完全音频旁路（2026-10-29）

面板追加 chunk_mode，已有控件顺序和前两个输出槽号保持不变。新节点默认 5s，
未传入此选项的旧 API 请求仍按 off 执行。

| chunk_mode | 单窗上限 | 后续窗口锁定的前缀 |
|---|---|---|
| 5s | 124 帧，约 5.167 秒 | 39 帧 / 12 个 video tokens |
| 8s | 192 帧，8 秒 | 同上 |
| 10s | 243 帧，10.125 秒 | 同上 |
| off | 整条输入 | 无 |

时间长度必须满足 H3 原生压缩网格；最后一窗可短于上限。窗口顺序执行，
逐窗读取、视频 VAE 编码、神经放大、采样。结果写入预分配 CPU latent，释放每窗的中间张量，
不会先把长视频全部解码或编码再切段。整个过程不创建临时视频、文件缓存或新依赖。

所有窗口给原生采样器的 seed 固定不变。噪声按“固定 seed + 绝对 video token 索引”
确定，因此同一个重叠位置不会随分窗重新随机；后续窗使用上一窗已重建的末尾 12 tokens，
以 noise_mask=0 锁住，并且只提交新区域，避免重复、漏帧或反复覆盖已完成前缀。
相比旧版本，噪声布局和音频条件均有变化，不能期待相同 seed 重现旧版本像素。

原音频不进入任何窗口。文件音轨仅一次解码成原采样率、原声道的 AUDIO；已存在的 AUDIO
直接返回原对象，不编码音频 latent、不按窗口切片。AV 输入的整条音频 latent 原样挂回。
原生 H3 联合模型内部仍需要 AV 接口，因此每窗只配等长的零音频占位 latent，并锁定其 mask；
占位音频被丢弃，不是对原音频做采样。最终 MP4 是否重编码音轨由下游合成节点决定。

固定噪声和硬前缀可减少接缝，但窗口上下文不同，不能保证完全无色差或与整段采样一致。
默认不做自动逐段色彩校正，以免改变真实光照变化。较短窗主要降低峰值内存和显存；
重叠和模型切换有开销，不能保证必然加速。最终完整视频 latent 仍驻留 CPU，
下游 VAE 解码/合成若展开整条 2K 视频，仍可能消耗较多内存，建议使用 VAEDecodeTiled。

## LoRA 完整性与转换

源发布包包含：
- 主适配器：Refiner + LightX2V FL2VA Turbo v1.2 + LMS Sharp v1.0，强度已经折入 A/B。
- 额外的 **50 个 Sharp 时间曲线分支**。
- 1025×8 的 adaln_t_table。

转换后保留 208 组原生线性投影：50 个 DiT block 与 2 个 text token-refiner block，
每个包含 QKV、attention output、FFN fc1/fc2。独立 Q/K/V 用秩拼接和块对角 B 精确组合；
SwiGLU fc1 的两半输出恢复为 ComfyUI 顺序。没有低秩再压缩或丢弃投影。

50 个曲线分支的 B@A 折为小型 FP32 [out,8] 矩阵，保留原曲线插值和各模态 timestep。
这保持数学映射，浮点计算顺序变化仍可能造成舍入差异。
加载器检查每组主投影和每个曲线分支；缺失、错形状、pruned curve 底模、重复加载都会报错。
曲线用克隆 MODEL 上的 ModelPatcher 补丁接入，执行结束清理临时时间坐标，不改磁盘底模。

转换命令（路径按自己的目录填写）：

```text
python tools/convert_h3_regenerate_lora.py --lora <原发布包>/lora/open_h3_regenerate_2k_lora.safetensors --curves <原发布包>/lora/sharp_curve_branches.safetensors --table <原发布包>/lora/adaln_t_table.bin --output <ComfyUI>/models/loras/open_h3_regenerate_2k_comfy.safetensors
```

输出约 17.76 GB（十进制），QKV 块对角表示会比源文件大。转换不会下载模型或覆盖已有文件。
**完整效果必须使用 JR Regenerate LoRA loader**；普通 LoRA Loader 可识别主线性层，
但不会执行额外 Sharp 曲线，因此不能把它当作完整等价加载。
不要再叠加 Turbo 或 Sharp。当前支持完整 time-embedding 的 BF16 H3；
不声称支持 pruned INT8 / curve-form / 其他量化底模。

## 节点内置 sigma，不接外部调度器

采样固定原生 Euler、CFG=1、video shift=6、audio shift=3。
原音频完全旁路；内部零音频占位通过每步 mask 锁定，不作为输出。
AV 输入最终重新挂回**同一个音频 tensor**，避免归一化回转或 dtype 转换改变输出值。
原视频流整体重建，旧 video noise_mask 不作为局部修复区域继承。

| sigma_schedule | 行为 |
|---|---|
| low_denoise_shift6（默认） | denoise 直接指定起始 sigma（0.15–0.35）。反解 shift-6，在未偏移域均匀取 steps 个区间，再重新 shift，最后到 0。 |
| release_original | 严格采用作者发布的原始 2／3 步路线。此模式忽略 denoise 控件，其他步数报错。 |

原始 2 步：[0.7, 0.49411764705882355, 0]。
原始 3 步：[0.909375, 0.8571428571428571, 0.6666666666666666, 0]。
低降噪 3 步 / 0.30：[0.30, 0.21818181818181817, 0.12, 0]。
状态输出显示实际 sigma，避免把 KSampler 的“截掉前面若干步”误当成这套路线。

默认 3 步、0.30 来自当前用户测试偏好，不是所有素材的画质保证。
原发布高降噪路线改动画面更明显；保真测试优先低降噪路线。
CLIP 提示词仍使用当前 ComfyUI 的 H3 编码路径；不同 VAE、文本编码精度、随机数布局和内核会影响最终像素，
这里不宣称与 Diffusers 桌面版逐像素相等。

## Attention 与 FFN

inherit 原样保留输入 MODEL 的 attention、FFN 及其他兼容补丁。
显式选择 Sage 时调用现有 JR acceleration adapter → 已安装 KJNodes。
`sageattn_qk_int8_pv_fp8_cuda++` 对应安装版的 `pv_accum_dtype="fp32+fp16"` 路线；
没有另装或替换 Sage，也不修改桌面版 attention。

上游已有 attention override / sparse block patch 时，禁止再覆盖，改用 inherit。
**LoRA 必须放在 Unified V2 之前**，让 V2 的所有权检查看到完整最终栈；
新节点不二次分块 FFN，不替换原有 MLP。
验证工作流：Sage 2++、Low VRAM attention 的 head_chunks=4、FFN chunks=32、threshold=4096、Sol 关闭。
FFN 由 Unified 控制，可按自己的显存调整。稀疏 attention 的速度和画质需独立验证。

放大复用现有 JR Neural Latent Upscaler，包含原有 VAE-domain / normalized-domain 转换。
keep 可接已经放大的 latent；scale / megapixels 内部先神经放大再二采。
2048² 选 4.194304 MP；2560×1440 选 3.6864 MP（依输入比例及原有网格对齐规则）。

## 分窗版验证记录（2026-10-29）

- 全仓回归 1262 passed / 9 skipped；随后补充 29.97 fps 逐窗帧一致性用例通过。
  新版覆盖完整范围、前缀锁定、全局噪声分区不变、固定种子复现、音频对象保留、
  44.1 kHz 立体声 PCM 逐值旁路和逐窗编码上限。38 节点注册与 36 工作流结构检查通过。
- 生产 Python 3.13 / Torch 2.11 / ComfyUI 0.43 运行时执行三个真实原生 H3 微型模型窗口通过，
  并实际读取指定 MP4 的两个窗口。API 示例通过原生 prompt 校验。
- RTX 5090：指定样片 800×800 / 175 帧 / 24 fps → 2048×2048 / 175 帧 / 24 fps。
  完整 BF16 Ref2VA、208 组主投影 + 50 个 Sharp 分支、现有 Sage 2++ / Unified FFN，
  3 步 / 0.30 / seed 1766580932 / 5s 档，共两个窗口（124 帧、90 帧；重叠 39 帧）。
- 输出原音频 AUDIO 与源解码 PCM 逐值相等，MP4 有音轨；最终合成会重新编码音轨。
  窗口交界附近检查未见明显全局色彩跳变，不能外推为所有素材都无接缝。
- 本次总耗时约 836 秒；含加载与逐窗重建约 739 秒。重建阶段 PyTorch peak allocated
  26.93 GiB、reserved 27.86 GiB。没有同素材整段 A/B 时间与内存基线，不能据此声称加速倍率。

## 上一版整段实现验证记录（2026-10-28）

- 开发源码先合并 GitHub main 7849887；既有未提交内容保存到本地 backup/pre-regenerate-20261028。
- 隔离运行时使用 JR 最新 CI 的 ComfyUI 0.39.0，926d828。
- 全仓 1225 passed / 9 skipped；38 节点注册、36 个工作流检查通过。
  四条旧工作流残留链接警告与本次无关。
- 新回归涵盖 QKV/SwiGLU 转换、Sharp 插值与异常清理、每步 clean audio、
  FP32/FP16/BF16 输入音频逐值保留、FFN 补丁复用、模型分支隔离、缺件拒绝和两套 sigma。
- RTX 5090：真实 BF16 Ref2VA + 本机 INT8 Qwen + 原生音视频 VAE +
  208 主投影 / 50 Sharp 分支 + 已安装 KJ Sage 2++ / FFN。
  真实 5 秒 800×800 输入 → 2048×2048 / 124 帧（补帧后），3 步 / 0.30。
  音频输出同一对象、逐值相等。三步采样约 400 秒，含加载、编码、放大和解码约 517 秒。
  PyTorch peak allocated 26.41 GiB、peak reserved 26.65 GiB；这些不是 NVML 全进程显存数。
- 独立 ComfyUI 服务的真实 VIDEO 输入工作流也执行成功：5 帧 → 2048×2048，完成音视频编码、节点采样、原生解码与 MP4 保存；原生 prompt 校验无错误。
- 以上证明当前本机流程可运行，不等于所有时长、量化模型或稀疏 attention 的质量验收。

转换依据：[Sana Open-H3-Regenerate-2K](https://github.com/NVlabs/Sana/tree/6e4a44002149418a547434b91d7280a8bbdcace3/models/Open-H3-Regenerate-2K)，
commit 6e4a44002149418a547434b91d7280a8bbdcace3 的 adapters.py、merge_adapters.py、sigmas.py。
权重来源 Efficient-Large-Model/Open-H3-Regenerate-2K，
revision 5809c33730c18b6b1c827b40974855a4288641b3。
本节点独立实现格式映射与 ComfyUI 集成；不随源码分发模型文件。
