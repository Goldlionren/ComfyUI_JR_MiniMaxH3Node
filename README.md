# ComfyUI JR MiniMax H3 Node

面向 MiniMax H3 工作流的 ComfyUI 自定义节点套件。当前版本注册 19 个 V1 Python 节点，覆盖混合模型加载、多模态导演时间线、标准媒体与 Director PIPE 互转、H3 提示词生成与校验、人工审核、原生 H3 conditioning、AV latent 构建与拆分、音频驱动 latent 注入与锁定、H3 neural latent 空间放大、顺序时间分块采样、模型加速、实验性缓存、分辨率规划、RTX 后处理、视频编码和末帧续接。

当前包版本：`0.15.0`。请以 Git 提交和 [CHANGELOG.md](CHANGELOG.md) 为准。

## 节点一览

| 显示名称 | 稳定 Node ID | 分类 | 主要输出 |
| --- | --- | --- | --- |
| JR MiniMax H3 Hybrid Loader | `JR_H3_HybridLoader` | Loaders | 一个原生 `MODEL` |
| JR MiniMax H3 Director Desk | `JR_H3_DirectorDesk` | Director | 原始 Director Prompt、`JR_H3_DIRECTOR_PIPE` |
| JR MiniMax H3 Director PIPE Builder | `JR_H3_DirectorPipeBuilder` | Director | 标准 STRING/IMAGE/VIDEO/AUDIO 组装的 PIPE |
| JR MiniMax H3 Director PIPE Unpack | `JR_H3_DirectorPipeUnpack` | Director | 透传 PIPE、prompt stages、选定的标准媒体 |
| JR MiniMax H3 Prompt Optimizer (OpenAI Compatible) | `JR_H3_OpenAICompatiblePromptOptimizer` | Prompt | 优化提示词、原提示词、状态、派生 PIPE |
| JR MiniMax H3 Prompt Review & Continue | `JR_H3_PromptReviewPause` | Prompt | 人工确认后的提示词、派生 PIPE |
| JR MiniMax H3 Directed Video Conditioning | `JR_H3_DirectedVideoConditioning` | Generation | 原生 H3 `CONDITIONING`、AV `LATENT` |
| JR MiniMax H3 Audio Driven Latent Builder | `JR_H3_AudioDrivenLatentBuilder` | Latent | 音频驱动并锁定 audio 分支的 H3 AV `LATENT`、状态 |
| JR MiniMax H3 Sequential Audio Chunk Driver | `JR_H3_SequentialAudioChunkDriver` | Sequential Audio | 当前音频驱动 AV `LATENT`、chunk context、seed、AUDIO slice、状态 |
| JR MiniMax H3 Sequential Continuation Guide | `JR_H3_SequentialContinuationGuide` | Sequential Audio | 末帧续接后的 `CONDITIONING` 与 `LATENT` |
| JR MiniMax H3 Sequential Latent Checkpoint | `JR_H3_SequentialLatentCheckpoint` | Sequential Audio | CPU/磁盘 checkpoint 后的 H3 AV `LATENT` |
| JR MiniMax H3 Sequential Video Output | `JR_H3_SequentialVideoOutput` | Sequential Audio | 分段提交状态与最终连续音频 MP4 路径 |
| JR MiniMax H3 AV Latent Builder | `JR_MiniMaxH3AVLatentBuilder` | Latent | H3 AV `LATENT`、校验状态 |
| JR MiniMax H3 Split AV Latent | `JR_H3_SplitAVLatent` | Latent | 独立 video/audio `LATENT` |
| JR MiniMax H3 Neural Latent Upscaler | `JR_MiniMaxH3NeuralLatentUpscaler` | Latent | neural 放大后的 video `LATENT`、状态 |
| JR MiniMax H3 Temporal Chunk Sampler | `JR_H3_TemporalChunkSampler` | Sampling | 分块采样后的 H3 AV `LATENT`、状态 |
| JR H3 Cache Config Router | `JR_H3_CacheConfigRouter` | Cache | 缓存配置、建议档位、分析 |
| JR H3 Adaptive Cache | `JR_H3_AdaptiveCache` | Cache | 已 patch 的 MODEL、实际档位、状态 |
| H3 Unified Acceleration | `JR_H3_UnifiedAcceleration` | Optimization | 已 patch 的 MODEL |
| JR MiniMax H3 Resolution Scale Calculator | `JR_H3_ResolutionScaleCalculator` | Scaling | 宽、高、缩放倍数、实际 MP |
| JR MiniMax H3 RTX Upscaler & Refiner | `JR_H3_RTXUpscalerRefiner` | Video | 后处理 IMAGE |
| JR MiniMax H3 Enhanced Video Combine | `JR_H3_EnhancedVideoCombine` | Video | IMAGE 帧、保存路径 |
| JR MiniMax H3 Last Frame | `JR_H3_LastFrame` | Utility | 最后一帧 IMAGE |

完整输入、默认值、范围和输出见 [节点参数参考](docs/NODE_REFERENCE.md)。

## 安装与更新

停止 ComfyUI 后，在它的 `custom_nodes` 目录执行：

```powershell
git clone https://github.com/Goldlionren/ComfyUI_JR_MiniMaxH3Node.git
<ComfyUI-Python> -m pip install -r .\ComfyUI_JR_MiniMaxH3Node\requirements.txt
```

必须使用 **运行 ComfyUI 的同一个 Python**。便携版、整合包和 Launcher 的 Python 路径可能不同，不要默认使用系统 Python。

如果目录本来就是从 GitHub 克隆的：

```powershell
cd ComfyUI_JR_MiniMaxH3Node
git pull origin main
<ComfyUI-Python> -m pip install -r .\requirements.txt
```

如果 `git pull` 提示没有 `origin`，说明这个目录不是正常克隆得到的仓库，或远程配置已丢失。最稳妥的做法是保留旧目录备份，然后重新 `git clone`；不要在没有确认来源的目录中强行合并。

更新后重启 ComfyUI，并对浏览器做一次强制刷新，以加载最新的预览和审核界面 JavaScript。

## 依赖

普通依赖：

- ComfyUI 自带 `torch`、`numpy` 和 Pillow。
- `imageio-ffmpeg>=0.5`，用于在系统 PATH 没有 FFmpeg 时提供可执行文件。
- Prompt Optimizer 和 Cache Config Router 需要 OpenAI 兼容的 `/v1/models` 与 `/v1/chat/completions` 服务。

可选 RTX 依赖仅支持合适的 Windows/NVIDIA 环境：

```powershell
<ComfyUI-Python> -m pip install -r .\requirements-rtx.txt
```

发行包名称是 `nvidia-vfx`，Python 导入名是 `nvvfx`。不同 binding 暴露的 `QualityLevel` 不完全一致：VSR 可用不代表 Denoise/Deblur 一定可用；节点会在执行相应效果时给出明确错误。

Unified Acceleration 的外部依赖不会由本仓库自动安装：

- [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
- KJNodes 所选 Sage 模式需要的 `sageattention` 或 `sageattn3`
- [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) 及其 Triton 运行环境

这些依赖均在节点执行时才解析；缺少它们不会阻止其他 JR 节点加载。本仓库不复制 KJNodes、Sol-Attn、SageAttention、Triton 或 NVIDIA SDK 源码。

## 推荐接线

### Hybrid Loader 与模型 patch 链

```text
JR MiniMax H3 Hybrid Loader
  -> Turbo LoRA（可选）
  -> H3 Unified Acceleration（可选）
  -> sampler / H3 workflow
```

Hybrid Loader 直接选择 `diffusion_models` 下的一份 FL2VA checkpoint 和一份 REF2VA checkpoint。Hybrid profile 始终只完整加载 FL：FL state dict 走当前 ComfyUI 原生 `load_torch_file`，因此会继承当前 AIMDO/mmap 或 stock fallback；REF 先扫描 safetensors header，再只读取计划内的 AdaLN tensor family，复制为自有 CPU tensor后立即关闭 REF，最终由 stock `load_diffusion_model_state_dict` 构造唯一 MODEL。它不会实例化两个完整 H3 MODEL。

#### 方法论：Hybrid 是参数来源策略，不是双模型拼接

这个节点把一次 Hybrid 加载拆成“规划、验证、选择性读取、原生构造”四个阶段：

```text
FL header ─┐
           ├─> HybridPlan：确定每个 selected tensor family 来自 FL 还是 REF
REF header ┘                    │
                                ├─> selected family 兼容性验证
FL ── ComfyUI native full load ─┤
REF ─ selected-only read/copy ──┘
                                ↓
                   stock ComfyUI MODEL construction
```

核心原则如下：

1. **FL 是唯一完整基座。** 模型架构、未选择参数、output heads 和默认 metadata 都以 FL 为权威；REF 只提供 profile 明确选中的参数，不会生成第二个完整 MODEL。
2. **先看 header，再读 tensor。** 节点先用两个 checkpoint 的 safetensors header 生成确定性的 `HybridPlan`，在读取大规模数据前完成 H3 layout、block 范围、key、shape、dtype 和量化表示验证。
3. **替换完整 tensor family，而不是孤立 weight。** 一个量化 linear 的 `weight`、`bias`、`weight_scale`、`.comfy_quant` 及 header 中实际存在的同族 metadata 必须保持同一来源，避免产生半 FL、半 REF 的无效量化表示。
4. **只要求被替换的 family 兼容。** FL/REF 全局 key set 可以因为剪枝或量化 metadata 而不同；只有真正选中的 family 必须拥有相同成员、shape 与 dtype。未证明安全的跨格式组合会 fail-closed，不做隐式反量化、重铸或猜测。
5. **REF 生命周期在 MODEL 构造前结束。** 计划内 REF tensor 被读取为独立 owned CPU copy 后立即关闭 REF handle；未选 REF tensor 从不调用 `get_tensor()`。这些 copy 覆盖进原生 FL state dict，未替换的 FL storage 不由插件 clone。
6. **最后仍交回 ComfyUI。** Hybrid state dict 和 FL metadata 交给 stock `load_diffusion_model_state_dict`，从而保留当前 ComfyUI 的模型识别、ModelPatcher、Dynamic VRAM 与缓存重建路径，而不是由插件自行实例化 MiniMax H3。

因此，`Recommended` 应理解为一个可复现、可审计的**实验性参数来源假设**：让 REF 的后半段 block AdaLN 参与条件调制，同时保留 FL 的主体与输出端。它不代表把 FL/REF 能力按固定比例相加，也不保证所有量化版本都有相同的质量或内存收益。日志中的 plan fingerprint、family 数、tensor 数和 selected bytes 才是本次加载实际发生了什么的依据。

- `Recommended`：REF blocks 25–49 AdaLN；Final AdaLN、video/audio output heads 及其余权重来自 FL。
- `All Block AdaLN`：REF blocks 0–49 AdaLN，Final AdaLN 来自 FL。
- `All Block AdaLN + Final`：REF blocks 0–49 AdaLN 加 Final AdaLN。
- `Custom Range`：REF 使用指定 block 闭区间；Final 由开关决定。
- `Pure FL` / `Pure REF`：直接调用 stock diffusion loader，且不会打开另一份 checkpoint。
- `Advanced Custom`：`custom_ref` 选择 prefix/glob，`custom_fl` 以完整 tensor-family 粒度强制退回 FL。

Header resolver 会让量化 weight、scale、`.comfy_quant` 等实际存在的 sibling 同源，并对 selected family 的 key、shape、dtype/quant representation 做 fail-closed 校验。全局不相关 key 不同不会导致整个 Hybrid 被拒绝。BF16、普通 INT8 ConvRot、pruned INT8 的 AdaLN 表示和内存量差异很大，不能跨格式混配；日志会报告实际 selected family/tensor/byte 数，不承诺固定 RAM 节省比例。`Recommended 25–49` 是 Scott Mudge 项目的实验建议，不是 MiniMax 官方推荐。详见 [Hybrid Loader](docs/H3_HYBRID_LOADER.md)。

### Director Desk、提示词与人工审核

```text
Director Desk.pip
  -> Prompt Optimizer.pip
       pip
         -> Prompt Review & Continue.pip
              pip
                -> Directed Video Conditioning.pipe
                     positive + latent
```

Director Desk 是不调用 LLM 的时间线编辑器。它把 Global Direction、Shot、图片、视频、音频和每项 Direction/Notes 确定性地编译为 raw `director_prompt`，并通过一根自定义类型的 `pip` 线把完整结构交给现有 Prompt Optimizer。Optimizer 将 `optimized_prompt` 写入一个新 PIPE，Review 将最终批准文本写入另一个新 PIPE，最后由 Directed Video Conditioning 直接调用当前 ComfyUI 原生 MiniMax H3 I2V/Ref2V conditioning。三个节点都不会原地修改上游 PIPE。

不需要 Director Desk 时间线时，可以用 `Director PIPE Builder` 把最终 prompt、duration/fps、首尾帧、Reference IMAGE batch、标准 VIDEO、Reference AUDIO 和 Driving AUDIO 直接组装为同一种 PIPE；输入 prompt 会逐字保存为当前 optimized stage，同时保留一个确定性的单 Shot Director context。`Director PIPE Unpack` 接受任何来源的 PIPE，原样透传总线，并按 1-based index 输出选定的标准 Reference Image/Video/Audio、首尾帧、prompt stages、时长、fps、尺寸和不含二进制的 registry JSON。索引不存在时对应媒体输出为 `None`，完整媒体仍保留在透传 PIPE 中。详见 [Director PIPE Builder / Unpack](docs/DIRECTOR_PIPE_IO.md)。

`director_prompt`、`optimized_prompt`、`reviewed_prompt` 等 STRING 输出用于监控、检查和调试；`JR_H3_DIRECTOR_PIPE` 才是 Director 主链唯一权威数据总线。最终提示词优先级固定为 `reviewed > optimized > director`。

工作流只保存轻量时间线和 ComfyUI `input/temp/output` 资产 descriptor；不会保存 Tensor、base64、音频 waveform 或视频字节。First Frame 是固定在 0.0 秒的唯一点锚；Visual 和 Reference Audio 可以重叠，Driving Audio 不允许重叠。拖动、resize、split、duplicate、delete、role 和 Direction 编辑都在节点内完成，节点只在首次创建时采用约 `1000×650` 默认尺寸，不会在执行后缩回。

PIP 连接后，PIP 的 prompt、duration、registry 和媒体是权威来源；Optimizer 的 legacy `duration_seconds` widget 必须与 PIPE timeline duration 相同，同时连接旧的 `first_frame`、`last_frame`、`ref_image_1..9` 或 `reference_instructions` 也会明确报冲突，避免静默覆盖、合并和重新编号。Review 的 STRING 只允许为空或与 PIPE 的权威审核文本完全相同。详情见 [Director Desk](docs/DIRECTOR_DESK.md) 和 [Director Pipeline](docs/DIRECTOR_PIPELINE.md)。

审核节点默认等待 `3600` 秒。每次排队都会再次审核；它需要发起任务的浏览器保持在线，不适合无人值守 API 队列。

### Router 与 Adaptive Cache

```text
Prompt Optimizer.optimized_prompt
  -> Cache Config Router.optimized_prompt

Cache Config Router.cache_config
  -> Adaptive Cache.cache_config

MiniMax H3 MODEL
  -> Adaptive Cache.model
       MODEL -> sampler
```

只需要把 Router 的 `cache_config` 接到 Adaptive Cache 的同名输入。Router 的 `selected_profile` 和 `analysis` 是供显示、记录或调试的 STRING 输出，**不需要**连接到 Adaptive Cache。连接 `cache_config` 后，Adaptive Cache 的手动 widgets 会被整组忽略。

Adaptive Cache 是实验性、内容相关的优化：档位被选中不等于必然命中，日志中的 `full_hits=0` 或 `block_hits=0` 可能只是当前采样变化超过阈值。不要把“选择了 dialogue_safe”误解为“保证加速”。

### 模型加速链

```text
Load Diffusion Model
  -> MiniMax H3 Turbo LoRA（外部）
  -> Reserved VRAM Setter（外部，可选）
  -> H3 Unified Acceleration
  -> JR H3 Adaptive Cache（可选且实验性）
  -> MiniMax H3 Sigma Shift（外部）
  -> Basic Guider / Basic Scheduler
```

Unified 节点内部顺序固定为：

```text
KJ Sage
  -> MiniMax H3 Low VRAM Attention
  -> MiniMax H3 Chunk FeedForward
  -> Sol-Attn
```

Sol 必须最后安装，才能把 Sage 保留为不适用场景的 previous dense backend。每个 enable 开关都是真正 bypass，而不是用 chunk 值模拟关闭。详情见 [H3 Unified Acceleration](docs/H3_UNIFIED_ACCELERATION.md)。

### 解码、放大和保存

```text
VAE Decode IMAGE
  -> Resolution Scale Calculator
  -> RTX Upscaler & Refiner
  -> Enhanced Video Combine
       frames（pass_frames=true）
         -> Last Frame
```

Resolution Calculator 的 `divisor` 是字符串下拉选项 `"8"`、`"16"`、`"32"`，同时兼容旧工作流保存的数值 `8/16/32`。

## Prompt Optimizer

Prompt Optimizer 是本地 H3 Prompt/Context 预处理器，不是 MiniMax 托管的 H3-Context-IR。它：

- 固定使用 MiniMax-H3 commit `8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea` 的 Prompt Writing 规范：LLM 只返回语义 JSON，Python 确定性生成最终官方 H3 字段、顺序、Shot、时间戳、对白、reference label 与 retention enum。
- 支持 `Auto`、`T2VA`、`I2VA`、`FL2VA`、`L2VA`、`Ref2VA`。
- 支持 `first_frame`、`last_frame` 和 `ref_image_1..9`；每个 reference slot 可以携带 IMAGE batch。
- 支持 optional `pip: JR_H3_DIRECTOR_PIPE`；PIP 不存在时旧工作流行为不变，新增的第四个 PIPE 输出为 `None`。
- PIP 模式成功后返回派生 PIPE，并写入 `optimized_prompt`；原 PIPE 的时间线、registry 和 runtime media 原样保留。
- 接受服务根地址、`/v1`、`/v1/models` 或完整 `/v1/chat/completions` 地址。
- `model` 留空时，仅在执行阶段查询 `/v1/models`。
- 语义 JSON 初次 schema 校验失败时最多进行 **一次** `temperature=0.1` 的结构化修复；随后 Python formatter 生成最终文本并运行严格 validator。
- 使用 closed-world 忠实改写规则：Director direction/notes/timing、显式用户要求和参考图中可直接观察的事实是完整真值源；profile 只能改变表达重点，不能新增人物关系、剧情动机、动作、姿势、表情、道具行为或音画事件。未指定内容必须省略，不能用 `or`、`likely`、`perhaps` 等备选或猜测表达补全。

成功状态：

```text
Success: model=<id>, mode=<mode>, repaired=0
Success: model=<id>, mode=<mode>, repaired=1
```

最终仍失败时：

- `Return Original`：返回原始用户提示词，状态为 `Fallback: <原因>`。
- `Stop Workflow`：抛出描述性错误并停止工作流。

修复不会无限重试，也不会为了减少失败而放宽 validator。不同 OpenAI-compatible 模型仍会带来不同语义质量，但最终 H3 结构不再由模型自由排版。对白原文由程序逐字保护，Base 对白只进入 `integrated_multimodal_description`，Ref2VA 对白只进入 `detailed_description`，不会重复到 `overall_soundscape`。

Auto 模式优先级：

| 已连接输入 | Auto 结果 |
| --- | --- |
| 任意 reference IMAGE，或 `reference_instructions` 中出现有效引用标签 | Ref2VA |
| 仅 `first_frame` | I2VA |
| `first_frame` + `last_frame` | FL2VA |
| 仅 `last_frame` | L2VA |
| 均无 | T2VA |

显式模式会拒绝冲突输入，不会偷偷切换模式。实现与升级边界见 [Official H3 Prompt Formatter](docs/H3_OFFICIAL_PROMPT_FORMATTER.md)；clean-room 格式来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、[固定规范来源](specs/minimax_h3_prompt/8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea/SOURCE.md) 与 [resources/minimax_h3_spec](resources/minimax_h3_spec/)。

## Prompt Review & Continue

审核节点同时支持旧 STRING 模式和 Director PIPE 模式。PIPE 模式按 `optimized_prompt > compiled_director_prompt` 选择审核文本，点击 **Next / Continue** 后返回 `reviewed_prompt: STRING`，并把同一批准文本写入新的 PIPE。

- 默认超时 `3600` 秒，范围 `60..86400`。
- 最小节点尺寸约为 `460×360`；前端不会把用户手动放大的节点缩回默认值。
- 刷新浏览器后会按 ComfyUI client ID 恢复仍在等待的审核。
- Stop、超时或关闭浏览器且不重连都会阻止下游继续。
- 提示词只保存在有限的内存状态中，不写入普通日志。

详见 [Prompt Review & Continue](docs/PROMPT_REVIEW_CONTINUE.md)。

## Directed Video Conditioning

`JR_H3_DirectedVideoConditioning` 直接消费审核后的 PIPE，并复用当前 ComfyUI 的 `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` 实现，输出可直接进入 H3 下游采样链的标准 `CONDITIONING` 与 AV `LATENT`。

- `Auto`：存在任意 Reference Image/Video/Audio 或 Driving Audio 时选择 Reference to Video；否则选择 Image to Video。
- 显式 Image to Video 遇到 Ref2V-only 媒体会明确报冲突，不会静默忽略。
- `Prefer Pipe`：从首个 Picture/Video 媒体尺寸推导画布；时长按固定 H3 `24 fps` 转成 `ceil(duration×24)` 帧，再由原生节点执行 `n % 17 == 5` 对齐。没有媒体尺寸时回退节点宽高。
- `Prefer Node`：使用节点 `width/height/length`。
- 原生限制为最多 9 张参考图、3 个参考视频、3 个独立参考音频；Ref2V 下首/尾帧也计入这 9 张 Picture 总额。`<Picture N>/<Video N>/<Audio N>` 顺序与实际送入原生节点的顺序一致。
- LLM 阶段不会上传 video/audio 二进制；Conditioning 阶段才按安全 descriptor 延迟解码并真正消费媒体。
- 参考视频必须解码为 24 fps，裁切后至少 5 帧；单条最多解码 15 秒，并受像素预算保护。原生实现随后按 `17k+5` 帧网格裁切参考帧。
- 文件音频在解码前必须具有可信的大小、时长、采样率和声道元数据，并受文件大小、时长和解码采样总量预算保护。
- `Prefer Pipe` 的 timeline 超过 150 秒会超过节点的 3600 帧输入上限并明确拒绝；这不是对完整 H3 工作流或模型能力的通用时长承诺。
- Ref2V 原生接口没有首尾帧硬锚。首/尾帧与其他 reference 同时出现时会作为普通参考图送入，不能声称仍有 I2V 硬锚语义。
- Driving Audio 映射到原生 standalone reference audio；它仍是 Director 的角色/提示词语义，不是模型级目标音轨替换或时间门控。
- 时间线 `start/end` 会保留在 PIPE 和提示词中；当前原生 H3 conditioning 不支持按 clip 区间对 tensor 条件做任意启停。

完整映射和限制见 [Director Pipeline](docs/DIRECTOR_PIPELINE.md)。

## AV Latent Builder、Audio Driven 与 Split AV Latent

`JR_MiniMaxH3AVLatentBuilder` 将上游分别编码好的 H3 video latent 与 audio latent 组装成官方两流 `NestedTensor` LATENT，适合 video-to-video 和 latent-to-latent 工作流。它不是 VAE 编码器、文件读取器、音频处理器或采样器。

```text
IMAGE frames -> H3 Video VAE Encode -> video_latent ┐
                                                     ├-> JR MiniMax H3 AV Latent Builder -> H3 sampler
AUDIO -> H3 Audio VAE Encode -> audio_latent        ┘
```

节点严格要求 video 为 `[B,24,T,H,W]`、audio 为 `[B,32,2,T_audio]`，batch、dtype 和 device 完全一致，数值全部 finite。官方 H3 时间网格为 `T_video=5k+2`，对应 `17k+5` 个 24 fps 原始帧；音频按 40 latent ticks/s 校验，并只容许 ±1 tick 的编码边界差异。节点不会 clone、cast 或移动输入 tensor。详见 [H3 AV Latent Builder](docs/H3_AV_LATENT_BUILDER.md)。

`JR_H3_AudioDrivenLatentBuilder` 用外部经 MiniMax H3 Audio VAE 编码的 audio latent 替换现有 H3 AV latent 的 audio 分支。它保留上游 video noise mask（缺失时才生成 `ones_like(video)`），并强制 audio mask 为 `zeros_like(audio)`，使采样时 video 可生成而 audio 被锁定。

```text
Load Audio -> VAE Encode Audio (MiniMax H3 Audio VAE) -> Audio Drive Latent ─┐
JR MiniMax H3 Directed Video Conditioning -> AV Latent ------------------├-> Audio Driven Latent Builder -> KSampler
```

音频时长以 AV latent 内 template audio 为准：过长从尾部截断，过短在尾部补零，不插值、不循环、不生成尾部。该节点不读取 waveform、不运行 Audio VAE 编码、不 decode 且不 mux；最终输出音质仍可在 Video Combine 中单独 mux 原始 waveform。详见 [H3 Audio Driven Latent Builder](docs/H3_AUDIO_DRIVEN_LATENT_BUILDER.md)。

`JR_H3_SplitAVLatent` 执行相反方向：它只接受 `samples` 为当前 ComfyUI 官方 `NestedTensor` 的 H3 AV LATENT，通过公开的 `unbind()` 按固定 `video, audio` 顺序拆成两个标准 LATENT 字典。节点检查两流数量、Tensor 类型、video `[B,24,T,H,W]`、audio `[B,32,2,T]`、batch 和 finite 值；输出直接引用原始 Tensor，不 clone、不 cast、不迁移 device，也不主动调用 `contiguous()`。因此两个输出均可直接接原生 `Save Latent`，由保存节点按自身标准路径处理连续布局。

```text
Load Latent / H3 sampler -> JR MiniMax H3 Split AV Latent
                              ├-> video_latent -> video latent workflow / Save Latent
                              └-> audio_latent -> audio latent workflow / Save Latent
```

跨工作流可先分别 `Load Latent` 两个输出，再接回 `JR MiniMax H3 AV Latent Builder` 重建官方 H3 AV LATENT。video 与 audio 是不同结构：图像/video latent 的空间放大、插值或 resize 节点只能处理 `video_latent`；`audio_latent` 没有 H/W 空间轴，绝不能送入空间放大链。详见 [H3 Split AV Latent](docs/H3_SPLIT_AV_LATENT.md)。

## Neural Latent Upscaler

```text
H3 AV LATENT -> JR MiniMax H3 Split AV Latent
                  ├-> video_latent -> JR MiniMax H3 Neural Latent Upscaler ┐
                  └-> audio_latent (unchanged) -----------------------------├-> AV Latent Builder -> Pass-2 sampler
```

`JR_MiniMaxH3NeuralLatentUpscaler` 只接受普通 H3 video LATENT `[B,24,T,H,W]`，不接受完整 AV `NestedTensor`，也不处理 audio、conditioning、noise、sigmas、MODEL 或 sampler。它使用用户单独放入 `ComfyUI/models/latent_upscale_models/` 的 H3-specific 3D neural checkpoint；节点不联网下载，缺少 checkpoint 时明确报错，绝不会悄悄退回 nearest/bilinear/bicubic。

- `scale`：宽和高各乘线性倍率，例如 `1.5x` 会使像素面积约乘 `2.25`。
- `megapixels`：目标是 VAE decode 后的 pixel-space MP，不是 latent-grid MP；保持输入宽高比并选择最接近目标 MP 的合法尺寸。
- 节点从当前 ComfyUI 原生 H3 类发现 VAE 16x 空间压缩和 DiT latent 2x2 patch，因而输出 latent H/W 对齐 2、pixel W/H 对齐 32。
- B/C/T、输入 dtype/device 和 LATENT 额外 metadata 保持不变；长时间 latent 内部按时间分块执行 3D 网络，但绝不做 temporal interpolation。
- 推理结束使用 ComfyUI 的 model-specific unload API 把 upscaler 移出 GPU，不调用全局卸载，也不把 `torch.cuda.empty_cache()` 当成模型管理方案。

例如 0.9 MP 一采可用 `scale=1.5` 得到约 2.0 MP，或直接选择 `megapixels=2.0`。checkpoint、license、尺寸算法和限制见 [H3 Neural Latent Upscaler](docs/H3_NEURAL_LATENT_UPSCALER.md)。

## Temporal Chunk Sampler

```text
MODEL ───────────────────────┐
original positive ───────────┤
MiniMax H3 video VAE ────────┤
Random Noise ────────────────┤
Sampler ─────────────────────┼-> JR MiniMax H3 Temporal Chunk Sampler -> sampled H3 AV LATENT
Sigmas ──────────────────────┤
H3 AV LATENT ────────────────┘
```

`JR_H3_TemporalChunkSampler` 不再接收外部 GUIDER。它必须拿到 original `positive`，以便为每一段建立真正独立的 conditioning 和 Basic Guider：Chunk 1 使用 original positive；Chunk 2+ 先把上一段完整 VAE decode 后的最终 RGB 帧通过官方 `MiniMaxH3AddGuide(frame_idx=0)` 加到 original positive，再用得到的 chunk-specific positive 新建 Basic Guider，最后调用原生 `SamplerCustomAdvanced`。

```text
Chunk 1: original positive -> new Basic Guider -> sample -> decoded final frame
Chunk 2+: previous final frame -> AddGuide(frame_idx=0) -> chunk positive -> new Basic Guider -> sample
```

节点仍只输出完整 H3 AV `LATENT + status`。每段 video/audio 结果直接复制到预分配的全长 CPU 缓冲区；处理中只额外保留一张上一段 CPU 末帧，不收集全部 chunk 后再 `torch.cat`。为了获得真实像素末帧，每个非末段都会多做一次完整 video VAE decode；下游最终再 decode 完整 latent 时会有额外 VAE 计算，这是保持现有 LATENT 输出链的明确代价。

H3 video/audio latent 的时间长度不同。内部切点对齐 video 的 5-token / 17-frame 周期，并从同一全局 24 fps frame boundary 换算 40 Hz audio boundary；最终 audio boundary 保留官方编码容许的 ±1 tick。旧 0.17 的 B/C overlap 实验已从节点 UI 移除，因为当前连续性方案只采用无 overlap 分段 + 上一段像素末帧 guide。

重要边界：

- 目标是限制采样期间随时间长度增长的 latent 与中间激活峰值；模型权重、conditioning、上游仍持有的整段 latent，以及当前块的原生 preview/x0 内存不包含在这项节省中。
- 不存在跨块 hidden-state/KV carry、全局时间位置偏移、latent blending 或 decoded crossfade；连续性来自官方 frame-0 image guide，不承诺与整段单次采样数值等价。
- 单块执行原样使用输入 NOISE，不改变原生 seed 语义。多块执行时，从 ComfyUI 实时节点注册表识别并调用官方 RandomNoise/DisableNoise，因此兼容核心 extra node 的路径模块加载身份；RandomNoise 使用 `base seed + absolute frame_start` 的稳定 uint64 派生子流，使相同 shape 的块可重复但不再逐位相同，DisableNoise 保持原生全零语义。其他 generic/custom NOISE 因 ComfyUI 没有公共 clone/offset/substream 协议而明确拒绝，不会静默读取或修改私有属性。状态输出会报告 `noise_mode=native_single`、`chunk_derived` 或 `native_zero`。
- `aggressive_memory_cleanup=false` 默认只依赖引用释放和 ComfyUI 正常内存管理；打开后才在每块结束调用 `soft_empty_cache`，通常更慢。
- 明确拒绝 `noise_mask`，因为不能安全猜测它在 H3 packed AV 双流中的时间映射。
- 多块 original positive 若已经含有 `minimax_keyframes` 会明确报错，避免绝对 full-timeline guide 与逐段局部 frame-0 guide 冲突；推荐使用没有目标帧 keyframe 的 Reference-to-Video positive。

完整算法、输入校验、60 秒规划示例和内存口径见 [H3 Temporal Chunk Sampler](docs/H3_TEMPORAL_CHUNK_SAMPLER.md)。

## Sequential Audio 长音频生成

```text
Directed Video Conditioning -> Sequential Audio Chunk Driver -> Sequential Continuation Guide
Load Audio + Audio VAE ------^                         | positive/latent + per-chunk seed
                                                        v
                                                     Sampler
                                                        v
                                              Sequential Latent Checkpoint
                                                        v
                                                    VAE Decode
                                                        v
                                              Sequential Video Output
```

该流程不是在单次执行中循环 KSampler，而是每块建立一个独立 ComfyUI prompt：默认四个严格 H3 档位为 `345/243/192/141` 帧，对应 `14.375/10.125/8.000/5.875` 秒。音频从全局样本边界切分，Audio VAE 输入只对最后一块补零；最终视频使用同编码器的静音 MP4 分段做 stream concat，并只把完整原始 PCM 编码/融合一次，因此块边界不会发生独立音频编码造成的重复或缺口。

默认 prompt 方法论是 `Same Audio Reactive Prompt`：Ref2VA 官方结构、单个开放式 `[Shot 1]`、不写块边界时间；所有块复用同一 reviewed/optimized prompt，真实变化来自当前 audio slice。`Previous Last Frame` 通过原生 `MiniMaxH3AddGuide` 把上一块末帧锚定到下一块局部第 0 帧；`Independent MV` 不强制续帧并允许不同派生 seed。

缓存默认位于 `ComfyUI/output/temp/JR_H3_audio_jobs`，可配置绝对路径。manifest 只会在 segment 验证成功后前移；浏览器关闭会在当前块后安全暂停，重新打开并手动 Queue 即可恢复。顺序分支的最终节点替代 Enhanced Video Combine，不会把所有 decoded IMAGE 一次性装回内存。详见 [H3 Sequential Audio Generation](docs/H3_SEQUENTIAL_AUDIO.md)。

## H3 Adaptive Cache

该节点面向 ComfyUI 原生 `comfy.ldm.minimax.model.MiniMaxH3Model`，并在运行时读取真实 Block 数量。它没有按模型文件名锁死，所以 bf16、int8、Ref2VA 等权重文件只要最终加载成兼容的原生 H3 模型结构即可；`strict_model_check=true` 时不兼容模型会明确报错。

`cache_device` 只控制大型 residual：

- Metric history 始终留在当前计算设备，使用 detached fp32 抽样。
- CPU residual 命中时才恢复到目标 tensor 的 device/dtype。
- `Auto` 只决定 residual 放 CPU 还是 GPU。
- 日志分别报告 `residual_to_cpu`、`residual_to_gpu` 和 `metric_migrations`；正常情况下 `metric_migrations=0`。

不要与 EasyCache、TeaCache、First Block Cache、CacheDiT、其他 DiT Block replacement cache 或第二个 JR Cache 叠加。Sage/Flash Attention、量化、Dynamic VRAM、CPU offload 和下游 RTX/视频节点不在该冲突列表中。

详见 [H3 Adaptive Cache](docs/H3_ADAPTIVE_CACHE.md)。

## Resolution 与 RTX

Resolution Calculator 按目标像素面积和宽高比计算最接近指定倍数的宽高；输出 `scale` 是面积等效的线性缩放比。

RTX 节点：

- 所有效果关闭时直接返回 RGB。
- Denoise、Deblur、VSR/High Bitrate 通过当前 `nvvfx.VideoSuperRes` binding 的 `QualityLevel` 枚举选择。
- 放大开启时才应用 `Same Size / Scale / Keep Ratio / Preset Ratio / Manual` 目标尺寸逻辑。
- `Center Crop (Fill)` 会裁切以填满目标比例；`Letterbox (Fit)` 会补边。
- 不在 import 阶段加载 `nvvfx` 或初始化 CUDA。

不同 SDK/binding 并不保证支持全部 Denoise/Deblur 枚举；可用功能以运行时检查结果为准。

## Enhanced Video Combine

该输出节点编码 IMAGE batch，并在节点内提供视频预览、Autoplay、Download、保存首帧和保存末帧控制。

- 视频：AV1、VP9、H.265、H.264。
- 容器：WebM、MKV、MP4、Animated WebP、Animated AVIF。
- 编码器顺序：NVENC → QSV → AMF → VAAPI → 软件编码器。
- 音频：Auto、AAC、Opus、MP3，码率 `64k..320k`。
- 支持 8/10-bit、ping-pong、metadata、日期/子目录文件名和 `crop_to_audio`。
- 输出计数扫描同一 basename 的全部扩展名与附加标记，重复运行不会覆盖旧视频。
- 每次执行返回新的 `preview_id`，防止浏览器继续显示上一轮缓存的视频。
- 后端使用 ComfyUI `gifs` 视频 UI payload，同时以 `images` 发布可选 PNG，兼容当前 Node 2.0 前端路径。

H.264 NVENC 常见最大宽度为 4096。横向拼接后出现 `4352×2880` 等超宽画面时，节点会把 Windows 的 `EPIPE/EINVAL (Errno 22)` 识别为当前编码器失败并继续回退到 `libx264`。软件回退能保存，但速度明显更慢。

浏览器不能直接播放的 HEVC、10-bit、MKV 等输出会通过临时 H.264 流预览，Download 始终指向原始保存文件。详见 [Enhanced Video Combine](docs/ENHANCED_VIDEO_COMBINE.md)。

## Last Frame

`Last Frame` 要求非空 `[B,H,W,C]` IMAGE batch，并保持 batch 轴返回最后一帧。若输入来自 Enhanced Video Combine，必须启用 `pass_frames=true`；`save_last_frame=true` 只负责写 PNG，不等同于图中的 IMAGE 输出。

## 示例

`examples/` 当前包含：

- `jr_minimax_h3_director_desk_workflow.json`
- `JR_MiniMax_H3_T2VA加速放大 (ver5.0).json`
- `JR_MiniMax_H3_文生视频&首尾帧生视频_加速放大.json`
- `JR_MiniMax_H3_ref加速放大.json`
- `jr_minimax_h3_prompt_review_workflow.json`
- [WORKFLOW_WIRING.md](examples/WORKFLOW_WIRING.md)

示例可能引用外部 custom nodes、模型和本地资源；导入后请更换缺失节点、模型路径、API 地址和媒体输入。示例参数是工作点，不是硬件上限或普适最佳值。

## 已知边界

- Adaptive Cache 和 Sol-Attn 都是实验性路径，不能承诺每个 prompt 都加速或画质无差异。
- 用户曾完成 RTX 4080 SUPER 16GB（约 0.8MP、15 秒）和 RTX 5090 32GB（1.5MP、15 秒）的工作流验证；两次 workload 不同，不能用总耗时直接比较 GPU。
- 低于约 0.6MP 不适合作为大幅后期放大的高质量起点，是用户经验，不是 MiniMax 官方限制。
- 超宽 H.264 软件回退可能很慢；若交付允许，可改用 H.265/AV1，或让单边宽度保持在硬件编码器限制内。
- Prompt Review 节点要求活动浏览器，不支持无人值守 API。

## 开发验证

```powershell
python -m pytest -q
python -m compileall -q .
python -m ruff check . --exclude .reference
```

import 阶段不会访问网络、加载模型、初始化 CUDA/RTX SDK 或运行 FFmpeg。真实 GPU、真实 H3、网络服务和编码器能力仍需在目标 ComfyUI 环境中单独验证。

## 发布到 Comfy Registry

Release process:

1. 在 `pyproject.toml` 中递增版本号。
2. 按 workflow 触发规则提交并 push 到 `main`。
3. GitHub Action 自动发布该不可变版本到 Comfy Registry。
4. 仓库 Actions Secrets 中必须存在 `REGISTRY_ACCESS_TOKEN`。

普通 Git commit 不等于 Registry release；没有版本号变更时不得覆盖已发布版本。

## 许可证与归属

本项目代码使用 [Apache License 2.0](LICENSE)。FFmpeg、ComfyUI、NVIDIA SDK/binding、KJNodes、SageAttention、Sol-Attn、Turbo LoRA、MiniMax H3 模型与提示词资料保留各自许可与使用条款。

参考仓库、commit、许可审计、clean-room 边界和未 vendoring 声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [NOTICE](NOTICE)。
