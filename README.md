# ComfyUI JR MiniMax H3 Node

A focused eight-node ComfyUI suite for MiniMax H3 prompt preparation, human review, scene-aware caching, resolution planning, RTX enhancement, video encoding, preview, and multi-segment continuity.

面向 MiniMax H3 视频工作流的 ComfyUI 节点套件：提示词优化、分辨率计算、RTX 放大与修复、视频合成预览，以及末帧续接。

## 功能概览

| 节点 | 用途 |
| --- | --- |
| **JR MiniMax H3 Prompt Optimizer (OpenAI Compatible)** | Uses an OpenAI-compatible `/v1/models` and `/v1/chat/completions` service to prepare local H3 Prompt/Context Preprocessor output from text, IMAGE references, and optional first/last anchors. |
| **JR MiniMax H3 Prompt Review & Continue** | 在工作流中暂停，让用户逐字审核或修改 H3 提示词，点击 Next / Continue 后才允许下游继续执行。 |
| **JR H3 Cache Config Router** | 对最终 H3 提示词发起独立的场景分类请求，并用本地版本化 Preset 生成类型安全的 Cache 配置。 |
| **JR H3 Adaptive Cache** | 为原生 MiniMax H3 音视频 DiT 提供 Visual Fast、Dialogue Safe、Action Safe、Balanced、Auto 和 Off 缓存路径。 |
| **JR MiniMax H3 Resolution Scale Calculator** | 按目标像素面积、宽高比和 8/16/32 倍数计算适合视频模型的宽高。 |
| **JR MiniMax H3 RTX Upscaler & Refiner** | 使用 NVIDIA Video Effects SDK 执行 Denoise、Deblur、VSR/High Bitrate 与尺寸调整；依赖按执行时加载。 |
| **JR MiniMax H3 Enhanced Video Combine** | 将 IMAGE 批次编码为视频或动画，支持节点内预览、Download、首尾帧保存、音频、metadata、ping-pong 和帧透传。 |
| **JR MiniMax H3 Last Frame** | 从 IMAGE 批次提取最后一帧，供下一段 H3 视频继续生成。 |

节点位于 `JR MiniMax H3` 分类；人工审核节点位于 `Prompt` 子分类，缓存节点位于 `Cache` 子分类。

## 安装

在 ComfyUI 停止运行时，进入它的 `custom_nodes` 目录：

```powershell
git clone https://github.com/Goldlionren/ComfyUI_JR_MiniMaxH3Node.git
```

然后使用 **ComfyUI 自己的 Python** 安装普通依赖：

```powershell
<ComfyUI-Python> -m pip install -r .\ComfyUI_JR_MiniMaxH3Node\requirements.txt
```

例如，Windows Portable 通常可以在 `ComfyUI\custom_nodes` 下执行：

```powershell
..\..\python_embeded\python.exe -m pip install -r .\ComfyUI_JR_MiniMaxH3Node\requirements.txt
```

重启 ComfyUI。升级插件时进入插件目录执行 `git pull`；如果前端预览控件没有立即更新，请重启 ComfyUI 并对浏览器执行一次强制刷新。

## 运行依赖

- ComfyUI 已提供 `torch`、`numpy` 和 Pillow，本项目不会重复固定这些大型依赖。
- 视频合成需要 FFmpeg。节点会使用系统 `ffmpeg.exe`，也会识别 `imageio-ffmpeg` 提供的可执行文件。
- Prompt Optimizer 需要一个 OpenAI 兼容的本地或远程服务；本地服务可以不填写 API Key。
- RTX 节点是可选功能，不影响其他七个节点加载。

### 可选 RTX 支持

需要 Windows、兼容的 NVIDIA RTX GPU/驱动，以及能够导入 `nvvfx` 的 NVIDIA Video Effects SDK Python binding：

```powershell
<ComfyUI-Python> -m pip install -r .\ComfyUI_JR_MiniMaxH3Node\requirements-rtx.txt
```

当前可安装发行包名为 `nvidia-vfx`，Python 导入名为 `nvvfx`。不同 SDK/binding 版本暴露的效果和枚举可能不同；节点会在执行时给出明确错误，不会在 ComfyUI 启动阶段初始化 CUDA 或 SDK。

## Prompt Optimizer

The node is a local H3-oriented Prompt/Context Preprocessor, not a Chinese-only image-to-video prompt template. It has two deliberate layers: a **JR Creative Director** layer (the selectable JR profiles and continuity choices), followed by a clean-room implementation of the current published MiniMax H3 prompt-format guidance (section names, labels, ordering, timing, and retention taxonomies).

`api_base_url` accepts a service root, `/v1`, or a full `/v1/models` or `/v1/chat/completions` URL; the node normalizes all of them without producing `/v1/v1`. If `model` is blank, `/v1/models` is queried at execution time. `max_tokens` defaults to **1800**; complex Ref2VA descriptions may need a larger value.

Every generated prompt is checked by the full local validator. If the first result fails only at that boundary, the node makes exactly one text-only repair request at `temperature=0.1`, instructing the model to correct H3 formatting without rewriting story/content or changing protected user literals, and then runs the same full validator again. During repair, protected literals found exactly or with whitespace-only mutations are temporarily replaced by counted immutable sentinels and restored locally before validation; removed or duplicated sentinels are rejected. Success status reports `repaired=0` or `repaired=1`. If the repaired result still fails, **Return Original** returns the unchanged user prompt with a concise final reason, while **Stop Workflow** raises a descriptive `ValueError`.

### Input modes

The `h3_input_mode` widget supports `Auto`, `T2VA`, `I2VA`, `FL2VA`, `L2VA`, and `Ref2VA`. In Auto, routing is deterministic:

| Inputs present | Resolved mode |
| --- | --- |
| Any reference IMAGE, or a labelled reference instruction | Ref2VA |
| No references; `first_frame` and `last_frame` absent | T2VA |
| `first_frame` only | I2VA |
| `first_frame` and `last_frame` | FL2VA |
| `last_frame` only | L2VA |

`first_frame` and `last_frame` are single-image anchors. `ref_image_1` through `ref_image_9` are reference images and can contain batches. The registry numbers media in this order—`first_frame`, `last_frame`, then reference-image slots and their batch items—so labels are stable (`<Picture 1>`, `<Picture 2>`, ...). `reference_instructions` may declare downstream `<Video N>`, `<Audio N>`, or `<Subject N>` labels; a `<Picture N>` declaration must resolve to a connected image. Explicit modes reject conflicting inputs instead of silently changing mode.

The profiles **Standard**, **Cinematic Drama**, **Action**, and **Character Consistency** are JR names for the Creative Director layer. They are not official MiniMax format names or an endorsement by MiniMax.

Typical prompts are concise and mode-specific:

```text
T2VA:  text-only scene description -> integrated_multimodal_description
I2VA:  first_frame -> image-anchored opening at 0.00 seconds
FL2VA: first_frame + last_frame -> opening and ending alignment
L2VA:  last_frame -> final-state alignment at the target duration
Ref2VA: ref_image_1..9 and/or labelled instructions -> subject/retention sections
```

The local node is **not** MiniMax's hosted proprietary **H3-Context-IR**, and it does not reproduce, replace, or claim compatibility with that internal system. It sends IMAGE inputs to the configured OpenAI-compatible prompt service. Video and Audio labels can be registered for downstream context, but this node does not upload or universally understand binary video/audio; backend and downstream support determine what those references mean.

Official guide prose and examples are not redistributed. The clean-room metadata records format facts and source hashes in [`resources/minimax_h3_spec`](resources/minimax_h3_spec/), pinned to [MiniMax-AI/MiniMax-H3 commit `8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea`](https://github.com/MiniMax-AI/MiniMax-H3/tree/8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea); see [`UPSTREAM.json`](resources/minimax_h3_spec/UPSTREAM.json) for the source paths and license link.

## Prompt Review & Continue

**JR MiniMax H3 Prompt Review & Continue** is the sixth node and provides a mandatory human-review checkpoint. Connect the Prompt Optimizer's `optimized_prompt` output to its `prompt` socket. When execution reaches this node, the workflow pauses and the incoming text appears in a large editor inside the node.

Edit the text as needed, then click **Next / Continue**. Only the approved text is emitted from `reviewed_prompt`; downstream MiniMax H3 nodes do not execute before approval. Unicode, line breaks, punctuation, and `<Picture N>` tags are preserved exactly.

The review runs again on every queue, even when the input is unchanged. **Stop** cancels the wait, and the configured timeout stops the workflow instead of silently returning the original prompt. This interactive node requires an active ComfyUI browser client and is intentionally unsupported in unattended/headless API workflows. Do not place it in automatic queues or unattended batch jobs. Browser refresh uses the same ComfyUI client ID to recover a still-pending review; closing the browser permanently leaves the workflow waiting until reconnect, Stop, or timeout.

```text
JR MiniMax H3 Prompt Optimizer (OpenAI Compatible)
    optimized_prompt
        -> JR MiniMax H3 Prompt Review & Continue
             reviewed_prompt
                 -> MiniMax H3 text prompt input
```

See [`docs/PROMPT_REVIEW_CONTINUE.md`](docs/PROMPT_REVIEW_CONTINUE.md) for interaction, timeout, cancellation, recovery, and API-mode behavior.
The importable example [`examples/jr_minimax_h3_prompt_review_workflow.json`](examples/jr_minimax_h3_prompt_review_workflow.json) demonstrates the complete Prompt Optimizer → review pause → downstream text preview chain. Set its API URL and model for your OpenAI-compatible service before running.

## H3 Adaptive Cache 与 Cache Config Router

推荐接线：

```text
Prompt Optimizer.optimized_prompt
    -> JR H3 Cache Config Router.optimized_prompt

JR H3 Cache Config Router.cache_config
    -> JR H3 Adaptive Cache.cache_config

MiniMax H3 MODEL
    -> JR H3 Adaptive Cache.model
    -> sampler MODEL
```

Router 会进行第二次、完全独立的 LLM 调用。它只分析已经完成的提示词，不会改写提示词，也不会调用或改变 Prompt Optimizer。LLM 只能返回受限的场景语义分类；阈值、Block 范围、窗口和连续命中限制全部来自本地、版本控制的 Preset。分类失败时默认使用 **Safe Balanced / Conservative**，不会改变 Prompt Optimizer 已生成的文本。

连接 `cache_config` 后，Adaptive Cache 完全采用 Router 配置并忽略节点上的手动 widget；不连接时所有设置均由手动模式、质量预设和高级参数决定。`enable=false` 时 Router 不发请求：`Disable Cache` 返回 Off，其余模式返回本地 Balanced 配置。

六种 Adaptive Cache 模式：

- **Auto**：先使用有效 `profile_hint`，否则按 Speech/Singing、Music/Ambient/None 或安全默认选择。
- **Visual Fast**：视频和音频分别判定，双方稳定才允许 Full-Step 命中并跳过整个 Transformer。
- **Dialogue Safe**：默认 F1-M47-B2；前部探测，缓存中段，尾部刷新，音频可单独否决。
- **Action Safe**：默认 F2-M46-B2；更低阈值、更窄窗口和最多一次连续 Block 命中。
- **Balanced**：低变化走 Full-Step Fast Path，中间灰区走 Block Probe Path，高变化走 Full Path。
- **Off**：不 clone、不添加 patch，原样返回 MODEL。

当前 ComfyUI 原生 `MiniMaxH3Model` 默认检测为 50 个 Block，但实现会在执行时读取真实 Block 数并检查前后区间。缓存 patch 使用官方 ModelPatcher clone、`DIFFUSION_MODEL` wrapper、DiT Block replacement 与 cleanup callback；不会修改 ComfyUI 核心文件。检测到 EasyCache、TeaCache、First Block Cache、CacheDiT、其他 DiT Block replacement 或第二个 JR Cache 时会拒绝叠加。SageAttention、FlashAttention、量化、Dynamic VRAM、CPU offload 和 RTX 后处理不属于 Cache 冲突。

**Diffusion timestep 不是视频时间轴。** Cache 模式控制每个 denoise step 的计算路径，不能在成片“前五秒对白、后五秒动作”之间按视频秒数切换。当前阈值来自本节点 relative-delta 量纲的保守初始校准，仍需在目标 GPU、量化和采样配置上 benchmark。完整状态机、失效条件、设备策略与限制见 [`docs/H3_ADAPTIVE_CACHE.md`](docs/H3_ADAPTIVE_CACHE.md)。

可用 `tools/h3_cache_benchmark.py` 手动记录 No Cache、EasyCache 与四种 JR 策略的真实运行数据，并导出 JSON、CSV 或 Markdown；该工具不会在 pytest 中加载模型或执行长时 benchmark。

## RTX Upscaler & Refiner

可组合 Denoise、Deblur 与 VSR/High Bitrate 放大。当前 `nvidia-vfx` binding 通过 `nvvfx.VideoSuperRes` 和不同的 `QualityLevel` 枚举选择相应处理模式；节点会复用批次内效果对象，并在执行结束后释放 SDK context。

所有效果关闭时，节点安全地执行 RGB 透传。建议先用较小的 IMAGE 批次验证当前 GPU、驱动、SDK 和 binding 组合。

## Enhanced Video Combine

主要能力：

- 节点内视频播放器、Autoplay、Download，以及分辨率、时长和 FPS 信息
- AV1、VP9、H.265、H.264；MP4、WebM、MKV；Animated WebP、Animated AVIF
- NVIDIA NVENC、Intel QSV、AMD AMF、VAAPI 到软件编码器的逐级回退
- 8/10-bit 输出、质量控制、ping-pong、metadata 和安全的日期/子目录文件名
- 可选 AUDIO、AAC/Opus/MP3、64k–320k bitrate 和 `crop_to_audio`
- 保存原生分辨率首帧/末帧，并将视频和图片发布到 ComfyUI Assets
- 按块向 FFmpeg 输送帧、ComfyUI 进度反馈、超时、卡死检测和失败文件清理

`codec=Auto` 会依次实际尝试 AV1/WebM、VP9/WebM、H.264/MP4。浏览器不支持直接播放的 AV1、HEVC、10-bit 或 MKV 文件会通过临时 H.264 兼容流预览，但 Download 始终下载原始输出文件。

完整参数和回退逻辑见 [`docs/ENHANCED_VIDEO_COMBINE.md`](docs/ENHANCED_VIDEO_COMBINE.md)。

## 末帧续接

如果要把合成节点的 `frames` 输出连接到 **JR MiniMax H3 Last Frame**，必须启用 `pass_frames`：

```text
MiniMax H3 IMAGE frames
  -> JR MiniMax H3 Enhanced Video Combine (images)
       pass_frames = true
       frames -> JR MiniMax H3 Last Frame (frames)
                    image -> Preview Image / 下一段 H3 的首帧输入
```

`save_last_frame=true` 保存的是磁盘 PNG，不等同于节点图中的 IMAGE 输出。更多说明见 [`examples/WORKFLOW_WIRING.md`](examples/WORKFLOW_WIRING.md)。

## 常见问题

- **找不到 FFmpeg：** 确认已安装 `requirements.txt`，或者把 `ffmpeg.exe` 加入 ComfyUI 进程的 PATH，然后重启 ComfyUI。
- **没有视频预览或 Download：** 确认整个 `js` 目录和根目录 `__init__.py` 已更新，并强制刷新浏览器。
- **Prompt Optimizer 连接失败：** 检查服务地址和端口；模型发现只在节点执行时发生。
- **HTTP 401：** 检查 API Key；错误消息不会回显密钥。
- **Last Frame 收到空批次：** 在 Enhanced Video Combine 中启用 `pass_frames`。
- **RTX 执行失败：** 检查 `nvidia-vfx`、NVIDIA Video Effects SDK、驱动和当前 binding 暴露的 `QualityLevel`。
- **Auto 选择了较低优先级编码器：** 更高优先级候选在真实运行测试中失败或没有产生进度，节点已自动继续回退。
- **Adaptive Cache 报告冲突：** 同一 MODEL 链中只能保留一个 Cache patch；Attention 或量化节点不需要移除。
- **CPU/Auto Cache：** `cache_device` 只控制大型 residual。抽样后的音频、视频 metric 始终留在当前计算设备；运行摘要会分别报告 residual CPU/GPU 传输和 metric migration。
- **命中始终为零且 resets 很高：** v0.3.2 已移除同一次采样中不稳定的 tensor 内存地址签名。正常 workflow 的首次初始化不计 reset，cleanup 后统计会重新从零开始。
- **Router 已选档但命中为零：** v0.3.3 已按真实 H3 数值尺度重新校准预设。摘要中的 `input_video/audio` 与 `probe_video/audio` 会显示 count/min/avg/max；仍无命中时可据此区分场景确实变化剧烈还是需要针对模型继续校准。

## 开发与验证

```powershell
python -m pytest -q
python -m compileall -q .
python -m ruff check . --exclude .reference
```

开发过程遵循延迟加载原则：import 阶段不会访问网络、加载模型、初始化 CUDA/RTX SDK 或启动 FFmpeg。网络请求、文件操作和 FFmpeg 子进程均包含异常处理、超时与资源清理。

## 许可证与来源

本项目源代码采用 [Apache License 2.0](LICENSE)。FFmpeg、NVIDIA SDK/binding、ComfyUI 和参考仓库保留各自的许可证与使用条款。

功能设计参考了 ComfyUI-DaSiWa-Nodes 与 Comfyui-minimaxh3-FBcache-shendumao。DaSiWa 的实现未被复制到本项目；H3 提示词约束策略根据参考实现重新组织和改写。具体参考 commit、观察到的许可证和归属说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与 [`NOTICE`](NOTICE)。
