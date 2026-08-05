# ComfyUI JR MiniMax H3 Node

A focused six-node ComfyUI suite for MiniMax H3 prompt preparation and review, resolution planning, RTX enhancement, video encoding, preview, and multi-segment continuity.

面向 MiniMax H3 视频工作流的 ComfyUI 节点套件：提示词优化、分辨率计算、RTX 放大与修复、视频合成预览，以及末帧续接。

## 功能概览

| 节点 | 用途 |
| --- | --- |
| **JR MiniMax H3 Prompt Optimizer (OpenAI Compatible)** | 通过 OpenAI 兼容的 `/v1/models` 与 `/v1/chat/completions` 接口，把简短创意和最多 9 路参考图整理成 H3 中文分镜提示词。 |
| **JR MiniMax H3 Prompt Review & Continue** | 在工作流中暂停，让用户逐字审核或修改 H3 提示词，点击 Next / Continue 后才允许下游继续执行。 |
| **JR MiniMax H3 Resolution Scale Calculator** | 按目标像素面积、宽高比和 8/16/32 倍数计算适合视频模型的宽高。 |
| **JR MiniMax H3 RTX Upscaler & Refiner** | 使用 NVIDIA Video Effects SDK 执行 Denoise、Deblur、VSR/High Bitrate 与尺寸调整；依赖按执行时加载。 |
| **JR MiniMax H3 Enhanced Video Combine** | 将 IMAGE 批次编码为视频或动画，支持节点内预览、Download、首尾帧保存、音频、metadata、ping-pong 和帧透传。 |
| **JR MiniMax H3 Last Frame** | 从 IMAGE 批次提取最后一帧，供下一段 H3 视频继续生成。 |

节点位于 `JR MiniMax H3` 分类；人工审核节点位于其 `Prompt` 子分类。

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
- RTX 节点是可选功能，不影响其他五个节点加载。

### 可选 RTX 支持

需要 Windows、兼容的 NVIDIA RTX GPU/驱动，以及能够导入 `nvvfx` 的 NVIDIA Video Effects SDK Python binding：

```powershell
<ComfyUI-Python> -m pip install -r .\ComfyUI_JR_MiniMaxH3Node\requirements-rtx.txt
```

当前可安装发行包名为 `nvidia-vfx`，Python 导入名为 `nvvfx`。不同 SDK/binding 版本暴露的效果和枚举可能不同；节点会在执行时给出明确错误，不会在 ComfyUI 启动阶段初始化 CUDA 或 SDK。

## Prompt Optimizer

`api_base_url` 可填写服务根地址、`/v1` 或完整的 `/v1/chat/completions` 地址，节点会统一规范化。`model` 留空时才会在执行阶段调用 `/v1/models` 自动选择模型。

支持四种优化档位：Standard、Cinematic Drama、Action、Character Consistency。内置 H3 中文导演提示词会处理 `<Picture N>` 映射、镜头时间轴、人物与场景连续性、硬约束和清晰的结束状态。

节点可连接最多 9 路参考 IMAGE。图片会经过 RGB/RGBA 校验、透明区域白底合成、限边缩放和 JPEG 编码，再按顺序发送。API Key、Authorization header、完整 base64 图片和完整用户提示词不会写入普通日志。

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
