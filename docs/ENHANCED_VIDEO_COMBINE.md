# JR MiniMax H3 Enhanced Video Combine

`JR_H3_EnhancedVideoCombine` 是一个 `OUTPUT_NODE`。它把 ComfyUI `IMAGE` batch 通过 FFmpeg 编码为视频或动画，并返回：

- `frames: IMAGE`：仅当 `pass_frames=true` 时返回实际帧序列；否则是空 batch。
- `filename: STRING`：最终文件的绝对路径。
- UI payload：视频使用 `gifs`，首尾帧 PNG 使用 `images`。

## 预览、下载与 Node 2.0

节点内播放器由 `js/enhanced_video_combine_preview.js` 提供，包含视频预览、分辨率/时长/FPS、Autoplay、Download、Save first frame 和 Save last frame。

当前后端把视频资源放在 ComfyUI 的 `gifs` 字段中，而不是伪装成图片资源。这避免 Node 2.0 或服务器 `/view` 路由尝试用 Pillow 打开 MP4。前端同时接受 `gifs` 和 `videos`，但当前 Python 返回 `gifs`。

每次保存后都会根据文件 mtime 和大小生成新的 `preview_id`，浏览器 URL 带 cache-busting 参数，因此连续执行不会继续播放上一轮缓存。

H.264/MP4、VP9/WebM 和常见 8-bit AV1/WebM 可以直接交给浏览器。HEVC、10-bit、MKV 等浏览器兼容性不足的格式通过 `/jr-h3/enhanced-video-preview` 临时转码成 fragment MP4 流；原始输出不被修改，Download 始终下载原文件。

## 自动选择与回退

`codec=Auto` 依次尝试 AV1/WebM、VP9/WebM、H.264/MP4。H.265 只在显式选择时尝试。

单一 codec 的编码器优先级：

```text
NVENC -> QSV -> AMF -> VAAPI -> software
```

显式组合全部失败后，节点最后尝试 H.264/MP4。可用编码器列表来自实际 FFmpeg `-encoders` 输出；“编译进 FFmpeg”不代表当前硬件一定能执行，所以每个候选仍进行真实编码测试。

### 超宽 H.264

部分 NVIDIA H.264 NVENC 实现最大宽度为 4096。横向拼接原始与放大画面可能得到 `4352×2880`，此时 FFmpeg 会提前退出。在 Windows 上，向已关闭 stdin 写帧可能表现为 `OSError: [Errno 22] Invalid argument`，而不是 `BrokenPipeError`。

当前实现把 FFmpeg 管道的 `EPIPE/EINVAL` 识别为“这个编码器失败”，保留 stderr 并继续尝试后续编码器；已验证 `4352×2880` 会从 `h264_nvenc` 回退到 `libx264`。其他 OSError（例如磁盘满）不会被吞掉。

软件编码超宽视频通常明显更慢。若工作流允许，可改用 H.265/AV1，或让单个 H.264 输出宽度不超过硬件限制。

## 输出命名

默认前缀：

```text
video/%date:yyyy-MM-dd%/%date:hhmmss%
```

支持安全子目录和 date token。路径分量会去除绝对路径、`..` 和 Windows 非法字符，确保目标仍位于 ComfyUI output/temp 目录内。

文件名为 `<basename>_00001.mp4`；连接音频时 basename 追加 `_audio`。计数器扫描同一 basename 的全部扩展名和附加标记，所以重复执行、音频版本和不同容器不会覆盖旧文件。

## 音频

可选 `AUDIO` 接受 ComfyUI waveform 字典，支持 `[B,C,T]`、`[C,T]` 或 `[T]`。编码前临时写为 float32 little-endian PCM，完成或异常后清理。

- Auto：WebM 优先 Opus，其他容器优先 AAC。
- 显式 AAC、Opus、MP3。
- 码率：64k、96k、128k、160k、192k、256k、320k。
- `crop_to_audio=true`：输出按音频时长截断。
- Animated WebP/AVIF 不容纳音频，连接的音频会被忽略。

## 帧、bit depth 与动画

- `pingpong=true` 输出原序列，再追加反向 interior frames；首尾不重复。
- `bit_depth=Auto` 在显式 codec 时抽样判断 8/10-bit；`codec=Auto` 固定为 8-bit。
- Animated WebP 使用 `libwebp_anim`。
- Animated AVIF 使用可用 AV1 编码器。
- `save_output=false` 写 ComfyUI temp；Download 仍指向实际输出。

## 超时与清理

- 硬件编码器启动后 8 秒无进度：终止该候选并回退。
- 软件编码器启动后 120 秒无进度：终止该候选。
- 已开始编码后 120 秒无进度：视为 stall。
- 单次 FFmpeg wait 上限：3600 秒。
- 失败候选文件、metadata 临时文件和 audio 临时文件会清理。
- 错误信息包含编码器、音频编码器和实际分辨率。

## 常见问题

**有保存文件但没有预览**：重启 ComfyUI 并强制刷新浏览器，确保 Python 与 `js` 目录来自同一提交。

**预览仍是第一条视频**：当前版本为每次输出设置新的 `preview_id`。若仍复现，检查是否混用了开发和生产目录的不同版本。

**MP4 被 Pillow 当图片打开**：这是旧版 UI payload/Node 2.0 兼容问题。当前版本使用 `gifs` 视频字段。

**横向拼接后 Errno 22**：通常是 H.264 NVENC 宽度限制。当前版本会尝试 `libx264`，速度变慢属于预期。
