# JR Cut Audio

独立音频上传、波形试听与精确剪辑节点。Node ID：`JR_CutAudio`；分类：`JR MiniMax H3/Audio`。无需 H3 模型、GPU 或额外音频模型。

## 使用

1. **Choose file to upload**：将 MP3、WAV、FLAC 等文件上传到当前 ComfyUI 的 `input/jr_cut_audio`。也可点 **Refresh files** 后选择现有 input 音频。
2. 等待本地解码与波形显示。点击波形定位白色播放线，或拖动播放进度条；Play/Pause 试听，Volume 仅调整试听音量。波形两端的青色边界可拖动。
3. 填写 **Start / End (seconds)**，允许小数秒。也可用 Cursor → start/end。时间基于解码后的源音频，结束点不包含在切片中。
4. 点击 **Cut & lock output**：选区变绿、起止输入锁定，默认试听切出部分。不会自动运行视频工作流。
5. **Download original** 下载原始文件；**Download cut WAV** 下载剪辑后的 float32 WAV。锁定后可切换 source/cut 试听。点击 **Unlock / reselect** 清除旧锁定，再选区、Cut。
6. `audio` 输出接 Director PIPE Builder 的 audio、Sequential Audio Chunk Driver 或其他接收标准 ComfyUI AUDIO 的节点。额外输出为 `duration_seconds` 和 `status`。点击正常 Queue 后才执行下游。

未 Cut 或解锁后，节点拒绝执行，避免把未确认的整首歌曲送进视频流程。播放音量、播放位置及未确认草稿不会改变 AUDIO 输出。

## 音质与保存

- 保留解码后原始采样率、声道及增益；不自动归一化、淡入淡出、重采样或转成单声道。输出形状为 `[1, channels, samples]`。
- 时间使用最近采样点的 half-up 舍入；切出 WAV 与 AUDIO 输出使用同一采样区间。MP3 等有损源已存在的压缩损失不可消除，但剪辑不会再次编码成 MP3。
- 剪辑边界不自动淡化，非零交叉点可能产生点击声；本版优先精确保留选定内容。
- 工作流 JSON 保存源文件相对路径、SHA-256、采样率及选区采样点，不嵌入音乐内容。迁移工作流时还需迁移源文件；同名文件被替换会要求重新 Cut。
- 试听与剪辑 WAV 缓存在 ComfyUI `temp/jr_cut_audio`，同一源/选区复用文件名。删除 temp 后恢复节点可重新生成；不同选区的临时文件可能累积，按正常 temp 管理清理。原始上传文件不自动删除。

## 边界

- 单文件最多 512 MiB；解码后 PCM 最多 256 MiB，最多 2 小时、1–8 声道（先达到任一限制即停止）。PCM 限额不是进程总内存上限；拼接、预览和输出副本还需要额外内存。
- 支持扩展名 MP3/WAV/FLAC/OGG/OPUS/M4A/AAC/AIF/AIFF/WMA；实际 codec 支持以 ComfyUI 已安装 PyAV/FFmpeg 为准。浏览器试听统一使用 WAV；多声道实际播放由浏览器/声卡决定，AUDIO 不下混。
- 解码在 CPU 后台线程，单个服务仅允许一个编辑器处理请求；忙时提示稍后重试，不排入无限队列。解码时限在音频帧之间检查，不能强制中断正在阻塞的底层 codec。
- 只读取 ComfyUI input 内允许扩展名的文件；不加载任意绝对路径或远程 URL。本节点不改变 ComfyUI 本身的访问权限、安全配置或上传服务限制。

## 开发验证

`tests/test_cut_audio.py`：合成音频、实际 PyAV 解码、样本完全一致、源变化、非法路径/范围、资源限额、HTTP 下载/Range、取消请求占用保护。

`node --test tests/test_cut_audio_frontend.mjs`：范围校验和 Comfy 扩展挂载/序列化/恢复/清理契约。

`tools/cut_audio_preview.py --comfy-root <ComfyUI> --scratch <dev-directory>`：localhost:8191 独立界面测试页，合成音频、不启动 ComfyUI、不运行模型。该测试页不替代实际 ComfyUI 中上传、画布缩放及多节点工作流验收。
