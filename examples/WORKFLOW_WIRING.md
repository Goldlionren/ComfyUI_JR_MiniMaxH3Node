# MiniMax H3 接线说明

## 提示词审核

```text
JR MiniMax H3 Director Desk.pip
  -> JR MiniMax H3 Prompt Optimizer.pip
       pip -> JR MiniMax H3 Prompt Review & Continue.pip
                pip -> JR MiniMax H3 Directed Video Conditioning.pipe
                         positive + latent -> H3 sampler chain
```

STRING 输出仅供监控和调试；PIPE 是主数据总线。审核节点默认超时 3600 秒，需要发起执行的浏览器在线。审核编辑器不序列化为下一次工作流的 socket 输入。

## Router 与 Adaptive Cache

```text
Prompt Optimizer.optimized_prompt -> Cache Config Router.optimized_prompt
Cache Config Router.cache_config -> Adaptive Cache.cache_config
MiniMax H3 MODEL -> Adaptive Cache.model -> sampler
```

Router 的 `selected_profile` 与 `analysis` 只用于显示或记录，不需要连接到 Adaptive Cache。连接 `cache_config` 后，Adaptive Cache 上的手动参数全部忽略。

## Unified Acceleration

```text
Load Diffusion Model
  -> MiniMax H3 Turbo LoRA（外部）
  -> Reserved VRAM Setter（外部，可选）
  -> H3 Unified Acceleration
  -> JR H3 Adaptive Cache（可选）
  -> MiniMax H3 Sigma Shift（外部）
  -> Basic Guider / Basic Scheduler
```

Unified 内部固定为 Sage -> Low VRAM Attention -> Chunk FFN -> Sol-Attn。不要把 Sol 放到 Sage 前面。

## 解码、放大与视频

```text
VAE Decode IMAGE
  -> Resolution Scale Calculator
  -> RTX Upscaler & Refiner
  -> Enhanced Video Combine
```

Resolution Calculator 的 `width`、`height` 可连接 RTX 节点 Manual 尺寸；也可以使用 RTX 的 Scale/Keep Ratio/Preset Ratio 模式。

把原始和放大后帧横向拼接会使宽度相加。例如 2176 宽的两路会得到 4352 宽，可能超过 H.264 NVENC 的 4096 限制。当前 Combine 会回退到 libx264，但会更慢。

## 末帧续接

手动创建：

```text
MiniMax H3 IMAGE frames
  -> JR MiniMax H3 Enhanced Video Combine (images)
       pass_frames = true
       frames -> JR MiniMax H3 Last Frame (frames)
                    image -> Preview Image or the next H3 segment's first-frame input
       filename -> optional downstream STRING consumer
```

`save_last_frame=true` 只把 PNG 写到视频旁边，与图中的 `frames` IMAGE 输出是两回事。连接 Last Frame 必须启用 `pass_frames=true`。

## 示例文件边界

`examples/*.json` 可能引用外部 custom nodes、模型、LoRA 和本地媒体。导入后出现 missing node/model 时，应安装对应外部项目或替换节点；这些 JSON 的参数是示例工作点，不是硬件上限或通用最佳设置。
