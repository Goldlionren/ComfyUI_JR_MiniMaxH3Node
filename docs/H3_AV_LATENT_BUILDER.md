# JR MiniMax H3 AV Latent Builder

## 用途

`JR_MiniMaxH3AVLatentBuilder` 接收两个已经分别编码完成的 MiniMax H3 latent：

- video `samples`: `[B,24,T_video,H,W]`
- audio `samples`: `[B,32,2,T_audio]`

校验通过后，节点返回标准 ComfyUI LATENT：

```python
{"samples": comfy.nested_tensor.NestedTensor((video, audio))}
```

stream 顺序固定为 video、audio。节点保留原 tensor 对象，不 clone、不 cast、不移动 device。

## 推荐接线

```text
IMAGE frames -> H3 Video VAE Encode -> video_latent ┐
                                                     ├-> AV Latent Builder -> H3 sampler
AUDIO -> H3 Audio VAE Encode -> audio_latent        ┘
```

典型用途包括 H3 video-to-video，以及从缓存或其他合法上游 latent 开始的 latent-to-latent 工作流。

## 校验契约

节点采用 fail-closed 策略：

1. 两个输入都必须是含 `samples` 的 LATENT mapping。
2. video 必须是 floating、strided、materialized tensor `[B,24,T,H,W]`。
3. audio 必须是 floating、strided、materialized tensor `[B,32,2,T]`。
4. batch、dtype 和 device 必须完全一致；不会自动 cast 或跨设备复制。
5. 所有 tensor 值必须为 finite，NaN/Inf 会在组装前拒绝。
6. video temporal grid 必须满足 `T_video=5k+2`。

当前 ComfyUI 官方 MiniMax H3 实现将 `17k+5` 个 24 fps 原始帧编码为 `5k+2` 个 video latent token，并按 40 latent ticks/s 计算 audio 长度：

```text
frame_count = 5 + 17k
expected_audio_t = round(frame_count × 40 / 24)
```

audio 允许相对该公式 ±1 tick，用于容纳编码边界的单 tick 舍入差异。例：video `T=37` 对应 124 帧，目标 audio `T=207`；`T=206..208` 可接受，而 `T=400` 会明确报 temporal mismatch。

## 非职责范围

本节点不读取视频或音频文件，不调用 FFmpeg，不抽帧、不 resize、不执行 video/audio VAE encode、不重采样、不 denoise、不 mux、不采样，也不负责原始音频 passthrough。

## 错误与状态

校验错误统一以 `JR MiniMax H3 AV Latent Builder:` 开头，并报告失败的流、实际 shape 或冲突属性。成功时 `status` 会列出两流 shape、batch、dtype、device、推导帧数与时间匹配结果。

实现依据当前 ComfyUI `comfy_extras.nodes_minimax_h3` 和 `comfy.nested_tensor.NestedTensor` 的公开运行结构；未复制第三方节点代码或文案。
