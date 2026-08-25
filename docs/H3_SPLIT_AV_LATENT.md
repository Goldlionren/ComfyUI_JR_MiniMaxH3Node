# JR MiniMax H3 Split AV Latent

## 用途

`JR_H3_SplitAVLatent` 将当前 ComfyUI 官方 MiniMax H3 两流 AV LATENT 拆成两个普通 LATENT：

```text
H3 AV LATENT {samples: NestedTensor(video, audio)}
  -> JR MiniMax H3 Split AV Latent
       ├-> video_latent {samples: video}
       └-> audio_latent {samples: audio}
```

stream 顺序固定为 video、audio。节点只使用官方 `comfy.nested_tensor.NestedTensor.unbind()`，不会接受仅仅碰巧带有 `unbind()` 方法的任意对象。

## 数据契约

- 输入 `av_latent` 必须是包含 `samples` 的标准 LATENT 字典。
- `samples` 必须是当前运行 ComfyUI 的精确官方 `NestedTensor` 类型。
- NestedTensor 必须恰好含两个 Tensor stream。
- video 必须是 floating、strided、materialized tensor `[B,24,T,H,W]`。
- audio 必须是 floating、strided、materialized tensor `[B,32,2,T]`。
- 两流 batch 必须一致，且所有值必须 finite。

输出字典是新的轻量容器，但其 `samples` 直接引用原始 stream Tensor。实现不会 clone、cast、迁移 device、改变 dtype/shape，也不会主动 contiguous。原生 `Save Latent` 会在保存时自行调用 `samples.contiguous()`，所以两个输出都能沿标准 ComfyUI LATENT 保存路径使用。

## Builder 往返与跨工作流

```text
video_latent ┐
             ├-> AV Latent Builder -> H3 AV LATENT -> Split AV Latent -> video_latent
audio_latent ┘                                                   └-----> audio_latent
```

Builder 与 Split 都保留原 Tensor 对象；在不经过其他处理节点时，往返不会改变数值、shape、dtype 或 device。

如需跨工作流保存和恢复：

1. Split 后分别把 video/audio LATENT 接到原生 `Save Latent`。
2. 在另一个工作流中分别使用 `Load Latent`。
3. 将两个已加载 LATENT 接回 `JR MiniMax H3 AV Latent Builder`。

Builder 会重新执行完整 H3 时间、dtype、device 和 finite 校验后再生成官方两流 NestedTensor。

## 重要边界

video latent `[B,24,T,H,W]` 有 H/W 空间轴；audio latent `[B,32,2,T]` 没有图像空间轴。任何 latent upscale、插值、resize 或面向图像/视频 latent 的空间处理只能连接 `video_latent`。不得把 `audio_latent` 接入空间放大链；音频时间变换必须使用真正理解 H3 audio latent 结构的专用实现。

本节点不解码、不编码、不读取或写入文件、不重采样、不 resize、不改变时间长度，也不检查 Builder 的 video/audio duration 对齐公式；它只验证官方两流容器和必要结构并拆分原始 stream。
