# H3 Temporal Chunk Sampler

`JR_H3_TemporalChunkSampler` 是一个独立的 MiniMax H3 AV 顺序时间分块采样节点。它面向“整段 H3 AV latent 可以创建，但把整段一次送入扩散采样时，时间维度相关的显存峰值过高”的场景。

Node ID：`JR_H3_TemporalChunkSampler`

显示名称：`JR MiniMax H3 Temporal Chunk Sampler`

分类：`JR MiniMax H3/Sampling`

## 接线

节点保持当前 ComfyUI Advanced Sampler 的核心接口：

```text
NOISE ──────┐
GUIDER ─────┤
SAMPLER ────┼-> JR MiniMax H3 Temporal Chunk Sampler -> output: LATENT
SIGMAS ─────┤                                      └-> status: STRING
H3 LATENT ──┘
```

输入 H3 LATENT 必须是官方双流 `NestedTensor`：

- video：`[B,24,T_video,H,W]`
- audio：`[B,32,2,T_audio]`
- `T_video = 5k + 2`
- 两流 batch、dtype、device 相同
- `T_audio` 与同一 24 fps / 40 Hz 时间线一致，允许编码边界产生 ±1 audio latent tick

## 原生采样复用

当前实现以本机 ComfyUI commit `de6b062fb5ed1c9b471a3ebcd614705d93d67560` 为核对基线，读取了：

- `comfy_extras/nodes_custom_sampler.py` 的 `SamplerCustomAdvanced.execute`
- `comfy/samplers.py` 的 `CFGGuider.sample` / `outer_sample`
- `comfy/nested_tensor.py` 的官方双流容器
- `comfy_extras/nodes_minimax_h3.py` 的 `FPS=24`、`AUDIO_LATENT_FPS=40` 与 `17k+5` 网格
- `comfy/latent_formats.py` 的 `MiniMaxH3AV.fix_empty_latent`

JR 节点没有复制或重写 denoising loop。每个块都构造成正常 LATENT mapping，然后直接调用当前安装版本的 `SamplerCustomAdvanced.execute`。这样继续继承原生 guider、sampler、sigma schedule、model patcher、callback/preview、进度条和 ComfyUI 模型清理路径，也不 monkey patch ComfyUI 核心。

## 双时间轴规划

H3 video 和 audio 共享内容时间，但不是共享 latent 下标：

```text
video token pattern: 1, 4, 4, 4, 4 frames  -> 5 tokens = 17 frames
video clock:                                  24 frames/s
audio latent clock:                           40 ticks/s
```

算法先把 `chunk_duration_seconds` 换算成完整 17-frame cycle 数，再只在完整 5-token 周期后切 video。每一个全局 video 边界先换算成 24 fps frame boundary，然后独立换算 audio boundary：

```text
audio_boundary = round(frame_boundary * 40 / 24)
```

最后一个 audio boundary 强制等于通过全局校验的实际 `T_audio`，从而保留官方编码链容许的 ±1 tick。尾块过小时会合并到前一块；因此 duration 是近似工作目标，不是硬上限。

60 秒级示例（实际 H3 网格为 1450 帧，约 60.417 秒）：

```text
video T = 427
audio T = 2417
chunk_duration_seconds = 15

#1 video [0:105]   audio [0:595]
#2 video [105:210] audio [595:1190]
#3 video [210:315] audio [1190:1785]
#4 video [315:427] audio [1785:2417]
```

这里 video/audio 的块长度不同，但每一对边界来自同一全局时间点。

## A/B/C 连续性实验

`temporal_mode` 被追加在旧 widgets 之后，默认 `A - Legacy No Overlap`，所以旧 workflow 的输入槽、widget 位置和输出槽保持不变。

### A - Legacy No Overlap

A 继续直接使用原来的 `plan_h3_temporal_chunks()`。duration 换算、尾块合并、native 调用次数、每块 absolute `frame_start` seed、CPU 回填和 metadata 语义没有改动。以 `T_video=217 / T_audio=1227 / requested=5.0s` 为例，范围仍是：

```text
video [0:35] [35:70] [70:105] [105:140] [140:175] [175:217]
audio [0:198] [198:397] [397:595] [595:793] [793:992] [992:1227]
```

### B/C exact window 数学

B/C 保留 A 的 global advance：

```text
cycles = max(1, floor(requested_seconds * 24 / 17))
stride_tokens = 5 * cycles
stride_frames = 17 * cycles
```

本地 window 同时满足官方 `T=5k+2 / frames=17k+5` 和整数 40 Hz audio length。可写成：

```text
window_tokens = 15n + 12
window_frames = 51n + 39
window_audio  = 85n + 65
```

实现选择严格大于 stride 的最小 window，并保守限制在当前原生节点文档所述约 124--362 帧 trained range 内：42/141/235、57/192/320、72/243/405、87/294/490、102/345/575。超过 345 帧时 B/C fail closed；A 不受这个限制。

Canonical 5 秒计划：

| # | sample video | keep video | sample audio | keep audio |
| ---: | --- | --- | --- | --- |
| 1 | `[0:42]` | `[0:42]` | `[0:235]` | `[0:235]` |
| 2 | `[35:77]` | `[42:77]` | `[198:433]` | `[235:433]` |
| 3 | `[70:112]` | `[77:112]` | `[397:632]` | `[433:632]` |
| 4 | `[105:147]` | `[112:147]` | `[595:830]` | `[632:830]` |
| 5 | `[140:182]` | `[147:182]` | `[793:1028]` | `[830:1028]` |
| 6 | `[175:217]` | `[182:217]` | `[992:1227]` | `[1028:1227]` |

最后一个 window 若会越过 global end，会改为 `[T_global-window:T_global]`。它的 overlap 可能比普通 overlap 大，keep 起点始终由 `already_committed_global_end` 决定，因此 video/audio 全局范围没有 gap、duplicate 或额外 token。

- B：完整 current window 都来自 original source latent；overlap 只提供 local attention context，重叠输出被丢弃。
- C：新 suffix 仍来自 original source，但 video/audio overlap 都从已完成的 CPU global output 读取。只搬运当前 overlap，不把完整 output 放回 GPU。
- B/C 使用相同 window、keep/discard、noise seed、guider、sampler 和 sigmas。C 没有 hard lock、mask 或 blending，因此唯一变量是 overlap context source。
- 全局 audio 若处于官方容许的 expected ±1 tick，普通 window 仍严格 exact，贴尾 window 保留真实 terminal tick，不 pad 或截断；status 会报告 delta。

## 顺序执行与输出内存

执行严格串行：

```text
plan timeline
  -> slice current video/audio views
  -> native SamplerCustomAdvanced(current chunk)
  -> allocate full CPU outputs once (from first native result dtype)
  -> copy current result directly into its final CPU slices
  -> delete current sampled result and slice references
  -> optional soft_empty_cache
  -> next chunk
  -> official NestedTensor(CPU video, CPU audio)
```

实现不保存“所有块结果”的 Python list，也不在最后调用 `torch.cat`。全长输出缓冲只在 CPU 各预分配一次。第一块返回后才分配，是为了服从原生 sampler 的实际输出 dtype，而不是擅自假设与输入 dtype 相同。

`aggressive_memory_cleanup=false` 时，每块完成后删除强引用，交给 PyTorch/ComfyUI 正常 allocator 管理。设为 `true` 时额外执行 Python GC 和 ComfyUI `soft_empty_cache()`；它可能降低缓存复用并明显变慢，因此默认关闭。节点不会调用 `unload_all_models`，也不会在导入时初始化 CUDA。

## 显存口径

这个节点能约束的是当前块进入原生 sampler 后产生的、随时间轴增大的 latent 与中间激活。下列内存不因此消失：

- H3 模型权重与 ModelPatcher 常驻量
- guider conditioning/reference tensors
- 上游节点或执行图仍持有的完整输入 latent（尤其当输入本来就在 GPU）
- 当前块的 noise、采样状态、preview/x0 与原生 sampler 临时量
- PyTorch allocator 的 reserved memory

因此，不能只看任务管理器或 `nvidia-smi` 的 reserved 数字判断释放是否成功。应同时观察 `torch.cuda.max_memory_allocated()`、块大小、分辨率、batch、采样器和模型 patch 配置。较小块通常降低峰值，但增加重复的原生 sampler 准备/清理成本。

## Noise 语义

当前 ComfyUI 的标准 `Noise_RandomNoise` 只保存 `seed`。每次 `generate_noise()` 都调用 `comfy.sample.prepare_noise()`，而该函数每次从同一 seed 重新创建 CPU generator。因此，相同 seed 与相同 latent shape 的连续调用会生成逐位相同的 noise；ComfyUI 的通用 `NOISE` 类型没有公共 clone、offset、skip 或 substream 协议。

节点采用以下确定性策略：

- 单块：原样把输入 NOISE 对象交给原生 sampler，不改变对象或 seed，状态为 `noise_mode=native_single`。
- 多块 + 官方 RandomNoise：以 ComfyUI 当前 `NODE_CLASS_MAPPINGS["RandomNoise"]` 的实际运行时输出类型和工厂为准，使用 base seed 与块的绝对全局 `frame_start` 通过 uint64 SplitMix64 permutation 派生 seed。这样既兼容核心 extra node 的路径模块加载身份，也保证相同 workflow/plan 可重复、不同时间起点的 seed 不同，状态为 `noise_mode=chunk_derived`。
- 多块 + 官方 DisableNoise：以实时注册的 `DisableNoise` 输出类型识别，原样保持全零 noise，状态为 `noise_mode=native_zero`。
- 多块 + generic/custom NOISE：采样前 fail closed。节点不假设它存在 `.seed`、不 clone、不 mutate，也不把标准 RandomNoise 实现复制进本项目。

派生只创建当前块的官方 NOISE provider 和当前块 noise；不会预生成完整长视频 noise，因此 bounded-memory 架构不变。它也不承诺等价于“生成一份整段 noise 再按时间切片”。

## 能力限制

- A 无 overlap；B/C 只有 sampling-context overlap，没有 cross-fade、latent blending 或 decoded boundary blending。
- 无跨块 hidden-state、KV 或 recurrent state carry。
- H3 当前原生 sampler/conditioning 没有公开的“全局 chunk 时间偏移”输入；每块在模型看来是独立短片段。
- 因此输出不与整段单次采样 bit-exact，也可能出现内容重启或块边界不连续。
- 同一个完整 sigma schedule 会独立用于每块；这是真采样，不是把已经采样完的 latent 切开。
- A/B/C 都拒绝 `noise_mask`，不会猜测 packed AV 双流 mask 的时间映射；C 也不对 overlap hard lock。
- 多块执行检测到 `minimax_keyframes` 时会明确拒绝。当前原生 keyframe 的 `resolved_frame_index` 以完整目标时间线为原点，而 sampler/guider 没有公开的 chunk offset 或无副作用重映射接口；把同一绝对索引直接交给每块会静默错位。普通 text 与 Ref2VA reference blocks 不使用这个目标帧锚点，可按原生 conditioning 继续传入。
- 节点只输出 sampled H3 AV LATENT，不解码视频或音频，不保存文件。

这些限制是能力边界，不应被文档或 UI 包装成“无缝长视频”。后续若要改善连续性，需要 H3 原生公开 position offset/state carry，或另行设计有实验证据的 overlap/blending 方案。

## 建议起点

- A 可先用 `chunk_duration_seconds=15`；B/C 建议从 `5.0` 开始做同 workflow 的肉眼对照。B/C 请求达到约 14.875 秒时会因下一 exact window 超出保守 trained range而明确拒绝。
- 保持 `aggressive_memory_cleanup=false`，只有确认 allocator 缓存造成压力时再开启。
- 用短 timeline 先比较原生单段与分块结果，检查工作流可运行性和边界质量。
- 如果输入 full latent 已经常驻 GPU，先处理上游 offload；本节点不能释放其他节点仍持有的引用。
- 记录分辨率、batch、video/audio T、sampler、steps、模型 dtype，以及 allocated/reserved peak，才有可复现的内存结论。
