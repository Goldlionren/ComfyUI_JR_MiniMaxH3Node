# 节点参数参考

本页按当前 Python 定义记录全部 23 个节点。保存工作流依赖稳定 Node ID，请不要用显示名称代替 Node ID。

## Hybrid Loader

Node ID：`JR_H3_HybridLoader`

分类：`JR MiniMax H3/Loaders`

输出：`model: MODEL`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `fl_model_name` | diffusion_models COMBO | 首项 | FL2VA authoritative base checkpoint |
| `ref_model_name` | diffusion_models COMBO | 首项 | REF2VA selective overlay checkpoint |
| `profile` | COMBO | Recommended | Recommended、All Block AdaLN、All Block AdaLN + Final、Custom Range、Pure FL、Pure REF、Advanced Custom |
| `weight_dtype` | COMBO | default | default、fp8_e4m3fn、fp8_e4m3fn_fast、fp8_e5m2；与本机 stock Load Diffusion Model 对齐 |
| `block_range_start` | INT | 25 | 0..49；Custom Range 使用 |
| `block_range_end` | INT | 49 | 0..49；Custom Range 使用 |
| `final_adaln_from_ref` | BOOLEAN | false | Custom Range/Advanced Custom 的 additive Final AdaLN 选择 |
| `custom_ref` | STRING | 空 | Advanced Custom 的逗号/换行 prefix 或 glob，最多 64 项/4096 UTF-8 bytes |
| `custom_fl` | STRING | 空 | Advanced Custom 强制退回 FL，按完整 tensor family 生效 |

Pure profile 只解析并 stock-load 被选 checkpoint。Hybrid profile 先执行 header-only H3/layout/family compatibility validation，然后 native-load FL、selected-only copy REF，最终 stock-construct 一个 MODEL。详见 [H3_HYBRID_LOADER.md](H3_HYBRID_LOADER.md)。

## Director Desk

Node ID：`JR_H3_DirectorDesk`

分类：`JR MiniMax H3/Director`

输出：`director_prompt: STRING`、`pip: JR_H3_DIRECTOR_PIPE`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `director_state_json` | STRING | 内置 10 秒/24 fps/1 Shot state | 前端隐藏的 schema-versioned 执行 state；实际编辑数据同步保存在 `node.properties.jr_h3_director_state` |

节点本身不调用 LLM。它在执行期验证时间轴、解析限定在 ComfyUI input/temp/output 根目录内的媒体 descriptor、只把 IMAGE 解码为 runtime tensor，并确定性地产生两个输出。First Frame 是唯一 0 秒点锚；Shot 不可重叠；Visual/Reference Audio 可重叠；Driving Audio 不可重叠。

## Director PIPE Builder

Node ID：`JR_H3_DirectorPipeBuilder`

分类：`JR MiniMax H3/Director`

输出：`pip: JR_H3_DIRECTOR_PIPE`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `prompt` | STRING | 空 | 必须非空；逐字写入当前 optimized stage |
| `duration_seconds` | FLOAT | 10.0 | 0.1..3600，step 0.1 |
| `fps` | FLOAT | 24.0 | 1..240；编辑 metadata，不改变 H3 原生 24 fps |
| `first_frame` | IMAGE | 未连接 | optional；必须恰含 1 张 RGB IMAGE |
| `last_frame` | IMAGE | 未连接 | optional；必须恰含 1 张 RGB IMAGE |
| `reference_images` | IMAGE | 未连接 | optional；batch 每项成为独立 Picture；含 anchors 总计最多 9 |
| `reference_video` | VIDEO | 未连接 | optional；标准 ComfyUI VIDEO，runtime-only |
| `reference_audio` | AUDIO | 未连接 | optional；标准 `[1,1|2,T]` AUDIO |
| `driving_audio` | AUDIO | 未连接 | optional；标准 `[1,1|2,T]` AUDIO |

Builder 创建单 Shot 合法 PIPE，不写 Tensor、waveform 或 VIDEO bytes 到 workflow JSON。详见 [DIRECTOR_PIPE_IO.md](DIRECTOR_PIPE_IO.md)。

## Director PIPE Unpack

Node ID：`JR_H3_DirectorPipeUnpack`

分类：`JR MiniMax H3/Director`

主要输出：原样 `pip`、final/director/optimized/reviewed prompt、duration/fps/width/height、首尾帧、选定的 Reference Image/Video/Audio/Driving Audio、`registry_json`、`status`。

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `pip` | JR_H3_DIRECTOR_PIPE | 必须连接 | 任何合法来源的 PIPE |
| `reference_image_index` | INT | 1 | 1..9；只在 reference_image 角色内计数 |
| `reference_video_index` | INT | 1 | 1..3 |
| `reference_audio_index` | INT | 1 | 1..3；只在 reference_audio 角色内计数 |
| `driving_audio_index` | INT | 1 | 1..3；只在 driving_audio 角色内计数 |

不存在的索引返回 `None`，但第一输出的 PIPE 始终保持完整且对象身份不变。若要同时拆出多个同类媒体，可并联多个 Unpack 并选择不同索引。

## Prompt Optimizer

Node ID：`JR_H3_OpenAICompatiblePromptOptimizer`

分类：`JR MiniMax H3/Prompt`

输出：`optimized_prompt: STRING`、`original_prompt: STRING`、`status: STRING`、`pip: JR_H3_DIRECTOR_PIPE`

执行契约：OpenAI-compatible 模型只生成严格语义 JSON；Python 按固定 MiniMax-H3 commit `8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea` 确定性生成最终 T2VA/I2VA/FL2VA/L2VA/Ref2VA 文本。字段名、section 顺序、Shot 时间、reference 编号、retention enum、对白原文/语言标签/稳定 speaker ID 不由模型自由决定。schema 不合法时最多一次低温结构化修复；formatter 输出再经过严格 validator。

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `prompt` | STRING | 空 | multiline |
| `enable` | BOOLEAN | `true` | 关闭时原样返回 |
| `api_base_url` | STRING | `http://127.0.0.1:10000` | OpenAI 兼容地址 |
| `model` | STRING | 空 | 空时执行期查询 models |
| `prompt_profile` | COMBO | `Standard` | Standard、Cinematic Drama、Action、Character Consistency |
| `duration_seconds` | INT | `10` | 1..60 |
| `target_width` | INT | `768` | 64..8192 |
| `target_height` | INT | `1152` | 64..8192 |
| `temperature` | FLOAT | `0.6` | 0..2，step 0.05 |
| `top_p` | FLOAT | `0.9` | 0..1，step 0.05 |
| `max_tokens` | INT | `1800` | 32..32768 |
| `timeout_seconds` | INT | `180` | 1..1800 |
| `image_send_size` | INT | `768` | 64..4096 |
| `fail_mode` | COMBO | `Return Original` | Return Original、Stop Workflow |
| `disable_reasoning` | BOOLEAN | `true` | 请求兼容字段 |
| `h3_input_mode` | COMBO | `Auto` | Auto、T2VA、I2VA、FL2VA、L2VA、Ref2VA |
| `reference_instructions` | STRING | 空 | multiline |
| `api_key` | STRING | 空 | optional |
| `ref_image_1..9` | IMAGE | 未连接 | optional，可输入 batch |
| `first_frame` | IMAGE | 未连接 | optional |
| `last_frame` | IMAGE | 未连接 | optional |
| `pip` | JR_H3_DIRECTOR_PIPE | 未连接 | optional；Director Desk 的权威结构化输入，必须位于 optional 列表末尾以保持旧 widget 顺序 |

## Prompt Review & Continue

Node ID：`JR_H3_PromptReviewPause`

分类：`JR MiniMax H3/Prompt`

输出：`reviewed_prompt: STRING`、`pip: JR_H3_DIRECTOR_PIPE`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `prompt` | STRING | 空 | multiline；legacy STRING 模式使用，PIPE 模式必须为空或与权威审核文本相同 |
| `timeout_seconds` | INT | `3600` | 60..86400 秒 |
| `pip` | JR_H3_DIRECTOR_PIPE | 未连接 | optional；审核 `optimized_prompt`，缺失时回退 compiled Director Prompt |

隐藏输入：`unique_id: UNIQUE_ID`。节点每次排队强制执行，不缓存人工审核结果。

## Directed Video Conditioning

Node ID：`JR_H3_DirectedVideoConditioning`

分类：`JR MiniMax H3/Generation`

输出：`positive: CONDITIONING`、`latent: LATENT`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `clip` | CLIP | 必须连接 | 当前 MiniMax H3 CLIP |
| `vae` | VAE | 必须连接 | MiniMax H3 video VAE |
| `pipe` | JR_H3_DIRECTOR_PIPE | 必须连接 | Review 输出的权威 PIPE |
| `mode_override` | COMBO | Auto | Auto、Image to Video、Reference to Video |
| `dimension_source` | COMBO | Prefer Pipe | Prefer Pipe、Prefer Node |
| `width` | INT | 1344 | 32..16384，step 32 |
| `height` | INT | 768 | 32..16384，step 32 |
| `length` | INT | 124 | 5..3600；原生按 24 fps 与 `n % 17 == 5` 对齐 |
| `ref_image_size` | COMBO | match | match、max；透传原生 Ref2V |
| `audio_vae` | VAE | 未连接 | optional；PIPE 含任意 Reference/Driving Audio 时必须连接 |

节点按 `reviewed > optimized > compiled director` 选择提示词，并直接调用当前 ComfyUI 原生 MiniMax H3 I2V/Ref2V conditioning。详情见 [DIRECTOR_PIPELINE.md](DIRECTOR_PIPELINE.md)。

## AV Latent Builder

Node ID：`JR_MiniMaxH3AVLatentBuilder`

分类：`JR MiniMax H3/Latent`

输出：`latent: LATENT`、`status: STRING`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `video_latent` | LATENT | 必须连接 | `samples` 必须是 floating tensor `[B,24,T,H,W]`，且 `T=5k+2` |
| `audio_latent` | LATENT | 必须连接 | `samples` 必须是 floating tensor `[B,32,2,T_audio]` |

两流必须具有相同 batch、dtype 和 device，并且全部数值 finite。节点按官方 24 fps / 40 Hz 时间结构检查同一 timeline，只封装官方 `NestedTensor((video, audio))`，不进行编码、clone、cast、设备迁移或文件 I/O。详见 [H3_AV_LATENT_BUILDER.md](H3_AV_LATENT_BUILDER.md)。

## Audio Driven Latent Builder

Node ID：`JR_H3_AudioDrivenLatentBuilder`

分类：`JR MiniMax H3/Latent`

输出：`audio_driven_av_latent: LATENT`、`status: STRING`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `av_latent` | LATENT | 必须连接 | 官方 H3 `NestedTensor(video, template_audio)`，通常来自 Directed Video Conditioning |
| `audio_drive_latent` | LATENT | 必须连接 | 使用 MiniMax H3 Audio VAE 编码的普通 audio latent `[B,32,2,T]` |

节点使用 template audio 的 T/device/dtype 为真值源，对外部 audio 做等长保留、过长尾截断或过短尾部补零；只允许 batch `N -> N` 或 `1 -> N`。输出保留原 video tensor 和有效的上游 video mask，将 audio mask 强制为零。节点不读取 waveform、不编解码且不 mux。详见 [H3_AUDIO_DRIVEN_LATENT_BUILDER.md](H3_AUDIO_DRIVEN_LATENT_BUILDER.md)。

## Sequential Audio Chunk Driver

Node ID：`JR_H3_SequentialAudioChunkDriver`

分类：`JR MiniMax H3/Sequential Audio`

输出：`audio_driven_av_latent: LATENT`、`chunk_context: JR_H3_AUDIO_CHUNK_CONTEXT`、`chunk_seed: INT`、`audio_slice: AUDIO`、`status: STRING`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `av_latent` | LATENT | 必须连接 | Directed Video Conditioning 输出；frame length 必须与 preset 相同 |
| `audio` | AUDIO | 必须连接 | 完整连续源音频；仅 batch 1、mono/stereo |
| `audio_vae` | VAE | 必须连接 | H3 Audio VAE；用于当前 slice 编码 |
| `chunk_preset` | COMBO | 14.375s / 345 frames / 575 ticks | 另有 10.125s/243、8s/192、5.875s/141 |
| `continuity_mode` | COMBO | Previous Last Frame | Previous Last Frame、Independent MV |
| `seed_mode` | COMBO | Derived per chunk | Derived per chunk、Fixed |
| `base_seed` | INT | 0 | unsigned 64-bit；建议连接 chunk_seed 到 Random Noise |
| `cache_path` | STRING | `temp/JR_H3_audio_jobs` | 相对路径落在 output 下；也接受绝对路径 |
| `job_name` | STRING | audio_sequence | 只作为清理后的安全目录名 |
| `run_id` | INT | 1 | 递增后创建新 run；旧 run 不删除、不覆盖 |

首次执行把原始 PCM 和全局一次性 resample 的 Audio VAE PCM 分开落盘；之后按全局 frame/sample 边界选择当前块。Driver 不前移 manifest，只有 Video Output 成功提交后才前移。默认 Same Audio Reactive Prompt 由上游保持不变。详见 [H3_SEQUENTIAL_AUDIO.md](H3_SEQUENTIAL_AUDIO.md)。

## Sequential Continuation Guide

Node ID：`JR_H3_SequentialContinuationGuide`

分类：`JR MiniMax H3/Sequential Audio`

输出：`positive: CONDITIONING`、`latent: LATENT`、原 `chunk_context`、`status`

输入为 Directed positive、Driver latent/context、video VAE，以及 optional `initial_frame`。Previous Last Frame 模式下，chunk 1 使用 initial frame，后续 chunk 从已提交缓存读取上一块末帧，并调用当前 ComfyUI 原生 `MiniMaxH3AddGuide` 锚定本块 frame 0；Independent MV 原样透传。该节点必须位于 Basic Guider 之前。

## Sequential Latent Checkpoint

Node ID：`JR_H3_SequentialLatentCheckpoint`

分类：`JR MiniMax H3/Sequential Audio`

输入 sampled H3 AV LATENT 与 chunk context；输出 CPU-backed 官方 AV LATENT、原 context 和状态。video/audio 两流原子保存为 `latents/chunk_NNNNN.safetensors`，不使用 pickle，不把 tensor 写入 workflow JSON。应放在 KSampler 与 VAE Decode 之间。

## Sequential Video Output

Node ID：`JR_H3_SequentialVideoOutput`

分类：`JR MiniMax H3/Sequential Audio`

这是顺序分支的 OUTPUT 节点，输入 decoded IMAGE 与 chunk context；另有 H.264 quality、8/10-bit、最终 AAC bitrate、filename prefix、自动续跑和 aggressive cleanup 控件。它验证并提交静音 MP4 segment、保存末帧、在活动浏览器中排队下一 prompt；最后使用相同编码器的 segment stream-copy，并把完整源 PCM 编码/融合一次。该分支不要再连接 Enhanced Video Combine。

## Split AV Latent

Node ID：`JR_H3_SplitAVLatent`

分类：`JR MiniMax H3/Latent`

输出：`video_latent: LATENT`、`audio_latent: LATENT`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `av_latent` | LATENT | 必须连接 | `samples` 必须是当前 ComfyUI 官方、恰含 video/audio 两流的 `NestedTensor` |

节点通过官方 `unbind()` 拆出 video `[B,24,T,H,W]` 与 audio `[B,32,2,T]`，验证 batch 与 finite 值，然后以新的标准 LATENT 字典返回原始 Tensor 引用；不 clone、cast、设备迁移或主动 contiguous。输出可直接连接原生 `Save Latent`。audio 没有图像空间轴，不得进入 video latent 的空间放大/resize 链。详见 [H3_SPLIT_AV_LATENT.md](H3_SPLIT_AV_LATENT.md)。

## Neural Latent Upscaler

Node ID：`JR_MiniMaxH3NeuralLatentUpscaler`

分类：`JR MiniMax H3/Latent`

输出：`video_latent: LATENT`、`status: STRING`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `video_latent` | LATENT | 必须连接 | 普通 H3 video tensor `[B,24,T,H,W]`；不能是完整 AV NestedTensor |
| `resize_mode` | COMBO | `scale` | `scale`、`megapixels` |
| `scale` | FLOAT | `1.5` | 1.0..4.0；宽高线性倍率，仅 scale 模式使用 |
| `target_megapixels` | FLOAT | `2.0` | 0.01..64.0；decode 后 pixel-space MP，仅 megapixels 模式使用 |

节点自动从 `models/latent_upscale_models/` 选择 H3 neural checkpoint，不联网下载且没有插值 fallback。输出只改变 H/W，保持 B/C/T、metadata、dtype/device；尺寸根据当前 ComfyUI 原生 H3 VAE 和 DiT patch 合同对齐。audio 必须绕过本节点后直接回到 Builder。详见 [H3_NEURAL_LATENT_UPSCALER.md](H3_NEURAL_LATENT_UPSCALER.md)。

## Temporal Chunk Sampler

Node ID：`JR_H3_TemporalChunkSampler`

分类：`JR MiniMax H3/Sampling`

输出：`output: LATENT`、`status: STRING`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `model` | MODEL | 必须连接 | MiniMax H3 model；节点为每段重新建立官方 Basic Guider |
| `positive` | CONDITIONING | 必须连接 | original positive；Chunk 1 原样使用，Chunk 2+ 派生 frame-0 continuation conditioning；不会原地修改 |
| `vae` | VAE | 必须连接 | MiniMax H3 video VAE；用于解码每个非末段的真实最终像素帧及编码下一段 guide |
| `noise` | NOISE | 必须连接 | 单块保持原生；多块支持官方 RandomNoise 的绝对时间派生子流及官方 DisableNoise，其他 custom NOISE fail closed |
| `sampler` | SAMPLER | 必须连接 | 原生 sampler object |
| `sigmas` | SIGMAS | 必须连接 | 每块使用同一完整 sigma schedule |
| `latent_image` | LATENT | 必须连接 | 官方 H3 两流 NestedTensor；video `[B,24,5k+2,H,W]`、audio `[B,32,2,T]` |
| `chunk_duration_seconds` | FLOAT | `15.0` | 1..3600，step 0.5；实际内部边界按 5 video tokens / 17 frames 对齐 |
| `aggressive_memory_cleanup` | BOOLEAN | `false` | 每块结束后调用 ComfyUI `soft_empty_cache`；更慢，通常保持关闭 |

节点按无 overlap H3 时间网格切片。Chunk 1 使用 original positive；每个后续 chunk 使用上一段完整 VAE decode 的最终 RGB 帧调用官方 `MiniMaxH3AddGuide(frame_idx=0)`，随后重新建立 Basic Guider 并委托当前 ComfyUI 原生 `SamplerCustomAdvanced`。结果直接写入预分配 CPU AV 缓冲。节点没有 hidden-state/KV carry 或 global position offset，并拒绝 `noise_mask`；多块 original positive 若已有绝对 `minimax_keyframes` 也会 fail closed。详见 [H3_TEMPORAL_CHUNK_SAMPLER.md](H3_TEMPORAL_CHUNK_SAMPLER.md)。

## Cache Config Router

Node ID：`JR_H3_CacheConfigRouter`

分类：`JR MiniMax H3/Cache`

输出：`cache_config: JR_H3_CACHE_CONFIG`、`selected_profile: STRING`、`analysis: STRING`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `optimized_prompt` | STRING | 必须连接 | multiline、force input |
| `enable` | BOOLEAN | `true` | 关闭时使用本地 fallback |
| `api_base_url` | STRING | `http://127.0.0.1:10000` | OpenAI 兼容地址 |
| `model` | STRING | 空 | 空时自动发现 |
| `api_key` | STRING | 空 | 不进入 cache_config |
| `temperature` | FLOAT | `0.0` | 0..2 |
| `top_p` | FLOAT | `1.0` | 0..1 |
| `max_tokens` | INT | `256` | 64..2048 |
| `timeout_seconds` | INT | `60` | 1..1800 |
| `disable_reasoning` | BOOLEAN | `true` | 请求兼容字段 |
| `quality_level` | COMBO | `Balanced` | Conservative、Balanced、Aggressive |
| `cache_device` | COMBO | `Auto` | Auto、GPU、CPU |
| `gpu_reserve_mb` | INT | `2048` | 0..131072，step 128 |
| `fail_mode` | COMBO | `Safe Balanced` | Safe Balanced、Disable Cache、Stop Workflow |
| `audio_content` | COMBO | `Auto` | Auto、None、Speech、Singing、Music、Ambient |
| `has_reference_audio` | BOOLEAN | `false` | 工作流事实 |
| `has_reference_video` | BOOLEAN | `false` | 工作流事实 |

接线规则：只把 `cache_config` 接到 Adaptive Cache。`selected_profile` 和 `analysis` 是显示/调试输出。

## Adaptive Cache

Node ID：`JR_H3_AdaptiveCache`

分类：`JR MiniMax H3/Cache`

输出：`MODEL`、`selected_profile: STRING`、`status: STRING`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `model` | MODEL | 必须连接 | 原生 MiniMax H3 |
| `mode` | COMBO | `Auto` | Auto、Visual Fast、Dialogue Safe、Action Safe、Balanced、Off |
| `quality_level` | COMBO | `Balanced` | Conservative、Balanced、Aggressive、Custom |
| `audio_content` | COMBO | `Auto` | Auto、None、Speech、Singing、Music、Ambient |
| `profile_hint` | STRING | 空 | Auto 手动提示；合法值为内部 profile 名 |
| `start_percent` | FLOAT | `0.10` | 0..0.99 |
| `end_percent` | FLOAT | `0.90` | 0.01..1 |
| `warmup_steps` | INT | `2` | 0..100 |
| `front_blocks` | INT | `1` | 0..48 |
| `back_blocks` | INT | `2` | 0..48 |
| `video_threshold` | FLOAT | `0.020` | 0..1，Custom 时生效 |
| `audio_threshold` | FLOAT | `0.012` | 0..1，Custom 时生效 |
| `fast_path_threshold` | FLOAT | `0.008` | 0..1，Custom 时生效 |
| `probe_path_threshold` | FLOAT | `0.035` | 0..1，Custom 时生效 |
| `max_full_step_hits` | INT | `1` | 0..20，Custom 时生效 |
| `max_block_hits` | INT | `2` | 0..20，Custom 时生效 |
| `video_metric_stride` | INT | `12` | 1..1024，Custom 时生效 |
| `audio_metric_stride` | INT | `6` | 1..1024，Custom 时生效 |
| `cache_device` | COMBO | `Auto` | Auto、GPU、CPU；仅大型 residual |
| `gpu_reserve_mb` | INT | `2048` | 0..131072 |
| `strict_model_check` | BOOLEAN | `true` | false 时不兼容模型安全返回 off |
| `verbose` | BOOLEAN | `false` | 详细运行日志 |
| `cache_config` | JR_H3_CACHE_CONFIG | 未连接 | optional；连接后忽略全部手动 widgets |

注意：界面显示的 Custom 默认阈值是历史手动初值；`quality_level` 不是 Custom 时，节点使用 `utils/h3_cache_config.py` 中当前版本化 preset，而不是这些 widgets。

## Unified Acceleration

Node ID：`JR_H3_UnifiedAcceleration`

显示名称：`H3 Unified Acceleration`

分类：`JR MiniMax H3/Optimization`

输出：`model: MODEL`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `model` | MODEL | 必须连接 | MiniMax H3 结构 |
| `enable` | BOOLEAN | `true` | 全局 bypass |
| `sage_attention` | COMBO | `sageattn_qk_int8_pv_fp8_cuda++` | disabled、auto 及当前 KJ Sage modes |
| `allow_compile` | BOOLEAN | `false` | 传给 KJ Sage |
| `enable_low_vram_attention` | BOOLEAN | `true` | 真 bypass |
| `head_chunks` | INT | `4` | 1..56 |
| `enable_low_vram_ffn` | BOOLEAN | `true` | 真 bypass |
| `ffn_chunks` | INT | `4` | 1..64 |
| `ffn_seq_threshold` | INT | `4096` | 256..262144，step 256 |
| `enable_sol_attn` | BOOLEAN | `true` | 真 bypass |
| `tau` | FLOAT | `1.3` | 0..4 |
| `start_percent` | FLOAT | `0.2` | 0..1 |
| `end_percent` | FLOAT | `0.9` | 0..1 |
| `min_tokens` | INT | `4096` | 0..1048576，step 512 |
| `int8_qk` | BOOLEAN | `true` | Sol 参数 |
| `int8_pv` | BOOLEAN | `true` | Sol 参数 |
| `sink_conditioning` | COMBO | `exact_kv_and_rows` | exact_kv、exact_kv_and_rows、off |
| `morton` | BOOLEAN | `false` | Sol 参数 |
| `morton_curve` | COMBO | `2d_frame` | 3d、2d_frame |
| `verbose` | BOOLEAN | `false` | Sol 参数 |
| `use_tma` | BOOLEAN | `false` | Sol 参数 |
| `dense_blocks` | STRING | 空 | Sol block selection string |
| `tau_profile` | STRING | 未连接/None | optional、force input；空字符串与未连接不同 |

## Resolution Scale Calculator

Node ID：`JR_H3_ResolutionScaleCalculator`

分类：`JR MiniMax H3/Scaling`

输出：`width: INT`、`height: INT`、`scale: FLOAT`、`actual_megapixels: FLOAT`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `source_width` | INT | `768` | 1..16384 |
| `source_height` | INT | `1152` | 1..16384 |
| `target_megapixels` | FLOAT | `0.88` | 0.001..256 |
| `aspect` | COMBO | `Source` | Source、1:1、2:3、3:2、16:9、9:16、Custom |
| `custom_aspect_width` | INT | `16` | 1..8192 |
| `custom_aspect_height` | INT | `9` | 1..8192 |
| `divisor` | COMBO | `"32"` | "8"、"16"、"32"；兼容旧数值 |

## RTX Upscaler & Refiner

Node ID：`JR_H3_RTXUpscalerRefiner`

分类：`JR MiniMax H3/Video`

输出：`images: IMAGE`

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `images` | IMAGE | 必须连接 | RGB/RGBA batch |
| `denoise` | BOOLEAN | `false` | 可选效果 |
| `denoise_quality` | COMBO | `Ultra` | Low、Medium、High、Ultra |
| `deblur` | BOOLEAN | `false` | 可选效果 |
| `deblur_quality` | COMBO | `Ultra` | Low、Medium、High、Ultra |
| `upscale` | COMBO | `VSR` | Off、VSR、High Bitrate |
| `upscale_quality` | COMBO | `Ultra` | Low、Medium、High、Ultra |
| `resize_type` | COMBO | `Scale` | Same Size、Scale、Keep Ratio、Preset Ratio、Manual |
| `scale` | FLOAT | `2.0` | 1..4 |
| `megapixels` | FLOAT | `2.0` | 0.01..64 |
| `width` | INT | `1920` | 64..8192 |
| `height` | INT | `1080` | 64..8192 |
| `divisible_by` | COMBO | `"8"` | 8、16、32、64、128 |
| `ratio_preset` | COMBO | `16:9` | 1:1、4:3、3:2、16:9、21:9 |
| `resize_method` | COMBO | `Center Crop (Fill)` | Center Crop (Fill)、Letterbox (Fit) |
| `device_id` | INT | `0` | 0..8，仍受实际 CUDA device count 限制 |

## Enhanced Video Combine

Node ID：`JR_H3_EnhancedVideoCombine`

分类：`JR MiniMax H3/Video`

输出：`frames: IMAGE`、`filename: STRING`

属性：`OUTPUT_NODE=true`，每次排队强制执行。

| 输入 | 类型 | 默认值 | 范围或说明 |
| --- | --- | --- | --- |
| `images` | IMAGE | 必须连接 | 非空 IMAGE batch |
| `audio` | AUDIO | 未连接 | optional |
| `frame_rate` | FLOAT | `24.0` | 0.1..240 |
| `codec` | COMBO | `Auto` | Auto、AV1、VP9、H.265 (HEVC)、H.264 |
| `container` | COMBO | `Auto` | Auto、WebM、MKV、MP4、Animated WebP、Animated AVIF |
| `bit_depth` | COMBO | `Auto` | Auto、8-bit、10-bit |
| `quality` | INT | `20` | 0..51 |
| `log_level` | COMBO | `Standard` | Standard、Verbose；当前执行路径不改变日志量 |
| `pingpong` | BOOLEAN | `false` | 追加反向 interior frames |
| `save_metadata` | BOOLEAN | `true` | 写 FFmetadata |
| `filename_prefix` | STRING | `video/%date:yyyy-MM-dd%/%date:hhmmss%` | 支持安全子目录与 date token |
| `save_output` | BOOLEAN | `true` | false 时写 ComfyUI temp |
| `pass_frames` | BOOLEAN | `false` | false 返回空 IMAGE batch |
| `crop_to_audio` | BOOLEAN | `false` | 按音频时长截断 |
| `audio_codec` | COMBO | `Auto` | Auto、AAC、Opus、MP3 |
| `audio_bitrate` | COMBO | `192k` | 64k、96k、128k、160k、192k、256k、320k |
| `save_first_frame` | BOOLEAN | `false` | 保存同目录 PNG |
| `save_last_frame` | BOOLEAN | `false` | 保存同目录 PNG |

隐藏输入：`prompt: PROMPT`、`extra_pnginfo: EXTRA_PNGINFO`。

## Last Frame

Node ID：`JR_H3_LastFrame`

分类：`JR MiniMax H3/Utility`

输入：`frames: IMAGE`

输出：`image: IMAGE`

输入必须是非空 `[B,H,W,C]` RGB/RGBA batch。输出保持 batch 轴，形状为 `[1,H,W,C]`。
