# 节点参数参考

本页按当前 `main` 的 Python 定义记录全部 13 个节点。保存工作流依赖稳定 Node ID，请不要用显示名称代替 Node ID。

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
