# ComfyTV 适配说明

本文档记录本节点包在 [ComfyTV](https://github.com/) (ComfyUI 的画布应用层) 下的适配结论、随附的 5 个工作流,以及 `server_auto_continue` 的设计。所有结论均经真机端到端验证(RTX 5090, ComfyUI 0.34.0, ComfyTV 1.9.0, 2026-08-30)。

## 核心结论:管线节点是 API-JSON 干净的

本包的全部执行管线节点(HybridLoader / UnifiedAcceleration / DirectedVideoConditioning / TemporalChunkSampler / NeuralLatentUpscaler / SplitAVLatent / AVLatentBuilder / AudioDrivenLatentBuilder / Sequential 系列 / EnhancedVideoCombine / DirectorDesk 执行端)**没有任何浏览器或前端依赖**。验证方式:从 history 抽出 28 节点 ref2va 全管线的 API prompt,不带 client_id 纯 HTTP POST `/prompt`,完整出片。

任何消费 ComfyUI API JSON 的编排器(ComfyTV、脚本、远端 Runner)都可以直接使用这些节点。真正的边界只有三处:

1. **`JR_H3_PromptReviewPause`**:按设计要求发起 client 有活跃 WebSocket。纯 API 提交时立刻报错;经 ComfyTV(带页面 client_id)提交时检查会通过,但审核 UI 无处渲染,**挂到 timeout_seconds 超时** — 编排场景请直接绕开此节点(STRING 进 STRING 出,删掉直连即可)。审核路由 `GET /jr_h3/prompt-review/pending` + `POST /jr_h3/prompt-review/continue` 是普通 HTTP,编排器想保留人工审可自行代理。
2. **无限时长的逐 chunk 自动续排**:原实现为前端 JS(`js/sequential_audio.js`)。现已提供服务端替代,见下节。
3. **examples/ 下的图 JSON 存在 widget 漂移**(旧版节点定义下保存),直接转 API 会产生空 combo 值被核心校验拒绝。本目录 `comfytv/workflows/` 下的图以当前节点定义重新序列化,无此问题。

## server_auto_continue(服务端逐 chunk 续排)

`JR MiniMax H3 Sequential Video Output` 新增 optional 布尔 `server_auto_continue`(默认 False,不影响既有工作流):

- commit 成功且还有后续 chunk 时,在 PromptServer 的 event loop 上挂一个 watcher,轮询 `prompt_queue.get_history(prompt_id)` **等待整个 prompt 成功完结**(与前端等 `execution_success` 的语义一致),然后把 history 中保存的 API prompt 通过 loopback POST 公共 `/prompt` 续排下一块;
- prompt 以 error/interrupt 结束则停;per-job 去重;总排队次数被 manifest 的 `total_chunks` 封顶;
- 开启时发给前端的 `chunk_committed` 事件会携带 `auto_queue_next=false`,避免浏览器双重排队;
- **wrapped-execution 守卫**:重放前检查 prompt 中是否含本节点。若不含(说明被编排器嵌套执行,如 ComfyTV stage —— 原样重放外层 prompt 会命中执行缓存空转),记日志跳过,由提交方驱动续跑。

实现:`utils/h3_sequential_server_continue.py`;测试:`tests/test_h3_sequential_server_continue.py`。

适用场景:直连 API 提交含 Sequential 链的工作流 → 一次 POST 自动跑完整首歌到终混,无需浏览器。

## 随附工作流(comfytv/workflows/)

5 个 GUI 格式图 JSON,以 ComfyUI 官方 H3 模板为骨架、嫁接本包增值节点,全部在 ComfyTV VideoStage 真机跑通。设计原则:不用 ReviewPause / Optimizer / Director Desk 编辑器等交互层,纯管线。

| 文件 | 能力 | 关键 JR 节点 |
|---|---|---|
| jr_h3_t2va_turbo_comfytv.json | 8 步文生视频+音频 | UnifiedAcceleration + fl2v 8-step Lightning |
| jr_h3_r2v_turbo_comfytv.json | 4 步参考生视频(9图/2视频/3音频参考) | UnifiedAcceleration + ref2v 4-step Lightning |
| jr_h3_dual_sample_upscale_comfytv.json | 低分辨率采样 → latent 放大 ~2MP → 3 步精修 | SplitAVLatent + NeuralLatentUpscaler + AVLatentBuilder |
| jr_h3_audio_mv_comfytv.json | 指定歌曲驱动画面的 MV | AudioDrivenLatentBuilder |
| jr_h3_sequential_mv_comfytv.json | 无限时长 MV(14.375s/块,尾帧续接,终混一次) | Sequential 全家 + 每块 CreateVideo/SaveVideo 预览 |

### ComfyTV 侧接入方式(link + bindings)

1. 把 JSON 放入 ComfyUI `user/default/workflows/`,`POST /comfytv/workflows/link {kind:"video", path, label}` 挂进工作流库;
2. 用 ComfyTV 的 workflow_edit(MCP)或工作流配置 GUI 配 bindings。各图的绑定要点(node id 见各 JSON):
   - `main_prompt` → PrimitiveStringMultiline(或 MiniMaxH3ImageToVideo.prompt);
   - `computed:width/height` → 条件节点的 width/height(sizing snap 32);
   - `option:duration_s` → Duration PrimitiveFloat;`option:seed` → RandomNoise.noise_seed(default random_int31);
   - 参考媒体 → `upstream_image:annotated[0..8]` / `upstream_video:annotated[0..1]` / `upstream_audio:annotated[0..2]`,并为条件节点的对应 autogrow 输入配 `prune_when_missing`(未接线的参考槽运行时剪掉);
   - result_node 指向 SaveVideo,result_type `ui_save_url`;meta `mention_style: minimax_tags`(提示词中 `@image_N` 自动展开为 `<Picture n>` 等模型原生标签)。
3. Sequential 图额外需要两个 custom stage param(ComfyTV `stage_params` 创建后按 `option:<key>` 绑定):
   - `seq_run_id` → 驱动节点 run_id(换新歌/从头重跑时 +1;已完成的 run 号会被拒绝,进度与产物永不覆盖);
   - `seq_song_seconds` → pipe 的 duration_seconds(歌曲全长;stage 自带 duration 上限 120s 不够用,故走 custom param)。

### Sequential 图在 ComfyTV 下的运行语义

**每次 Run 渲染一块**,卡片显示该块预览;块数 = 音频长度 ÷ 14.375 向上取整;manifest 落盘,中断后 Run 即续;最后一块提交后自动 concat 全部分段并把原始完整音轨混音一次,成片落 `output/video/<日期>/`。ComfyTV 嵌套执行下 `server_auto_continue` 会被守卫跳过(见上),连续 Run 可由用户点击或编排器循环 `run_stage` 完成 — 实测 3 分 37 秒歌曲 16 块连跑零失败。

## 模型依赖

- 基础 H3 模型同官方模板(Comfy-Org/MiniMax-H3);
- Lightning LoRA:`minimax_h3_fl2v_turbo_8step_v1.0` / `minimax_h3_ref2v_turbo_4step_v0.1`;
- 双采样图需 `LBH-123-AI/Minimax_h3_latent_Upscaler` 的 checkpoint 置于 `models/latent_upscale_models/`(Apache-2.0,不随本仓库分发)。

## 已知注意事项

- ComfyTV 的 `custom_params` widget 若经程序设置,必须是 JSON **字符串**且形如 `{"items":[{"key":...,"value":...}]}`;传对象会被字符串化为非 JSON 而静默丢弃。
- ComfyTV stage 的输出仅追踪其自身提交的 prompt;Sequential 的终混由资产扫描收录,不出现在最后一次 Run 的卡片上。
