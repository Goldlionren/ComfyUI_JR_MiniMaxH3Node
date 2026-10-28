# VEDA 接入最新 JR Sol-H3 的可行性评估

研究日期：2026-10-09。状态：研究完成；按用户最新要求实现独立 JR_H3_VedaAttention 适配节点，尚未做完整模型的 VEDA GPU 对照。

用户决定：VEDA 是全新的节点，保留 Sol-H3。最终接入方案见 [独立节点说明](H3_VEDA_ATTENTION.md)；下文关于在 V2 内编排的段落仅解释直接串接为何不成立，不是当前实现路线。

## 结论

VEDA 值得作为独立实验节点研发。它与 Core Sol-H3 使用平行 MODEL 分支，Sage 可继续承担全注意力及异常回退。现有 V2 不能直接在前后串官方 VEDA 节点。

目前没有证据证明 VEDA 比 JR 当前 Core Sol-H3 更快或更省显存。公开的 VEDA/Sage 对比支持开展实验，不能据此替换生产默认值。

## 最新基线

- JR 远程 main：`9c960a958ef48089c8c7782a3303537b96f8e07b`。本次已把工作区主仓库从 bf3422e 快进至该提交。新增的是 ver3.1/ver3.2 示例，Sol-H3 核心代码没有差异。
- 最新 ver3.2 双采示例的两个 Unified V2 均显式选择 core。研究基于这一状态，而非早期仅 Sage 或 legacy Sol Triton。
- 核对了上次 sol-h3-v2-dev 开发/部署记录、主仓库、F:/AI/custom_nodes 的 JR 和秋叶部署。V2 两个核心文件忽略 CRLF/LF 后一致；秋叶部署 HEAD 也是 9c960a9。
- VEDA ComfyUI 源码固定为 `60bfae688897dc41bc8685ca9f3130d7ed8f3b0c`，声明版本 0.3.0。
- 本机 RTX 5090 约 32 GB 显存；秋叶 ComfyUI 0.39.0，满足官方要求 >=0.38.0。检查时 Comfy MCP 指向的服务未运行；研究开始时两套 custom_nodes 未发现 VEDA 目录；用户随后在秋叶生产环境安装了官方 VEDA。

JR 最新方法还包括 Progressive/Guided、TaoMate 二采、Hard AV Prefix、外部 TRT 解码和独立的 H3→LTX Bridge。VEDA 接入 H3 MODEL 的注意力计算，不改变这些采样与媒体处理算法。

来源：[最新 JR 提交](https://github.com/Goldlionren/ComfyUI_JR_MiniMaxH3Node/commit/9c960a958ef48089c8c7782a3303537b96f8e07b)、[Unified V2](H3_UNIFIED_ACCELERATION_V2.md)、[Bridge](H3_LTX_BRIDGE.md)、[VEDA 版本与依赖](https://github.com/veda-sparse/Veda-on-ComfyUI/blob/60bfae688897dc41bc8685ca9f3130d7ed8f3b0c/pyproject.toml)。

## 方法与性能证据

| 路线 | 主要作用 | JR 中的角色 |
|---|---|---|
| Sage | 用量化等方式加速全注意力 | dense 层、dense 步和回退 |
| Core Sol-H3 | 稀疏计算；分块 QKV producer 直接生成 kernel 的量化数据 | 同时优化计算和 QKV 峰值显存 |
| VEDA | 蒸馏的 predictor 选择重要 tile，并按 head 的时空分块方案执行 | 新的稀疏选择后端，需要专用 predictor |

VEDA 从平均/最大/最小值构成 tile 统计，蒸馏全注意力的 tile 分布。它不修改 H3 主干权重，不是 LoRA。论文主要测试 Waver/Wan；H3 应看专门的工程和 predictor 证据，不能套用论文的 5.1×。固定保留比例也不意味着严格线性复杂度。

来源：[论文](https://arxiv.org/html/2605.30325v1)、[H3 predictor](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview)。

官方 ComfyUI 对照：RTX 5070 12 GB / Windows 11，1344×768、124 帧、8 步 Turbo、90% 稀疏。默认注意力 342 秒，Sage 231 秒，VEDA 130 秒；相对该 Sage 配置约 1.78×、耗时减少约 44%。这是作者单一负载的记录，不是本机结果，也不是 Core Sol-H3 对照。

上次 JR 小负载暖跑请求中位数：Core 14.538 秒、legacy 14.563 秒、sparse disabled 15.345 秒，记录在工作区 sol-h3-v2-dev/warm-comparison-summary.json。两组负载不同，不能跨表相除。

来源：[VEDA ComfyUI 性能表](https://github.com/veda-sparse/Veda-on-ComfyUI/blob/60bfae688897dc41bc8685ca9f3130d7ed8f3b0c/README.zh-CN.md)。

## 接入阻塞

### V2 所有权检查与 Core producer

`utils/h3_sparse_backend.py::validate_clean_input()` 拒绝已有 attention override、DiT block replacement 或相关 object patch。`attach_report()` 在 on_pre_run 检查下游是否修改 override、block 或 object patch。因此直接把 VEDA 放在 V2 前后都不构成支持的接线。

Core H3 稀疏层通过 block 的 attention= 注入 producer，不经过普通 optimized_attention。VEDA 的 override 无法接管这些已走 Core producer 的调用。

独立 VEDA 节点自行编排可选 Sage → LowVRAM → FFN → VEDA，再登记自己的最终栈；不安装或修改 Core/Sol 稀疏补丁。

来源：[JR 后端](../utils/h3_sparse_backend.py)、[V2 节点](../nodes/h3_unified_acceleration_v2.py)。另已核对本机 F:/ComfyUI-aki-v3/ComfyUI/comfy_extras/nodes_sparse_attention.py 的 h3_sparse_attention() 和 make_h3_block_patch()。

### 显存不能沿用 Core 的结论

Core 按约 4K token 投影 QKV，避免构造完整 Q/K/V。VEDA 当前接收完整 Q/K/V；本机 KJ LowVRAM forward 也先完整执行 qkv_proj(x)。VEDA 内部 head 分块限制的是后续工作缓冲，不等同于 Core producer。

因此 VEDA 可能减少计算却增加相对 Core 的峰值显存或 offload 压力。16 GB 工作流必须测整个过程，包括 MLP 和权重搬运。官方已记录并缩小过 VEDA 工作集以处理 MLP OOM，但这不证明它比 JR Core 省显存。

来源：[VEDA 接入及显存说明](https://github.com/veda-sparse/Veda-on-ComfyUI/blob/60bfae688897dc41bc8685ca9f3130d7ed8f3b0c/docs/features/comfyui_node.md)、[硬件验证范围](https://github.com/veda-sparse/Veda-on-ComfyUI/blob/60bfae688897dc41bc8685ca9f3130d7ed8f3b0c/docs/hardware.md)。

### 最新 R2VA predictor 与现有 loader 不兼容

已核对新发布的 R2VA predictor，revision `1598b407905b7c0af85aa0c13acdfa6f01ff2982`。只通过 HTTP Range 读取 safetensors 文件头：142144 字节，HTTP 206；未下载完整权重。

文件头：format=miowtion-veda-predictor-v1，50 层、56 头、head_dim=128、FP8 存储、12 个 plan。生成和参考预算都是 tiles:32，**没有 keep_ratio 字段**。

当前官方 `veda_comfy/core/bundle.py::load_bundle()` 必读 `float(metadata['keep_ratio'])`，缺失会抛出 BundleError: incomplete metadata；predictors.py 的已知默认文件仍只有 T2VA predictor。因此新 R2VA 文件不能仅放进 models/veda 就声称兼容。

研发需要兼容旧 keep-ratio metadata 与新 target/ref 固定 tile 预算，并正确展示/继承训练预算。不能随意补 0.1 或沿用两个 90% 默认值。R2VA 模型卡描述的配置是每 query/head 最多 32 个参考 tile、32 个生成 tile。修复后仍须验证 plan 和实际推理。

现有 T2VA predictor 的官方模型卡允许用于 T2VA/FL2VA/R2VA 和不同步数；这与“新 R2VA 专用文件已在 ComfyUI 验证”是不同结论。

来源：[R2VA 模型卡](https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview)、[固定 revision 文件](https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/blob/1598b407905b7c0af85aa0c13acdfa6f01ff2982/minimax_h3_r2va_veda_preview_fp8.safetensors)、[当前 loader](https://github.com/veda-sparse/Veda-on-ComfyUI/blob/60bfae688897dc41bc8685ca9f3130d7ed8f3b0c/veda_comfy/core/bundle.py)、[已知 predictor](https://github.com/veda-sparse/Veda-on-ComfyUI/blob/60bfae688897dc41bc8685ca9f3130d7ed8f3b0c/veda_comfy/predictors.py)。

## 兼容判断

| 功能 | 当前判断与边界 |
|---|---|
| Sage | VEDA 可捕获 previous override，保留 dense fallback；没有证据说明删除 Sage 更好。 |
| KJ LowVRAM / FFN | VEDA 有 head_chunks 接管代码；FFN 独立。JR 编排后仍须验证头数和显存峰值。 |
| TST | 不能直接叠加：TST 要独占 override，VEDA 每步又把自己放到最上层，JR wrapper 会报错。首版拒绝 active TST。 |
| Progressive / Guided | 布局层面可评估；根据真实 grid 选 plan。低分辨率、TaoMate 少步二采和混合权重仍需画质验收。 |
| Adaptive Cache | 首轮关闭。通用 cache 文档允许 attention backend，不等于 V2 的 block 栈检查允许任意顺序叠加。 |
| Temporal Chunk / Hard AV Prefix | 原生分窗可单独评估；检查条件 span、音频锁定、首尾窗口和接缝。尚未 VEDA 联调。 |
| Streaming Sparse KV | 首版不承诺：该 runtime 也独占 override，KV 路径会有 Q/K 长度不同的调用，VEDA 当前会拒绝。 |
| H3→LTX / TRT | VEDA 仅作用 H3 MODEL；H3 predictor 不支持 LTX，保留 LTX 自己的后端。TRT 解码独立。 |

来源：[TST](../utils/h3_temporal_transport.py)、[Streaming](../utils/h3_stream_attention.py)、[Progressive](H3_PROGRESSIVE_SAMPLER.md)、[Cache](H3_ADAPTIVE_CACHE.md)、[VEDA 接入代码](https://github.com/veda-sparse/Veda-on-ComfyUI/blob/60bfae688897dc41bc8685ca9f3130d7ed8f3b0c/veda_comfy/comfy_patch.py)。

## 研发与验收方案

1. 新增独立 VEDA 节点，保留 V2 实现、默认值和旧工作流；使用平行 MODEL 分支。调用外部官方实现，不复制 kernels。
2. 首版关闭 compile/TST，保持 KJ FFN 与 dense Sage 参数。区分 VEDA 的 sparsity/tile budget 与 Sol 的 tau/extra_tokens，不能互相换算。
3. 先用 T2VA predictor 验证原生 H3，再适配新 R2VA metadata 和默认预算。缺依赖/错格式应在采样前报清楚；记录 sparse/dense 调用和回退，不能把全 dense 渲染当成 VEDA 成功。
4. 比较 Sage dense、当前 Core Sol-H3、VEDA 三路；固定模型/LoRA、conditioning、seed、sigma、帧数、分辨率、步数、FFN、解码。预热后测完整请求；隔离 Stage2 时使用相同 Stage1 latent。
5. 先测 5 秒基线，再测 ver3.2 实际 10–30 秒负载；覆盖 T2VA、参考人物/首尾帧、音频驱动。检查身份、动作、闪烁、口型、音画同步和接缝。音频行 dense 不保证联合模型的音频输出不受视觉近似影响。
6. 记录 Stage1/Stage2/总时间、峰值显存/RAM、offload 与 fallback。GPU 接口测试、重复运行和 cleanup 检查后，以真实媒体对照决定是否推荐。

新增独立 VEDA 适配节点及接口测试；原 Unified/Sol-H3、采样实现和旧工作流未修改。未下载完整 predictor；用户安装官方 VEDA 后，已按要求部署独立 JR 节点到秋叶生产目录，完成导入/接口检查，生成测试由用户执行。开发目录 junction 指向 F:/AI，该套同步状态亦已核对。已有未跟踪文件保留。
