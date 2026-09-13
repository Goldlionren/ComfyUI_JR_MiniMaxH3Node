# JR TST：第二阶段默认关闭实验

不需要新采样节点，也不需要安装 SelfLift。在已有 `JR_H3_UnifiedAcceleration` 中启用 `enable_tst` 即可；输出仍接原来的 Progressive Guided 或原生采样流程。

## 第一轮用户对照

1. 先保存稳定工作流，固定 seed、提示词、参考图/首尾帧/音频、分辨率、时长、steps、transition_step，以及现有 Sol/Sage 参数。
2. 保留用户暂定甜品点 `lowres_scale=0.6`。分别跑 `enable_tst=false`、`true + tst_strength=0.1`、`true + tst_strength=0.2`。
3. 首轮关闭 Adaptive Cache，`morton=false`、`allow_compile=false`；不要再接其他 attention 覆盖节点。保持 Sol 的 `tau` 不变。
4. 比较身份、服装、logo/纹理、运动幅度、提示词遵从和口型。TST 是纠正实验，不是身份硬锁；强度增加不保证更好。
5. 每类至少 3 个种子，加入遮挡、大运动、主动换装/镜头切换案例。另做 scale=1 对照，区分 TST 与分辨率切换的效果。
6. 记录暖机后的 sampler 和整个 prompt 耗时、峰值显存。首次 Triton 编译不要混入正式对比。上游开销声明不是 JR 实测。

更新后刷新浏览器；若旧节点实例未显示新增 widget，可新建一个 Unified 节点并复制原参数。旧保存文件无需增加必填字段。

## 实现与边界

- 参数默认关闭；`tst_strength=0` 不安装分发器、不计算谱统计。旧 widget 顺序及 Sol `tau` 保留。
- 读取归一化/RoPE 后的目标视频 Q/K，按 latent 帧和 head 做空间均值池化；计算谱熵差并生成 Q 缩放。K/V、参考图/关键帧/文本/音频 Q 不直接修改。AV 联合计算仍可能间接改变生成音频。
- Unified 保留 Sage → LowVRAM → FFN → Sol 的后端安装顺序；最后安装的组合分发器先变换 Q，再调用捕获的 Sol，fallback 沿原链回 Sage/原生 attention。不是两个完整 attention 串联。
- 独立实现依据 Tang 等人的 [TST 论文](https://arxiv.org/html/2609.08505v1)，未复制 SelfLift source。H3 池化统计是完整联合注意力的近似，不是论文 Wan 结果的复现保证。
- 根据真实 `block_index` 计算层权重，head_chunks 不影响层计数。时间权重取完整 schedule；JR Progressive 两个阶段不重置时间轴。普通采样使用 native sample_sigmas；独立时间窗口各自有 schedule，不承诺跨窗口校正。
- 使用 PackedLayout 及原生 (1,2,2) patch 网格定位视频。只支持 batch=1、head-major 的未加 attention mask 的 H3 调用。Guided 的 latent 音频锁定不是 attention mask，仍支持。
- 不支持 Morton token 重排、compile 或绕开 optimized_attention 的外部 forward 补丁；明确报错，不显示开启却静默失效。保留 Unified 为该路径最后一个 attention 配置节点。
- Q 采用非原地副本，避免破坏共享 QKV/缓存所有权；不复制 K/V，也不构造完整 token×token attention。此选择有显存/耗时成本，需要全模型测量。
- 配置本身不存储采样进度或输入张量；每个 forward 的上下文独立并在异常/完成时清理。Adaptive Cache 对 TST 设置和 Progressive 时间轴做签名隔离；缓存命中不会假装新增 TST 调用。已有缓存近似误差仍存在，先分别验证，再组合。

## 本机验证与未验证项

完整测试 855 passed、1 skipped（非 CUDA RTX 路径）。新增测试覆盖 Q/K 非突变、熵极值、分块一致性、后端/容器一次消费、异常清理、两阶段全时间轴、AV 引导重跑及 Cache 两种顺序。

RTX 5090 隔离进程使用本机 KJ Sage/LowVRAM/FFN、Sol、实际视频 VAE、实际 Neural Upscaler 与小型随机 H3：全部引导组合通过，两次 AV 重跑最大差值均为 0，锁定音频精确不变；额外 4114-token attention 检查实际 Sol 稀疏分支与 Sage 回退，Sol errors=0。该小负载不是完整 H3 画质、速度或显存上限测试。

2026-09-14 发布记录：用户反馈 TST `0.2` 在已测素材上效果良好。这是用户视觉验收，不是多素材统计结论；默认仍关闭、默认强度仍为 `0.1`。更广泛的画质收益、长期稳定性与运行开销仍待 A/B 验证；没有宣称达到 SelfLift 的 1–2% 开销。
