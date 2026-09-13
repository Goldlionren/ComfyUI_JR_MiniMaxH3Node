# H3 VAE 可选兼容与 final decode 探测

JR 不内置、安装或编译 H3VAE_TRT，也不引入 TensorRT requirements。正式工作流的最终解码仍使用原生 VAEDecode。当前新增的是内部能力报告和独立命令行探测，不新增节点。

## 用户验收与集成决定（2026-09-14）

用户已安装 engine 并确认外部 H3VAE_TRT 解码正常、速度有明显提升。此为用户实测反馈，未提供严格耗时对照或整体加速倍率，也未证明 JR 探测脚本、Guided 往返或 tiled API 全部通过。

继续直接使用 [上游 H3VAE_TRT](https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT) 的 Loader、编译工具和模型资源。**不将 TRT Loader 或模型/engine 吸收进 JR**；JR 仅保留可选协议检查、探测和文档，常规最终解码继续由原生 VAEDecode 接收外部 VAE。

## 已识别的契约缺口

审查版本 [H3VAE_TRT 4360e00](https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT/tree/4360e00867eca86ab61b3899216c0ec281367b46) 的 `ComfyTRTVAE` 缺少完整 H3 元数据，并允许只配置 encoder/decoder 一方。`VAE` 端口相同不代表 Guided 往返和继承的 tiled API 可用。

Guided 保留 24 latent channels / 16 倍空间压缩 / encode+decode 的检查。缺失属性或角色时给出可读错误；不伪造属性、不修改共享 VAE、不放宽任意 VAE 的验证。当前**不声明该 TRT 包装器可以直接用于 Guided**；建议先由上游补完整协议，再独立验证关键帧重编码。

`inspect_h3_video_vae(vae)` 不进行编解码或 engine 加载；`engine_tested=false` 明确区分协议检查与真实 TRT 验证。

## 只使用已有本机资源的探测

在 JR 仓库目录执行，替换示例绝对路径：

```powershell
& 'F:\ComfyUI-aki-v3\python\python.exe' tools/probe_h3_vae.py --comfy-root 'F:\ComfyUI-aki-v3\ComfyUI' --native-vae 'F:\ComfyUI-aki-v3\ComfyUI\models\vae\minimax_h3_video_vae_fp16.safetensors'
```

只检查元数据可增加 `--inspect-only`。默认使用零 video latent `[1,24,2,16,16]`，输出应为 5 帧 256×256；可通过 `--latent-t/h/w` 改尺寸。两次解码分别报告 cold/warm 秒数，但合成输入只能验证接口，不能评价真实视频质量。

已有可信外部源文件与本机编译好的 engine 后，可用以下显式入口（用户反馈不等同于此脚本已通过）：

```powershell
& 'F:\ComfyUI-aki-v3\python\python.exe' tools/probe_h3_vae.py --comfy-root 'F:\ComfyUI-aki-v3\ComfyUI' --trt-node-file 'F:\path\ComfyUI-H3VAE_TRT\minimax_trt_node.py' --decoder-engine 'F:\path\decoder.engine' --latent-t 7
```

探测脚本在独立进程载入指定外部代码和 engine；不调用上游 Compiler、不下载、不提交生产队列，不修改 ComfyUI 核心。请在 GPU 空闲时执行。

## 首轮解码边界

- 输入可以是视频 LATENT 或 AV NestedTensor；只提取视频，克隆后送入 VAE，避免外部实现修改原 AV 值。
- 返回必须是正确时间长度、16 倍空间尺寸的有限 IMAGE；兼容 native batch-1 5D 图像输出。
- TRT 首轮仅接受已审查 runtime 的 256px tile、固定 `[1,24,7,16,16]` 输入 profile、fp16 绑定及匹配的 `[1,3,28,256,256]` 内部输出缓冲。内部输出还须经上游时间裁剪，不能把 28 当作最终视频帧数。
- 不接受小于 256px 的画布、32×32 profile 或未知绑定，防止 runtime 与 engine 不一致。不把 tile 限制误说成整个画布必须固定 256×256。
- 单帧、短视频时间填充及裁剪特别需要验证；最终形状不对即失败，不通过改变目标时长来迁就 engine。

## 后续验证边界

当前保留已验证的外部最终解码方案，不扩大到 JR 内置 TRT。若未来要在 Guided 关键帧往返或其他像素往返流程中使用外部 TRT VAE，应另行验证 encode/decode 能力、原生 VAE 图像误差、tile 接缝、人物观感、音频锁定及显存加载/卸载，不能直接套用最终解码结论。

初始开发验证未安装 H3VAE_TRT 或运行 engine，只有协议测试；之后用户完成了上述实际解码验收。本次 GitHub 发布不安装、编译或运行 TRT。上游“最高 1.7×”属于 VAE encode/decode，不是整个工作流。
