"""Generate three editable UI workflows using installed native input nodes."""

import json

from build_progressive_example import PROJECT, build


def node(node_id, kind, title, inputs=(), outputs=(), widgets=(), pos=(0, 0)):
    return dict(id=node_id, type=kind, title=title, pos=list(pos), size=[400, 210], flags={}, order=0, mode=0,
                inputs=[dict(name=n, type=t, link=None) for n, t in inputs],
                outputs=[dict(name=n, type=t, links=[]) for n, t in outputs],
                properties={"Node name for S&R": kind}, widgets_values=list(widgets))


def build_guided(mode):
    workflow = build(write=False)
    nodes = {n["id"]: n for n in workflow["nodes"]}
    sampler = nodes[125]
    sampler.update(type="JR_H3_ProgressiveGuidedSampler", title="JR Progressive Guided / scale 1 = native baseline",
                   properties={"Node name for S&R": "JR_H3_ProgressiveGuidedSampler"})
    sampler["inputs"].append(dict(name="vae", type="VAE", link=None))
    links = workflow["links"]

    def connect(source, slot, target, name, kind):
        target_slot = next(i for i, v in enumerate(nodes[target]["inputs"]) if v["name"] == name)
        links[:] = [v for v in links if v[3:5] != [target, target_slot]]
        links.append([max(v[0] for v in links) + 1, source, slot, target, target_slot, kind])

    def add(value):
        nodes[value["id"]] = value
        workflow["nodes"].append(value)

    connect(119, 0, 125, "vae", "VAE")
    nodes[131]["title"] = "First frame / optional last frame"
    nodes[131]["widgets_values"][0] = (
        "A single continuous cinematic shot. Preserve the subject's identity and natural proportions. "
        "Gentle believable motion and a stable camera. No cuts, no subtitles."
    )
    if mode == "Ref2VA":
        # Same autogrow socket naming as the shipped native Ref2VA workflow.
        old = nodes[131]
        old.update(type="MiniMaxH3ReferenceToVideo", title="Independent reference (not a first frame)",
                   properties={"Node name for S&R": "MiniMaxH3ReferenceToVideo"})
        old["inputs"] = [dict(name=n, type=t, link=None) for n, t in (
            ("clip", "CLIP"), ("vae", "VAE"), ("audio_vae", "VAE"), ("ref_images.ref_image_0", "IMAGE"),
            ("prompt", "STRING"), ("width", "INT"), ("height", "INT"), ("length", "INT"), ("ref_image_size", "COMBO"))]
        for item in old["inputs"][4:]:
            item["widget"] = {"name": item["name"]}
        old["inputs"][3].update(shape=7, label="ref_image_0")
        old["widgets_values"] = [
            "<Picture 1> provides the subject's appearance. A single continuous cinematic shot of the same subject, "
            "natural proportions and gentle believable movement. Stable camera, no cuts or subtitles.", 1344, 768, 124, "match"]
        # Replace every incoming link after changing the schema's input order.
        links[:] = [v for v in links if v[3] != 131]
        connect(128, 0, 131, "clip", "CLIP")
        connect(119, 0, 131, "vae", "VAE")
        connect(120, 0, 131, "audio_vae", "VAE")
        connect(132, 1, 131, "length", "INT")
        nodes[127]["widgets_values"][0] = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
        nodes[134]["widgets_values"][0] = "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
        nodes[127]["properties"].pop("models", None)
        nodes[134]["properties"].pop("models", None)
    add(node(150, "LoadImage", "Choose your reference / first-frame image", outputs=(("IMAGE", "IMAGE"), ("MASK", "MASK")),
             widgets=("", "image"), pos=(-2600, 5500)))
    connect(150, 0, 131, "ref_images.ref_image_0" if mode == "Ref2VA" else "first_frame", "IMAGE")
    if mode == "FirstLast":
        add(node(151, "LoadImage", "Choose last frame (disconnect for first-frame-only)",
                 outputs=(("IMAGE", "IMAGE"), ("MASK", "MASK")), widgets=("", "image"), pos=(-2100, 5500)))
        connect(151, 0, 131, "last_frame", "IMAGE")
    if mode == "AudioDrive":
        add(node(152, "LoadAudio", "Choose driving audio (set duration to the segment to generate)",
                 outputs=(("AUDIO", "AUDIO"),), widgets=("", None, ""), pos=(-2600, 5800)))
        add(node(153, "VAEEncodeAudio", "Encode with H3 AUDIO VAE", inputs=(("audio", "AUDIO"), ("vae", "VAE")),
                 outputs=(("LATENT", "LATENT"),), pos=(-2100, 5800)))
        add(node(154, "JR_H3_AudioDrivenLatentBuilder", "Video generates / audio LOCKED",
                 inputs=(("av_latent", "LATENT"), ("audio_drive_latent", "LATENT")),
                 outputs=(("audio_driven_av_latent", "LATENT"), ("status", "STRING")), pos=(-1450, 5500)))
        connect(152, 0, 153, "audio", "AUDIO")
        connect(120, 0, 153, "vae", "VAE")
        connect(131, 1, 154, "av_latent", "LATENT")
        connect(153, 0, 154, "audio_drive_latent", "LATENT")
        connect(154, 0, 125, "latent_image", "LATENT")
        nodes[131]["widgets_values"][0] += " The subject performs in time with the provided driving audio."
    nodes[92]["widgets_values"][0] = f"video/JR_H3_Progressive_{mode}"
    nodes[117]["widgets_values"] = [
        "# Local experimental workflow\nSelect the appropriate installed model/LoRA pair: Ref2VA for independent "
        "references, FL2VA for first/last frames. Media selectors are intentionally empty: choose your own files.\n\n"
        "AudioDrive: first frame is connected; disconnect it for audio-only drive. Add another LoadImage to last_frame "
        "for first+last+audio. For Ref2VA combined with keyframes, insert native MiniMaxH3AddGuide before the sampler."
    ]
    nodes[116]["widgets_values"] = [
        f"# JR Progressive Guided — {mode}\n\n"
        "默认 duration=5 秒，H3 对齐后为 124 帧（约 5.17 秒）。先选择你自己的图片／音频，再替换提示词。\n\n"
        "A/B：固定 seed，先 lowres_scale=1，再改 0.5；仅改变这一项。8 步、transition_step=3。"
        "scheduler 和 sampler 接同一路 MODEL。Unified 默认关闭，先验收引导与画质，再开 Sage/Sol。\n\n"
        "首尾帧：采样器 vae 必须与 conditioning 的视频 VAE 一致。独立参考图保持原尺寸。"
        "音频驱动：保留完整锁定音频 latent；时长按 duration 裁切／补齐，不自动变成整首歌。"
        "不支持 hard-prefix、局部 mask 或分块续接。原音频 PCM 如需无损保留，可接到 CreateVideo 的 audio 输入。\n\n"
        "这是运行接口已验证的实验功能，不是画质无损保证。先检查人脸、肢体、首尾帧贴合、口型和运动。"
    ]
    # Rebuild both sides, discarding inherited dangling output links and shadow widgets.
    for n in workflow["nodes"]:
        n.pop("widgets_values_named", None)
        for item in n.get("inputs", []):
            item["link"] = None
        for item in n.get("outputs", []):
            item["links"] = []
    for link_id, source, slot, target, target_slot, _ in links:
        nodes[source]["outputs"][slot]["links"].append(link_id)
        nodes[target]["inputs"][target_slot]["link"] = link_id
    workflow.update(last_node_id=max(nodes), last_link_id=max(v[0] for v in links))
    return workflow


if __name__ == "__main__":
    for mode in ("Ref2VA", "FirstLast", "AudioDrive"):
        target = PROJECT / f"examples/JR_H3_Progressive_{mode}_Experimental.json"
        target.write_text(json.dumps(build_guided(mode), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(target)
