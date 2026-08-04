import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "JR_H3_EnhancedVideoCombine";
const DIRECT_PLAYBACK = new Set(["H.264|MP4|8", "VP9|WebM|8", "AV1|WebM|8"]);

function assetUrl(asset) {
    const query = new URLSearchParams({
        filename: asset.filename,
        subfolder: asset.subfolder || "",
        type: asset.type || "output",
    });
    return api.apiURL(`/view?${query.toString()}`);
}

function compatibilityPreviewUrl(asset) {
    const query = new URLSearchParams({
        filename: asset.filename,
        subfolder: asset.subfolder || "",
        type: asset.type || "output",
    });
    return api.apiURL(`/jr-h3/enhanced-video-preview?${query.toString()}`);
}

function needsCompatibilityPreview(asset) {
    return !DIRECT_PLAYBACK.has(`${asset.codec}|${asset.container}|${asset.bit_depth ?? 8}`);
}

function preventCanvasGesture(element) {
    for (const name of ["pointerdown", "mousedown", "touchstart", "wheel"]) {
        element.addEventListener(name, (event) => event.stopPropagation());
    }
}

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function setBooleanWidget(node, name, value) {
    const widget = findWidget(node, name);
    if (!widget) return;
    widget.value = Boolean(value);
    widget.callback?.(widget.value);
    node.graph?.setDirtyCanvas(true, true);
}

function collapseBackendWidget(node, name) {
    const widget = findWidget(node, name);
    if (!widget) return;
    widget.draw = () => {};
    widget.computeSize = () => [0, -4];
}

function clock(seconds) {
    if (!Number.isFinite(seconds)) return "--:--";
    const rounded = Math.max(0, Math.floor(seconds));
    return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, "0")}`;
}

function resizeForPreview(node) {
    const computed = node.computeSize([node.size[0], node.size[1]]);
    node.setSize([Math.max(420, node.size[0]), computed[1]]);
    node.graph?.setDirtyCanvas(true, true);
}

function showHelp() {
    const dialog = document.createElement("dialog");
    dialog.style.cssText = [
        "max-width:650px", "padding:18px", "border:1px solid #596273", "border-radius:9px",
        "background:#20242b", "color:#e8edf5", "font:13px/1.5 sans-serif",
    ].join(";");
    dialog.innerHTML = `
        <form method="dialog" style="float:right"><button aria-label="Close">×</button></form>
        <h3 style="margin:0 32px 10px 0">JR Enhanced Video Combine</h3>
        <p>把 IMAGE 批次编码为视频，并在节点内预览、保存和下载。</p>
        <ul style="padding-left:20px;margin-bottom:0">
          <li>Auto 会实际测试 AV1、VP9、H.264，并在硬件编码不可用时回退软件编码。</li>
          <li>显式编码支持自动 8/10-bit；Animated WebP/AVIF 不包含音频。</li>
          <li>不受浏览器支持的格式会通过 FFmpeg 实时转成兼容预览，不改变保存的原文件。</li>
          <li>Save first/last frame 会把原尺寸 PNG 和视频放在同一输出目录。</li>
        </ul>`;
    document.body.append(dialog);
    dialog.addEventListener("close", () => dialog.remove(), { once: true });
    dialog.showModal();
}

function overHelpIcon(node, position) {
    const titleHeight = globalThis.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
    return position[0] >= node.size[0] - 25 && position[0] <= node.size[0] - 3
        && position[1] >= -titleHeight && position[1] <= 0;
}

function migrateLegacyWidgetValues(node) {
    const bitDepth = findWidget(node, "bit_depth");
    const oldSchemaSignature = typeof bitDepth?.value === "number"
        && typeof findWidget(node, "quality")?.value === "boolean"
        && typeof findWidget(node, "log_level")?.value === "boolean"
        && typeof findWidget(node, "pingpong")?.value === "string";
    if (!oldSchemaSignature) return false;

    const legacy = {
        quality: bitDepth.value,
        pingpong: findWidget(node, "quality")?.value,
        save_metadata: findWidget(node, "log_level")?.value,
        filename_prefix: findWidget(node, "pingpong")?.value,
        save_output: findWidget(node, "save_metadata")?.value,
        pass_frames: findWidget(node, "filename_prefix")?.value,
        crop_to_audio: findWidget(node, "save_output")?.value,
        save_first_frame: findWidget(node, "pass_frames")?.value,
        save_last_frame: findWidget(node, "crop_to_audio")?.value,
    };
    bitDepth.value = "Auto";
    findWidget(node, "log_level").value = "Standard";
    findWidget(node, "audio_codec").value = "Auto";
    findWidget(node, "audio_bitrate").value = "192k";
    for (const [name, value] of Object.entries(legacy)) {
        const widget = findWidget(node, name);
        if (widget && value !== undefined) widget.value = value;
    }
    return true;
}

function refreshPreviewCheckboxes(node) {
    if (!node.jrH3Preview) return;
    node.jrH3Preview.first.checked = Boolean(findWidget(node, "save_first_frame")?.value);
    node.jrH3Preview.last.checked = Boolean(findWidget(node, "save_last_frame")?.value);
}

function buildPreview(node) {
    collapseBackendWidget(node, "log_level");
    collapseBackendWidget(node, "save_first_frame");
    collapseBackendWidget(node, "save_last_frame");

    const panel = document.createElement("section");
    panel.style.cssText = "width:100%;box-sizing:border-box;color:var(--input-text,#e6e6e6);font:11px sans-serif";

    const video = document.createElement("video");
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.controls = false;
    video.disablePictureInPicture = true;
    video.controlsList = "nodownload nofullscreen noremoteplayback";
    video.style.cssText = "display:block;width:100%;min-height:80px;background:#080a0d;border-radius:5px;object-fit:contain";

    const status = document.createElement("div");
    status.style.cssText = "display:flex;gap:9px;min-height:16px;padding:4px 2px;color:#cbd5e1";
    const dimensions = document.createElement("span");
    const duration = document.createElement("span");
    const fps = document.createElement("span");
    status.append(dimensions, duration, fps);

    const toolbar = document.createElement("div");
    toolbar.style.cssText = [
        "display:flex", "align-items:center", "gap:7px", "flex-wrap:wrap", "padding:6px",
        "background:rgba(25,30,40,.85)", "border:1px solid rgba(255,255,255,.12)", "border-radius:5px",
    ].join(";");

    function checkbox(text, initial, changed) {
        const label = document.createElement("label");
        label.style.cssText = "display:flex;gap:4px;align-items:center;white-space:nowrap;cursor:pointer";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = Boolean(initial);
        input.addEventListener("change", () => changed(input.checked));
        label.append(input, text);
        return { label, input };
    }

    const first = checkbox("Save first frame", findWidget(node, "save_first_frame")?.value, (value) => {
        setBooleanWidget(node, "save_first_frame", value);
    });
    const last = checkbox("Save last frame", findWidget(node, "save_last_frame")?.value, (value) => {
        setBooleanWidget(node, "save_last_frame", value);
    });
    const autoplay = checkbox("Autoplay", true, () => {});
    autoplay.label.style.marginLeft = "auto";

    const download = document.createElement("a");
    download.textContent = "Download";
    download.href = "#";
    download.download = "video";
    download.style.cssText = [
        "padding:3px 8px", "border:1px solid rgba(255,255,255,.22)", "border-radius:4px",
        "background:rgba(255,255,255,.1)", "color:inherit", "text-decoration:none",
    ].join(";");
    download.addEventListener("click", (event) => {
        if (!video.dataset.filename) event.preventDefault();
    });

    toolbar.append(first.label, last.label, autoplay.label, download);
    panel.append(video, status, toolbar);
    [panel, video, status, toolbar, first.label, last.label, autoplay.label, download].forEach(preventCanvasGesture);

    let widget;
    const requiredHeight = () => (node.size[0] - 20) / (widget?.aspectRatio || 16 / 9) + 96;
    widget = node.addDOMWidget("jr_h3_video_preview", "video-preview", panel, {
        serialize: false,
        hideOnZoom: false,
        getHeight: requiredHeight,
    });
    widget.aspectRatio = 16 / 9;
    widget.computeSize = (width) => [width, panel.hidden ? -4 : requiredHeight()];

    const selectSource = (asset, forceCompatibility = false) => {
        video.dataset.fallback = forceCompatibility ? "1" : "0";
        video.src = forceCompatibility || needsCompatibilityPreview(asset)
            ? compatibilityPreviewUrl(asset)
            : assetUrl(asset);
        video.load();
    };

    video.addEventListener("loadedmetadata", () => {
        if (video.videoWidth && video.videoHeight) widget.aspectRatio = video.videoWidth / video.videoHeight;
        dimensions.textContent = `${video.videoWidth || "?"}×${video.videoHeight || "?"}`;
        duration.textContent = clock(video.duration);
        fps.textContent = video.dataset.fps ? `${video.dataset.fps} fps` : "";
        resizeForPreview(node);
        if (autoplay.input.checked) video.play().catch(() => {});
    });
    video.addEventListener("error", () => {
        const asset = node.jrH3VideoAsset;
        if (asset && video.dataset.fallback !== "1") {
            selectSource(asset, true);
            return;
        }
        dimensions.textContent = "Preview unavailable";
        duration.textContent = "Check FFmpeg/browser decoder";
        fps.textContent = "";
    });

    let playbackWatch;
    const clearPlaybackWatch = () => {
        if (playbackWatch) globalThis.clearTimeout(playbackWatch);
        playbackWatch = undefined;
    };
    video.addEventListener("timeupdate", clearPlaybackWatch);
    video.addEventListener("playing", () => {
        clearPlaybackWatch();
        playbackWatch = globalThis.setTimeout(() => {
            if (video.currentTime === 0 && video.dataset.fallback !== "1" && node.jrH3VideoAsset) {
                selectSource(node.jrH3VideoAsset, true);
            }
        }, 2500);
    });
    video.addEventListener("mouseenter", () => {
        video.controls = true;
        video.muted = false;
    });
    video.addEventListener("mouseleave", () => {
        video.muted = true;
        video.controls = false;
    });
    video.addEventListener("dblclick", (event) => event.preventDefault());

    node.jrH3Preview = { panel, video, widget, download, first: first.input, last: last.input, selectSource };
    if (node.size[0] < 420) node.setSize([420, node.size[1]]);
}

app.registerExtension({
    name: "JR.MiniMaxH3.EnhancedVideoCombinePreview",
    loadedGraphNode(node) {
        if (node.comfyClass !== NODE_ID && node.type !== NODE_ID) return;
        if (migrateLegacyWidgetValues(node)) node.graph?.setDirtyCanvas(true, true);
        refreshPreviewCheckboxes(node);
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        const originalExecuted = nodeType.prototype.onExecuted;
        const originalRemoved = nodeType.prototype.onRemoved;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            buildPreview(this);

            const previousDraw = this.onDrawForeground;
            const previousMouseDown = this.onMouseDown;
            this.onDrawForeground = function (context) {
                previousDraw?.apply(this, arguments);
                const titleHeight = globalThis.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
                const x = this.size[0] - 14;
                const y = -titleHeight / 2;
                context.save();
                context.fillStyle = "#dbeafe";
                context.beginPath();
                context.arc(x, y, 8, 0, Math.PI * 2);
                context.fill();
                context.fillStyle = "#172033";
                context.font = "bold 12px sans-serif";
                context.textAlign = "center";
                context.textBaseline = "middle";
                context.fillText("?", x, y + 0.5);
                context.restore();
            };
            this.onMouseDown = function (_event, position) {
                if (overHelpIcon(this, position)) {
                    showHelp();
                    return true;
                }
                return previousMouseDown?.apply(this, arguments);
            };
            return result;
        };

        nodeType.prototype.onExecuted = function (message) {
            const result = originalExecuted?.apply(this, arguments);
            const asset = message?.gifs?.[0] ?? message?.videos?.[0];
            if (!asset?.filename || !this.jrH3Preview) return result;
            this.jrH3VideoAsset = asset;
            const { video, widget, download, selectSource } = this.jrH3Preview;
            video.pause();
            video.dataset.filename = asset.filename;
            video.dataset.fps = asset.fps ?? "";
            if (asset.width && asset.height) widget.aspectRatio = asset.width / asset.height;
            download.href = assetUrl(asset);
            download.download = asset.filename;
            selectSource(asset);
            return result;
        };

        nodeType.prototype.onRemoved = function () {
            if (this.jrH3Preview?.video) {
                this.jrH3Preview.video.pause();
                this.jrH3Preview.video.removeAttribute("src");
                this.jrH3Preview.video.load();
            }
            return originalRemoved?.apply(this, arguments);
        };
    },
});
