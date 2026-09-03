import { app } from "../../scripts/app.js";

const NODE_ID = "JR_H3_TemporalChunkSampler";
const HARD_MODE = "Hard AV Latent Prefix";
const HIDDEN_WIDGET_TYPE = "jr-temporal-hidden";
const ORIGINAL_STATE = Symbol("jrTemporalOriginalWidgetState");

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (!widget[ORIGINAL_STATE]) {
        widget[ORIGINAL_STATE] = {
            type: widget.type,
            computeSize: widget.computeSize,
            hidden: widget.hidden,
        };
    }
    const original = widget[ORIGINAL_STATE];
    if (visible) {
        widget.type = original.type;
        widget.computeSize = original.computeSize;
        widget.hidden = false;
    } else {
        widget.type = HIDDEN_WIDGET_TYPE;
        widget.computeSize = () => [0, -4];
        widget.hidden = true;
    }
}

function syncModeWidgets(node, selectedMode) {
    const mode = node.widgets?.find((widget) => widget.name === "continuity_mode");
    const legacyDuration = node.widgets?.find((widget) => widget.name === "chunk_duration_seconds");
    const hardPreset = node.widgets?.find((widget) => widget.name === "hard_chunk_preset");
    if (!mode || !legacyDuration || !hardPreset) return;

    const modeValue = typeof selectedMode === "string" ? selectedMode : mode.value;
    const hard = modeValue === HARD_MODE;
    node.__jrTemporalLastMode = modeValue;
    setWidgetVisible(legacyDuration, !hard);
    setWidgetVisible(hardPreset, hard);

    requestAnimationFrame(() => {
        const computed = node.computeSize([node.size?.[0] || 360, node.size?.[1] || 0]);
        node.setSize([Math.max(360, node.size?.[0] || computed[0]), computed[1]]);
        node.setDirtyCanvas?.(true, true);
    });
}

function installModeCallback(node) {
    const mode = node.widgets?.find((widget) => widget.name === "continuity_mode");
    if (!mode || mode.__jrTemporalCallbackWrapped) return;

    const originalCallback = mode.callback;
    mode.callback = function () {
        const selectedMode = arguments[0];
        const callbackResult = originalCallback?.apply(this, arguments);
        scheduleModeWidgetSync(node, selectedMode);
        return callbackResult;
    };
    mode.__jrTemporalCallbackWrapped = true;
}

function scheduleModeWidgetSync(node, selectedMode) {
    // LiteGraph versions differ on whether a combo callback runs immediately
    // before or after widget.value is committed. Use the callback value first,
    // then confirm against the committed widget value on the next frame.
    queueMicrotask(() => {
        installModeCallback(node);
        syncModeWidgets(node, selectedMode);
    });
    requestAnimationFrame(() => {
        installModeCallback(node);
        syncModeWidgets(node);
    });
}

app.registerExtension({
    name: "JR.MiniMaxH3.TemporalChunkSamplerPresets",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const originalOnCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnCreated?.apply(this, arguments);
            scheduleModeWidgetSync(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            scheduleModeWidgetSync(this);
            return result;
        };

        const originalOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            const result = originalOnDrawForeground?.apply(this, arguments);
            const mode = this.widgets?.find((widget) => widget.name === "continuity_mode");
            if (mode && this.__jrTemporalLastMode !== mode.value) {
                installModeCallback(this);
                syncModeWidgets(this, mode.value);
            }
            return result;
        };
    },

    loadedGraphNode(node) {
        if (node.comfyClass === NODE_ID) scheduleModeWidgetSync(node);
    },
});
