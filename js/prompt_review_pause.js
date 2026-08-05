import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "JR_H3_PromptReviewPause";
const REQUEST_EVENT = "jr_h3_prompt_review_requested";
const STATUS_EVENT = "jr_h3_prompt_review_status";
const MAX_TEXT_LENGTH = 100000;
const pendingByNode = new Map();
let recoveryInFlight = false;
let recoveryTimer;

function sameNodeId(left, right) {
    return String(left) === String(right);
}

function findNode(nodeId) {
    return app.graph?._nodes?.find((node) => sameNodeId(node.id, nodeId));
}

function stopCanvasGesture(element) {
    for (const name of ["pointerdown", "mousedown", "touchstart", "wheel", "keydown"]) {
        element.addEventListener(name, (event) => event.stopPropagation());
    }
}

function stateColor(state) {
    return {
        "Waiting for review": "#fbbf24",
        Submitting: "#60a5fa",
        Approved: "#4ade80",
        "Timed out": "#fb7185",
        Cancelled: "#f97316",
        Error: "#fb7185",
    }[state] || "#94a3b8";
}

function resizeNode(node) {
    const size = node.computeSize([Math.max(460, node.size[0]), node.size[1]]);
    node.setSize([Math.max(460, node.size[0]), Math.max(size[1], 360)]);
    node.graph?.setDirtyCanvas(true, true);
}

function setReviewState(node, state, message = "") {
    const review = node.jrH3PromptReview;
    if (!review) return;
    review.state = state;
    review.status.textContent = message ? `${state} — ${message}` : state;
    review.status.style.color = stateColor(state);
    review.panel.style.borderColor = stateColor(state);
    const waiting = state === "Waiting for review";
    review.textarea.disabled = !waiting;
    review.button.disabled = !waiting;
    review.button.textContent = waiting ? "Next / Continue" : state === "Approved" ? "Approved" : "Next";
    node.graph?.setDirtyCanvas(true, true);
}

async function submitReview(node) {
    const review = node.jrH3PromptReview;
    if (!review || review.state !== "Waiting for review" || !review.reviewId) return;
    const text = review.textarea.value;
    if (!text.trim()) {
        setReviewState(node, "Error", "Prompt cannot be empty");
        review.textarea.disabled = false;
        review.button.disabled = false;
        review.button.textContent = "Next / Continue";
        review.state = "Waiting for review";
        return;
    }
    if (text.length > MAX_TEXT_LENGTH) {
        setReviewState(node, "Error", "Prompt is too long");
        review.textarea.disabled = false;
        review.button.disabled = false;
        review.button.textContent = "Next / Continue";
        review.state = "Waiting for review";
        return;
    }
    const reviewId = review.reviewId;
    setReviewState(node, "Submitting");
    try {
        const response = await api.fetchApi("/jr_h3/prompt-review/continue", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ review_id: reviewId, text }),
        });
        if (!response.ok) {
            let code = "request_failed";
            try {
                code = (await response.json())?.error || code;
            } catch {
                // Keep the generic code; never print the submitted prompt or body.
            }
            if (response.status === 404 || response.status === 409) {
                setReviewState(node, "Error", "Review is no longer pending");
            } else {
                setReviewState(node, "Error", code.replaceAll("_", " "));
                review.state = "Waiting for review";
                review.textarea.disabled = false;
                review.button.disabled = false;
                review.button.textContent = "Next / Continue";
            }
            return;
        }
        if (review.reviewId === reviewId) setReviewState(node, "Approved");
    } catch {
        if (review.reviewId === reviewId) {
            setReviewState(node, "Error", "Submission failed");
            review.state = "Waiting for review";
            review.textarea.disabled = false;
            review.button.disabled = false;
            review.button.textContent = "Next / Continue";
        }
    }
}

function beginReview(node, payload) {
    if (!node?.jrH3PromptReview || !payload?.review_id) return false;
    const review = node.jrH3PromptReview;
    review.reviewId = payload.review_id;
    review.textarea.value = typeof payload.text === "string" ? payload.text : "";
    review.textarea.placeholder = "Review or edit the prompt, then click Next.";
    setReviewState(node, "Waiting for review");
    resizeNode(node);
    return true;
}

function acceptReviewRequest(payload) {
    if (!payload?.node_id || !payload?.review_id) return;
    pendingByNode.set(String(payload.node_id), payload);
    const node = findNode(payload.node_id);
    if (node && beginReview(node, payload)) pendingByNode.delete(String(payload.node_id));
}

function acceptStatus(payload) {
    if (!payload?.node_id || !payload?.status) return;
    const node = findNode(payload.node_id);
    const review = node?.jrH3PromptReview;
    if (!review || (payload.review_id && review.reviewId !== payload.review_id)) return;
    setReviewState(node, payload.status);
    if (payload.status !== "Waiting for review") review.reviewId = null;
}

async function recoverPendingReviews() {
    if (recoveryInFlight || !api.clientId) return;
    recoveryInFlight = true;
    try {
        const query = new URLSearchParams({ client_id: api.clientId });
        const response = await api.fetchApi(`/jr_h3/prompt-review/pending?${query.toString()}`, { cache: "no-store" });
        if (!response.ok) return;
        const body = await response.json();
        for (const payload of body.reviews || []) acceptReviewRequest(payload);
    } catch {
        // A reconnect can race server startup; the next status/reconnect event retries.
    } finally {
        recoveryInFlight = false;
    }
}

function scheduleRecovery() {
    globalThis.clearTimeout(recoveryTimer);
    recoveryTimer = globalThis.setTimeout(recoverPendingReviews, 150);
}

function buildReviewWidget(node) {
    const panel = document.createElement("section");
    panel.style.cssText = [
        "width:100%", "box-sizing:border-box", "padding:8px", "border:2px solid #64748b",
        "border-radius:7px", "background:rgba(15,23,42,.88)", "font:12px/1.4 sans-serif",
    ].join(";");

    const status = document.createElement("div");
    status.textContent = "Idle";
    status.style.cssText = "font-weight:700;margin-bottom:6px;color:#94a3b8";

    const textarea = document.createElement("textarea");
    textarea.name = "review_text";
    textarea.placeholder = "Waiting for execution...";
    textarea.disabled = true;
    textarea.maxLength = MAX_TEXT_LENGTH;
    textarea.spellcheck = false;
    textarea.style.cssText = [
        "display:block", "width:100%", "height:210px", "min-height:150px", "resize:vertical",
        "box-sizing:border-box", "padding:8px", "border:1px solid #475569", "border-radius:5px",
        "background:#090f1d", "color:#e2e8f0", "font:12px/1.45 ui-monospace,monospace",
    ].join(";");

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Next";
    button.disabled = true;
    button.style.cssText = "margin-top:7px;width:100%;padding:7px;border-radius:5px;font-weight:700";
    button.addEventListener("click", () => submitReview(node));

    panel.append(status, textarea, button);
    [panel, textarea, button].forEach(stopCanvasGesture);
    const widget = node.addDOMWidget("review_text", "prompt-review", panel, {
        serialize: false,
        hideOnZoom: false,
        getHeight: () => 285,
    });
    widget.serializeValue = () => undefined;
    node.jrH3PromptReview = { panel, status, textarea, button, widget, reviewId: null, state: "Idle" };
    resizeNode(node);
}

app.registerExtension({
    name: "JR.MiniMaxH3.PromptReviewPause",
    async setup() {
        api.addEventListener(REQUEST_EVENT, ({ detail }) => acceptReviewRequest(detail));
        api.addEventListener(STATUS_EVENT, ({ detail }) => acceptStatus(detail));
        api.addEventListener("reconnected", scheduleRecovery);
        api.addEventListener("status", scheduleRecovery);
        api.addEventListener("execution_interrupted", ({ detail }) => {
            const node = findNode(detail?.node_id);
            if (node?.jrH3PromptReview?.reviewId) setReviewState(node, "Cancelled");
        });
        api.addEventListener("execution_error", ({ detail }) => {
            const node = findNode(detail?.node_id);
            if (node?.jrH3PromptReview?.reviewId) setReviewState(node, "Error");
        });
        scheduleRecovery();
    },
    async afterConfigureGraph() {
        scheduleRecovery();
    },
    loadedGraphNode(node) {
        if (node.comfyClass !== NODE_ID && node.type !== NODE_ID) return;
        const payload = pendingByNode.get(String(node.id));
        if (payload && beginReview(node, payload)) pendingByNode.delete(String(node.id));
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        const originalExecutionStart = nodeType.prototype.onExecutionStart;
        const originalRemoved = nodeType.prototype.onRemoved;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            buildReviewWidget(this);
            const payload = pendingByNode.get(String(this.id));
            if (payload && beginReview(this, payload)) pendingByNode.delete(String(this.id));
            return result;
        };

        nodeType.prototype.onExecutionStart = function () {
            const result = originalExecutionStart?.apply(this, arguments);
            if (this.jrH3PromptReview) {
                this.jrH3PromptReview.reviewId = null;
                this.jrH3PromptReview.textarea.value = "";
                this.jrH3PromptReview.textarea.placeholder = "Waiting for upstream prompt...";
                setReviewState(this, "Idle");
            }
            return result;
        };

        nodeType.prototype.onRemoved = function () {
            pendingByNode.delete(String(this.id));
            return originalRemoved?.apply(this, arguments);
        };
    },
});
