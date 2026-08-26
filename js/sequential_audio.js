import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EVENT = "jr_h3.sequential_audio.chunk_committed";
let pending = null;
let queueing = false;

function toast(severity, summary, detail) {
    app.extensionManager?.toast?.add?.({ severity, summary, detail, life: 5000 });
}

async function queueNextChunk(item) {
    if (queueing || !item?.has_next || !item?.auto_queue_next) return;
    queueing = true;
    try {
        const prompt = await app.graphToPrompt();
        await api.queuePrompt(0, prompt);
        toast(
            "info",
            "JR H3 queued next audio chunk",
            `Chunk ${Number(item.chunk_index) + 2}/${item.total_chunks}`,
        );
    } catch (error) {
        console.error("[JR H3 Sequential Audio] Could not queue next chunk", error);
        toast(
            "error",
            "JR H3 sequential audio paused",
            "The next prompt could not be queued. Queue the workflow manually to resume from disk.",
        );
    } finally {
        queueing = false;
    }
}

app.registerExtension({
    name: "JR.MiniMaxH3.SequentialAudio",

    async setup() {
        api.addEventListener(EVENT, ({ detail }) => {
            if (!detail?.has_next) {
                pending = null;
                if (detail?.filename) {
                    toast("success", "JR H3 audio sequence complete", detail.filename);
                }
                return;
            }
            pending = detail;
        });

        api.addEventListener("execution_success", async () => {
            const item = pending;
            pending = null;
            if (!item) return;
            await new Promise((resolve) => setTimeout(resolve, 250));
            await queueNextChunk(item);
        });

        const clearPending = () => {
            pending = null;
        };
        api.addEventListener("execution_error", clearPending);
        api.addEventListener("execution_interrupted", clearPending);
    },
});
