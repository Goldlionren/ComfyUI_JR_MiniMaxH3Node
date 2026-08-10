import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "JR_H3_DirectorDesk";
const PROP_KEY = "jr_h3_director_state";
const STATE_WIDGET = "director_state_json";
const SNAP = 0.1;
const MAX_TIMELINE_SECONDS = 3600;
const instances = new WeakMap();

const deepClone = (value) => JSON.parse(JSON.stringify(value));
const uid = (prefix) => `${prefix}-${globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`}`;
const snap = (value) => Number((Math.round(Math.max(0, Number(value) || 0) * 10 + Number.EPSILON) / 10).toFixed(1));
const snapDelta = (value) => {
    const scaled = (Number(value) || 0) * 10;
    return Number(((Math.sign(scaled) * Math.round(Math.abs(scaled))) / 10).toFixed(1));
};
const fmt = (value) => snap(value).toFixed(1);
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

function defaultState() {
    return {
        schema: "jr_h3_director_state",
        schema_version: 1,
        timeline: { duration_seconds: 10.0, fps: 24.0 },
        global_direction: "",
        shots: [{ id: "shot-1", start: 0.0, end: 10.0, direction: "", notes: "" }],
        visual_items: [],
        audio_items: [],
        ui: { selected_item_id: null, inspector_tab: "item", zoom: 1.0, lane_order: { visual: [], audio: [] } },
    };
}

function normalizeAsset(raw, kind) {
    const asset = raw && typeof raw === "object" ? raw : {};
    const identifier = (value, field) => {
        if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value)) {
            throw new Error(`${field} is not a valid stable identifier.`);
        }
        return value;
    };
    const stringField = (value, fallback, field, allowEmpty = false) => {
        if (value === undefined) value = fallback;
        if (typeof value !== "string" || (!allowEmpty && !value)) throw new Error(`${field} must be text.`);
        return value;
    };
    const optionalNumber = (value, field, minimum, integer = false) => {
        if (value == null) return null;
        if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || (integer && !Number.isInteger(value))) {
            throw new Error(`${field} must be ${integer ? "an integer" : "a finite number"} of at least ${minimum}.`);
        }
        return value;
    };
    const storedKind = stringField(asset.kind, null, "asset.kind");
    if (storedKind !== kind) throw new Error(`Asset kind ${storedKind} does not match timeline item kind ${kind}.`);
    const status = stringField(asset.status, "ready", "asset.status");
    if (!["ready", "missing", "invalid", "probe_unavailable"].includes(status)) throw new Error(`Unsupported asset status: ${status}.`);
    const filename = stringField(asset.filename, null, "asset.filename");
    if (filename.includes("/") || filename.includes("\\") || filename === "." || filename === "..") {
        throw new Error("asset.filename must be a basename; use subfolder separately.");
    }
    const subfolder = stringField(asset.subfolder, "", "asset.subfolder", true).replaceAll("\\", "/");
    if (subfolder && (/^(?:\/|[A-Za-z]:)/.test(subfolder) || subfolder.split("/").some((part) => !part || part === "." || part === ".."))) {
        throw new Error("asset.subfolder must be a safe relative ComfyUI path.");
    }
    const folderType = stringField(asset.type, "input", "asset.type");
    if (!["input", "output", "temp"].includes(folderType)) throw new Error(`Unsupported asset type: ${folderType}.`);
    return {
        id: identifier(asset.id, "asset.id"),
        kind: storedKind,
        filename,
        subfolder,
        type: folderType,
        display_name: stringField(asset.display_name, filename, "asset.display_name"),
        mime_type: stringField(asset.mime_type, "", "asset.mime_type", true),
        duration_seconds: optionalNumber(asset.duration_seconds, "asset.duration_seconds", 0),
        width: optionalNumber(asset.width, "asset.width", 1, true),
        height: optionalNumber(asset.height, "asset.height", 1, true),
        status,
    };
}

function normalizeState(raw) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("Director state must be an object.");
    if (!Object.keys(raw).length) return defaultState();
    if (new TextEncoder().encode(JSON.stringify(raw)).length > 512 * 1024) throw new Error("Director state exceeds the 512 KiB limit.");
    if (raw.schema !== "jr_h3_director_state" || raw.schema_version !== 1) {
        throw new Error(`Unsupported Director state schema/version: ${String(raw.schema)}/${String(raw.schema_version)}`);
    }
    for (const key of ["shots", "visual_items", "audio_items"]) {
        if (Array.isArray(raw[key]) && raw[key].length > 200) throw new Error(`${key} exceeds the 200-item limit.`);
    }
    const timelineRaw = Object.hasOwn(raw, "timeline") ? raw.timeline : {};
    const uiRaw = Object.hasOwn(raw, "ui") ? raw.ui : {};
    if (!timelineRaw || typeof timelineRaw !== "object" || Array.isArray(timelineRaw)) throw new Error("timeline must be an object.");
    if (!uiRaw || typeof uiRaw !== "object" || Array.isArray(uiRaw)) throw new Error("ui must be an object.");
    const laneRaw = Object.hasOwn(uiRaw, "lane_order") ? uiRaw.lane_order : {};
    if (!laneRaw || typeof laneRaw !== "object" || Array.isArray(laneRaw)) throw new Error("ui.lane_order must be an object.");
    const finite = (value, fallback, field) => {
        if (value === undefined) return fallback;
        if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${field} must be a finite number.`);
        return value;
    };
    const time = (value, fallback, field) => {
        const result = finite(value, fallback, field);
        if (result < 0) throw new Error(`${field} must be at least 0.0.`);
        return result;
    };
    const identifier = (value, field) => {
        if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value)) {
            throw new Error(`${field} is not a valid stable identifier.`);
        }
        return value;
    };
    const duration = Math.max(SNAP, snap(time(timelineRaw.duration_seconds, 10, "timeline.duration_seconds")));
    if (duration > MAX_TIMELINE_SECONDS) throw new Error(`Timeline duration exceeds ${MAX_TIMELINE_SECONDS} seconds.`);
    const fps = clamp(finite(timelineRaw.fps, 24, "timeline.fps"), 1, 240);
    const text = (value, limit = 32768) => {
        if (value == null) return "";
        if (typeof value !== "string") throw new Error("Director text fields must be strings.");
        if (value.includes("\0")) throw new Error("Director text contains a NUL character.");
        if (value.length > limit) throw new Error(`Director text exceeds ${limit} characters.`);
        return value;
    };
    if (!Array.isArray(raw.shots) || !raw.shots.length) throw new Error("At least one Shot is required.");
    const shots = raw.shots.map((item, index) => {
        if (!item || typeof item !== "object" || Array.isArray(item)) throw new Error(`shots[${index}] must be an object.`);
        return {
            id: identifier(item.id, `shots[${index}].id`),
            start: snap(time(item.start, 0, `shots[${index}].start`)),
            end: snap(time(item.end, 0, `shots[${index}].end`)),
            direction: text(item.direction), notes: text(item.notes), _order: index,
        };
    });
    if (Object.hasOwn(raw, "visual_items") && !Array.isArray(raw.visual_items)) throw new Error("visual_items must be an array.");
    const visual = Array.isArray(raw.visual_items) ? raw.visual_items.slice(0, 200).map((item, index) => {
        if (!item || typeof item !== "object" || Array.isArray(item)) throw new Error(`visual_items[${index}] must be an object.`);
        if (!["image", "video"].includes(item.kind)) throw new Error(`Unsupported visual kind: ${String(item.kind)}.`);
        const kind = item.kind;
        if (!["reference_image", "first_frame", "reference_video"].includes(item.role)) throw new Error(`Unsupported visual role: ${String(item.role)}.`);
        const role = item.role;
        return {
            id: identifier(item.id, `visual_items[${index}].id`), kind, role,
            start: role === "first_frame" ? 0 : snap(time(item.start, 0, `visual_items[${index}].start`)),
            end: role === "first_frame" ? 0 : snap(time(item.end, 0, `visual_items[${index}].end`)),
            source_in: item.source_in == null ? null : snap(time(item.source_in, 0, `visual_items[${index}].source_in`)),
            source_out: item.source_out == null ? null : snap(time(item.source_out, 0, `visual_items[${index}].source_out`)),
            direction: text(item.direction), notes: text(item.notes),
            registry_order: (() => { const value = finite(item.registry_order, index + 1, `visual_items[${index}].registry_order`); if (!Number.isInteger(value) || value < 0) throw new Error("registry_order must be a non-negative integer."); return value; })(),
            asset: normalizeAsset(item.asset, kind),
        };
    }) : [];
    if (Object.hasOwn(raw, "audio_items") && !Array.isArray(raw.audio_items)) throw new Error("audio_items must be an array.");
    const audio = Array.isArray(raw.audio_items) ? raw.audio_items.slice(0, 200).map((item, index) => {
        if (!item || typeof item !== "object" || Array.isArray(item)) throw new Error(`audio_items[${index}] must be an object.`);
        return {
        id: identifier(item.id, `audio_items[${index}].id`),
        role: (() => { if (!["reference_audio", "driving_audio"].includes(item.role)) throw new Error(`Unsupported audio role: ${String(item.role)}.`); return item.role; })(),
        start: snap(time(item.start, 0, `audio_items[${index}].start`)), end: snap(time(item.end, duration, `audio_items[${index}].end`)),
        source_in: item.source_in == null ? null : snap(time(item.source_in, 0, `audio_items[${index}].source_in`)),
        source_out: item.source_out == null ? null : snap(time(item.source_out, 0, `audio_items[${index}].source_out`)),
        direction: text(item.direction), notes: text(item.notes),
        registry_order: (() => { const value = finite(item.registry_order, index + 1, `audio_items[${index}].registry_order`); if (!Number.isInteger(value) || value < 0) throw new Error("registry_order must be a non-negative integer."); return value; })(),
        asset: normalizeAsset(item.asset, "audio"),
        };
    }) : [];
    if (shots.length + visual.length + audio.length > 200) throw new Error("Director timeline exceeds the 200-item total limit.");
    const selected = uiRaw.selected_item_id;
    if (selected != null && selected !== "") identifier(selected, "ui.selected_item_id");
    const normalizeLaneOrder = (value, items, field) => {
        if (value === undefined) value = [];
        if (!Array.isArray(value) || value.some((entry) => typeof entry !== "string")) throw new Error(`${field} must be an item id array.`);
        const known = new Set(items.map((item) => item.id));
        if (value.some((entry) => !known.has(entry)) || new Set(value).size !== value.length) throw new Error(`${field} contains unknown or duplicate item ids.`);
        return [...value, ...items.map((item) => item.id).filter((id) => !value.includes(id))];
    };
    return {
        schema: "jr_h3_director_state", schema_version: 1,
        timeline: { duration_seconds: duration, fps },
        global_direction: text(raw.global_direction, 65536),
        shots: shots.length ? shots.map(({ _order, ...item }) => item) : defaultState().shots,
        visual_items: visual, audio_items: audio,
        ui: {
            selected_item_id: selected == null || selected === "" ? null : selected,
            inspector_tab: text(uiRaw.inspector_tab || "item"),
            zoom: clamp(finite(uiRaw.zoom, 1, "ui.zoom"), 0.75, 3),
            lane_order: {
                visual: normalizeLaneOrder(laneRaw.visual, visual, "ui.lane_order.visual"),
                audio: normalizeLaneOrder(laneRaw.audio, audio, "ui.lane_order.audio"),
            },
        },
    };
}

function validateState(state) {
    const duration = state.timeline.duration_seconds;
    if (!(duration > 0)) return "Timeline duration must be greater than zero.";
    if (duration > MAX_TIMELINE_SECONDS) return `Timeline duration cannot exceed ${MAX_TIMELINE_SECONDS} seconds.`;
    if (!state.shots.length) return "At least one Shot is required.";
    const ids = new Set();
    const shots = [...state.shots].sort((a, b) => a.start - b.start || a.end - b.end || a.id.localeCompare(b.id));
    for (let i = 0; i < shots.length; i++) {
        const item = shots[i];
        if (ids.has(item.id)) return `Duplicate item id: ${item.id}`;
        ids.add(item.id);
        if (item.start < 0 || item.end <= item.start || item.end > duration) return `Invalid Shot range: ${item.id}`;
        if (i && item.start < shots[i - 1].end) return `Shots ${shots[i - 1].id} and ${item.id} overlap.`;
    }
    const firstFrames = state.visual_items.filter((item) => item.role === "first_frame");
    if (firstFrames.length > 1) return "Only one First Frame may exist.";
    if (firstFrames.length && shots[0].start !== 0) return "A First Frame requires the first Shot to begin at 0.0s.";
    for (const item of state.visual_items) {
        if (ids.has(item.id)) return `Duplicate item id: ${item.id}`;
        ids.add(item.id);
        if (item.role === "first_frame") {
            if (item.kind !== "image" || item.start !== 0 || item.end !== 0) return "First Frame must be an IMAGE point at 0.0s.";
        } else {
            const expectedKind = item.role === "reference_video" ? "video" : "image";
            if (item.kind !== expectedKind || item.asset?.kind !== item.kind) return `Role/media type mismatch: ${item.id}`;
            if (item.start < 0 || item.end <= item.start || item.end > duration) return `Invalid visual range: ${item.id}`;
        }
        if ((item.source_in == null) !== (item.source_out == null)) return `Set both source in and source out for ${item.id}.`;
        if (item.source_in != null && item.source_out <= item.source_in) return `Invalid source range: ${item.id}`;
        if (item.source_out != null && item.asset?.duration_seconds != null && item.source_out > item.asset.duration_seconds) return `Source out exceeds media duration: ${item.id}`;
    }
    const driving = [];
    for (const item of state.audio_items) {
        if (ids.has(item.id)) return `Duplicate item id: ${item.id}`;
        ids.add(item.id);
        if (item.start < 0 || item.end <= item.start || item.end > duration) return `Invalid audio range: ${item.id}`;
        if ((item.source_in == null) !== (item.source_out == null)) return `Set both source in and source out for ${item.id}.`;
        if (item.source_in != null && item.source_out <= item.source_in) return `Invalid source range: ${item.id}`;
        if (item.asset?.kind !== "audio") return `Audio item has a non-audio asset: ${item.id}`;
        if (item.source_out != null && item.asset?.duration_seconds != null && item.source_out > item.asset.duration_seconds) return `Source out exceeds media duration: ${item.id}`;
        if (item.role === "driving_audio") driving.push(item);
    }
    driving.sort((a, b) => a.start - b.start || a.end - b.end || a.id.localeCompare(b.id));
    for (let i = 1; i < driving.length; i++) {
        if (driving[i].start < driving[i - 1].end) return `Driving Audio ${driving[i - 1].id} and ${driving[i].id} overlap.`;
    }
    return "";
}

function laneLayout(items, preferredOrder = []) {
    const rank = new Map(preferredOrder.map((id, index) => [id, index]));
    const ordered = [...items].sort((a, b) => a.start - b.start || a.end - b.end || (rank.get(a.id) ?? 1e9) - (rank.get(b.id) ?? 1e9) || a.id.localeCompare(b.id));
    const ends = [];
    const result = new Map();
    for (const item of ordered) {
        if (item.role === "first_frame") {
            result.set(item.id, 0);
            continue;
        }
        let lane = ends.findIndex((end) => end <= item.start);
        if (lane < 0) lane = ends.length;
        ends[lane] = item.end;
        result.set(item.id, lane);
    }
    return { lanes: Math.max(1, ends.length, items.some((item) => item.role === "first_frame") ? 1 : 0), result };
}

function assetUrl(asset) {
    const query = new URLSearchParams({ filename: asset.filename, subfolder: asset.subfolder, type: asset.type });
    return api.apiURL(`/view?${query.toString()}`);
}

function setStatus(instance, message, error = false) {
    instance.status.textContent = message || "Ready";
    instance.status.classList.toggle("error", Boolean(error));
}

function commit(instance, next, { undo = true, render = true } = {}) {
    if (instance.loadError) {
        setStatus(instance, "Persisted state is invalid. Use Reset Director State before editing.", true);
        return false;
    }
    let normalized;
    try { normalized = normalizeState(next); }
    catch (error) { setStatus(instance, error?.message || "Director state is invalid.", true); return false; }
    const error = validateState(normalized);
    if (error) {
        setStatus(instance, error, true);
        return false;
    }
    const node = instance.node;
    const graph = node.graph;
    if (undo) graph?.beforeChange?.(node);
    instance.syncing = true;
    instance.state = normalized;
    node.properties ||= {};
    node.properties[PROP_KEY] = deepClone(normalized);
    if (instance.stateWidget) instance.stateWidget.value = JSON.stringify(normalized);
    instance.syncing = false;
    if (undo) graph?.afterChange?.(node);
    graph?.setDirtyCanvas?.(true, true);
    setStatus(instance, "Ready");
    if (render) renderAll(instance);
    return true;
}

function selectItem(instance, id, { render = true } = {}) {
    const next = deepClone(instance.state);
    next.ui.selected_item_id = id;
    commit(instance, next, { undo: false, render });
    if (!render) renderInspector(instance);
}

function findItem(state, id) {
    for (const key of ["shots", "visual_items", "audio_items"]) {
        const index = state[key].findIndex((item) => item.id === id);
        if (index >= 0) return { key, index, item: state[key][index] };
    }
    return null;
}

function maxRegistryOrder(state, family) {
    const source = family === "audio" ? state.audio_items : state.visual_items.filter((item) => item.kind === family);
    return source.reduce((value, item) => Math.max(value, Number(item.registry_order) || 0), 0);
}

function updateItem(instance, id, changes) {
    const next = deepClone(instance.state);
    const found = findItem(next, id);
    if (!found) return false;
    const wasFirstFrame = found.item.role === "first_frame";
    Object.assign(found.item, changes);
    if (found.item.role === "first_frame") Object.assign(found.item, { start: 0, end: 0, source_in: null, source_out: null });
    else if (wasFirstFrame && found.item.end <= found.item.start) Object.assign(found.item, { start: 0, end: next.timeline.duration_seconds });
    return commit(instance, next);
}

function deleteItem(instance, id) {
    const next = deepClone(instance.state);
    const found = findItem(next, id);
    if (!found) return;
    if (found.key === "shots" && next.shots.length === 1) return setStatus(instance, "At least one Shot is required.", true);
    next[found.key].splice(found.index, 1);
    if (found.key === "visual_items") next.ui.lane_order.visual = next.ui.lane_order.visual.filter((itemId) => itemId !== id);
    if (found.key === "audio_items") next.ui.lane_order.audio = next.ui.lane_order.audio.filter((itemId) => itemId !== id);
    next.ui.selected_item_id = null;
    commit(instance, next);
}

function duplicateItem(instance, id) {
    const next = deepClone(instance.state);
    const found = findItem(next, id);
    if (!found) return;
    const copy = deepClone(found.item);
    copy.id = uid(found.key === "shots" ? "shot" : found.key === "visual_items" ? "visual" : "audio");
    const length = Math.max(SNAP, found.item.end - found.item.start);
    if (found.key === "shots" || copy.role === "driving_audio") {
        copy.start = snap(found.item.end);
        copy.end = snap(copy.start + length);
        if (copy.end > next.timeline.duration_seconds) return setStatus(instance, "No non-overlapping room is available for this duplicate.", true);
    }
    if (found.key === "visual_items") copy.registry_order = maxRegistryOrder(next, copy.kind) + 1;
    if (found.key === "audio_items") copy.registry_order = maxRegistryOrder(next, "audio") + 1;
    next[found.key].splice(found.index + 1, 0, copy);
    if (found.key === "visual_items") next.ui.lane_order.visual.push(copy.id);
    if (found.key === "audio_items") next.ui.lane_order.audio.push(copy.id);
    next.ui.selected_item_id = copy.id;
    commit(instance, next);
}

function splitItem(instance, id) {
    const next = deepClone(instance.state);
    const found = findItem(next, id);
    if (!found || found.item.role === "first_frame") return;
    const midpoint = snap((found.item.start + found.item.end) / 2);
    if (midpoint <= found.item.start || midpoint >= found.item.end) return setStatus(instance, "The item is too short to split at 0.1s precision.", true);
    const second = deepClone(found.item);
    second.id = uid(found.key === "shots" ? "shot" : found.key === "visual_items" ? "visual" : "audio");
    if (found.item.source_in != null) {
        const ratio = (midpoint - found.item.start) / (found.item.end - found.item.start);
        const sourceMid = snap(found.item.source_in + (found.item.source_out - found.item.source_in) * ratio);
        found.item.source_out = sourceMid;
        second.source_in = sourceMid;
    }
    found.item.end = midpoint;
    second.start = midpoint;
    if (found.key === "visual_items") second.registry_order = maxRegistryOrder(next, second.kind) + 1;
    if (found.key === "audio_items") second.registry_order = maxRegistryOrder(next, "audio") + 1;
    next[found.key].splice(found.index + 1, 0, second);
    if (found.key === "visual_items") next.ui.lane_order.visual.push(second.id);
    if (found.key === "audio_items") next.ui.lane_order.audio.push(second.id);
    next.ui.selected_item_id = second.id;
    commit(instance, next);
}

function reorderItem(instance, id, delta) {
    const next = deepClone(instance.state);
    const found = findItem(next, id);
    if (!found) return;
    if (found.key === "shots") {
        const ordered = [...next.shots].sort((a, b) => a.start - b.start || a.id.localeCompare(b.id));
        const index = ordered.findIndex((item) => item.id === id);
        const other = ordered[index + delta];
        if (!other) return;
        const currentRange = [found.item.start, found.item.end];
        found.item.start = other.start; found.item.end = other.end;
        other.start = currentRange[0]; other.end = currentRange[1];
    } else {
        const laneKey = found.key === "audio_items" ? "audio" : "visual";
        const order = next.ui.lane_order[laneKey];
        const index = order.indexOf(id);
        const otherIndex = index + delta;
        if (index < 0 || otherIndex < 0 || otherIndex >= order.length) return;
        [order[index], order[otherIndex]] = [order[otherIndex], order[index]];
    }
    commit(instance, next);
}

function startDrag(instance, event, item, mode) {
    if (event.button !== 0 || item.role === "first_frame") return;
    event.preventDefault(); event.stopPropagation();
    event.currentTarget.setPointerCapture?.(event.pointerId);
    const before = deepClone(instance.state);
    const draft = deepClone(instance.state);
    const found = findItem(draft, item.id);
    const startX = event.clientX;
    const duration = draft.timeline.duration_seconds;
    const trackWidth = Math.max(1, instance.shotTrack.getBoundingClientRect().width);
    const original = { start: found.item.start, end: found.item.end };
    instance.dragging = true;
    const move = (moveEvent) => {
        const delta = snapDelta((moveEvent.clientX - startX) / trackWidth * duration);
        const length = original.end - original.start;
        if (mode === "move") {
            const start = clamp(snap(original.start + delta), 0, duration - length);
            found.item.start = start; found.item.end = snap(start + length);
        } else if (mode === "start") {
            found.item.start = clamp(snap(original.start + delta), 0, found.item.end - SNAP);
        } else {
            found.item.end = clamp(snap(original.end + delta), found.item.start + SNAP, duration);
        }
        instance.state = draft;
        const error = validateState(draft);
        setStatus(instance, error || `${fmt(found.item.start)}s → ${fmt(found.item.end)}s`, Boolean(error));
        renderTimeline(instance);
    };
    const detach = () => {
        globalThis.removeEventListener("pointermove", move);
        globalThis.removeEventListener("pointerup", finish);
        globalThis.removeEventListener("pointercancel", cancel);
        instance.activeDragCancel = null;
    };
    const finish = () => {
        detach();
        instance.dragging = false;
        const error = validateState(draft);
        instance.state = before;
        if (error) {
            setStatus(instance, error, true); renderAll(instance); return;
        }
        if (JSON.stringify(draft) === JSON.stringify(before)) return renderAll(instance);
        commit(instance, draft);
    };
    const cancel = () => {
        detach();
        instance.dragging = false; instance.state = before; renderAll(instance);
    };
    instance.activeDragCancel?.();
    instance.activeDragCancel = cancel;
    globalThis.addEventListener("pointermove", move);
    globalThis.addEventListener("pointerup", finish, { once: true });
    globalThis.addEventListener("pointercancel", cancel, { once: true });
}

function itemTitle(item) {
    if (item.role === "first_frame") return `◆ First Frame · ${item.asset.display_name}`;
    if (item.asset) return `${item.role.replaceAll("_", " ")} · ${item.asset.display_name}`;
    return `Shot · ${item.id}`;
}

function itemElement(instance, item, lane) {
    const element = document.createElement("div");
    element.className = `jr-dd-item ${item.role || "shot"}${instance.state.ui.selected_item_id === item.id ? " selected" : ""}`;
    if (item.asset?.status === "missing" || item.asset?.status === "invalid") element.classList.add("missing");
    element.dataset.itemId = item.id;
    element.title = `${itemTitle(item)}\n${fmt(item.start)}s – ${fmt(item.end)}s`;
    const duration = instance.state.timeline.duration_seconds;
    if (item.role === "first_frame") {
        element.classList.add("point");
        element.style.left = "0%"; element.style.width = "18px";
    } else {
        element.style.left = `${item.start / duration * 100}%`;
        element.style.width = `${Math.max(0.4, (item.end - item.start) / duration * 100)}%`;
    }
    element.style.top = `${lane * 34 + 3}px`;
    const label = document.createElement("span"); label.textContent = itemTitle(item); element.append(label);
    if (item.role !== "first_frame") {
        for (const side of ["start", "end"]) {
            const handle = document.createElement("i"); handle.className = `jr-dd-handle ${side}`;
            handle.addEventListener("pointerdown", (event) => startDrag(instance, event, item, side));
            element.append(handle);
        }
    }
    element.addEventListener("pointerdown", (event) => {
        if (event.target.classList.contains("jr-dd-handle")) return;
        selectItem(instance, item.id, { render: false });
        startDrag(instance, event, item, "move");
    });
    element.addEventListener("dblclick", (event) => {
        event.stopPropagation(); selectItem(instance, item.id);
        requestAnimationFrame(() => instance.inspector.querySelector("textarea")?.focus());
    });
    element.addEventListener("contextmenu", (event) => showContextMenu(instance, event, item));
    return element;
}

function renderRuler(instance) {
    instance.ruler.replaceChildren();
    const duration = instance.state.timeline.duration_seconds;
    const tickStep = Math.max(1, Math.ceil(duration / 100));
    const steps = Math.max(1, Math.ceil(duration / tickStep));
    for (let index = 0; index <= steps; index++) {
        const second = Math.min(duration, index * tickStep);
        const tick = document.createElement("span");
        tick.style.left = `${Math.min(100, second / duration * 100)}%`;
        tick.textContent = `${second}s`; instance.ruler.append(tick);
    }
}

function renderTrack(instance, target, items, layout) {
    target.replaceChildren();
    const height = Math.max(38, layout.lanes * 34 + 6);
    target.style.height = `${height}px`;
    for (const item of items) target.append(itemElement(instance, item, layout.result.get(item.id) || 0));
}

function renderTimeline(instance) {
    const zoom = instance.state.ui.zoom;
    instance.timelineInner.style.width = `${zoom * 100}%`;
    renderRuler(instance);
    const shots = [...instance.state.shots].sort((a, b) => a.start - b.start || a.id.localeCompare(b.id));
    renderTrack(instance, instance.shotTrack, shots, { lanes: 1, result: new Map(shots.map((item) => [item.id, 0])) });
    renderTrack(instance, instance.visualTrack, instance.state.visual_items, laneLayout(instance.state.visual_items, instance.state.ui.lane_order.visual));
    renderTrack(instance, instance.audioTrack, instance.state.audio_items, laneLayout(instance.state.audio_items, instance.state.ui.lane_order.audio));
}

function field(labelText, input) {
    const label = document.createElement("label");
    const span = document.createElement("span"); span.textContent = labelText;
    label.append(span, input); return label;
}

function numberInput(value, change, { min = 0, max = 3600 } = {}) {
    const input = document.createElement("input"); input.type = "number"; input.step = "0.1";
    input.min = String(min); input.max = String(max); input.value = fmt(value);
    input.addEventListener("change", () => change(snap(input.value)));
    return input;
}

function textArea(value, change, placeholder) {
    const area = document.createElement("textarea"); area.value = value || ""; area.placeholder = placeholder;
    area.maxLength = 32768;
    area.addEventListener("blur", () => change(area.value));
    return area;
}

function button(text, action, className = "") {
    const value = document.createElement("button"); value.type = "button"; value.textContent = text;
    if (className) value.className = className; value.addEventListener("click", action); return value;
}

function releaseMediaElement(instance, media) {
    try { media.pause?.(); media.removeAttribute("src"); media.removeAttribute("poster"); media.load?.(); }
    catch { /* detached browser media cleanup is best effort */ }
    instance.mediaElements?.delete(media);
}

function mediaPreview(instance, item) {
    const wrap = document.createElement("div"); wrap.className = "jr-dd-preview";
    const asset = item.asset;
    const assetFingerprint = `${asset.kind}|${asset.type}|${asset.subfolder}|${asset.filename}`;
    const isCurrentAsset = () => {
        const current = findItem(instance.state, item.id);
        if (!current) return false;
        const value = current.item.asset;
        return `${value.kind}|${value.type}|${value.subfolder}|${value.filename}` === assetFingerprint;
    };
    if (!asset) return wrap;
    let media;
    if (asset.kind === "image") { media = document.createElement("img"); media.alt = asset.display_name; }
    else if (asset.kind === "video") { media = document.createElement("video"); media.controls = true; media.preload = "metadata"; }
    else { media = document.createElement("audio"); media.controls = true; media.preload = "metadata"; }
    instance.mediaElements.add(media);
    const meta = document.createElement("small");
    meta.textContent = `${asset.display_name}${asset.duration_seconds ? ` · ${Number(asset.duration_seconds).toFixed(2)}s` : ""} · ${asset.status}`;
    media.src = assetUrl(asset);
    if (asset.kind !== "image") media.addEventListener("loadedmetadata", () => {
        if (instance.destroyed || !isCurrentAsset()) return;
        const duration = Number.isFinite(media.duration) ? Number(media.duration) : null;
        if (asset.kind === "video" && duration > 0) {
            media.currentTime = Math.min(0.1, duration / 2);
            media.addEventListener("seeked", () => {
                if (instance.destroyed || !media.videoWidth || !media.videoHeight) return;
                const scale = Math.min(1, 480 / media.videoWidth);
                const canvas = document.createElement("canvas");
                canvas.width = Math.max(1, Math.round(media.videoWidth * scale));
                canvas.height = Math.max(1, Math.round(media.videoHeight * scale));
                canvas.getContext("2d")?.drawImage(media, 0, 0, canvas.width, canvas.height);
                media.poster = canvas.toDataURL("image/jpeg", 0.75);
            }, { once: true });
        }
        if (asset.duration_seconds == null && duration > 0) {
            const next = deepClone(instance.state); const found = findItem(next, item.id);
            if (found) { found.item.asset.duration_seconds = duration; commit(instance, next, { undo: false, render: false }); }
            meta.textContent = `${asset.display_name} · ${duration.toFixed(2)}s · ${asset.status}`;
        }
    });
    media.addEventListener("error", () => {
        if (instance.destroyed || !isCurrentAsset()) return;
        const next = deepClone(instance.state);
        const found = findItem(next, item.id);
        if (found && found.item.asset.status !== "missing") {
            found.item.asset.status = "missing";
            commit(instance, next, { undo: false, render: false });
            renderTimeline(instance);
        }
        meta.textContent = `${asset.display_name} · missing`;
        setStatus(instance, `Missing or unreadable asset: ${asset.display_name}. Relink or remove it.`, true);
    });
    wrap.append(media);
    wrap.append(meta);
    wrap.append(button("Relink", () => chooseAsset(instance, asset.kind, item.id)));
    return wrap;
}

function renderInspector(instance) {
    const inspector = instance.inspector;
    inspector.querySelectorAll("img,video,audio").forEach((media) => releaseMediaElement(instance, media));
    inspector.replaceChildren();
    const found = findItem(instance.state, instance.state.ui.selected_item_id);
    const title = document.createElement("h4"); title.textContent = "Inspector"; inspector.append(title);
    if (!found) {
        const empty = document.createElement("p"); empty.textContent = "Select an item to edit its role, timing, direction and notes.";
        inspector.append(empty); return;
    }
    const item = found.item;
    const badge = document.createElement("div"); badge.className = "jr-dd-badge"; badge.textContent = itemTitle(item); inspector.append(badge);
    if (item.asset) inspector.append(mediaPreview(instance, item));
    if (found.key !== "shots") {
        const select = document.createElement("select");
        const roles = found.key === "visual_items"
            ? (item.kind === "image" ? ["reference_image", "first_frame"] : ["reference_video"])
            : ["reference_audio", "driving_audio"];
        for (const role of roles) { const option = document.createElement("option"); option.value = role; option.textContent = role.replaceAll("_", " "); select.append(option); }
        select.value = item.role;
        select.addEventListener("change", () => updateItem(instance, item.id, { role: select.value }));
        inspector.append(field("Role", select));
    }
    if (item.role !== "first_frame") {
        inspector.append(field("Timeline start (s)", numberInput(item.start, (value) => updateItem(instance, item.id, { start: value }))));
        inspector.append(field("Timeline end (s)", numberInput(item.end, (value) => updateItem(instance, item.id, { end: value }), { max: instance.state.timeline.duration_seconds })));
    }
    if (item.asset && item.kind !== "image") {
        inspector.append(field("Source in (s)", numberInput(item.source_in ?? 0, (value) => updateItem(instance, item.id, { source_in: value, source_out: item.source_out ?? item.asset.duration_seconds ?? value + SNAP }))));
        inspector.append(field("Source out (s)", numberInput(item.source_out ?? item.asset.duration_seconds ?? item.end - item.start, (value) => updateItem(instance, item.id, { source_in: item.source_in ?? 0, source_out: value }))));
    }
    inspector.append(field("Direction", textArea(item.direction, (value) => updateItem(instance, item.id, { direction: value }), "What should happen or be preserved here?")));
    inspector.append(field("Notes", textArea(item.notes, (value) => updateItem(instance, item.id, { notes: value }), "Production notes, continuity, caveats…")));
    const actions = document.createElement("div"); actions.className = "jr-dd-actions";
    actions.append(
        button("Duplicate", () => duplicateItem(instance, item.id)),
        button("Split", () => splitItem(instance, item.id)),
        button("↑", () => reorderItem(instance, item.id, -1)),
        button("↓", () => reorderItem(instance, item.id, 1)),
        button("Delete", () => deleteItem(instance, item.id), "danger"),
    );
    inspector.append(actions);
}

function renderAll(instance) {
    if (instance.dragging) return renderTimeline(instance);
    instance.durationInput.value = fmt(instance.state.timeline.duration_seconds);
    instance.fpsInput.value = String(instance.state.timeline.fps);
    instance.zoomInput.value = String(instance.state.ui.zoom);
    if (instance.globalInput !== document.activeElement) instance.globalInput.value = instance.state.global_direction;
    renderTimeline(instance); renderInspector(instance);
}

function showContextMenu(instance, event, item) {
    event.preventDefault(); event.stopPropagation();
    instance.closeContextMenu?.();
    const menu = document.createElement("div"); menu.className = "jr-dd-menu";
    menu.style.left = `${event.clientX}px`; menu.style.top = `${event.clientY}px`;
    let dismiss; let dismissTimer;
    const close = () => {
        if (dismissTimer) clearTimeout(dismissTimer);
        if (dismiss) document.removeEventListener("pointerdown", dismiss);
        menu.remove(); instance.contextMenu = null; instance.closeContextMenu = null;
    };
    const action = (label, fn) => menu.append(button(label, () => { close(); fn(); }));
    action("Edit in Inspector", () => selectItem(instance, item.id));
    action("Duplicate", () => duplicateItem(instance, item.id));
    if (item.role !== "first_frame") action("Split at midpoint", () => splitItem(instance, item.id));
    if (item.kind === "image" && item.role !== "first_frame") action("Set as First Frame", () => updateItem(instance, item.id, { role: "first_frame", start: 0, end: 0 }));
    action("Move Up", () => reorderItem(instance, item.id, -1));
    action("Move Down", () => reorderItem(instance, item.id, 1));
    action("Delete", () => deleteItem(instance, item.id));
    document.body.append(menu); instance.contextMenu = menu;
    instance.closeContextMenu = close;
    dismiss = (pointerEvent) => { if (!menu.contains(pointerEvent.target)) close(); };
    dismissTimer = setTimeout(() => document.addEventListener("pointerdown", dismiss), 0);
}

async function browserDuration(asset) {
    if (asset.kind === "image") return null;
    return await new Promise((resolve) => {
        const media = document.createElement(asset.kind === "video" ? "video" : "audio");
        let timer;
        const finish = (value) => {
            clearTimeout(timer); media.onloadedmetadata = null; media.onerror = null;
            media.removeAttribute("src"); media.load?.(); resolve(value);
        };
        timer = setTimeout(() => finish(null), 5000);
        media.preload = "metadata";
        media.onloadedmetadata = () => finish(Number.isFinite(media.duration) ? media.duration : null);
        media.onerror = () => finish(null);
        media.src = assetUrl(asset);
    });
}

async function uploadAsset(file, kind) {
    const allowed = file.type.startsWith(`${kind}/`) || (kind === "image" && file.type.startsWith("image/"));
    if (!allowed && file.type) throw new Error(`The selected file is not ${kind} media.`);
    const form = new FormData(); form.append("image", file); form.append("type", "input"); form.append("subfolder", "jr_h3_director");
    const upload = await api.fetchApi("/upload/image", { method: "POST", body: form });
    if (!upload.ok) throw new Error(`ComfyUI upload failed (${upload.status}).`);
    const saved = await upload.json();
    let asset = {
        id: uid("asset"), kind, filename: saved.name, subfolder: saved.subfolder || "",
        type: saved.type || "input", display_name: file.name, mime_type: file.type || "",
        duration_seconds: null, width: null, height: null, status: "ready",
    };
    const probe = await api.fetchApi("/jr-h3/director/probe", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(asset),
    });
    if (!probe.ok) throw new Error((await probe.text()) || `Media inspection failed (${probe.status}).`);
    asset = normalizeAsset(await probe.json(), kind);
    if (asset.duration_seconds == null) asset.duration_seconds = await browserDuration(asset);
    return asset;
}

async function chooseAsset(instance, kind, replaceItemId = null) {
    const input = document.createElement("input"); input.type = "file"; input.accept = `${kind}/*`;
    input.onchange = async () => {
        const file = input.files?.[0]; if (!file) return;
        setStatus(instance, `Uploading ${file.name}…`);
        try {
            const asset = await uploadAsset(file, kind);
            if (instance.destroyed) return;
            const next = deepClone(instance.state);
            if (replaceItemId) {
                const found = findItem(next, replaceItemId);
                if (!found || found.item.asset.kind !== kind) throw new Error("Relink media type does not match the selected item.");
                asset.id = found.item.asset.id; found.item.asset = asset;
            } else if (kind === "image") {
                const item = {
                    id: uid("visual"), kind: "image", role: "reference_image", start: 0,
                    end: next.timeline.duration_seconds, source_in: null, source_out: null,
                    direction: "", notes: "", registry_order: maxRegistryOrder(next, "image") + 1, asset,
                };
                next.visual_items.push(item); next.ui.lane_order.visual.push(item.id); next.ui.selected_item_id = item.id;
            } else if (kind === "video") {
                const span = Math.min(next.timeline.duration_seconds, asset.duration_seconds || next.timeline.duration_seconds);
                const item = {
                    id: uid("visual"), kind: "video", role: "reference_video", start: 0, end: snap(span),
                    source_in: 0, source_out: snap(asset.duration_seconds || span), direction: "", notes: "",
                    registry_order: maxRegistryOrder(next, "video") + 1, asset,
                };
                next.visual_items.push(item); next.ui.lane_order.visual.push(item.id); next.ui.selected_item_id = item.id;
            } else {
                const span = Math.min(next.timeline.duration_seconds, asset.duration_seconds || next.timeline.duration_seconds);
                const item = {
                    id: uid("audio"), role: "reference_audio", start: 0, end: snap(span),
                    source_in: 0, source_out: snap(asset.duration_seconds || span), direction: "", notes: "",
                    registry_order: maxRegistryOrder(next, "audio") + 1, asset,
                };
                next.audio_items.push(item); next.ui.lane_order.audio.push(item.id); next.ui.selected_item_id = item.id;
            }
            commit(instance, next);
        } catch (error) { if (!instance.destroyed) setStatus(instance, error?.message || "Asset import failed.", true); }
    };
    input.click();
}

function addShot(instance) {
    const next = deepClone(instance.state);
    const ordered = [...next.shots].sort((a, b) => a.end - b.end);
    const lastEnd = ordered.at(-1)?.end ?? 0;
    const end = Math.min(next.timeline.duration_seconds, snap(lastEnd + 1));
    if (end <= lastEnd) return setStatus(instance, "No free time remains after the last Shot. Resize or split an existing Shot.", true);
    const item = { id: uid("shot"), start: lastEnd, end, direction: "", notes: "" };
    next.shots.push(item); next.ui.selected_item_id = item.id; commit(instance, next);
}

function buildPanel(node, stateWidget) {
    const panel = document.createElement("div"); panel.className = "jr-dd";
    const toolbar = document.createElement("div"); toolbar.className = "jr-dd-toolbar";
    const durationInput = document.createElement("input"); durationInput.type = "number"; durationInput.min = "0.1"; durationInput.step = "0.1";
    durationInput.max = String(MAX_TIMELINE_SECONDS);
    const fpsInput = document.createElement("input"); fpsInput.type = "number"; fpsInput.min = "1"; fpsInput.max = "240"; fpsInput.step = "1";
    const zoomInput = document.createElement("input"); zoomInput.type = "range"; zoomInput.min = "0.75"; zoomInput.max = "3"; zoomInput.step = "0.25";
    toolbar.append(
        field("Duration", durationInput), field("FPS", fpsInput), field("Zoom", zoomInput),
        button("+ Shot", () => addShot(instance)),
        button("+ Image", () => chooseAsset(instance, "image")),
        button("+ Video", () => chooseAsset(instance, "video")),
        button("+ Audio", () => chooseAsset(instance, "audio")),
        button("Reset", () => {
            if (!globalThis.confirm?.("Reset all Director Desk timeline state?")) return;
            instance.loadError = "";
            commit(instance, defaultState());
        }),
    );
    const globalInput = textArea("", (value) => {
        const next = deepClone(instance.state); next.global_direction = value; commit(instance, next);
    }, "Global direction for the whole video: style, performance, continuity, camera language…");
    globalInput.className = "jr-dd-global";
    globalInput.maxLength = 65536;
    const globalSection = document.createElement("section");
    const globalTitle = document.createElement("strong"); globalTitle.textContent = "Global Direction";
    globalSection.append(globalTitle, globalInput);
    const status = document.createElement("div"); status.className = "jr-dd-status"; status.textContent = "Ready";
    const main = document.createElement("div"); main.className = "jr-dd-main";
    const timelineScroll = document.createElement("div"); timelineScroll.className = "jr-dd-timeline-scroll";
    const timelineInner = document.createElement("div"); timelineInner.className = "jr-dd-timeline";
    const ruler = document.createElement("div"); ruler.className = "jr-dd-ruler"; timelineInner.append(ruler);
    const makeRow = (name) => {
        const row = document.createElement("div"); row.className = "jr-dd-row";
        const label = document.createElement("b"); label.textContent = name;
        const track = document.createElement("div"); track.className = `jr-dd-track ${name.toLowerCase()}`;
        row.append(label, track); timelineInner.append(row); return track;
    };
    const shotTrack = makeRow("SHOT"), visualTrack = makeRow("VISUAL"), audioTrack = makeRow("AUDIO");
    timelineScroll.append(timelineInner);
    const inspector = document.createElement("aside"); inspector.className = "jr-dd-inspector";
    main.append(timelineScroll, inspector); panel.append(toolbar, globalSection, status, main);
    const sourceState = node.properties?.[PROP_KEY] || safeWidgetState(stateWidget);
    let initialState; let loadError = "";
    try { initialState = normalizeState(sourceState); }
    catch (error) { initialState = defaultState(); loadError = error?.message || "Director state could not be loaded."; }
    const instance = {
        node, panel, stateWidget, state: initialState, loadError,
        toolbar, durationInput, fpsInput, zoomInput, globalInput, status, main, timelineScroll,
        timelineInner, ruler, shotTrack, visualTrack, audioTrack, inspector,
        syncing: false, dragging: false, destroyed: false, contextMenu: null,
        closeContextMenu: null, activeDragCancel: null, mediaElements: new Set(), cleanup: [],
    };
    durationInput.addEventListener("change", () => {
        const next = deepClone(instance.state); const value = Math.max(SNAP, snap(durationInput.value));
        const previous = next.timeline.duration_seconds; next.timeline.duration_seconds = value;
        for (const item of [...next.shots, ...next.visual_items, ...next.audio_items]) {
            if (item.role !== "first_frame" && item.end === previous) item.end = value;
        }
        commit(instance, next);
    });
    fpsInput.addEventListener("change", () => { const next = deepClone(instance.state); next.timeline.fps = clamp(Number(fpsInput.value) || 24, 1, 240); commit(instance, next); });
    zoomInput.addEventListener("change", () => { const next = deepClone(instance.state); next.ui.zoom = clamp(Number(zoomInput.value) || 1, .75, 3); commit(instance, next, { undo: false }); });
    for (const type of ["pointerdown", "mousedown", "touchstart", "wheel", "keydown"]) panel.addEventListener(type, (event) => event.stopPropagation());
    panel.addEventListener("dragover", (event) => { event.preventDefault(); event.stopPropagation(); });
    panel.addEventListener("drop", (event) => {
        event.preventDefault(); event.stopPropagation();
        const file = event.dataTransfer?.files?.[0]; if (!file) return;
        const kind = file.type.startsWith("image/") ? "image" : file.type.startsWith("video/") ? "video" : file.type.startsWith("audio/") ? "audio" : null;
        if (!kind) return setStatus(instance, "Drop an image, video, or audio file.", true);
        uploadAsset(file, kind).then((asset) => {
            if (instance.destroyed) return;
            // Reuse the same add code through a synthetic one-file picker would be fragile;
            // direct drops are imported as reference items here.
            const next = deepClone(instance.state);
            const span = Math.min(next.timeline.duration_seconds, asset.duration_seconds || next.timeline.duration_seconds);
            let item;
            if (kind === "image") { item = { id: uid("visual"), kind, role: "reference_image", start: 0, end: next.timeline.duration_seconds, source_in: null, source_out: null, direction: "", notes: "", registry_order: maxRegistryOrder(next, "image") + 1, asset }; next.visual_items.push(item); }
            else if (kind === "video") { item = { id: uid("visual"), kind, role: "reference_video", start: 0, end: snap(span), source_in: 0, source_out: snap(asset.duration_seconds || span), direction: "", notes: "", registry_order: maxRegistryOrder(next, "video") + 1, asset }; next.visual_items.push(item); }
            else { item = { id: uid("audio"), role: "reference_audio", start: 0, end: snap(span), source_in: 0, source_out: snap(asset.duration_seconds || span), direction: "", notes: "", registry_order: maxRegistryOrder(next, "audio") + 1, asset }; next.audio_items.push(item); }
            next.ui.lane_order[kind === "audio" ? "audio" : "visual"].push(item.id);
            next.ui.selected_item_id = item.id;
            commit(instance, next);
        }).catch((error) => { if (!instance.destroyed) setStatus(instance, error?.message || "Drop import failed.", true); });
    });
    if (loadError) setStatus(instance, `Persisted Director state was not overwritten: ${loadError}`, true);
    return instance;
}

function safeWidgetState(widget) {
    try { return JSON.parse(String(widget?.value || "{}")); } catch { return {}; }
}

function hideStateWidget(widget) {
    if (!widget) return;
    widget.type = "jr-hidden-state";
    widget.computeSize = () => [0, -4];
    widget.serializeValue = () => widget.value;
}

function syncFromNode(instance) {
    if (instance.syncing) return;
    instance.activeDragCancel?.();
    const source = instance.node.properties?.[PROP_KEY] || safeWidgetState(instance.stateWidget);
    let normalized;
    try {
        normalized = normalizeState(source);
        const validationError = validateState(normalized);
        if (validationError) throw new Error(validationError);
    }
    catch (error) {
        instance.loadError = error?.message || "Director state could not be loaded.";
        setStatus(instance, `Persisted Director state was not overwritten: ${instance.loadError}`, true);
        return;
    }
    instance.state = normalized;
    instance.loadError = "";
    instance.syncing = true;
    instance.node.properties ||= {};
    instance.node.properties[PROP_KEY] = deepClone(instance.state);
    if (instance.stateWidget) instance.stateWidget.value = JSON.stringify(instance.state);
    instance.syncing = false;
    renderAll(instance);
}

function installStyles() {
    if (document.getElementById("jr-h3-director-desk-styles")) return;
    const style = document.createElement("style"); style.id = "jr-h3-director-desk-styles";
    style.textContent = `
.jr-dd{box-sizing:border-box;width:100%;height:100%;min-height:620px;padding:8px;color:#e7edf7;background:#151a22;font:12px system-ui;overflow:auto;display:flex;flex-direction:column;gap:7px}.jr-dd *{box-sizing:border-box}.jr-dd button,.jr-dd input,.jr-dd select,.jr-dd textarea{font:inherit;color:#e7edf7;background:#242b36;border:1px solid #46536a;border-radius:4px}.jr-dd button{padding:5px 8px;cursor:pointer}.jr-dd button:hover{background:#34435a}.jr-dd button.danger{color:#ffb8b8;border-color:#8d4242}.jr-dd-toolbar{display:flex;align-items:end;gap:6px;flex-wrap:wrap}.jr-dd-toolbar label{width:90px}.jr-dd-toolbar label span,.jr-dd-inspector label span{display:block;color:#aeb9ca;margin-bottom:2px}.jr-dd-toolbar input{width:100%;padding:4px}.jr-dd-global{width:100%;height:68px;resize:vertical;padding:6px}.jr-dd-status{min-height:20px;padding:3px 6px;background:#1c2531;color:#86d5aa;border-left:3px solid #3baf77}.jr-dd-status.error{color:#ffb4b4;border-color:#e05c5c}.jr-dd-main{min-height:0;min-width:928px;flex:1;display:grid;grid-template-columns:minmax(620px,1fr) 300px;gap:8px}.jr-dd-timeline-scroll{min-width:0;overflow:auto;border:1px solid #303a49;background:#10141b}.jr-dd-timeline{position:relative;min-width:100%;padding-top:30px}.jr-dd-ruler{position:absolute;left:82px;right:0;top:0;height:28px;border-bottom:1px solid #384356}.jr-dd-ruler span{position:absolute;bottom:2px;transform:translateX(-50%);color:#8995a8}.jr-dd-ruler span:before{content:"";position:absolute;left:50%;bottom:-6px;height:5px;border-left:1px solid #59677d}.jr-dd-row{display:grid;grid-template-columns:82px 1fr;border-bottom:1px solid #283140;min-height:40px}.jr-dd-row>b{padding:12px 7px;color:#9bacbf;background:#171d26}.jr-dd-track{position:relative;min-width:500px;background:repeating-linear-gradient(90deg,transparent 0,transparent calc(10% - 1px),#242c39 calc(10% - 1px),#242c39 10%)}.jr-dd-item{position:absolute;height:29px;min-width:16px;border:1px solid #6e87ad;border-radius:4px;background:#344866;color:#fff;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;padding:6px 9px;cursor:grab;user-select:none;z-index:1}.jr-dd-item.selected{outline:2px solid #f5c451;z-index:2}.jr-dd-item.reference_image{background:#365b52}.jr-dd-item.reference_video{background:#563f70}.jr-dd-item.reference_audio{background:#3f526d}.jr-dd-item.driving_audio{background:#7a4c34}.jr-dd-item.point{clip-path:polygon(50% 0,100% 50%,50% 100%,0 50%);border-radius:0;padding:0;background:#e2b441;color:transparent}.jr-dd-handle{position:absolute;top:0;bottom:0;width:7px;background:#aec1dd55;cursor:ew-resize}.jr-dd-handle.start{left:0}.jr-dd-handle.end{right:0}.jr-dd-inspector{overflow:auto;border:1px solid #303a49;background:#1a202a;padding:8px}.jr-dd-inspector h4{margin:0 0 7px}.jr-dd-inspector label{display:block;margin:7px 0}.jr-dd-inspector input,.jr-dd-inspector select,.jr-dd-inspector textarea{width:100%;padding:5px}.jr-dd-inspector textarea{height:84px;resize:vertical}.jr-dd-badge{padding:5px;background:#2c3748;border-radius:4px;overflow:hidden;text-overflow:ellipsis}.jr-dd-preview{display:flex;flex-direction:column;gap:4px;margin:7px 0}.jr-dd-preview img,.jr-dd-preview video{width:100%;max-height:170px;object-fit:contain;background:#090b10}.jr-dd-preview audio{width:100%}.jr-dd-actions{display:flex;gap:4px;flex-wrap:wrap;margin-top:8px}.jr-dd-menu{position:fixed;z-index:100000;display:flex;flex-direction:column;min-width:170px;padding:4px;background:#202731;border:1px solid #56647a;box-shadow:0 8px 24px #0008}.jr-dd-menu button{text-align:left;color:#eee;background:transparent;border:0;padding:6px}.jr-dd-menu button:hover{background:#34435a}
`;
    style.textContent += ".jr-dd-item.missing{outline:2px dashed #ef6565;filter:saturate(.45)}";
    document.head.append(style);
}

app.registerExtension({
    name: "JR.MiniMaxH3.DirectorDesk",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        const originalConfigure = nodeType.prototype.onConfigure;
        const originalPropertyChanged = nodeType.prototype.onPropertyChanged;
        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            installStyles();
            const stateWidget = this.widgets?.find((widget) => widget.name === STATE_WIDGET);
            hideStateWidget(stateWidget);
            const instance = buildPanel(this, stateWidget);
            instances.set(this, instance);
            const widget = this.addDOMWidget("jr_h3_director_editor", "director-desk", instance.panel, {
                serialize: false, hideOnZoom: false,
                getMinHeight: () => 620,
                getHeight: () => Math.max(620, Number(this.size?.[1] || 650) - 42),
                afterResize: () => renderAll(instance),
            });
            instance.domWidget = widget;
            this.properties ||= {};
            if (!this.properties[PROP_KEY] && !instance.loadError) this.properties[PROP_KEY] = deepClone(instance.state);
            if (this.size?.[0] < 1000 || this.size?.[1] < 650) this.setSize([Math.max(1000, this.size?.[0] || 0), Math.max(650, this.size?.[1] || 0)]);
            if (!instance.loadError) commit(instance, instance.state, { undo: false });
            return result;
        };
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            const instance = instances.get(this); if (instance) syncFromNode(instance);
            return result;
        };
        nodeType.prototype.onPropertyChanged = function (name, value) {
            const result = originalPropertyChanged?.apply(this, arguments);
            const instance = instances.get(this);
            if (instance && name === PROP_KEY && !instance.syncing) {
                instance.activeDragCancel?.();
                try {
                    const normalized = normalizeState(value);
                    const validationError = validateState(normalized);
                    if (validationError) throw new Error(validationError);
                    instance.syncing = true;
                    instance.state = normalized;
                    this.properties ||= {};
                    this.properties[PROP_KEY] = deepClone(normalized);
                    if (instance.stateWidget) instance.stateWidget.value = JSON.stringify(instance.state);
                    instance.syncing = false;
                    instance.loadError = ""; renderAll(instance);
                } catch (error) {
                    instance.syncing = false;
                    instance.loadError = error?.message || "Director state could not be loaded.";
                    setStatus(instance, `Persisted Director state was not overwritten: ${instance.loadError}`, true);
                }
            }
            return result;
        };
        nodeType.prototype.onRemoved = function () {
            const instance = instances.get(this);
            if (instance) {
                instance.destroyed = true;
                instance.activeDragCancel?.();
                instance.closeContextMenu?.();
                for (const media of instance.mediaElements) releaseMediaElement(instance, media);
                for (const cleanup of instance.cleanup) cleanup();
                instances.delete(this);
            }
            return originalRemoved?.apply(this, arguments);
        };
    },
    loadedGraphNode(node) {
        const instance = instances.get(node); if (instance) syncFromNode(instance);
    },
    afterConfigureGraph() {
        for (const node of app.graph?._nodes || []) {
            const instance = instances.get(node); if (instance) syncFromNode(instance);
        }
    },
});
