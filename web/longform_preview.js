import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_NAMES = new Set([
    "MiniMaxH3LongFormRef2VZi",
    "MiniMaxH3LongFormReferenceVideoZi",
]);
const EVENT_NAME = "h3_longform_preview";
const states = new Map();

function assetUrl(asset, revision) {
    const params = new URLSearchParams({
        filename: asset.filename,
        subfolder: asset.subfolder || "",
        type: asset.type || "temp",
        t: String(revision || Date.now()),
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function title(text) {
    const element = document.createElement("div");
    element.textContent = text;
    element.style.font = "600 12px sans-serif";
    element.style.margin = "2px 0 5px";
    element.style.opacity = "0.9";
    return element;
}

function panel() {
    const element = document.createElement("div");
    element.style.display = "flex";
    element.style.flexDirection = "column";
    element.style.gap = "5px";
    element.style.padding = "6px";
    element.style.border = "1px solid rgba(255,255,255,0.12)";
    element.style.borderRadius = "5px";
    element.style.background = "rgba(0,0,0,0.18)";
    return element;
}

function buildPreviewWidget(node) {
    const root = document.createElement("div");
    root.style.display = "grid";
    root.style.gridTemplateColumns = "1fr 1fr";
    root.style.gap = "8px";
    root.style.width = "100%";
    root.style.boxSizing = "border-box";

    const currentPanel = panel();
    const currentTitle = title("Current chunk — waiting");
    const currentImage = document.createElement("img");
    currentImage.alt = "Current chunk preview";
    currentImage.style.width = "100%";
    currentImage.style.minHeight = "120px";
    currentImage.style.objectFit = "contain";
    currentImage.style.background = "#111";
    currentImage.style.borderRadius = "3px";
    currentPanel.append(currentTitle, currentImage);

    const completedPanel = panel();
    const completedTitle = title("Completed output — waiting");
    const completedVideo = document.createElement("video");
    completedVideo.controls = true;
    completedVideo.muted = true;
    completedVideo.playsInline = true;
    completedVideo.preload = "auto";
    completedVideo.style.width = "100%";
    completedVideo.style.minHeight = "120px";
    completedVideo.style.background = "#111";
    completedVideo.style.borderRadius = "3px";
    completedPanel.append(completedTitle, completedVideo);

    root.append(currentPanel, completedPanel);
    node.addDOMWidget("longform_live_previews", "div", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 260,
    });

    const state = {
        node,
        currentTitle,
        currentImage,
        completedTitle,
        completedVideo,
        segments: [],
        segmentKeys: new Set(),
        playingIndex: -1,
    };

    completedVideo.addEventListener("ended", () => {
        playSegment(state, state.playingIndex + 1, true);
    });
    completedVideo.addEventListener("error", () => {
        if (state.playingIndex + 1 < state.segments.length) {
            playSegment(state, state.playingIndex + 1, true);
        }
    });

    states.set(String(node.id), state);
    return state;
}

function resetState(state) {
    state.currentTitle.textContent = "Current chunk — waiting";
    state.currentTitle.removeAttribute("title");
    state.currentImage.removeAttribute("src");
    state.completedTitle.textContent = "Completed output — waiting";
    state.completedVideo.pause();
    state.completedVideo.removeAttribute("src");
    state.completedVideo.load();
    state.segments.length = 0;
    state.segmentKeys.clear();
    state.playingIndex = -1;
}

function playSegment(state, index, autoplay) {
    if (index < 0 || index >= state.segments.length) return;
    state.playingIndex = index;
    const segment = state.segments[index];
    state.completedVideo.src = segment.url;
    state.completedVideo.load();
    if (autoplay) {
        state.completedVideo.play().catch(() => {
            // Browser autoplay policy can still require a click despite muted.
        });
    }
}

function onCurrent(state, detail) {
    const mode = detail.mode === "vae" ? "VAE" : "latent approx";
    state.currentTitle.textContent =
        `Current chunk ${detail.chunk_index + 1} — step ${detail.step}/${detail.total_steps} — ${mode}`;
    state.currentTitle.title = detail.fallback_reason || "";
    state.currentImage.src = assetUrl(detail.asset, detail.revision);
}

function onCurrentError(state, detail) {
    const message = detail.message || "unknown preview error";
    state.currentTitle.textContent =
        `Current chunk ${detail.chunk_index + 1} — preview failed`;
    state.currentTitle.title = message;
    state.currentImage.removeAttribute("src");
    console.warn("[H3 Extended] current chunk preview failed:", message);
}

function onCompleted(state, detail) {
    const key = `${detail.chunk_index}:${detail.asset.filename}`;
    if (!state.segmentKeys.has(key)) {
        state.segmentKeys.add(key);
        state.segments.push({
            chunkIndex: detail.chunk_index,
            url: assetUrl(detail.asset, detail.revision),
        });
        state.segments.sort((a, b) => a.chunkIndex - b.chunkIndex);
    }
    state.completedTitle.textContent =
        `Completed output — ${state.segments.length} chunks, ${detail.completed_frames} frames`;
    if (state.playingIndex < 0 || state.completedVideo.ended) {
        playSegment(state, 0, true);
    }
}

api.addEventListener(EVENT_NAME, (event) => {
    const detail = event.detail || {};
    const state = states.get(String(detail.node_id));
    if (!state) return;
    if (detail.kind === "reset") {
        resetState(state);
        return;
    }
    if (detail.kind === "current_chunk_error") {
        onCurrentError(state, detail);
        return;
    }
    if (!detail.asset) return;
    if (detail.kind === "current_chunk") onCurrent(state, detail);
    if (detail.kind === "completed_chunk") onCompleted(state, detail);
});

app.registerExtension({
    name: "H3Extended.LongFormDualPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name)) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            buildPreviewWidget(this);
            const width = Math.max(this.size?.[0] || 360, 560);
            const height = Math.max(this.size?.[1] || 300, 560);
            this.setSize([width, height]);
            return result;
        };
        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            states.delete(String(this.id));
            return originalRemoved?.apply(this, arguments);
        };
    },
});
