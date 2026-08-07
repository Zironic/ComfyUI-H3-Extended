import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// Every node that builds a LongFormPreviewPublisher has to be listed here, or
// it publishes preview events that no pane exists to receive.
const NODE_NAMES = new Set([
    "MiniMaxH3LongFormRef2VZi",
    "MiniMaxH3LongFormReferenceVideoZi",
    "MiniMaxH3LongFormAVContinuationZi",
]);
const EVENT_NAME = "h3_longform_preview";
const STATE_KEY = "__h3LongFormPreviewState";

// The state is attached to the node object, never keyed by id. `onNodeCreated`
// fires inside `LiteGraph.createNode`, and `LGraph.configure` only assigns
// `node.id` *after* that call returns — so at widget-build time every node
// still reports the unassigned id -1. Keying a Map there registered all state
// under "-1" and no incoming event could ever match it.
function graphNodes() {
    const graph = app.graph;
    return graph?.nodes || graph?._nodes || [];
}

function findState(detail) {
    const candidates = graphNodes().filter((node) => {
        const name = node.comfyClass || node.type || node.constructor?.comfyClass;
        return NODE_NAMES.has(name) && node[STATE_KEY];
    });

    const raw = detail?.node_id;
    if (raw !== undefined && raw !== null && raw !== "None") {
        const match = candidates.find((node) => String(node.id) === String(raw));
        if (match) return match[STATE_KEY];
    }

    // A single long-form node in the graph is unambiguous; keep the panes alive
    // rather than dropping the event when the id does not line up.
    return candidates.length === 1 ? candidates[0][STATE_KEY] : null;
}

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
    // Two elements, one visible at a time. TAEH3 can decode a whole chunk per
    // update, and only MP4 makes that affordable to encode - Pillow needs ~9 s
    // to palettize a 105-frame GIF where ffmpeg needs ~0.08 s. GIF is still the
    // fallback when ffmpeg cannot be resolved, and an <img> is the only thing
    // that plays one.
    const currentImage = document.createElement("img");
    currentImage.alt = "Current chunk preview";
    currentImage.style.width = "100%";
    currentImage.style.minHeight = "120px";
    currentImage.style.objectFit = "contain";
    currentImage.style.background = "#111";
    currentImage.style.borderRadius = "3px";
    currentImage.style.display = "none";

    const currentVideo = document.createElement("video");
    currentVideo.autoplay = true;
    currentVideo.loop = true;
    currentVideo.muted = true;
    currentVideo.playsInline = true;
    currentVideo.preload = "auto";
    currentVideo.style.width = "100%";
    currentVideo.style.minHeight = "120px";
    currentVideo.style.objectFit = "contain";
    currentVideo.style.background = "#111";
    currentVideo.style.borderRadius = "3px";
    currentPanel.append(currentTitle, currentVideo, currentImage);

    const completedPanel = panel();
    const completedTitle = title("Completed output — waiting");
    const completedVideo = document.createElement("video");
    completedVideo.controls = true;
    // Muted is required for autoplay, and the pane autoplays. Segments now
    // carry an audio track, so the controls' unmute button does something; the
    // choice is remembered across the source swaps a growing stitch causes.
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
        currentVideo,
        completedTitle,
        completedVideo,
        segments: [],
        segmentKeys: new Set(),
        playingIndex: -1,
        stitchedUrl: null,
    };

    completedVideo.addEventListener("ended", () => {
        playSegment(state, state.playingIndex + 1, true);
    });
    completedVideo.addEventListener("error", () => {
        if (state.playingIndex + 1 < state.segments.length) {
            playSegment(state, state.playingIndex + 1, true);
        }
    });

    node[STATE_KEY] = state;
    return state;
}

function resetState(state) {
    state.currentTitle.textContent = "Current chunk — waiting";
    state.currentTitle.removeAttribute("title");
    state.currentImage.removeAttribute("src");
    state.currentVideo.pause();
    state.currentVideo.removeAttribute("src");
    state.currentVideo.load();
    state.completedTitle.textContent = "Completed output — waiting";
    state.completedVideo.pause();
    state.completedVideo.removeAttribute("src");
    state.completedVideo.load();
    state.segments.length = 0;
    state.segmentKeys.clear();
    state.playingIndex = -1;
    state.stitchedUrl = null;
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

const CURRENT_MODES = {
    taeh3: "TAEH3",
    preview_vae: "VAE",
    vae: "VAE",
    latent: "latent approx",
};

function onCurrent(state, detail) {
    const mode = CURRENT_MODES[detail.mode] || "latent approx";
    const rate = detail.preview_fps ? ` @ ${detail.preview_fps} fps` : "";
    const count = detail.frames ? ` — ${detail.frames}f` : "";
    state.currentTitle.textContent =
        `Current chunk ${detail.chunk_index + 1} — step ${detail.step}/${detail.total_steps} — ${mode}${count}${rate}`;
    state.currentTitle.title = detail.fallback_reason || "";

    const url = assetUrl(detail.asset, detail.revision);
    const isVideo = (detail.format || "gif") === "mp4";
    state.currentVideo.style.display = isVideo ? "" : "none";
    state.currentImage.style.display = isVideo ? "none" : "";
    if (isVideo) {
        state.currentImage.removeAttribute("src");
        state.currentVideo.src = url;
        state.currentVideo.load();
        state.currentVideo.play().catch(() => {
            // Autoplay policy can still require a click despite muted.
        });
    } else {
        state.currentVideo.removeAttribute("src");
        state.currentImage.src = url;
    }
}

function onCurrentError(state, detail) {
    const message = detail.message || "unknown preview error";
    state.currentTitle.textContent =
        `Current chunk ${detail.chunk_index + 1} — preview failed`;
    state.currentTitle.title = message;
    state.currentImage.removeAttribute("src");
    state.currentVideo.removeAttribute("src");
    console.warn("[H3 Extended] current chunk preview failed:", message);
}

// The server stitches every finished chunk into one growing MP4, so the pane
// swaps in a longer file each time rather than playing a per-chunk playlist.
// The swap keeps the viewer's position and play state: a reload that jumped
// back to zero on every chunk would be as unwatchable as showing one chunk.
function onStitched(state, detail) {
    const video = state.completedVideo;
    const hadStitch = Boolean(state.stitchedUrl);
    const resumeAt = hadStitch ? video.currentTime || 0 : 0;
    const wasPlaying = hadStitch ? !video.paused || video.ended : true;

    state.stitchedUrl = assetUrl(detail.asset, detail.revision);
    state.segments.length = 0;
    state.segmentKeys.clear();
    state.playingIndex = -1;

    const onMetadata = () => {
        video.removeEventListener("loadedmetadata", onMetadata);
        if (resumeAt > 0 && resumeAt < video.duration) {
            try {
                video.currentTime = resumeAt;
            } catch (error) {
                // Seeking before the container is seekable is not worth failing.
            }
        }
        if (wasPlaying) {
            video.play().catch(() => {
                // Browser autoplay policy can still require a click.
            });
        }
    };
    video.addEventListener("loadedmetadata", onMetadata);
    video.src = state.stitchedUrl;
    video.load();

    const sound = detail.has_audio
        ? video.muted
            ? " — sound (unmute)"
            : " — sound"
        : "";
    state.completedTitle.textContent =
        `Completed output — ${detail.chunk_index + 1} chunks, ${detail.completed_frames} frames${sound}`;
    state.completedTitle.title = detail.audio_error || "";
}

function onCompleted(state, detail) {
    if (detail.stitched) {
        onStitched(state, detail);
        return;
    }
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
    const state = findState(detail);
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
            delete this[STATE_KEY];
            return originalRemoved?.apply(this, arguments);
        };
    },
});
