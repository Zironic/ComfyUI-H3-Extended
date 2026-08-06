import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const EVENT_NAME = "h3_longform_preview";
const NODE_NAMES = new Set([
    "MiniMaxH3LongFormRef2VZi",
    "MiniMaxH3LongFormReferenceVideoZi",
]);

function assetUrl(asset, revision) {
    const params = new URLSearchParams({
        filename: asset.filename,
        subfolder: asset.subfolder || "",
        type: asset.type || "temp",
        t: String(revision || Date.now()),
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function findNode(detail) {
    const raw = detail?.node_id;
    const numeric = Number(raw);
    let node = app.graph?.getNodeById?.(raw);
    if (!node && Number.isFinite(numeric)) {
        node = app.graph?.getNodeById?.(numeric);
    }
    if (node) return node;

    // Hidden unique_id behavior differs across Comfy builds.  A single active
    // long-form preview node is an unambiguous fallback and prevents a server
    // event with a missing/stringified id from disappearing silently.
    const candidates = (app.graph?._nodes || []).filter((candidate) => {
        const name = candidate.comfyClass || candidate.type || candidate.constructor?.comfyClass;
        return NODE_NAMES.has(name);
    });
    return candidates.length === 1 ? candidates[0] : null;
}

function completedElements(detail) {
    const node = findNode(detail);
    const widget = node?.widgets?.find(
        (candidate) => candidate.name === "longform_live_previews",
    );
    const root = widget?.element || widget?.inputEl;
    const panel = root?.children?.[1];
    if (!panel) return null;

    const title = panel.children?.[0] || panel.querySelector("div");
    const video = panel.querySelector("video");
    let image = panel.querySelector("img.h3-completed-fallback");
    if (!image) {
        image = document.createElement("img");
        image.className = "h3-completed-fallback";
        image.alt = "Completed chunk preview fallback";
        image.style.display = "none";
        image.style.width = "100%";
        image.style.minHeight = "120px";
        image.style.objectFit = "contain";
        image.style.background = "#111";
        image.style.borderRadius = "3px";
        panel.append(image);
    }
    return { title, video, image };
}

function showFallback(detail) {
    const elements = completedElements(detail);
    if (!elements || !detail.asset) return;
    const { title, video, image } = elements;
    if (video) {
        video.pause();
        video.style.display = "none";
    }
    image.style.display = "block";
    image.src = assetUrl(detail.asset, detail.revision);
    title.textContent =
        `Completed output — ${detail.chunk_index + 1} chunks, ${detail.completed_frames} frames — GIF fallback`;
    title.title = detail.fallback_reason || "MP4 preview failed; showing GIF fallback";
}

function showError(detail) {
    const elements = completedElements(detail);
    if (!elements) return;
    const message = detail.message || "unknown completed-preview error";
    elements.title.textContent =
        `Completed output — chunk ${detail.chunk_index + 1} preview failed`;
    elements.title.title = message;
    console.warn("[H3 Extended] completed chunk preview failed:", message);
}

function restoreVideo(detail) {
    const elements = completedElements(detail);
    if (!elements) return;
    elements.image.style.display = "none";
    elements.image.removeAttribute("src");
    if (elements.video) elements.video.style.display = "block";
    elements.title.removeAttribute("title");
}

function reset(detail) {
    const elements = completedElements(detail);
    if (!elements) return;
    elements.image.style.display = "none";
    elements.image.removeAttribute("src");
    if (elements.video) elements.video.style.display = "block";
    elements.title.removeAttribute("title");
}

api.addEventListener(EVENT_NAME, (event) => {
    const detail = event.detail || {};
    if (detail.kind === "completed_chunk_fallback") {
        showFallback(detail);
    } else if (detail.kind === "completed_chunk_error") {
        showError(detail);
    } else if (detail.kind === "completed_chunk") {
        restoreVideo(detail);
    } else if (detail.kind === "reset") {
        reset(detail);
    }
});
