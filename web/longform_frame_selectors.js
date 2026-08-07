import { app } from "/scripts/app.js";

const NODE_NAMES = new Set([
    "MiniMaxH3LongFormRef2VZi",
    "MiniMaxH3LongFormReferenceVideoZi",
    "MiniMaxH3ChunkPromptTimelineZi",
]);

// H3 video generation lengths are 17k+5. The long-form nodes deliberately
// start at 22 frames because five frames is not a useful production chunk.
const CHUNK_FRAMES = Array.from(
    { length: Math.floor((362 - 22) / 17) + 1 },
    (_, index) => 22 + index * 17,
);

// Video is 24 fps and H3 audio latents are 40 Hz. Their time boundaries meet
// every three video frames (five audio latents), so an exact A/V chunk length
// must also be divisible by three.
const AUDIO_ALIGNED_CHUNK_FRAMES = CHUNK_FRAMES.filter(
    (frames) => frames % 3 === 0,
);

// A carried overlap must begin on a video-latent boundary. For a legal H3
// chunk, the possible suffix lengths repeat with residues 0, 4, 5, 9 and 13
// modulo 17. O=4 is the smallest possible suffix and carries one latent.
const OVERLAP_RESIDUES = new Set([0, 4, 5, 9, 13]);
const OVERLAP_FRAMES = Array.from(
    { length: 180 - 4 + 1 },
    (_, index) => index + 4,
).filter((frames) => OVERLAP_RESIDUES.has(frames % 17));
const AUDIO_ALIGNED_OVERLAP_FRAMES = OVERLAP_FRAMES.filter(
    (frames) => frames % 3 === 0,
);

function nearestAllowed(value, allowed, preferLargerOnTie = false) {
    if (!allowed.length) return undefined;
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return allowed[0];

    return allowed.reduce((best, candidate) => {
        const candidateDistance = Math.abs(candidate - numeric);
        const bestDistance = Math.abs(best - numeric);
        if (candidateDistance < bestDistance) return candidate;
        if (candidateDistance === bestDistance) {
            return preferLargerOnTie
                ? Math.max(best, candidate)
                : Math.min(best, candidate);
        }
        return best;
    }, allowed[0]);
}

function makeComboWidget(widget, values) {
    if (!widget) return;

    // Change only the visible widget. Do not rewrite nodeData.input.required:
    // doing that changes the socket type from INT to COMBO and makes the
    // timeline node's INT passthrough outputs impossible to connect.
    widget.type = "combo";

    // Mutate the existing options object instead of replacing it. Nodes 2.0
    // keeps a second reference to this exact object in the widget value store
    // (BaseWidget copies the reference once, at construction) and the Vue
    // select reads its option list from that copy. Assigning a fresh object
    // here leaves the store holding the original INT options, which have no
    // `values` — that is what renders as "No Results Found".
    const options = widget.options || (widget.options = {});
    // The select compares option values to the widget value with ===, and it
    // stringifies every option, so keep both sides strings. execution.py
    // coerces INT widget values with int() before the node runs.
    options.values = values.map((value) => String(value));
    // min/max/step/step2 stay. The select ignores them, and useIntWidget's
    // onValueChange callback still runs underneath: it snaps to
    // round((v - min % step2) / step2) * step2 + min % step2. Without min, the
    // 22-frame offset is lost and every legal 17k+5 pick collapses onto a bare
    // multiple of 17 (141 -> 136).
}

function setWidgetValue(node, widget, value) {
    if (!widget || value === undefined) return;

    const next = String(value);
    if (widget.value === next) return;

    const previous = widget.value;
    widget.value = next;
    widget.callback?.(next);
    // onValueChange writes a rounded number back over the value; restore the
    // string so the select recognises it as one of its options.
    widget.value = next;
    node.onWidgetChanged?.(
        widget.name,
        next,
        previous,
        widget,
    );
}

function audioAlignmentEnabled(node) {
    const widget = node.widgets?.find(
        (candidate) => candidate.name === "align_audio_chunks",
    );
    // New schemas default this to true. Treat a missing widget as false so old
    // node definitions still retain their previous selector behavior.
    return widget ? Boolean(widget.value) : false;
}

function activeChunkFrames(node) {
    return audioAlignmentEnabled(node)
        ? AUDIO_ALIGNED_CHUNK_FRAMES
        : CHUNK_FRAMES;
}

function activeOverlapFrames(node) {
    return audioAlignmentEnabled(node)
        ? AUDIO_ALIGNED_OVERLAP_FRAMES
        : OVERLAP_FRAMES;
}

function syncFrameChoices(node) {
    const chunkWidget = node.widgets?.find((widget) => widget.name === "chunk_frames");
    const overlapWidget = node.widgets?.find((widget) => widget.name === "overlap_frames");
    if (!chunkWidget || !overlapWidget) return;

    const aligned = audioAlignmentEnabled(node);
    const chunks = activeChunkFrames(node);
    makeComboWidget(chunkWidget, chunks);
    setWidgetValue(
        node,
        chunkWidget,
        chunks.includes(Number(chunkWidget.value))
            ? Number(chunkWidget.value)
            : nearestAllowed(chunkWidget.value, chunks, aligned),
    );

    const overlaps = activeOverlapFrames(node).filter(
        (frames) => frames < Number(chunkWidget.value),
    );
    makeComboWidget(overlapWidget, overlaps);

    // Always run the value back through setWidgetValue, even when it is already
    // legal: a widget loaded as the number 4 has to become the string "4" or the
    // select treats it as an unknown value and flags the widget as invalid.
    setWidgetValue(
        node,
        overlapWidget,
        overlaps.includes(Number(overlapWidget.value))
            ? Number(overlapWidget.value)
            : nearestAllowed(overlapWidget.value, overlaps, aligned),
    );
}

function installLegalSelectors(node) {
    const chunkWidget = node.widgets?.find((widget) => widget.name === "chunk_frames");
    const overlapWidget = node.widgets?.find((widget) => widget.name === "overlap_frames");
    const alignWidget = node.widgets?.find((widget) => widget.name === "align_audio_chunks");
    if (!chunkWidget || !overlapWidget) return;

    if (!chunkWidget.__h3LegalFramesWrapped) {
        const originalCallback = chunkWidget.callback;
        chunkWidget.callback = function (value, ...args) {
            const aligned = audioAlignmentEnabled(node);
            const allowed = activeChunkFrames(node);
            const legal = nearestAllowed(value, allowed, aligned);
            const next = legal === undefined ? value : String(legal);
            this.value = next;
            const result = originalCallback?.call(this, next, ...args);
            this.value = next;
            syncFrameChoices(node);
            return result;
        };
        chunkWidget.__h3LegalFramesWrapped = true;
    }

    if (!overlapWidget.__h3LegalFramesWrapped) {
        const originalCallback = overlapWidget.callback;
        overlapWidget.callback = function (value, ...args) {
            const aligned = audioAlignmentEnabled(node);
            const allowed = activeOverlapFrames(node).filter(
                (frames) => frames < Number(chunkWidget.value),
            );
            const legal = nearestAllowed(value, allowed, aligned);
            const next = legal === undefined ? value : String(legal);
            this.value = next;
            const result = originalCallback?.call(this, next, ...args);
            this.value = next;
            return result;
        };
        overlapWidget.__h3LegalFramesWrapped = true;
    }

    if (alignWidget && !alignWidget.__h3AudioAlignmentWrapped) {
        const originalCallback = alignWidget.callback;
        alignWidget.callback = function (value, ...args) {
            this.value = value;
            const result = originalCallback?.call(this, value, ...args);
            this.value = value;
            syncFrameChoices(node);
            return result;
        };
        alignWidget.__h3AudioAlignmentWrapped = true;
    }

    syncFrameChoices(node);
}

app.registerExtension({
    name: "H3Extended.LongFormLegalFrameSelectors",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name)) return;

        // Keep the backend schema as INT so primitive outputs remain compatible.
        // Only the instantiated widgets are converted to legal-value dropdowns.
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            installLegalSelectors(this);
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            installLegalSelectors(this);
            return result;
        };
    },
});
