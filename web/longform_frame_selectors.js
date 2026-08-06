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

// A carried overlap must begin on a video-latent boundary. For a legal H3
// chunk, the possible suffix lengths repeat with residues 0, 4, 5, 9 and 13
// modulo 17. O=4 is the smallest possible suffix and carries one latent.
const OVERLAP_RESIDUES = new Set([0, 4, 5, 9, 13]);
const OVERLAP_FRAMES = Array.from(
    { length: 180 - 4 + 1 },
    (_, index) => index + 4,
).filter((frames) => OVERLAP_RESIDUES.has(frames % 17));

function nearestAllowed(value, allowed) {
    if (!allowed.length) return undefined;
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return allowed[0];

    return allowed.reduce((best, candidate) => {
        const candidateDistance = Math.abs(candidate - numeric);
        const bestDistance = Math.abs(best - numeric);
        if (candidateDistance < bestDistance) return candidate;
        if (candidateDistance === bestDistance && candidate < best) return candidate;
        return best;
    }, allowed[0]);
}

function makeComboWidget(widget, values) {
    if (!widget) return;

    // Change only the visible widget. Do not rewrite nodeData.input.required:
    // doing that changes the socket type from INT to COMBO and makes the
    // timeline node's INT passthrough outputs impossible to connect.
    widget.type = "combo";
    widget.options = {
        ...(widget.options || {}),
        values,
    };
    delete widget.options.min;
    delete widget.options.max;
    delete widget.options.step;
}

function setWidgetValue(node, widget, value) {
    if (!widget || value === undefined || Number(widget.value) === Number(value)) {
        return;
    }

    const previous = widget.value;
    widget.value = value;
    widget.callback?.(value);
    node.onWidgetChanged?.(
        widget.name,
        value,
        previous,
        widget,
    );
}

function syncOverlapChoices(node) {
    const chunkWidget = node.widgets?.find((widget) => widget.name === "chunk_frames");
    const overlapWidget = node.widgets?.find((widget) => widget.name === "overlap_frames");
    if (!chunkWidget || !overlapWidget) return;

    const allowed = OVERLAP_FRAMES.filter(
        (frames) => frames < Number(chunkWidget.value),
    );
    makeComboWidget(overlapWidget, allowed);

    if (!allowed.includes(Number(overlapWidget.value))) {
        setWidgetValue(
            node,
            overlapWidget,
            nearestAllowed(overlapWidget.value, allowed),
        );
    }
}

function installLegalSelectors(node) {
    const chunkWidget = node.widgets?.find((widget) => widget.name === "chunk_frames");
    const overlapWidget = node.widgets?.find((widget) => widget.name === "overlap_frames");
    if (!chunkWidget || !overlapWidget) return;

    makeComboWidget(chunkWidget, CHUNK_FRAMES);
    if (!CHUNK_FRAMES.includes(Number(chunkWidget.value))) {
        setWidgetValue(
            node,
            chunkWidget,
            nearestAllowed(chunkWidget.value, CHUNK_FRAMES),
        );
    }

    if (!chunkWidget.__h3LegalFramesWrapped) {
        const originalCallback = chunkWidget.callback;
        chunkWidget.callback = function (value, ...args) {
            const legal = nearestAllowed(value, CHUNK_FRAMES);
            if (legal !== undefined) this.value = legal;
            const result = originalCallback?.call(this, legal, ...args);
            syncOverlapChoices(node);
            return result;
        };
        chunkWidget.__h3LegalFramesWrapped = true;
    }

    if (!overlapWidget.__h3LegalFramesWrapped) {
        const originalCallback = overlapWidget.callback;
        overlapWidget.callback = function (value, ...args) {
            const allowed = OVERLAP_FRAMES.filter(
                (frames) => frames < Number(chunkWidget.value),
            );
            const legal = nearestAllowed(value, allowed);
            if (legal !== undefined) this.value = legal;
            return originalCallback?.call(this, legal, ...args);
        };
        overlapWidget.__h3LegalFramesWrapped = true;
    }

    syncOverlapChoices(node);
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
