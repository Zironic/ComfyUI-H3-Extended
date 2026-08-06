import { app } from "/scripts/app.js";

const NODE_NAMES = new Set([
    "MiniMaxH3LongFormRef2VZi",
    "MiniMaxH3LongFormReferenceVideoZi",
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

function replaceInputWithCombo(nodeData, name, values) {
    const required = nodeData?.input?.required;
    const current = required?.[name];
    if (!current) return;

    const options = { ...(current[1] || {}) };
    delete options.min;
    delete options.max;
    delete options.step;
    required[name] = [values, options];
}

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

function syncOverlapChoices(node) {
    const chunkWidget = node.widgets?.find((widget) => widget.name === "chunk_frames");
    const overlapWidget = node.widgets?.find((widget) => widget.name === "overlap_frames");
    if (!chunkWidget || !overlapWidget) return;

    const allowed = OVERLAP_FRAMES.filter(
        (frames) => frames < Number(chunkWidget.value),
    );
    overlapWidget.options = {
        ...(overlapWidget.options || {}),
        values: allowed,
    };

    const current = Number(overlapWidget.value);
    if (allowed.includes(current)) return;

    const previous = overlapWidget.value;
    const replacement = nearestAllowed(current, allowed);
    if (replacement === undefined) return;

    overlapWidget.value = replacement;
    overlapWidget.callback?.(replacement);
    node.onWidgetChanged?.(
        overlapWidget.name,
        replacement,
        previous,
        overlapWidget,
    );
}

function installDynamicOverlap(node) {
    const chunkWidget = node.widgets?.find((widget) => widget.name === "chunk_frames");
    if (!chunkWidget) return;

    if (!chunkWidget.__h3LegalFramesWrapped) {
        const originalCallback = chunkWidget.callback;
        chunkWidget.callback = function (value, ...args) {
            const result = originalCallback?.call(this, value, ...args);
            syncOverlapChoices(node);
            return result;
        };
        chunkWidget.__h3LegalFramesWrapped = true;
    }

    syncOverlapChoices(node);
}

app.registerExtension({
    name: "H3Extended.LongFormLegalFrameSelectors",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name)) return;

        replaceInputWithCombo(nodeData, "chunk_frames", CHUNK_FRAMES);
        replaceInputWithCombo(nodeData, "overlap_frames", OVERLAP_FRAMES);

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            installDynamicOverlap(this);
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            installDynamicOverlap(this);
            return result;
        };
    },
});
