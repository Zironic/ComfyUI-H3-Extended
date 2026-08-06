import { app } from "/scripts/app.js";

const NODE_NAME = "MiniMaxH3ChunkPromptTimelineZi";
const STORAGE_WIDGET = "chunk_prompts_json";
const FPS = 24;

function hideWidget(widget) {
    if (!widget) return;
    widget.type = "hidden";
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
}

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function positiveInt(value, fallback) {
    const numeric = Math.trunc(Number(value));
    return Number.isFinite(numeric) && numeric > 0 ? numeric : fallback;
}

function calculateSchedule(node) {
    const outputSeconds = positiveInt(getWidget(node, "output_seconds")?.value, 30);
    const chunkFrames = positiveInt(getWidget(node, "chunk_frames")?.value, 90);
    const overlapFrames = positiveInt(getWidget(node, "overlap_frames")?.value, 4);
    const strideFrames = Math.max(1, chunkFrames - overlapFrames);
    const targetFrames = outputSeconds * FPS;
    const chunkCount = targetFrames <= chunkFrames
        ? 1
        : 1 + Math.ceil((targetFrames - chunkFrames) / strideFrames);
    return {
        outputSeconds,
        chunkFrames,
        overlapFrames,
        strideFrames,
        targetFrames,
        chunkCount,
    };
}

function parseStoredPrompts(widget) {
    const text = widget?.value;
    if (!text) return [];
    try {
        const parsed = JSON.parse(text);
        const values = Array.isArray(parsed) ? parsed : parsed?.prompts;
        return Array.isArray(values) ? values.map((value) => String(value ?? "")) : [];
    } catch (_) {
        return [];
    }
}

function formatSeconds(frames) {
    return `${(frames / FPS).toFixed(3)}s`;
}

class ChunkPromptEditor {
    constructor(node, container) {
        this.node = node;
        this.container = container;
        this.storage = getWidget(node, STORAGE_WIDGET);
        this.prompts = parseStoredPrompts(this.storage);
        this.boundWidgets = [];
        this.render();
        this.bindGeometryWidgets();
    }

    bindGeometryWidgets() {
        for (const name of ["output_seconds", "chunk_frames", "overlap_frames"]) {
            const widget = getWidget(this.node, name);
            if (!widget || widget.__h3ChunkTimelineWrapped) continue;
            const original = widget.callback;
            widget.callback = (value, ...args) => {
                const result = original?.call(widget, value, ...args);
                this.render();
                return result;
            };
            widget.__h3ChunkTimelineWrapped = true;
            this.boundWidgets.push(widget);
        }
    }

    persist() {
        if (!this.storage) return;
        this.storage.value = JSON.stringify({ version: 1, prompts: this.prompts });
        this.node.graph?.setDirtyCanvas?.(true, true);
    }

    render() {
        const schedule = calculateSchedule(this.node);
        while (this.prompts.length < schedule.chunkCount) this.prompts.push("");
        this.persist();

        this.container.replaceChildren();
        this.container.style.cssText = [
            "height:520px",
            "overflow:auto",
            "box-sizing:border-box",
            "padding:8px",
            "background:#171717",
            "border:1px solid #333",
            "border-radius:4px",
            "font-family:sans-serif",
            "color:#ddd",
        ].join(";");

        const summary = document.createElement("div");
        summary.style.cssText = "position:sticky;top:0;z-index:2;padding:6px 4px 10px;background:#171717;font-size:12px;color:#bbb";
        summary.textContent = `${schedule.chunkCount} chunks · C=${schedule.chunkFrames} · O=${schedule.overlapFrames} · S=${schedule.strideFrames} · ${schedule.outputSeconds}s`;
        this.container.appendChild(summary);

        for (let index = 0; index < schedule.chunkCount; index += 1) {
            const start = index * schedule.strideFrames;
            const modelStop = start + schedule.chunkFrames;
            const realStop = Math.min(modelStop, schedule.targetFrames);
            const padded = Math.max(0, modelStop - schedule.targetFrames);

            const card = document.createElement("div");
            card.style.cssText = "margin-bottom:10px;padding:8px;background:#222;border:1px solid #3a3a3a;border-radius:4px";

            const label = document.createElement("div");
            label.style.cssText = "margin-bottom:6px;font-size:11px;color:#aaa";
            label.textContent = [
                `Chunk ${index}`,
                `frames ${start}–${Math.max(start, realStop - 1)}`,
                `${formatSeconds(start)}–${formatSeconds(realStop)}`,
                padded ? `${padded} padded model frames` : null,
            ].filter(Boolean).join(" · ");
            card.appendChild(label);

            const textarea = document.createElement("textarea");
            textarea.value = this.prompts[index] ?? "";
            textarea.placeholder = "Prompt instruction for this chunk…";
            textarea.style.cssText = [
                "display:block",
                "width:100%",
                "min-height:84px",
                "box-sizing:border-box",
                "resize:vertical",
                "padding:7px",
                "background:#111",
                "color:#eee",
                "border:1px solid #444",
                "border-radius:3px",
                "font:12px sans-serif",
            ].join(";");
            textarea.addEventListener("input", () => {
                this.prompts[index] = textarea.value;
                this.persist();
            });
            textarea.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });
            textarea.addEventListener("pointerdown", (event) => event.stopPropagation());
            card.appendChild(textarea);
            this.container.appendChild(card);
        }
    }

    restore() {
        this.prompts = parseStoredPrompts(this.storage);
        this.render();
    }
}

app.registerExtension({
    name: "H3Extended.ChunkPromptTimeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            hideWidget(getWidget(this, STORAGE_WIDGET));

            const container = document.createElement("div");
            this._h3ChunkPromptWidget = this.addDOMWidget(
                "h3_chunk_prompt_editor",
                "H3ChunkPromptEditor",
                container,
                {
                    serialize: false,
                    hideOnZoom: false,
                    getMinHeight: () => 520,
                    getHeight: () => 520,
                },
            );

            setTimeout(() => {
                try {
                    this._h3ChunkPromptEditor = new ChunkPromptEditor(this, container);
                } catch (error) {
                    console.error("[H3 Extended] chunk prompt editor failed:", error);
                }
            }, 0);
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            hideWidget(getWidget(this, STORAGE_WIDGET));
            setTimeout(() => this._h3ChunkPromptEditor?.restore(), 10);
            return result;
        };
    },
});
