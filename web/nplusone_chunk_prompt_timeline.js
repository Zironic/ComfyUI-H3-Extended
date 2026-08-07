import { app } from "/scripts/app.js";

const NODE_NAME = "MiniMaxH3NPlusOneChunkPromptTimelineZi";
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
    const chunkFrames = positiveInt(getWidget(node, "chunk_frames")?.value, 141);
    const targetFrames = outputSeconds * FPS;
    const chunkCount = Math.max(1, Math.ceil(targetFrames / chunkFrames));
    return { outputSeconds, chunkFrames, targetFrames, chunkCount };
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

class NPlusOnePromptEditor {
    constructor(node, container) {
        this.node = node;
        this.container = container;
        this.storage = getWidget(node, STORAGE_WIDGET);
        this.prompts = parseStoredPrompts(this.storage);
        this.render();
        this.bindGeometryWidgets();
    }

    bindGeometryWidgets() {
        for (const name of ["output_seconds", "chunk_frames"]) {
            const widget = getWidget(this.node, name);
            if (!widget || widget.__h3NPlusOneWrapped) continue;
            const original = widget.callback;
            widget.callback = (value, ...args) => {
                const result = original?.call(widget, value, ...args);
                this.render();
                return result;
            };
            widget.__h3NPlusOneWrapped = true;
        }
    }

    persist() {
        if (!this.storage) return;
        this.storage.value = JSON.stringify({ version: 2, prompts: this.prompts });
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
        summary.textContent = `${schedule.chunkCount} chunks · C=${schedule.chunkFrames} · no overlap · N+1 AV continuation · ${schedule.outputSeconds}s`;
        this.container.appendChild(summary);

        const hint = document.createElement("div");
        hint.style.cssText = "margin:0 0 10px;padding:8px;background:#202020;border:1px solid #343434;border-radius:4px;font-size:11px;line-height:1.35;color:#aaa";
        hint.textContent = "Write only what should happen during each generated chunk. Chunk 1 uses static references; chunk 2+ automatically receive the immediately preceding generated clip as dynamic <Video N+1> and <Audio M+1> continuation references.";
        this.container.appendChild(hint);

        for (let index = 0; index < schedule.chunkCount; index += 1) {
            const start = index * schedule.chunkFrames;
            const modelStop = start + schedule.chunkFrames;
            const realStop = Math.min(modelStop, schedule.targetFrames);
            const generatedBeyondOutput = Math.max(0, modelStop - schedule.targetFrames);

            const card = document.createElement("div");
            card.style.cssText = "margin-bottom:10px;padding:8px;background:#222;border:1px solid #3a3a3a;border-radius:4px";

            const label = document.createElement("div");
            label.style.cssText = "margin-bottom:6px;font-size:11px;color:#aaa";
            label.textContent = [
                `Chunk ${index}`,
                `frames ${start}–${Math.max(start, realStop - 1)}`,
                `${formatSeconds(start)}–${formatSeconds(realStop)}`,
                index === 0 ? "static refs only" : "N+1 AV continuation",
                generatedBeyondOutput ? `${generatedBeyondOutput} generated tail frames discarded` : null,
            ].filter(Boolean).join(" · ");
            card.appendChild(label);

            const textarea = document.createElement("textarea");
            textarea.value = this.prompts[index] ?? "";
            textarea.placeholder = index === 0
                ? "Opening chunk action…"
                : "What happens next? Dynamic continuation refs are added automatically…";
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
    name: "H3Extended.NPlusOneChunkPromptTimeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            hideWidget(getWidget(this, STORAGE_WIDGET));

            const container = document.createElement("div");
            this._h3NPlusOnePromptWidget = this.addDOMWidget(
                "h3_nplusone_prompt_editor",
                "H3NPlusOnePromptEditor",
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
                    this._h3NPlusOnePromptEditor = new NPlusOnePromptEditor(this, container);
                } catch (error) {
                    console.error("[H3 Extended] N+1 prompt editor failed:", error);
                }
            }, 0);
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            hideWidget(getWidget(this, STORAGE_WIDGET));
            setTimeout(() => this._h3NPlusOnePromptEditor?.restore(), 10);
            return result;
        };
    },
});
