import { app } from "/scripts/app.js";

const NODE_NAME = "MiniMaxH3NPlusOneChunkPromptTimelineZi";
const STORAGE_WIDGET = "chunk_prompts_json";
const FPS = 24;
const STORE_VERSION = 2;

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
    const videoReferenceFrames = positiveInt(
        getWidget(node, "video_reference_frames")?.value,
        90,
    );
    const audioReferenceSeconds = Math.max(
        0.025,
        Number(getWidget(node, "audio_reference_seconds")?.value ?? 4.0),
    );
    const audioReferenceLatents = Math.min(
        Math.round(audioReferenceSeconds * 40),
        Math.round((chunkFrames / FPS) * 40),
    );
    const targetFrames = outputSeconds * FPS;
    const chunkCount = Math.max(1, Math.ceil(targetFrames / chunkFrames));
    return {
        outputSeconds,
        chunkFrames,
        videoReferenceFrames,
        audioReferenceSeconds,
        audioReferenceLatents,
        targetFrames,
        chunkCount,
    };
}

function parsePromptStore(text) {
    if (text === undefined || text === null || String(text).trim() === "") {
        return { ok: true, prompts: [] };
    }
    let parsed;
    try {
        parsed = JSON.parse(text);
    } catch (error) {
        return { ok: false, error: error.message };
    }
    const values = Array.isArray(parsed) ? parsed : parsed?.prompts;
    if (!Array.isArray(values)) {
        return { ok: false, error: 'expected an array, or an object with a "prompts" array' };
    }
    return { ok: true, prompts: values.map((value) => String(value ?? "")) };
}

function serializePrompts(prompts) {
    return JSON.stringify({ version: STORE_VERSION, prompts }, null, 2);
}

function samePrompts(left, right) {
    return left.length === right.length && left.every((value, index) => value === right[index]);
}

function formatSeconds(frames) {
    return `${(frames / FPS).toFixed(3)}s`;
}

/**
 * Make every write to `widget.value` observable, including programmatic ones
 * (workflow load, undo/redo, other extensions) that never fire the callback.
 */
function watchWidgetValue(widget, notify) {
    if (!widget) return;
    widget.__h3NPlusOneNotify = notify;
    if (widget.__h3NPlusOneValueWatched) return;

    let descriptor = null;
    for (let target = widget; target && !descriptor; target = Object.getPrototypeOf(target)) {
        descriptor = Object.getOwnPropertyDescriptor(target, "value");
    }

    if (descriptor?.get && descriptor?.set) {
        const { get, set } = descriptor;
        Object.defineProperty(widget, "value", {
            configurable: true,
            enumerable: descriptor.enumerable ?? true,
            get() {
                return get.call(this);
            },
            set(next) {
                const previous = get.call(this);
                set.call(this, next);
                if (previous !== next) this.__h3NPlusOneNotify?.(next);
            },
        });
    } else {
        let stored = widget.value;
        Object.defineProperty(widget, "value", {
            configurable: true,
            enumerable: true,
            get() {
                return stored;
            },
            set(next) {
                const previous = stored;
                stored = next;
                if (previous !== next) this.__h3NPlusOneNotify?.(next);
            },
        });
    }
    widget.__h3NPlusOneValueWatched = true;
}

class NPlusOnePromptEditor {
    constructor(node, container) {
        this.node = node;
        this.container = container;
        this.storage = getWidget(node, STORAGE_WIDGET);
        this.suspend = false;

        const parsed = parsePromptStore(this.storage?.value);
        this.prompts = parsed.ok ? parsed.prompts : [];

        this.buildShell();
        this.render();
        this.bindGeometryWidgets();
        this.watchStorage();
        if (!parsed.ok) this.setStatus(`invalid JSON — ${parsed.error}`, true);
    }

    // ---------------------------------------------------------------- shell

    buildShell() {
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

        this.summaryEl = document.createElement("div");
        this.summaryEl.style.cssText =
            "position:sticky;top:0;z-index:2;padding:6px 4px 10px;background:#171717;font-size:12px;color:#bbb";
        this.container.appendChild(this.summaryEl);

        const hint = document.createElement("div");
        hint.style.cssText =
            "margin:0 0 10px;padding:8px;background:#202020;border:1px solid #343434;border-radius:4px;font-size:11px;line-height:1.35;color:#aaa";
        hint.textContent =
            "Write only what should happen during each generated chunk. Chunk 1 uses static references; chunk 2+ automatically receive the immediately preceding generated clip as dynamic <Video N+1> and <Audio M+1> continuation references.";
        this.container.appendChild(hint);

        this.cardsHost = document.createElement("div");
        this.container.appendChild(this.cardsHost);

        this.container.appendChild(this.buildJsonPanel());
    }

    buildJsonPanel() {
        const panel = document.createElement("details");
        panel.style.cssText =
            "margin-top:4px;background:#1c1c1c;border:1px solid #343434;border-radius:4px";

        const title = document.createElement("summary");
        title.textContent = "Raw JSON (editable — edits flow back into the chunks above)";
        title.style.cssText =
            "padding:7px 8px;cursor:pointer;font-size:11px;color:#aaa;user-select:none";
        panel.appendChild(title);

        const body = document.createElement("div");
        body.style.cssText = "padding:0 8px 8px";

        this.jsonInput = document.createElement("textarea");
        this.jsonInput.spellcheck = false;
        this.jsonInput.value = serializePrompts(this.prompts);
        this.jsonInput.style.cssText = [
            "display:block",
            "width:100%",
            "min-height:140px",
            "box-sizing:border-box",
            "resize:vertical",
            "padding:7px",
            "background:#111",
            "color:#eee",
            "border:1px solid #444",
            "border-radius:3px",
            "font:11px/1.4 ui-monospace,Consolas,monospace",
            "white-space:pre",
        ].join(";");
        this.jsonInput.addEventListener("input", () => this.onJsonInput(this.jsonInput.value));
        this.jsonInput.addEventListener("blur", () => this.onJsonBlur());
        this.jsonInput.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });
        this.jsonInput.addEventListener("pointerdown", (event) => event.stopPropagation());
        this.jsonInput.addEventListener("keydown", (event) => event.stopPropagation());
        body.appendChild(this.jsonInput);

        this.statusEl = document.createElement("div");
        this.statusEl.style.cssText = "margin-top:5px;font-size:11px;color:#888";
        body.appendChild(this.statusEl);

        panel.appendChild(body);
        return panel;
    }

    setStatus(message, isError) {
        if (!this.statusEl) return;
        if (!message) {
            const count = this.prompts.length;
            this.statusEl.textContent = `in sync · ${count} prompt${count === 1 ? "" : "s"} stored`;
            this.statusEl.style.color = "#888";
            if (this.jsonInput) this.jsonInput.style.borderColor = "#444";
            return;
        }
        this.statusEl.textContent = message;
        this.statusEl.style.color = isError ? "#e06c6c" : "#888";
        if (this.jsonInput) this.jsonInput.style.borderColor = isError ? "#a33" : "#444";
    }

    // ----------------------------------------------------------- data plumbing

    bindGeometryWidgets() {
        for (const name of ["output_seconds", "chunk_frames", "video_reference_frames", "audio_reference_seconds"]) {
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

    watchStorage() {
        if (!this.storage) return;
        watchWidgetValue(this.storage, (value) => this.onStorageChanged(value));
        if (this.storage.__h3NPlusOneCallbackWrapped) return;
        const original = this.storage.callback;
        this.storage.callback = (value, ...args) => {
            const result = original?.call(this.storage, value, ...args);
            this.onStorageChanged(value ?? this.storage.value);
            return result;
        };
        this.storage.__h3NPlusOneCallbackWrapped = true;
    }

    /** Write `text` to the hidden widget without re-entering our own watcher. */
    writeRaw(text) {
        if (!this.storage || this.storage.value === text) return;
        this.suspend = true;
        try {
            this.storage.value = text;
        } finally {
            this.suspend = false;
        }
        this.node.graph?.setDirtyCanvas?.(true, true);
    }

    /** Canonicalise the model into the widget, and into the JSON box unless it is being typed in. */
    persist() {
        const text = serializePrompts(this.prompts);
        this.writeRaw(text);
        if (this.jsonInput && document.activeElement !== this.jsonInput) {
            this.jsonInput.value = text;
            this.setStatus(null);
        }
    }

    padPrompts(count) {
        while (this.prompts.length < count) this.prompts.push("");
    }

    /** A write to the widget that did not originate here: workflow load, undo/redo, API. */
    onStorageChanged(value) {
        if (this.suspend) return;
        const parsed = parsePromptStore(value);
        if (!parsed.ok) {
            if (this.jsonInput && document.activeElement !== this.jsonInput) {
                this.jsonInput.value = String(value ?? "");
            }
            this.setStatus(`invalid JSON — ${parsed.error}`, true);
            return;
        }
        if (samePrompts(parsed.prompts, this.prompts)) {
            this.setStatus(null);
            return;
        }
        this.prompts = parsed.prompts;
        if (this.jsonInput && document.activeElement !== this.jsonInput) {
            this.jsonInput.value = String(value ?? "");
        }
        this.renderCards();
        this.setStatus(null);
    }

    /**
     * Typing in the JSON box. The raw text goes straight to the widget so what is
     * shown is what runs — a broken document fails loudly at queue time instead of
     * silently executing a stale one. The chunk cards only follow valid JSON.
     */
    onJsonInput(text) {
        this.writeRaw(text);
        const parsed = parsePromptStore(text);
        if (!parsed.ok) {
            this.setStatus(`invalid JSON — ${parsed.error} (chunks below not updated)`, true);
            return;
        }
        if (!samePrompts(parsed.prompts, this.prompts)) {
            this.prompts = parsed.prompts;
            this.renderCards();
        }
        this.setStatus(null);
    }

    /** Leaving the JSON box normalises it — padding to chunk count, canonical formatting. */
    onJsonBlur() {
        const parsed = parsePromptStore(this.jsonInput?.value);
        if (!parsed.ok) return;
        this.prompts = parsed.prompts;
        this.padPrompts(calculateSchedule(this.node).chunkCount);
        this.persist();
        this.renderCards();
    }

    // -------------------------------------------------------------- rendering

    render() {
        const schedule = calculateSchedule(this.node);
        this.padPrompts(schedule.chunkCount);
        this.persist();
        this.renderSummary(schedule);
        this.renderCards(schedule);
    }

    renderSummary(schedule) {
        const referenceSummary =
            schedule.videoReferenceFrames === schedule.chunkFrames
                ? `${schedule.videoReferenceFrames} (${formatSeconds(schedule.videoReferenceFrames)}) complete previous chunk`
                : `${schedule.videoReferenceFrames} (${formatSeconds(schedule.videoReferenceFrames)}) video tail`;
        this.summaryEl.textContent = [
            `${schedule.chunkCount} chunks`,
            `C=${schedule.chunkFrames}`,
            `V=${referenceSummary}`,
            `A=${schedule.audioReferenceLatents} latents (${(schedule.audioReferenceLatents / 40).toFixed(3)}s) history`,
            "no overlap",
            "N+1 AV continuation",
            `${schedule.outputSeconds}s`,
        ].join(" · ");
    }

    renderCards(schedule) {
        schedule = schedule ?? calculateSchedule(this.node);
        this.padPrompts(schedule.chunkCount);
        this.cardsHost.replaceChildren();

        for (let index = 0; index < schedule.chunkCount; index += 1) {
            const start = index * schedule.chunkFrames;
            const modelStop = start + schedule.chunkFrames;
            const realStop = Math.min(modelStop, schedule.targetFrames);
            const generatedBeyondOutput = Math.max(0, modelStop - schedule.targetFrames);

            const card = document.createElement("div");
            card.style.cssText =
                "margin-bottom:10px;padding:8px;background:#222;border:1px solid #3a3a3a;border-radius:4px";

            const continuationText =
                index === 0
                    ? "static refs only"
                    : schedule.videoReferenceFrames === schedule.chunkFrames
                    ? `N+1 AV continuation | previous complete previous chunk`
                    : `N+1 AV continuation | previous ${formatSeconds(schedule.videoReferenceFrames)} video tail + ${(schedule.audioReferenceLatents / 40).toFixed(3)}s audio history`;

            const label = document.createElement("div");
            label.style.cssText = "margin-bottom:6px;font-size:11px;color:#aaa";
            label.textContent = [
                `Chunk ${index}`,
                `frames ${start}–${Math.max(start, realStop - 1)}`,
                `${formatSeconds(start)}–${formatSeconds(realStop)}`,
                continuationText,
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
            this.cardsHost.appendChild(card);
        }
    }

    restore() {
        const parsed = parsePromptStore(this.storage?.value);
        if (parsed.ok) {
            this.prompts = parsed.prompts;
            this.render();
            this.setStatus(null);
        } else {
            if (this.jsonInput) this.jsonInput.value = String(this.storage?.value ?? "");
            this.renderSummary(calculateSchedule(this.node));
            this.setStatus(`invalid JSON — ${parsed.error}`, true);
        }
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
