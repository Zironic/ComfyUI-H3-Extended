#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${COMFY_ROOT:-/comfyui}"
COMFYUI_REPO="${COMFYUI_REPO:-https://github.com/Comfy-Org/ComfyUI.git}"
COMFYUI_REF="${COMFYUI_REF:-v0.31.0}"
H3_REPO="${H3_REPO:-https://github.com/Zironic/ComfyUI-H3-Extended.git}"
H3_EXTENDED_REF="${H3_EXTENDED_REF:-agent/runpod-serverless}"
H3_DEST="${COMFY_ROOT}/custom_nodes/ComfyUI-H3-Extended"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
UV_BIN="${UV_BIN:-uv}"

log() {
    printf '[h3-runpod bootstrap] %s\n' "$*"
}

checkout_ref() {
    local repo="$1"
    local ref="$2"
    local dest="$3"

    rm -rf "$dest"
    mkdir -p "$dest"
    git -C "$dest" init -q
    git -C "$dest" remote add origin "$repo"
    git -C "$dest" fetch -q --depth=1 --filter=blob:none origin "$ref"
    git -C "$dest" checkout -q --detach FETCH_HEAD
}

replace_comfyui() {
    local next="/tmp/comfyui-h3-runpod-next"
    local keep="/tmp/comfyui-h3-runpod-keep"

    log "Fetching ComfyUI ref ${COMFYUI_REF}"
    checkout_ref "$COMFYUI_REPO" "$COMFYUI_REF" "$next"

    rm -rf "$keep"
    mkdir -p "$keep"

    if [[ -d "$COMFY_ROOT" ]]; then
        for name in models input output user; do
            if [[ -e "${COMFY_ROOT}/${name}" ]]; then
                mv "${COMFY_ROOT}/${name}" "${keep}/${name}"
            fi
        done
        rm -rf "$COMFY_ROOT"
    fi

    mv "$next" "$COMFY_ROOT"

    for name in models input output user; do
        if [[ -e "${keep}/${name}" ]]; then
            rm -rf "${COMFY_ROOT:?}/${name}"
            mv "${keep}/${name}" "${COMFY_ROOT}/${name}"
        fi
    done
    rm -rf "$keep"

    mkdir -p \
        "$COMFY_ROOT/models" \
        "$COMFY_ROOT/input" \
        "$COMFY_ROOT/output" \
        "$COMFY_ROOT/user" \
        "$COMFY_ROOT/custom_nodes"

    log "Updating ComfyUI Python dependencies in the stock RunPod venv"
    "$UV_BIN" pip install --python "$PYTHON_BIN" \
        -r "$COMFY_ROOT/requirements.txt" \
        "transformers>=4.50.3,<5" \
        "huggingface-hub<1.0"
}

install_h3_extended() {
    log "Fetching ComfyUI-H3-Extended ref ${H3_EXTENDED_REF}"
    checkout_ref "$H3_REPO" "$H3_EXTENDED_REF" "$H3_DEST"
}

write_deployment_state() {
    local state_dir="${COMFY_ROOT}/user/runpod"
    local comfy_sha="unknown"
    local h3_sha="unknown"

    mkdir -p "$state_dir"
    if [[ -d "${COMFY_ROOT}/.git" ]]; then
        comfy_sha="$(git -C "$COMFY_ROOT" rev-parse HEAD)"
    fi
    if [[ -d "${H3_DEST}/.git" ]]; then
        h3_sha="$(git -C "$H3_DEST" rev-parse HEAD)"
    fi

    "$PYTHON_BIN" - "$state_dir/deployment.json" "$COMFYUI_REF" "$comfy_sha" "$H3_EXTENDED_REF" "$h3_sha" <<'PY'
import json
import sys

path, comfy_ref, comfy_sha, h3_ref, h3_sha = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "comfyui_ref": comfy_ref,
            "comfyui_commit": comfy_sha,
            "h3_extended_ref": h3_ref,
            "h3_extended_commit": h3_sha,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
}

main() {
    if [[ "${RUNPOD_SKIP_COMFY_UPDATE:-0}" != "1" ]]; then
        replace_comfyui
    else
        log "RUNPOD_SKIP_COMFY_UPDATE=1; keeping the ComfyUI bundled in the worker image"
        mkdir -p \
            "$COMFY_ROOT/custom_nodes" \
            "$COMFY_ROOT/models" \
            "$COMFY_ROOT/input" \
            "$COMFY_ROOT/output" \
            "$COMFY_ROOT/user"
    fi

    install_h3_extended

    log "Linking H3 files from RunPod Cached Models"
    "$PYTHON_BIN" "$H3_DEST/deploy/runpod/link_h3_models.py"

    write_deployment_state

    log "Installing the H3-aware RunPod handler"
    cp "$H3_DEST/deploy/runpod/handler.py" /handler.py

    if [[ "${RUNPOD_BOOTSTRAP_SMOKE_TEST:-0}" == "1" ]]; then
        log "Running optional CPU import smoke test"
        (
            cd "$COMFY_ROOT"
            timeout 300 "$PYTHON_BIN" main.py --quick-test-for-ci --cpu
        )
    fi

    log "Handing control back to the stock RunPod Comfy worker start script"
    exec /start.sh
}

main "$@"
