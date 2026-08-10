#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${COMFY_ROOT:-/comfyui}"
COMFYUI_REPO="${COMFYUI_REPO:-https://github.com/Comfy-Org/ComfyUI.git}"
COMFYUI_REF="${COMFYUI_REF:-v0.31.0}"
H3_REPO="${H3_REPO:-https://github.com/Zironic/ComfyUI-H3-Extended.git}"
H3_EXTENDED_REF="${H3_EXTENDED_REF:-main}"
H3_DEST="${COMFY_ROOT}/custom_nodes/ComfyUI-H3-Extended"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
UV_BIN="${UV_BIN:-uv}"
SPARGE_REPO="${SPARGE_REPO:-https://github.com/woct0rdho/SpargeAttn.git}"
SPARGE_REF="${SPARGE_REF:-067d80cb6b76345c7b8be40e86c7d19a3cf7c4eb}"
SPARGE_BUILD_JOBS="${SPARGE_BUILD_JOBS:-}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
SPARGE_MARKER="${COMFY_ROOT}/user/runpod/sparge-attn.json"

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

sparge_runtime() {
    "$PYTHON_BIN" "$H3_DEST/deploy/runpod/sparge_runtime.py" "$@"
}

install_sparge() {
    mkdir -p "${COMFY_ROOT}/user/runpod"
    local capability
    local architecture
    capability="$(sparge_runtime probe --field capability)"
    architecture="$(sparge_runtime probe --field architecture)"
    TORCH_CUDA_ARCH_LIST="$(sparge_runtime probe --field arch_list)"
    export TORCH_CUDA_ARCH_LIST
    log "Detected ${architecture} (compute capability ${capability})"

    if sparge_runtime check \
        --marker "$SPARGE_MARKER" --repo "$SPARGE_REPO" --ref "$SPARGE_REF" \
        --h3-root "$H3_DEST"; then
        log "Pinned SpargeAttention is already installed for ${architecture}"
        retun
    else
        local status=$?
        if [[ "$status" -ne 3 ]]; then
            retun "$status"
        fi
    fi

    if ! command -v nvcc >/dev/null 2>&1; then
        if [[ "$(id -u)" -ne 0 ]]; then
            log "nvcc is missing and bootstrap must run as root to install CUDA build packages"
            retun 1
        fi
        if ! command -v apt-get >/dev/null 2>&1; then
            log "nvcc is missing and apt-get is unavailable; use the stock Ubuntu CUDA 12.8 RunPod image"
            retun 1
        fi
        log "Installing the minimal CUDA 12.8 compiler toolchain on the stock image"
        if ! DEBIAN_FRONTEND=noninteractive apt-get update; then
            log "apt-get update failed while installing the CUDA 12.8 compiler toolchain"
            retun 1
        fi
        if ! DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            build-essential python3-dev ninja-build cuda-nvcc-12-8 cuda-cudart-dev-12-8; then
            log "apt-get could not install build-essential, python3-dev, ninja-build, cuda-nvcc-12-8, and cuda-cudart-dev-12-8"
            retun 1
        fi
    fi

    if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
        log "CUDA_HOME=${CUDA_HOME} has no nvcc; install CUDA 12.8 toolkit/compiler packages or set CUDA_HOME"
        retun 1
    fi
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    if [[ -n "$SPARGE_BUILD_JOBS" ]]; then
        export MAX_JOBS="$SPARGE_BUILD_JOBS"
        export CMAKE_BUILD_PARALLEL_LEVEL="$SPARGE_BUILD_JOBS"
    fi
    if ! "$PYTHON_BIN" - "$CUDA_HOME/bin/nvcc" <<'PY'
import re
import subprocess
import sys

nvcc = sys.argv[1]
try:
    output = subprocess.run(
        [nvcc, "--version"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout
except (OSError, subprocess.CalledProcessError) as exc:
    raise SystemExit(f"CUDA compiler {nvcc} is not callable: {exc}")
match = re.search(r"release\s+(\d+)\.(\d+)", output)
if match is None:
    raise SystemExit(f"Could not parse CUDA compiler version from {nvcc} --version")
version = tuple(int(part) for part in match.groups())
if version < (12, 8):
    raise SystemExit(
        f"Pinned SpargeAttention bootstrap requires nvcc >= 12.8; found {version[0]}.{version[1]}"
    )
PY
    then
        log "CUDA compiler validation failed for ${CUDA_HOME}/bin/nvcc"
        retun 1
    fi

    log "Installing SpargeAttention build requirements into the stock /opt/venv"
    if ! "$UV_BIN" pip install --python "$PYTHON_BIN" setuptools wheel ninja packaging; then
        log "Could not install Python build requirements into ${PYTHON_BIN}"
        retun 1
    fi
    log "Building pinned SpargeAttention ${SPARGE_REF} for ${architecture} (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST})"
    if ! "$UV_BIN" pip install --python "$PYTHON_BIN" --no-build-isolation --no-deps \
        "git+${SPARGE_REPO}@${SPARGE_REF}"; then
        log "SpargeAttention source build failed; verify CUDA_HOME, nvcc, and compiler packages"
        retun 1
    fi
    sparge_runtime verify \
        --marker "$SPARGE_MARKER" --repo "$SPARGE_REPO" --ref "$SPARGE_REF" \
        --h3-root "$H3_DEST"
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

    local sparge_capability
    local sparge_architecture
    sparge_capability="$(sparge_runtime probe --field capability)"
    sparge_architecture="$(sparge_runtime probe --field architecture)"

    "$PYTHON_BIN" - "$state_dir/deployment.json" "$COMFYUI_REF" "$comfy_sha" "$H3_EXTENDED_REF" "$h3_sha" "$SPARGE_REPO" "$SPARGE_REF" "$sparge_capability" "$sparge_architecture" <<'PY'
import json
import sys

path, comfy_ref, comfy_sha, h3_ref, h3_sha, sparge_repo, sparge_ref, sparge_capability, sparge_architecture = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "comfyui_ref": comfy_ref,
            "comfyui_commit": comfy_sha,
            "h3_extended_ref": h3_ref,
            "h3_extended_commit": h3_sha,
            "sparge_repo": sparge_repo,
            "sparge_ref": sparge_ref,
            "sparge_capability": sparge_capability,
            "sparge_architecture": sparge_architecture,
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

    install_sparge

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
