#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

CACHE_ROOT = Path(os.environ.get("RUNPOD_HF_CACHE", "/runpod-volume/huggingface-cache/hub"))
MODEL_REPO = os.environ.get("RUNPOD_H3_MODEL_REPO", "Comfy-Org/MiniMax-H3")
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/comfyui"))
REQUIRED = os.environ.get("RUNPOD_H3_REQUIRED", "0") == "1"

MODEL_TARGETS = {
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors": COMFY_ROOT / "models/diffusion_models",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors": COMFY_ROOT / "models/diffusion_models",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": COMFY_ROOT / "models/text_encoders",
    "minimax_h3_video_vae_fp16.safetensors": COMFY_ROOT / "models/vae",
    "minimax_h3_audio_vae_fp32.safetensors": COMFY_ROOT / "models/vae",
}


def repo_cache_dir(repo_id: str) -> Path:
    return CACHE_ROOT / ("models--" + repo_id.replace("/", "--"))


def snapshots(repo_id: str) -> list[Path]:
    root = repo_cache_dir(repo_id) / "snapshots"
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)


def find_file(snapshot: Path, filename: str) -> Path | None:
    direct = list(snapshot.rglob(filename))
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        raise RuntimeError(f"Cached model contains more than one {filename}: {direct}")
    return None


def link_file(source: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source)
    return target


def main() -> int:
    candidates = snapshots(MODEL_REPO)
    if not candidates:
        message = f"RunPod cached model {MODEL_REPO!r} was not found under {CACHE_ROOT}"
        if REQUIRED:
            raise SystemExit(message)
        print(f"[h3-runpod linker] WARNING: {message}")
        return 0

    snapshot = candidates[0]
    print(f"[h3-runpod linker] using cached snapshot {snapshot}")

    linked = 0
    missing: list[str] = []
    for filename, target_dir in MODEL_TARGETS.items():
        source = find_file(snapshot, filename)
        if source is None:
            missing.append(filename)
            continue
        target = link_file(source, target_dir)
        linked += 1
        print(f"[h3-runpod linker] {target} -> {source}")

    if missing:
        print("[h3-runpod linker] not present in this cache snapshot:")
        for filename in missing:
            print(f"  - {filename}")

    if REQUIRED and missing:
        raise SystemExit(
            "Required H3 cached-model files are missing: " + ", ".join(missing)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
