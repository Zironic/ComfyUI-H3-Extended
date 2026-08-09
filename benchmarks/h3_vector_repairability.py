"""Capture H3 trajectories, replay repairability branches, and write a profile.

The injected ``module:function`` factory receives parsed arguments and returns
``model``, ``x``, ``sigmas``, ``latent_shapes`` and optional ``extra_args`` and
``metadata``. The script performs the Comfy2 GPU preflight before importing it.
"""

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path


PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACK_ROOT.parents[1]
sys.path.insert(0, str(PACK_ROOT))

from h3_vector_accel.repairability import (  # noqa: E402
    PROFILE_ROOT,
    build_repairability_profile,
    capture_native_trajectory,
    profile_json,
    resolve_profile_path,
    run_repairability_sweep,
)
from h3_vector_accel.sampler import resolve_h3_sampling  # noqa: E402


def _load_factory(spec):
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("factory must use module:function syntax")
    return getattr(importlib.import_module(module_name), function_name)


def _gpu_preflight():
    script = COMFY_ROOT / ".agents" / "skills" / "operate-comfy2-install" / "scripts" / "comfy_gpu_preflight.ps1"
    subprocess.run([
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)
    ], cwd=COMFY_ROOT, check=True)


def _steps(value):
    return tuple(int(item) for item in value.split(",") if item.strip())


def _profile_rows(results):
    grouped = {}
    for result in results:
        if result.branch_type != "normalized_perturbation":
            continue
        grouped.setdefault(result.step, {})[result.modality] = result
    rows = []
    for step, modalities in sorted(grouped.items()):
        if not all(name in modalities for name in ("video", "audio", "joint")):
            raise ValueError(f"step {step} is missing normalized joint/video/audio branches")
        rows.append({
            "step": step,
            "progress": modalities["video"].progress,
            "survival": {
                "video": modalities["video"].survival["video"],
                "audio": modalities["audio"].survival["audio"],
                "joint_conservative_max": modalities["joint"].survival["joint_conservative_max"],
            },
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--method", choices=("hold", "linear_velocity", "vde"), default="linear_velocity")
    parser.add_argument("--conditioning-mode", required=True)
    parser.add_argument("--steps", type=_steps, default=(2, 5, 8, 11, 14, 16, 18))
    parser.add_argument("--target-rms", type=float, required=True)
    parser.add_argument("--tolerance-conservative", type=float, required=True)
    parser.add_argument("--tolerance-balanced", type=float, required=True)
    parser.add_argument("--tolerance-aggressive", type=float, required=True)
    parser.add_argument("--approve-vde-adaptive", action="store_true")
    parser.add_argument("--authorize-gpu", action="store_true")
    args = parser.parse_args()
    if not args.authorize_gpu:
        raise SystemExit("--authorize-gpu is required because repairability replay loads H3 and allocates VRAM")
    if args.method == "vde" and args.approve_vde_adaptive is False:
        adaptive_methods = []
    else:
        adaptive_methods = [args.method]
    _gpu_preflight()
    payload = _load_factory(args.factory)(args)
    required = ("model", "x", "sigmas", "latent_shapes")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError("factory omitted: " + ", ".join(missing))
    model = payload["model"]
    context = resolve_h3_sampling(model, latent_shapes=payload["latent_shapes"])
    trajectory = capture_native_trajectory(
        model, payload["x"], payload["sigmas"],
        extra_args=payload.get("extra_args"),
        latent_shapes=context.latent_shapes,
        metadata=payload.get("metadata"),
    )
    results = run_repairability_sweep(
        model, trajectory, method=args.method, steps=args.steps,
        target_rms=args.target_rms, extra_args=payload.get("extra_args"),
    )
    profile = build_repairability_profile(
        _profile_rows(results), sigmas=trajectory.sigmas,
        model_fingerprint=context.model_fingerprint,
        video_shift=context.video_shift, audio_shift=context.audio_shift,
        nominal_steps=trajectory.logical_steps,
        predictor_method=args.method, conditioning_mode=args.conditioning_mode,
        quality_presets={
            "conservative": args.tolerance_conservative,
            "balanced": args.tolerance_balanced,
            "aggressive": args.tolerance_aggressive,
        },
        adaptive_methods=adaptive_methods,
        target_rms=args.target_rms,
        source_metadata=payload.get("metadata", {}),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "branches.json").open("w", encoding="utf-8") as handle:
        json.dump([result.as_dict() for result in results], handle, sort_keys=True, indent=2, allow_nan=False)
    path = resolve_profile_path(args.profile_name)
    PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(profile_json(profile) + "\n")
    with (out / "profile.json").open("w", encoding="utf-8") as handle:
        handle.write(profile_json(profile) + "\n")


if __name__ == "__main__":
    main()
