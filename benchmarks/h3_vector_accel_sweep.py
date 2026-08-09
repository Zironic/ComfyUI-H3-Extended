"""Run the fixed H3 vector-acceleration study through an injected live runner.

The runner is ``module:function`` and receives one ``StudyArm``. It must return
true NFE, wall time, explicit quality_pass, and separate video/audio metric
dictionaries. This script performs the Comfy2 GPU preflight before importing
the runner.
"""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACK_ROOT.parents[1]
sys.path.insert(0, str(PACK_ROOT))

from h3_vector_accel.study import (  # noqa: E402
    adaptive_comparison_arms,
    fixed_policy_arms,
    run_study_arms,
    write_study_result,
)


def _load_runner(spec):
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("runner must use module:function syntax")
    return getattr(importlib.import_module(module_name), function_name)


def _gpu_preflight():
    script = COMFY_ROOT / ".agents" / "skills" / "operate-comfy2-install" / "scripts" / "comfy_gpu_preflight.ps1"
    subprocess.run([
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)
    ], cwd=COMFY_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-vde", action="store_true")
    parser.add_argument("--adaptive-profile")
    parser.add_argument("--adaptive-method", choices=("hold", "linear_velocity", "vde"), default="linear_velocity")
    parser.add_argument("--conditioning-mode", default="default")
    parser.add_argument("--best-fixed-method", choices=("hold", "linear_velocity", "vde"), default="linear_velocity")
    parser.add_argument("--best-fixed-profile", default="conservative_12")
    parser.add_argument("--authorize-gpu", action="store_true")
    args = parser.parse_args()
    if not args.authorize_gpu:
        raise SystemExit("--authorize-gpu is required because the injected runner may load H3 and allocate VRAM")
    _gpu_preflight()
    runner = _load_runner(args.runner)
    arms = list(fixed_policy_arms(include_vde=args.include_vde))
    if args.adaptive_profile:
        arms.extend(adaptive_comparison_arms(
            args.best_fixed_method, args.best_fixed_profile,
            args.adaptive_profile, args.adaptive_method,
            args.conditioning_mode,
        ))
    result = run_study_arms(runner, arms)
    write_study_result(result, args.out)


if __name__ == "__main__":
    main()
