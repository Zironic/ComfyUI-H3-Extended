#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SUPPORTED = {
    (8, 0): {"architecture": "sm80", "arch_list": "8.0", "minimum_cuda": (12, 0)},
    (8, 6): {"architecture": "sm86", "arch_list": "8.6", "minimum_cuda": (12, 0)},
    (8, 7): {"architecture": "sm87", "arch_list": "8.7", "minimum_cuda": (12, 0)},
    (8, 9): {"architecture": "sm89", "arch_list": "8.9", "minimum_cuda": (12, 4)},
    (9, 0): {"architecture": "sm90", "arch_list": "9.0", "minimum_cuda": (12, 3)},
    (12, 0): {"architecture": "sm120", "arch_list": "12.0", "minimum_cuda": (12, 8)},
}


def _version(value: str) -> tuple[int, int]:
    try:
        parts = tuple(int(part) for part in value.split(".")[:2])
    except ValueError as exc:
        raise RuntimeError(f"could not parse CUDA version {value!r}") from exc
    if len(parts) != 2:
        raise RuntimeError(f"could not parse CUDA version {value!r}")
    return parts


def runtime_info() -> dict:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"cannot import torch from the stock worker venv: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Hybrid Sparse bootstrap requires CUDA")
    capability = tuple(int(part) for part in torch.cuda.get_device_capability())
    contract = SUPPORTED.get(capability)
    if contract is None:
        raise RuntimeError(
            "H3 Hybrid Sparse does not support CUDA capability "
            f"{capability[0]}.{capability[1]}"
        )
    cuda_text = torch.version.cuda or "0.0"
    cuda_version = _version(cuda_text)
    if cuda_version < contract["minimum_cuda"]:
        required = ".".join(str(part) for part in contract["minimum_cuda"])
        raise RuntimeError(
            f"{contract['architecture']} requires Torch CUDA >= {required}; found {cuda_text}"
        )
    return {
        "architecture": contract["architecture"],
        "arch_list": contract["arch_list"],
        "capability": f"{capability[0]}.{capability[1]}",
        "torch": torch.__version__,
        "torch_cuda": cuda_text,
    }


def load_spec(h3_root: Path):
    sys.path.insert(0, str(h3_root))
    try:
        from h3_attention.hybrid.sparse_sage import preflight_sparse_sage

        return preflight_sparse_sage()
    finally:
        sys.path.pop(0)


def marker_payload(info: dict, spec, repo: str, ref: str) -> dict:
    return {
        "repo": repo,
        "ref": ref,
        **info,
        "kernel_architecture": spec.architecture,
        "kernel_name": spec.kernel_name,
        "extension_layout": spec.extension_layout,
        "q_tile": spec.q_tile,
        "kv_tile": spec.kv_tile,
        "v_format": spec.v_format,
        "accumulator": spec.accumulator,
    }


def marker_valid(marker: Path, expected: dict, h3_root: Path) -> bool:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    for name in ("repo", "ref", "architecture", "arch_list", "capability", "torch", "torch_cuda"):
        if payload.get(name) != expected.get(name):
            return False
    try:
        spec = load_spec(h3_root)
    except Exception:
        return False
    return (
        spec.architecture == payload.get("kernel_architecture")
        and list(spec.capability) == [int(part) for part in expected["capability"].split(".")]
        and spec.kernel_name == payload.get("kernel_name")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe and verify the RunPod Sparse Sage runtime")
    parser.add_argument("mode", choices=("probe", "check", "verify"))
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--repo")
    parser.add_argument("--ref")
    parser.add_argument("--h3-root", type=Path)
    parser.add_argument(
        "--field",
        choices=("architecture", "arch_list", "capability", "torch", "torch_cuda"),
    )
    args = parser.parse_args()

    try:
        info = runtime_info()
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.mode == "probe":
        print(info[args.field] if args.field else json.dumps(info, sort_keys=True))
        return 0
    if args.marker is None or not args.repo or not args.ref or args.h3_root is None:
        parser.error("check/verify require --marker, --repo, --ref, and --h3-root")

    expected = {"repo": args.repo, "ref": args.ref, **info}
    if args.mode == "check":
        if marker_valid(args.marker, expected, args.h3_root):
            print(
                f"Sparse Sage marker and {info['architecture']} extension contract are valid"
            )
            return 0
        print("Sparse Sage marker or extension contract is stale; rebuilding")
        return 3

    try:
        spec = load_spec(args.h3_root)
    except Exception as exc:
        raise SystemExit(f"Sparse Sage extension verification failed: {exc}") from exc
    payload = marker_payload(info, spec, args.repo, args.ref)
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    args.marker.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Sparse Sage {spec.architecture} extension verified: "
        f"{spec.q_tile}Qx{spec.kv_tile}KV {spec.v_format} V"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
