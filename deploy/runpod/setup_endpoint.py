#!/usr/bin/env python3
"""Create or reuse the RunPod Serverless endpoint used by H3-Extended.

This script is intended to run on the local machine. It uses RunPod's official
`runpodctl` CLI because the CLI exposes cached-model attachment through
`--model-reference`, while the worker-side code only consumes the cache after
RunPod has mounted it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterator
from typing import Any


DEFAULT_IMAGE = "runpod/worker-comfyui:5.8.6-base-cuda12.8.1"
DEFAULT_GPU_PROFILE = "rtx4090"
DEFAULT_MODEL_REFERENCE = "https://huggingface.co/Comfy-Org/MiniMax-H3:main"
DEFAULT_BRANCH = "main"
DEFAULT_COMFY_REF = "v0.31.0"


@dataclass(frozen=True)
class GPUProfile:
    gpu_id: str
    capability: str
    memory_gb: int


GPU_PROFILES = {
    "rtx4090": GPUProfile("NVIDIA GeForce RTX 4090", "8.9", 24),
    "rtx5090": GPUProfile("NVIDIA GeForce RTX 5090", "12.0", 32),
    "rtx6000-ada": GPUProfile("NVIDIA RTX 6000 Ada Generation", "8.9", 48),
    "l40s": GPUProfile("NVIDIA L40S", "8.9", 48),
    "a100-80gb": GPUProfile("NVIDIA A100-SXM4-80GB", "8.0", 80),
}
UNSUPPORTED_GPU_PROFILES = {
    "b200": "B200 is SM100, which the current H3 Hybrid Sparse backend does not support",
}
DEFAULT_GPU = GPU_PROFILES[DEFAULT_GPU_PROFILE].gpu_id
DEFAULT_ENDPOINT_NAME = f"h3-extended-{DEFAULT_GPU_PROFILE}"
DEFAULT_TEMPLATE_NAME = f"{DEFAULT_ENDPOINT_NAME}-comfy-serverless"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "custom-gpu"


def _resolve_gpu(
    profile_name: str,
    raw_gpu: str | None,
    endpoint_name: str | None,
    template_name: str | None,
) -> dict[str, Any]:
    if raw_gpu:
        name = f"custom-{_slug(raw_gpu)}"
        gpu_id = raw_gpu
        capability = "runtime-detected"
        memory_gb = None
    else:
        if profile_name in UNSUPPORTED_GPU_PROFILES:
            raise ValueError(UNSUPPORTED_GPU_PROFILES[profile_name])
        profile = GPU_PROFILES[profile_name]
        name = profile_name
        gpu_id = profile.gpu_id
        capability = profile.capability
        memory_gb = profile.memory_gb
    endpoint = endpoint_name or f"h3-extended-{name}"
    template = template_name or f"{endpoint}-comfy-serverless"
    return {
        "profile": name,
        "gpu": gpu_id,
        "capability": capability,
        "memory_gb": memory_gb,
        "endpoint_name": endpoint,
        "template_name": template,
    }


def _profile_payload() -> dict[str, dict[str, Any]]:
    payload = {
        name: {
            "gpu_id": profile.gpu_id,
            "capability": profile.capability,
            "memory_gb": profile.memory_gb,
            "supported": True,
        }
        for name, profile in GPU_PROFILES.items()
    }
    payload.update(
        {
            name: {"supported": False, "reason": reason}
            for name, reason in UNSUPPORTED_GPU_PROFILES.items()
        }
    )
    return payload


def _run(args: list[str]) -> str:
    print("+", shlex.join(args), file=sys.stderr)
    result = subprocess.run(args, check=True, text=True, capture_output=True)
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)
    return result.stdout


def _run_json(args: list[str]) -> Any:
    output = _run(args).strip()
    if not output:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "runpodctl did not return JSON. Current runpodctl defaults to JSON; "
            "upgrade runpodctl or inspect the command output above."
        ) from exc


def _objects(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def _find_named(value: Any, name: str) -> dict[str, Any] | None:
    candidates = [obj for obj in _objects(value) if obj.get("name") == name]
    if not candidates:
        return None
    return candidates[0]


def _id_from(value: Any, *, name: str | None = None) -> str | None:
    if name is not None:
        match = _find_named(value, name)
        if match is not None and isinstance(match.get("id"), str):
            return match["id"]
    for obj in _objects(value):
        identifier = obj.get("id")
        if isinstance(identifier, str) and identifier:
            return identifier
    return None


def _bootstrap_command(h3_ref: str, comfy_ref: str) -> str:
    bootstrap_url = (
        "https://raw.githubusercontent.com/Zironic/ComfyUI-H3-Extended/"
        f"{h3_ref}/deploy/runpod/bootstrap.sh"
    )
    return (
        f"wget -qO /tmp/h3-runpod-bootstrap.sh {shlex.quote(bootstrap_url)}"
        " && chmod +x /tmp/h3-runpod-bootstrap.sh"
        f" && H3_EXTENDED_REF={shlex.quote(h3_ref)}"
        f" COMFYUI_REF={shlex.quote(comfy_ref)}"
        " RUNPOD_H3_REQUIRED=1"
        " /tmp/h3-runpod-bootstrap.sh"
    )


def _require_runpodctl() -> None:
    if shutil.which("runpodctl") is None:
        raise SystemExit(
            "runpodctl is required. Install it using RunPod's official CLI instructions, "
            "then configure it with: runpodctl config --apiKey YOUR_KEY"
        )
    _run(["runpodctl", "version"])


def _existing_endpoint(name: str) -> dict[str, Any] | None:
    data = _run_json(["runpodctl", "serverless", "list", "--include-template"])
    return _find_named(data, name)


def _existing_template(name: str) -> dict[str, Any] | None:
    data = _run_json(
        ["runpodctl", "template", "list", "--type", "user", "--limit", "100"]
    )
    return _find_named(data, name)


def _create_template(args: argparse.Namespace) -> str:
    docker_start = "bash,-lc," + _bootstrap_command(args.h3_ref, args.comfy_ref)
    command = [
        "runpodctl",
        "template",
        "create",
        "--name",
        args.template_name,
        "--image",
        args.image,
        "--container-disk-in-gb",
        str(args.container_disk_gb),
        "--docker-start-cmd",
        docker_start,
        "--serverless",
    ]
    created = _run_json(command)
    template_id = _id_from(created, name=args.template_name)
    if not template_id:
        # Re-query rather than depending on the exact create-response envelope.
        match = _existing_template(args.template_name)
        template_id = match.get("id") if match else None
    if not isinstance(template_id, str) or not template_id:
        raise RuntimeError("RunPod created the template but its ID could not be resolved")
    return template_id


def _ensure_template(args: argparse.Namespace) -> str:
    existing = _existing_template(args.template_name)
    if existing is not None:
        template_id = existing.get("id")
        if isinstance(template_id, str) and template_id:
            print(f"Using existing template {args.template_name}: {template_id}", file=sys.stderr)
            return template_id
    return _create_template(args)


def _create_endpoint(args: argparse.Namespace, template_id: str) -> str:
    command = [
        "runpodctl",
        "serverless",
        "create",
        "--name",
        args.endpoint_name,
        "--template-id",
        template_id,
        "--gpu-id",
        args.gpu,
        "--gpu-count",
        "1",
        "--workers-min",
        "0",
        "--workers-max",
        str(args.workers_max),
        "--idle-timeout",
        str(args.idle_timeout),
        "--flash-boot=true",
        "--execution-timeout",
        str(args.execution_timeout),
        "--min-cuda-version",
        args.min_cuda_version,
        "--scale-by",
        "delay",
        "--scale-threshold",
        str(args.scale_threshold),
        "--model-reference",
        args.model_reference,
    ]
    created = _run_json(command)
    endpoint_id = _id_from(created, name=args.endpoint_name)
    if not endpoint_id:
        match = _existing_endpoint(args.endpoint_name)
        endpoint_id = match.get("id") if match else None
    if not isinstance(endpoint_id, str) or not endpoint_id:
        raise RuntimeError("RunPod created the endpoint but its ID could not be resolved")
    return endpoint_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create/reuse the H3-Extended RunPod Serverless endpoint. "
            "Submitting a job later is what actually causes a GPU worker to scale from zero."
        )
    )
    parser.add_argument("--endpoint-name")
    parser.add_argument("--template-name")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--gpu-profile",
        choices=sorted(GPU_PROFILES | UNSUPPORTED_GPU_PROFILES),
        default=DEFAULT_GPU_PROFILE,
    )
    parser.add_argument("--gpu", help="exact RunPod GPU ID; overrides --gpu-profile")
    parser.add_argument("--list-gpu-profiles", action="store_true")
    parser.add_argument("--model-reference", default=DEFAULT_MODEL_REFERENCE)
    parser.add_argument("--h3-ref", default=DEFAULT_BRANCH)
    parser.add_argument("--comfy-ref", default=DEFAULT_COMFY_REF)
    parser.add_argument("--container-disk-gb", type=int, default=30)
    parser.add_argument("--workers-max", type=int, default=1)
    parser.add_argument("--idle-timeout", type=int, default=5)
    parser.add_argument("--execution-timeout", type=int, default=7200, help="seconds")
    parser.add_argument("--min-cuda-version", default="12.8")
    parser.add_argument("--scale-threshold", type=int, default=1, help="queue delay in seconds")
    parser.add_argument(
        "--force-new-endpoint",
        action="store_true",
        help="create a new endpoint even when an endpoint with --endpoint-name already exists",
    )
    args = parser.parse_args()

    if args.list_gpu_profiles:
        print(json.dumps(_profile_payload(), indent=2, sort_keys=True))
        return 0

    try:
        selected = _resolve_gpu(
            args.gpu_profile, args.gpu, args.endpoint_name, args.template_name
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.gpu_profile = selected["profile"]
    args.gpu = selected["gpu"]
    args.endpoint_name = selected["endpoint_name"]
    args.template_name = selected["template_name"]

    _require_runpodctl()

    if not args.force_new_endpoint:
        endpoint = _existing_endpoint(args.endpoint_name)
        if endpoint is not None:
            endpoint_id = endpoint.get("id")
            if isinstance(endpoint_id, str) and endpoint_id:
                print(json.dumps(endpoint, indent=2))
                print(f"\nexport RUNPOD_ENDPOINT_ID={shlex.quote(endpoint_id)}")
                return 0

    template_id = _ensure_template(args)
    endpoint_id = _create_endpoint(args, template_id)

    print(
        json.dumps(
            {
                "endpoint_id": endpoint_id,
                "endpoint_name": args.endpoint_name,
                "template_id": template_id,
                "gpu": args.gpu,
                "gpu_profile": args.gpu_profile,
                "expected_capability": selected["capability"],
                "memory_gb": selected["memory_gb"],
                "model_reference": args.model_reference,
                "workers_min": 0,
                "workers_max": args.workers_max,
                "idle_timeout": args.idle_timeout,
                "execution_timeout": args.execution_timeout,
            },
            indent=2,
        )
    )
    print(f"\nexport RUNPOD_ENDPOINT_ID={shlex.quote(endpoint_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
