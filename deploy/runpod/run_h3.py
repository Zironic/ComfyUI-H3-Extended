#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import binascii
import json
import mimetypes
import os
import re
import time
from pathlib import Path, PurePosixPath

import requests

from prepare_workflow import prepare_workflow


TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
DEFAULT_MAX_REQUEST_BYTES = 19_000_000
ASSET_PATTERN = re.compile(r"\{\{ASSET:([^{}]+)\}\}")


def headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def safe_asset_name(name: str) -> str:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Invalid input name {name!r}")
    return path.as_posix()


def parse_input(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, raw_path = spec.split("=", 1)
    else:
        raw_path = spec
        name = Path(raw_path).name
    name = safe_asset_name(name)
    path = Path(raw_path)
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    return name, path


def workflow_asset_names(value) -> set[str]:
    names = set()
    if isinstance(value, dict):
        for item in value.values():
            names.update(workflow_asset_names(item))
    elif isinstance(value, list):
        for item in value:
            names.update(workflow_asset_names(item))
    elif isinstance(value, str):
        names.update(safe_asset_name(match) for match in ASSET_PATTERN.findall(value))
    return names


def inferred_input_roots(workflow_path: Path) -> list[Path]:
    candidates = []
    configured = os.environ.get("COMFY_INPUT_DIR")
    if configured:
        candidates.append(Path(configured))
    for parent in workflow_path.resolve().parents:
        if parent.name.casefold() == "user":
            candidates.extend((parent.parent / "input", parent.parent / "Input"))
    candidates.extend((Path.cwd() / "input", Path.cwd() / "Input"))
    return candidates


def resolve_input_root(workflow_path: Path, override: Path | None) -> Path:
    candidates = [override] if override is not None else inferred_input_roots(workflow_path)
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate.resolve()
    checked = ", ".join(str(path) for path in candidates)
    raise ValueError(
        "Could not locate the local Comfy input directory. "
        f"Checked: {checked or '<none>'}. Use --input-root or COMFY_INPUT_DIR."
    )


def resolve_saved_input(root: Path, name: str) -> Path:
    relative = PurePosixPath(safe_asset_name(name))
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Saved workflow input escapes the Comfy input directory: {name!r}")
    if not path.is_file():
        raise ValueError(f"Saved workflow input does not exist: {path}")
    return path


def encode_input(name: str, path: Path) -> dict:
    data = path.read_bytes()
    return {
        "name": name,
        "bytes": len(data),
        "content_type": mimetypes.guess_type(name)[0] or "application/octet-stream",
        "base64": base64.b64encode(data).decode("ascii"),
    }


def result_output(result: dict) -> dict:
    output = result.get("output", result)
    if not isinstance(output, dict):
        raise ValueError("RunPod result output is not an object")
    return output


def save_artifacts(result: dict, output_dir: Path) -> list[Path]:
    output = result_output(result)
    if output.get("error"):
        raise RuntimeError(str(output["error"]))
    artifacts = output.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("RunPod result artifacts is not a list")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    names = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("RunPod returned an invalid artifact entry")
        name = str(artifact.get("name", ""))
        if not name or Path(name).name != name:
            raise ValueError(f"RunPod returned an invalid artifact name: {name!r}")
        if name in names:
            raise ValueError(f"RunPod returned duplicate artifact name: {name!r}")
        names.add(name)
        encoded = artifact.get("base64")
        if not isinstance(encoded, str):
            raise ValueError(f"RunPod artifact {name!r} has no base64 data")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"RunPod artifact {name!r} contains invalid base64 data") from exc
        expected = artifact.get("bytes")
        if expected is not None and expected != len(data):
            raise ValueError(
                f"RunPod artifact {name!r} declared {expected} bytes but returned {len(data)}"
            )
        destination = output_dir / name
        destination.write_bytes(data)
        saved.append(destination)
    return saved


def printable_result(result: dict) -> dict:
    sanitized = json.loads(json.dumps(result))
    try:
        artifacts = result_output(sanitized).get("artifacts", [])
    except ValueError:
        return sanitized
    for artifact in artifacts:
        if isinstance(artifact, dict) and isinstance(artifact.get("base64"), str):
            artifact["base64"] = f"<{len(artifact['base64'])} base64 characters>"
    return sanitized


def poll(base: str, job_id: str, api_key: str, poll_seconds: float) -> dict:
    while True:
        time.sleep(poll_seconds)
        response = requests.get(f"{base}/status/{job_id}", headers=headers(api_key), timeout=30)
        response.raise_for_status()
        result = response.json()
        state = result.get("status")
        print(f"{job_id}: {state}", flush=True)
        if state in TERMINAL_STATES:
            return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a ComfyUI API workflow on the H3 RunPod endpoint with inline files"
    )
    parser.add_argument("workflow", type=Path, help="ComfyUI workflow exported in API format")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="[NAME=]PATH",
        help="Override an automatically discovered workflow input. Repeatable.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Local Comfy input directory; normally inferred from the workflow path",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runpod-output"))
    parser.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--execution-timeout-ms", type=int, default=7_200_000)
    parser.add_argument("--sync-wait-ms", type=int, default=300_000)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=DEFAULT_MAX_REQUEST_BYTES,
        help="Local guard below RunPod's 20 MB /runsync payload limit",
    )
    args = parser.parse_args()

    if not args.endpoint:
        parser.error("--endpoint or RUNPOD_ENDPOINT_ID is required")
    if not args.api_key:
        parser.error("--api-key or RUNPOD_API_KEY is required")
    if not args.workflow.is_file():
        parser.error(f"Workflow does not exist: {args.workflow}")
    if not 1_000 <= args.sync_wait_ms <= 300_000:
        parser.error("--sync-wait-ms must be between 1000 and 300000")

    source_workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    if not isinstance(source_workflow, dict):
        parser.error("workflow must contain a JSON object")
    try:
        workflow = prepare_workflow(source_workflow)
        required_assets = workflow_asset_names(workflow)
        overrides = dict(parse_input(spec) for spec in args.input)
        unexpected = set(overrides) - required_assets
        if unexpected:
            raise ValueError(
                "--input names are not used by the connected workflow: "
                + ", ".join(sorted(unexpected))
            )
        missing = required_assets - set(overrides)
        automatic = {}
        if missing:
            input_root = resolve_input_root(args.workflow, args.input_root)
            automatic = {
                name: resolve_saved_input(input_root, name)
                for name in missing
            }
        selected = automatic | overrides
        assets = [encode_input(name, selected[name]) for name in sorted(required_assets)]
    except ValueError as exc:
        parser.error(str(exc))

    request = {
        "input": {
            "workflow": workflow,
            "assets": assets,
        },
        "policy": {
            "executionTimeout": args.execution_timeout_ms,
        },
    }
    body = json.dumps(request, separators=(",", ":")).encode("utf-8")
    if len(body) > args.max_request_bytes:
        parser.error(
            f"encoded request is {len(body)} bytes; limit is {args.max_request_bytes}. "
            "Use a smaller input or an external-storage transport."
        )

    base = f"https://api.runpod.ai/v2/{args.endpoint}"
    response = requests.post(
        f"{base}/runsync?wait={args.sync_wait_ms}",
        headers=headers(args.api_key),
        data=body,
        timeout=args.sync_wait_ms / 1000 + 30,
    )
    response.raise_for_status()
    result = response.json()
    state = result.get("status")
    if state not in TERMINAL_STATES:
        job_id = result.get("id")
        if not job_id:
            raise RuntimeError(f"RunPod did not return a terminal result or job ID: {result}")
        result = poll(base, str(job_id), args.api_key, args.poll_seconds)

    print(json.dumps(printable_result(result), indent=2))
    if result.get("status") != "COMPLETED":
        return 1
    try:
        saved = save_artifacts(result, args.output_dir)
    except (RuntimeError, ValueError) as exc:
        print(f"RunPod result error: {exc}")
        return 1
    for path in saved:
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
