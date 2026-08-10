#!/usr/bin/env python3
from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import os
import shutil
import time
import uuid
from pathlib import Path, PurePosixPath

import requests
import runpod
import websocket

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/comfyui"))
INPUT_ROOT = COMFY_ROOT / "input" / "runpod"
OUTPUT_ROOT = COMFY_ROOT / "output" / "runpod"


def log(message: str) -> None:
    print(f"[h3-runpod handler] {message}", flush=True)


def wait_for_comfy(timeout_s: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://{COMFY_HOST}/"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"ComfyUI did not become ready at {url}: {last_error}")


def safe_name(name: str) -> str:
    base = Path(name).name
    if not base or base in {".", ".."}:
        raise ValueError(f"Invalid asset name: {name!r}")
    return base


def safe_asset_name(name: str) -> str:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Invalid asset name: {name!r}")
    return path.as_posix()


def hybrid_mode_for_capability(capability: tuple[int, int]) -> str:
    return "sage128_fused_qkv" if tuple(capability) == (8, 9) else "sage128"


def current_hybrid_mode() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Hybrid Sparse workflow requires CUDA")
    return hybrid_mode_for_capability(tuple(torch.cuda.get_device_capability()))


def decode_asset(job_id: str, asset: dict) -> dict:
    name = safe_asset_name(str(asset.get("name", "")))
    encoded = asset.get("base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"Asset {name!r} must provide non-empty base64 data")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Asset {name!r} contains invalid base64 data") from exc

    job_input = INPUT_ROOT / job_id
    job_input.mkdir(parents=True, exist_ok=True)
    destination = job_input.joinpath(*PurePosixPath(name).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)

    return {
        "name": name,
        "path": str(destination),
        "comfy_path": f"runpod/{job_id}/{name}",
        "bytes": len(data),
    }


def replace_strings(value, replacements: dict[str, str]):
    if isinstance(value, dict):
        return {key: replace_strings(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_strings(item, replacements) for item in value]
    if isinstance(value, str):
        for needle, replacement in replacements.items():
            value = value.replace(needle, replacement)
        return value
    return value


def prepare_workflow(
    job_id: str,
    workflow: dict,
    downloaded: list[dict],
    hybrid_mode: str | None = None,
) -> tuple[dict, str]:
    output_prefix = f"runpod/{job_id}/result"
    hybrid_mode = current_hybrid_mode() if hybrid_mode is None else hybrid_mode
    replacements = {
        "{{RUNPOD_JOB_ID}}": job_id,
        "{{RUNPOD_OUTPUT_PREFIX}}": output_prefix,
        "{{RUNPOD_HYBRID_MODE}}": hybrid_mode,
    }
    for item in downloaded:
        replacements[f"{{{{ASSET:{item['name']}}}}}"] = item["comfy_path"]
    return replace_strings(workflow, replacements), output_prefix


def queue_workflow(workflow: dict, client_id: str) -> str:
    response = requests.post(
        f"http://{COMFY_HOST}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    if response.status_code == 400:
        raise ValueError(f"ComfyUI workflow validation failed: {response.text}")
    response.raise_for_status()
    payload = response.json()
    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {payload}")
    return prompt_id


def wait_for_execution(client_id: str, prompt_id: str) -> None:
    ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
    attempts = int(os.environ.get("RUNPOD_WS_RECONNECT_ATTEMPTS", "5"))
    delay = float(os.environ.get("RUNPOD_WS_RECONNECT_DELAY", "2"))
    ws: websocket.WebSocket | None = None

    def connect() -> websocket.WebSocket:
        connection = websocket.WebSocket()
        connection.connect(ws_url, timeout=15)
        connection.settimeout(30)
        return connection

    try:
        ws = connect()
        reconnects = 0
        while True:
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException:
                reconnects += 1
                if reconnects > attempts:
                    raise
                time.sleep(delay)
                ws = connect()
                continue

            if not isinstance(raw, str):
                continue
            message = json.loads(raw)
            msg_type = message.get("type")
            data = message.get("data", {})
            if data.get("prompt_id") not in (None, prompt_id):
                continue
            if msg_type == "execution_error":
                raise RuntimeError(
                    f"ComfyUI execution failed at node {data.get('node_id')} "
                    f"({data.get('node_type')}): {data.get('exception_message')}"
                )
            if msg_type == "executing" and data.get("node") is None and data.get("prompt_id") == prompt_id:
                return
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


def get_history(prompt_id: str) -> dict:
    response = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
    response.raise_for_status()
    payload = response.json()
    if prompt_id not in payload:
        raise RuntimeError(f"Prompt {prompt_id} missing from ComfyUI history")
    return payload[prompt_id]


def discover_history_files(history: dict) -> set[Path]:
    files: set[Path] = set()
    for output in history.get("outputs", {}).values():
        if not isinstance(output, dict):
            continue
        for value in output.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                filename = item.get("filename")
                if not filename:
                    continue
                subfolder = item.get("subfolder", "")
                kind = item.get("type", "output")
                root = COMFY_ROOT / kind if kind in {"input", "output", "temp"} else COMFY_ROOT / "output"
                candidate = root / subfolder / filename
                if candidate.is_file():
                    files.add(candidate.resolve())
    return files


def discover_job_files(job_id: str) -> set[Path]:
    root = OUTPUT_ROOT / job_id
    if not root.exists():
        return set()
    return {path.resolve() for path in root.rglob("*") if path.is_file()}


def encode_artifact(path: Path) -> dict:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = path.read_bytes()
    return {
        "name": path.name,
        "bytes": len(data),
        "content_type": content_type,
        "base64": base64.b64encode(data).decode("ascii"),
    }


def cleanup_job_files(job_id: str) -> None:
    for root in (INPUT_ROOT, OUTPUT_ROOT):
        path = root / job_id
        if path.is_dir():
            shutil.rmtree(path)


def handler(job: dict) -> dict:
    started = time.monotonic()
    job_id = safe_name(str(job.get("id") or uuid.uuid4()))
    payload = job.get("input")
    if not isinstance(payload, dict):
        return {"error": "input must be an object"}

    workflow = payload.get("workflow")
    if not isinstance(workflow, dict):
        return {"error": "input.workflow must be a ComfyUI API workflow object"}

    assets = payload.get("assets", [])
    if not isinstance(assets, list):
        return {"error": "input.assets must be a list"}

    try:
        wait_for_comfy(float(os.environ.get("RUNPOD_COMFY_READY_TIMEOUT", "300")))
        decoded = [decode_asset(job_id, asset) for asset in assets]
        workflow, output_prefix = prepare_workflow(job_id, workflow, decoded)

        client_id = str(uuid.uuid4())
        prompt_id = queue_workflow(workflow, client_id)
        log(f"queued prompt {prompt_id} for job {job_id}")
        wait_for_execution(client_id, prompt_id)
        history = get_history(prompt_id)

        files = discover_job_files(job_id) | discover_history_files(history)
        artifacts = [encode_artifact(path) for path in sorted(files)]

        result = {
            "status": "success",
            "job_id": job_id,
            "prompt_id": prompt_id,
            "output_prefix": output_prefix,
            "inputs": [
                {"name": item["name"], "bytes": item["bytes"]}
                for item in decoded
            ],
            "artifacts": artifacts,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        result_bytes = len(json.dumps(result, separators=(",", ":")).encode("utf-8"))
        max_result_bytes = int(os.environ.get("RUNPOD_MAX_INLINE_RESULT_BYTES", "19000000"))
        if result_bytes > max_result_bytes:
            raise ValueError(
                f"inline result is {result_bytes} bytes; limit is {max_result_bytes}. "
                "Reduce the output size or use an external-storage transport."
            )
        return result
    except Exception as exc:
        log(f"job {job_id} failed: {exc}")
        return {
            "error": str(exc),
            "job_id": job_id,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        cleanup_job_files(job_id)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
