#!/usr/bin/env python3
from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import requests
import runpod
import websocket
from runpod.serverless.utils import rp_upload

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


def download_asset(job_id: str, asset: dict) -> dict:
    name = safe_name(str(asset.get("name", "")))
    url = asset.get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError(f"Asset {name!r} must provide an http(s) URL")

    job_input = INPUT_ROOT / job_id
    job_input.mkdir(parents=True, exist_ok=True)
    destination = job_input / name

    log(f"downloading input asset {name}")
    with requests.get(url, stream=True, timeout=(15, 600)) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)

    return {
        "name": name,
        "path": str(destination),
        "comfy_path": f"runpod/{job_id}/{name}",
        "bytes": destination.stat().st_size,
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


def prepare_workflow(job_id: str, workflow: dict, downloaded: list[dict]) -> tuple[dict, str]:
    output_prefix = f"runpod/{job_id}/result"
    replacements = {
        "{{RUNPOD_JOB_ID}}": job_id,
        "{{RUNPOD_OUTPUT_PREFIX}}": output_prefix,
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


def upload_artifact(job_id: str, path: Path) -> dict:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    bucket_name = os.environ.get("BUCKET_NAME") or None
    url = rp_upload.upload_file_to_bucket(
        path.name,
        str(path),
        bucket_name=bucket_name,
        prefix=job_id,
        extra_args={"ContentType": content_type},
    )
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "content_type": content_type,
        "url": url,
    }


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
        downloaded = [download_asset(job_id, asset) for asset in assets]
        workflow, output_prefix = prepare_workflow(job_id, workflow, downloaded)

        client_id = str(uuid.uuid4())
        prompt_id = queue_workflow(workflow, client_id)
        log(f"queued prompt {prompt_id} for job {job_id}")
        wait_for_execution(client_id, prompt_id)
        history = get_history(prompt_id)

        files = discover_job_files(job_id) | discover_history_files(history)
        artifacts = [upload_artifact(job_id, path) for path in sorted(files)]

        return {
            "status": "success",
            "job_id": job_id,
            "prompt_id": prompt_id,
            "output_prefix": output_prefix,
            "inputs": downloaded,
            "artifacts": artifacts,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        log(f"job {job_id} failed: {exc}")
        return {
            "error": str(exc),
            "job_id": job_id,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
