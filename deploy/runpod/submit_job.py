#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests


def headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit a ComfyUI API workflow to the H3 RunPod endpoint")
    parser.add_argument("workflow", type=Path, help="ComfyUI workflow exported in API format")
    parser.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="Remote input asset. Workflow may reference it as {{ASSET:NAME}}. Repeatable.",
    )
    parser.add_argument("--execution-timeout-ms", type=int, default=7_200_000)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    if not args.endpoint:
        parser.error("--endpoint or RUNPOD_ENDPOINT_ID is required")
    if not args.api_key:
        parser.error("--api-key or RUNPOD_API_KEY is required")

    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    assets = []
    for spec in args.asset:
        if "=" not in spec:
            parser.error(f"Invalid --asset {spec!r}; expected NAME=URL")
        name, url = spec.split("=", 1)
        assets.append({"name": name, "url": url})

    base = f"https://api.runpod.ai/v2/{args.endpoint}"
    request = {
        "input": {
            "workflow": workflow,
            "assets": assets,
        },
        "policy": {
            "executionTimeout": args.execution_timeout_ms,
        },
    }

    response = requests.post(f"{base}/run", headers=headers(args.api_key), json=request, timeout=30)
    response.raise_for_status()
    job = response.json()
    print(json.dumps(job, indent=2))

    if args.no_wait:
        return 0

    job_id = job["id"]
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
    while True:
        time.sleep(args.poll_seconds)
        status_response = requests.get(f"{base}/status/{job_id}", headers=headers(args.api_key), timeout=30)
        status_response.raise_for_status()
        status = status_response.json()
        state = status.get("status")
        print(f"{job_id}: {state}", flush=True)
        if state in terminal:
            print(json.dumps(status, indent=2))
            return 0 if state == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
