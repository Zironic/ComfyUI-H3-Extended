"""Read-only catalog of ComfyUI H3 video output settings.

The catalog deliberately uses ffprobe and the embedded API prompt only.  It
does not import ComfyUI, contact a server, or inspect the generated pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


DEFAULT_VIDEO_ROOT = Path(r"D:\AI\ComfyUI\Output\video")
LOCAL_FFPROBE = Path(r"D:\yt-dlp\ffmpeg-9.0-full_build\bin\ffprobe.exe")
SAMPLER_PREFIX = "SamplerCustom"
OUTPUT_HINTS = ("save", "video", "combine", "preview", "output")


def _json_value(value: Any) -> Any:
    """Return a JSON-stable primitive/list/dict representation."""
    if isinstance(value, dict):
        return {str(k): _json_value(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_link(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 2 and isinstance(value[1], int) and (
        isinstance(value[0], (str, int))
    )


def _literal(value: Any) -> Any:
    if _is_link(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return _json_value(value)


def resolve_video_root(value: str | Path | None = None) -> Path:
    if value:
        return Path(value).expanduser()
    configured = os.environ.get("COMFYUI_OUTPUT_DIR")
    if configured:
        root = Path(configured).expanduser()
        if root.is_dir() and not any(root.glob("*.mp4")) and (root / "video").is_dir():
            root = root / "video"
        return root
    return DEFAULT_VIDEO_ROOT


def resolve_ffprobe(value: str | Path | None = None) -> str:
    if value:
        return str(value)
    for name in ("COMFYUI_FFPROBE", "H3_FFPROBE", "FFPROBE", "FFPROBE_PATH"):
        if os.environ.get(name):
            return os.environ[name]
    found = shutil.which("ffprobe")
    if found:
        return found
    return str(LOCAL_FFPROBE)


def probe_tags(path: Path, ffprobe: str | Path) -> dict[str, Any]:
    command = [str(ffprobe), "-v", "error", "-show_entries", "format_tags=prompt:format_tags=workflow", "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = (result.stderr or "ffprobe failed").strip().splitlines()[-1]
        raise RuntimeError(detail)
    payload = json.loads(result.stdout)
    tags = payload.get("format", {}).get("tags", {})
    if not isinstance(tags, dict):
        raise ValueError("ffprobe returned no format tags")
    return {str(key).lower(): value for key, value in tags.items()}


def _parse_json_tag(tags: dict[str, Any], name: str) -> Any:
    value = tags.get(name)
    if value is None:
        raise ValueError(f"missing embedded {name} metadata")
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def _nodes(prompt: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(prompt, dict):
        raise ValueError("prompt metadata is not an API object")
    result = {}
    for node_id, node in prompt.items():
        if not isinstance(node, dict) or "class_type" not in node:
            continue
        result[str(node_id)] = node
    if not result:
        raise ValueError("prompt metadata has no API nodes")
    return result


def _workflow_nodes(workflow: Any) -> dict[str, dict[str, Any]]:
    """Accept the API-shaped workflow object used by some video writers."""
    if isinstance(workflow, dict) and all(isinstance(value, dict) and "class_type" in value for value in workflow.values()):
        return _nodes(workflow)
    if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
        raise ValueError("workflow metadata is not an API object")
    result = {}
    for item in workflow["nodes"]:
        if not isinstance(item, dict) or "id" not in item or "type" not in item:
            continue
        result[str(item["id"])] = {"class_type": item["type"], "inputs": {}}
    if not result:
        raise ValueError("workflow metadata has no nodes")
    return result


def _deps(value: Any) -> list[str]:
    if _is_link(value):
        return [str(value[0])]
    if isinstance(value, list):
        found = []
        for item in value:
            found.extend(_deps(item))
        return found
    if isinstance(value, dict):
        found = []
        for item in value.values():
            found.extend(_deps(item))
        return found
    return []


def _active_sampler(nodes: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    dependencies = {
        node_id: {dep for value in node.get("inputs", {}).values() for dep in _deps(value)}
        for node_id, node in nodes.items()
    }
    output_ids = [
        node_id for node_id, node in nodes.items()
        if not str(node.get("class_type", "")).startswith(SAMPLER_PREFIX)
        and any(hint in str(node.get("class_type", "")).lower() for hint in OUTPUT_HINTS)
    ]
    seen: set[str] = set()
    stack = output_ids or list(nodes)
    while stack:
        node_id = str(stack.pop())
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(dependencies.get(node_id, ()))
    candidates = [
        (node_id, node) for node_id, node in nodes.items()
        if str(node.get("class_type", "")).startswith(SAMPLER_PREFIX) and node_id in seen
    ]
    if not candidates:
        candidates = [
            (node_id, node) for node_id, node in nodes.items()
            if str(node.get("class_type", "")).startswith(SAMPLER_PREFIX)
        ]
    if not candidates:
        raise ValueError("no active SamplerCustom node")
    candidates.sort(key=lambda pair: (not pair[0].isdigit(), int(pair[0]) if pair[0].isdigit() else pair[0]))
    return candidates[-1]


def _linked_node(nodes: dict[str, dict[str, Any]], value: Any) -> tuple[str, dict[str, Any]] | None:
    if not _is_link(value):
        return None
    node_id = str(value[0])
    node = nodes.get(node_id)
    return (node_id, node) if node is not None else None


def _text_digest(nodes: dict[str, dict[str, Any]]) -> str | None:
    texts = []
    for node in nodes.values():
        class_type = str(node.get("class_type", "")).lower()
        if "text" in class_type or "prompt" in class_type or "string" in class_type:
            inputs = node.get("inputs", {})
            for name in ("text", "value", "prompt"):
                value = inputs.get(name)
                if isinstance(value, str) and value:
                    texts.append(value)
    if not texts:
        return None
    canonical = "\n".join(sorted(texts)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


def _node_settings(nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    settings = {}
    for node_id in sorted(nodes, key=lambda key: (not str(key).isdigit(), int(key) if str(key).isdigit() else str(key))):
        node = nodes[node_id]
        literals = {}
        for name, value in sorted(node.get("inputs", {}).items()):
            literal = _literal(value)
            if literal is not None:
                literals[str(name)] = literal
        settings[str(node_id)] = {"class_type": str(node.get("class_type", "")), "inputs": literals}
    return settings


def catalog_file(path: Path, ffprobe: str | Path) -> dict[str, Any]:
    base: dict[str, Any] = {"filename": path.name, "video_number": _video_number(path.name)}
    try:
        tags = probe_tags(path, ffprobe)
        try:
            prompt = _parse_json_tag(tags, "prompt")
            nodes = _nodes(prompt)
        except ValueError:
            nodes = _workflow_nodes(_parse_json_tag(tags, "workflow"))
        sampler_id, sampler = _active_sampler(nodes)
        sampler_inputs = sampler.get("inputs", {})
        sampler_link = _linked_node(nodes, sampler_inputs.get("sampler"))
        sampler_impl = sampler_link[1] if sampler_link else sampler
        sampler_impl_inputs = sampler_impl.get("inputs", {})
        is_vector = sampler_impl.get("class_type") == "MiniMaxH3VectorAccelSamplerZi"
        sampler_method = str(sampler_impl_inputs.get("method", sampler_impl_inputs.get("sampler_name", "stock"))) if is_vector else str(sampler_impl_inputs.get("sampler_name", "stock"))
        sampler_name = None
        if sampler_link and not is_vector:
            sampler_name = sampler_impl_inputs.get("sampler_name")
        sigma_link = _linked_node(nodes, sampler_inputs.get("sigmas"))
        sigma_inputs = sigma_link[1].get("inputs", {}) if sigma_link else {}
        noise_link = _linked_node(nodes, sampler_inputs.get("noise"))
        noise_inputs = noise_link[1].get("inputs", {}) if noise_link else {}
        profile = sampler_impl_inputs.get("evaluation_profile")
        evaluation_profile = profile or ("native_20" if sampler_name == "native" else "stock")
        vector_settings = {k: _json_value(v) for k, v in sampler_impl_inputs.items() if not _is_link(v)} if is_vector else {}
        prompt_digest = _text_digest(nodes)
        classification = "vector" if is_vector else ("stock_res20" if sampler_method == "res_multistep" and str(sigma_inputs.get("steps")) == "20" else "stock")
        result = {
            **base,
            "active_sampler": {"node_id": sampler_id, "kind": sampler.get("class_type"), "method": sampler_method, "name": sampler_name, "source_kind": sampler_impl.get("class_type")},
            "sampler_method": sampler_method,
            "classification": classification,
            "evaluation_profile": evaluation_profile,
            "scheduler": sigma_inputs.get("scheduler"),
            "scheduler_steps": sigma_inputs.get("steps"),
            "seed": noise_inputs.get("noise_seed", noise_inputs.get("seed")),
            "vector_settings": vector_settings,
            "effective_settings": {"method": sampler_method, "evaluation_profile": evaluation_profile, **vector_settings},
            "prompt_digest": prompt_digest,
            "node_settings": _node_settings(nodes),
            "status": "ok",
        }
        attention = {}
        for node in nodes.values():
            class_type = str(node.get("class_type", ""))
            if "H3" in class_type or "ModelSampling" in class_type:
                attention[class_type] = {k: _json_value(v) for k, v in node.get("inputs", {}).items() if not _is_link(v)}
        result["h3_settings"] = attention
        result["model_settings"] = {
            class_type: {k: _json_value(v) for k, v in node.get("inputs", {}).items() if not _is_link(v)}
            for node in nodes.values()
            for class_type in [str(node.get("class_type", ""))]
            if "model" in class_type.lower() or "checkpoint" in class_type.lower()
        }
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return {**base, "status": "error", "error": error}


def _video_number(name: str) -> int | None:
    stem = Path(name).stem
    digits = ""
    for char in reversed(stem):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    return int(digits) if digits else None


def catalog(video_root: str | Path | None = None, *, last: int | None = None, ffprobe: str | Path | None = None) -> list[dict[str, Any]]:
    root = resolve_video_root(video_root)
    probe = resolve_ffprobe(ffprobe)
    files = sorted((path for path in root.glob("*.mp4") if path.is_file()), key=lambda path: path.name.casefold())
    if last is not None:
        files = [] if last <= 0 else files[-last:]
    return [catalog_file(path, probe) for path in files]


def render_text(rows: list[dict[str, Any]]) -> str:
    headers = ("video", "sampler", "method", "profile", "scheduler", "steps", "seed", "prompt", "status")
    values = []
    for row in rows:
        active = row.get("active_sampler", {})
        values.append((
            f"{row.get('video_number') or '-'}:{row.get('filename', '')}",
            f"{active.get('source_kind') or active.get('kind', '?')}", active.get("name") or active.get("method") or "-",
            row.get("evaluation_profile", "-"), row.get("scheduler") or "-", row.get("scheduler_steps") or "-",
            row.get("seed") if row.get("seed") is not None else "-", row.get("prompt_digest") or "-", row.get("status", "error"),
        ))
    widths = [max(len(str(header)), *(len(str(row[i])) for row in values)) for i, header in enumerate(headers)] if values else [len(h) for h in headers]
    lines = ["  ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)), "  ".join("-" * width for width in widths)]
    lines.extend("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) for row in values)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--last", type=int)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--ffprobe")
    args = parser.parse_args(argv)
    rows = catalog(args.video_root, last=args.last, ffprobe=args.ffprobe)
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(render_text(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
