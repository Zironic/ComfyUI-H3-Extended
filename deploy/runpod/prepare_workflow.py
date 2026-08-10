#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePath, PurePosixPath


PREVIEW_NODE = "SamplerCustomAdvancedMiniMaxPreview"
CORE_SAMPLER_NODE = "SamplerCustomAdvanced"
HYBRID_NODE = "MiniMaxH3HybridSparseAttentionZi"
ZIRO_SCALE_NODE = "ZiroScaleImageToNativeCanvas"
SAMPLER_INPUTS = {"noise", "guider", "sampler", "sigmas", "latent_image"}
MODEL_INPUTS = {
    "UNETLoader": "unet_name",
    "VAELoader": "vae_name",
}
RUNPOD_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
OUTPUT_NODE_TYPES = {"SaveVideo"}
ASSET_NODE_INPUTS = {
    "LoadImage": "image",
    "LoadImageMask": "image",
    "LoadAudio": "audio",
    "LoadVideo": "file",
}


def filename(value: str) -> str:
    return PurePath(value.replace("\\", "/")).name


def replace_output_reference(value, old_node: str, replacement):
    if isinstance(value, dict):
        return {
            key: replace_output_reference(item, old_node, replacement)
            for key, item in value.items()
        }
    if isinstance(value, list):
        if len(value) == 2 and str(value[0]) == old_node and value[1] == 0:
            return list(replacement)
        return [replace_output_reference(item, old_node, replacement) for item in value]
    return value


def upstream_node_ids(value, node_ids: set[str]) -> set[str]:
    found = set()
    if isinstance(value, dict):
        for item in value.values():
            found.update(upstream_node_ids(item, node_ids))
    elif isinstance(value, list):
        if (
            len(value) == 2
            and str(value[0]) in node_ids
            and isinstance(value[1], int)
        ):
            found.add(str(value[0]))
        else:
            for item in value:
                found.update(upstream_node_ids(item, node_ids))
    return found


def connected_node_ids(workflow: dict) -> set[str]:
    node_ids = {str(node_id) for node_id in workflow}
    roots = [
        str(node_id)
        for node_id, node in workflow.items()
        if node.get("class_type") in OUTPUT_NODE_TYPES
    ]
    if not roots:
        raise ValueError("workflow has no supported output node (SaveVideo)")

    connected = set()
    pending = list(roots)
    while pending:
        node_id = pending.pop()
        if node_id in connected:
            continue
        connected.add(node_id)
        node = workflow[node_id]
        pending.extend(upstream_node_ids(node.get("inputs", {}), node_ids) - connected)
    return connected


def asset_name(value: str) -> str:
    value = value.strip()
    if value.startswith("{{ASSET:") and value.endswith("}}"):
        value = value[8:-2]
    for suffix in (" [input]", " [output]", " [temp]"):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
            break
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid saved input path: {value!r}")
    return path.as_posix()


def prepare_workflow(workflow: dict, input_name: str | None = None) -> dict:
    if not isinstance(workflow, dict):
        raise ValueError("workflow must be a ComfyUI API workflow object")

    prepared = json.loads(json.dumps(workflow))
    hybrid = []
    ziro_nodes = []

    for node_id, node in prepared.items():
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            raise ValueError(f"node {node_id!r} is not an API-format ComfyUI node")
        class_type = node.get("class_type")
        inputs = node["inputs"]
        if class_type == HYBRID_NODE:
            hybrid.append(node)
        elif class_type == PREVIEW_NODE:
            node["class_type"] = CORE_SAMPLER_NODE
            node["inputs"] = {key: value for key, value in inputs.items() if key in SAMPLER_INPUTS}
        elif class_type == ZIRO_SCALE_NODE:
            image = inputs.get("image")
            if not isinstance(image, list) or len(image) != 2:
                raise ValueError(f"{ZIRO_SCALE_NODE} node {node_id} has no linked image input")
            ziro_nodes.append((str(node_id), image))

    if len(hybrid) != 1:
        raise ValueError(f"expected exactly one {HYBRID_NODE}; found {len(hybrid)}")
    if hybrid[0]["inputs"].get("enabled") is not True:
        raise ValueError(f"{HYBRID_NODE} must remain enabled")
    for node_id, image in ziro_nodes:
        prepared = replace_output_reference(prepared, node_id, image)
        del prepared[node_id]

    connected = connected_node_ids(prepared)
    asset_nodes = [
        (str(node_id), node, ASSET_NODE_INPUTS[node.get("class_type")])
        for node_id, node in prepared.items()
        if node.get("class_type") in ASSET_NODE_INPUTS and str(node_id) in connected
    ]
    if not asset_nodes:
        raise ValueError("workflow has no connected supported media loader")
    if input_name is not None and len(asset_nodes) != 1:
        raise ValueError("--input-name can only override a workflow with one connected media loader")
    for _node_id, node, input_key in asset_nodes:
        saved = node["inputs"].get(input_key)
        if not isinstance(saved, str) or not saved:
            raise ValueError(f"connected {node['class_type']} has no saved {input_key} filename")
        name = asset_name(input_name if input_name is not None else saved)
        node["inputs"][input_key] = f"{{{{ASSET:{name}}}}}"

    for node in prepared.values():
        class_type = node.get("class_type")
        inputs = node["inputs"]
        model_input = MODEL_INPUTS.get(class_type)
        if model_input and isinstance(inputs.get(model_input), str):
            inputs[model_input] = filename(inputs[model_input])
        if class_type == "CLIPLoader":
            inputs["clip_name"] = RUNPOD_CLIP
        if class_type == "SaveVideo":
            inputs["filename_prefix"] = "{{RUNPOD_OUTPUT_PREFIX}}"
        if class_type == HYBRID_NODE and "run_tag" in inputs:
            inputs["run_tag"] = "{{RUNPOD_JOB_ID}}"
        if class_type == HYBRID_NODE:
            inputs["mode"] = "{{RUNPOD_HYBRID_MODE}}"

    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize an exported MiniMax H3 API workflow for the storage-free RunPod worker"
    )
    parser.add_argument("source", type=Path, help="ComfyUI workflow exported in API format")
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--input-name",
        help="Override the saved filename when the workflow has exactly one connected media loader",
    )
    args = parser.parse_args()

    if not args.source.is_file():
        parser.error(f"source workflow does not exist: {args.source}")
    workflow = json.loads(args.source.read_text(encoding="utf-8"))
    try:
        prepared = prepare_workflow(workflow, args.input_name)
    except ValueError as exc:
        parser.error(str(exc))
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(prepared, indent=2) + "\n", encoding="utf-8")
    print(args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
