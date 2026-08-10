"""Report candidate temporal measurements from captured H3 x0 video latents."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from typing import Any

from safetensors import safe_open


PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT / "h3_vector_accel"))

from temporal_descriptor import (  # noqa: E402
    compare_temporal_descriptors,
    descriptor_summary,
    extract_temporal_descriptor,
    transition_summary,
)


FORMAT_ID = "h3-temporal-descriptor-probe-v1"


def _status_ok(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        value = value.get("status_str", value.get("status", value.get("state")))
    if value is None:
        return True
    return str(value).lower() in {"ok", "success", "completed", "complete", "available"}


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest.json below {run_dir}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain an object")
    if not _status_ok(manifest.get("status")):
        raise ValueError(f"run status is not successful: {manifest.get('status')!r}")
    callbacks = manifest.get("callbacks")
    if not isinstance(callbacks, list) or not callbacks:
        raise ValueError("manifest has no callback tensors")
    return manifest


def _artifact_path(run_dir: Path, callback: dict[str, Any]) -> Path:
    value = callback.get("artifact")
    if not isinstance(value, str) or not value:
        raise ValueError("callback is missing an artifact path")
    path = (run_dir / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"callback artifact escapes run directory: {value}") from exc
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"callback artifact is not safetensors: {path.name}")
    if not path.is_file():
        raise FileNotFoundError(f"callback artifact is missing: {path}")
    return path


def _integer_field(callback: dict[str, Any], keys: tuple[str, ...], fallback: int) -> int:
    value: Any = fallback
    for key in keys:
        if key in callback:
            value = callback[key]
            break
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"callback field {keys[0]} is not an integer: {value!r}") from exc


def _declared_video(callback: dict[str, Any]) -> tuple[tuple[int, ...], str | None]:
    tensors = callback.get("tensors")
    if not isinstance(tensors, dict) or not isinstance(tensors.get("x0.video"), dict):
        raise ValueError("callback manifest must describe tensors.x0.video")
    entry = tensors["x0.video"]
    shape = entry.get("shape")
    if not isinstance(shape, list):
        raise ValueError("callback manifest must declare tensors.x0.video.shape")
    try:
        declared_shape = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("declared x0.video shape must contain integers") from exc
    return declared_shape, str(entry["dtype"]) if entry.get("dtype") is not None else None


def _load_video(path: Path, declared_shape: tuple[int, ...]):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if "x0.video" not in handle.keys():
            raise ValueError(f"callback artifact does not contain x0.video: {path.name}")
        stored_shape = tuple(int(value) for value in handle.get_slice("x0.video").get_shape())
        if stored_shape != declared_shape:
            raise ValueError(
                f"manifest tensor shape {declared_shape} disagrees with {stored_shape} for {path.name}"
            )
        video = handle.get_tensor("x0.video")
    if video.ndim != 5 or int(video.shape[0]) != 1:
        raise ValueError(f"x0.video must have batch-one [1,C,T,H,W] shape: {tuple(video.shape)}")
    return video


def _metadata_value(callback: dict[str, Any], key: str) -> Any:
    metadata = callback.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    return callback.get(key)


def run_probe(run_dir: str | Path, *, max_callbacks: int | None = None,
              pool_size: int = 8, motion_energy_floor: float = 1e-6,
              structure_threshold: float | None = None) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {root}")
    if max_callbacks is not None and int(max_callbacks) <= 0:
        raise ValueError("max_callbacks must be positive")
    manifest = _read_manifest(root)
    selected = manifest["callbacks"] if max_callbacks is None else manifest["callbacks"][:int(max_callbacks)]

    callbacks_output = []
    transitions_output = []
    previous_descriptor = None
    previous_order = None
    seen_orders: set[int] = set()
    for position, callback in enumerate(selected):
        if not isinstance(callback, dict):
            raise ValueError(f"callback {position} must be an object")
        if not _status_ok(callback.get("status")):
            raise ValueError(f"callback {position} is not successful")
        order = _integer_field(callback, ("order", "index", "callback_index", "callback"), position)
        if order in seen_orders:
            raise ValueError(f"duplicate callback order: {order}")
        seen_orders.add(order)
        callback_index = _integer_field(callback, ("callback", "callback_index", "index"), order)
        step = _integer_field(callback, ("step", "logical_step"), order)
        artifact = _artifact_path(root, callback)
        declared_shape, declared_dtype = _declared_video(callback)
        video = _load_video(artifact, declared_shape)
        descriptor = extract_temporal_descriptor(video, pool_size=pool_size)
        callbacks_output.append({
            "order": order,
            "callback": callback_index,
            "step": step,
            "sigma": callback.get("sigma"),
            "nominal_source_sigma": callback.get("nominal_source_sigma"),
            "artifact": str(artifact.relative_to(root)),
            "declared_dtype": declared_dtype,
            "declared_shape": list(declared_shape),
            "true_nfe": _metadata_value(callback, "h3_vector_true_nfe"),
            "descriptor": descriptor_summary(descriptor, structure_threshold=structure_threshold),
        })
        if previous_descriptor is not None:
            transition = compare_temporal_descriptors(
                previous_descriptor,
                descriptor,
                motion_energy_floor=motion_energy_floor,
            )
            transitions_output.append({
                "from_order": previous_order,
                "to_order": order,
                "transition": transition_summary(transition),
            })
        previous_descriptor = descriptor
        previous_order = order

    first_callback = selected[0]
    return {
        "format": FORMAT_ID,
        "latent_proxy_notice": (
            "Measurements come from latent x0.video tensors and do not establish decoded quality, "
            "timeline lock, or safe sampling intervals."
        ),
        "configuration": {
            "pool_size": int(pool_size),
            "motion_energy_floor": float(motion_energy_floor),
            "structure_threshold": structure_threshold,
        },
        "source": {
            "run_dir": str(root),
            "manifest": "manifest.json",
            "run_id": manifest.get("run_id", root.name),
            "status": manifest.get("status"),
            "seed": manifest.get("seed"),
            "source_sigmas": manifest.get("source_sigmas"),
            "settings": manifest.get("settings"),
            "method": _metadata_value(first_callback, "h3_vector_method"),
            "profile": _metadata_value(first_callback, "h3_vector_profile"),
            "source_sigma_hash": _metadata_value(first_callback, "h3_vector_source_sigma_hash"),
        },
        "callbacks": callbacks_output,
        "transitions": transitions_output,
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            flattened.update(_flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(_flatten(item, f"{prefix}.{index}"))
        return flattened
    return {prefix: value}


def render_output(payload: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if output_format == "csv":
        flattened = _flatten(payload)
        columns = sorted(flattened)
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerow([flattened[column] for column in columns])
        return stream.getvalue()
    if output_format != "text":
        raise ValueError(f"unsupported output format: {output_format}")
    lines = [
        f"{payload['format']}: {len(payload['callbacks'])} callbacks, "
        f"{len(payload['transitions'])} adjacent transitions",
        payload["latent_proxy_notice"],
    ]
    for callback in payload["callbacks"]:
        structure = callback["descriptor"]["spatial_structure_score"]
        lines.append(
            f"order {callback['order']} step {callback['step']}: "
            f"structure p10={structure['p10']:.6g} p50={structure['p50']:.6g} "
            f"p95={structure['p95']:.6g}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--max-callbacks", type=int)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--motion-energy-floor", type=float, default=1e-6)
    parser.add_argument("--structure-threshold", type=float)
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text", dest="output_format")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = run_probe(
        args.run_dir,
        max_callbacks=args.max_callbacks,
        pool_size=args.pool_size,
        motion_energy_floor=args.motion_energy_floor,
        structure_threshold=args.structure_threshold,
    )
    rendered = render_output(payload, args.output_format)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
