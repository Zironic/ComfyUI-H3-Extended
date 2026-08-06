"""Estimate H3 trajectory-AdaLN table geometry from a checkpoint header.

The runtime provider reports measured table and projection bytes when enabled.
This utility answers the earlier question: whether precomputing the complete
trajectory is even a memory win for a particular full-width or curve checkpoint.
"""

import argparse
import json
import math
import os
import struct


def read_header(path):
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length))


def dtype_bytes(name):
    name = str(name).upper()
    if "64" in name:
        return 8
    if "32" in name:
        return 4
    if "16" in name:
        return 2
    if "8" in name:
        return 1
    return 0


def tensor_bytes(entry):
    return math.prod(entry["shape"]) * dtype_bytes(entry["dtype"])


def find_projection_entries(header):
    return {
        key: value
        for key, value in header.items()
        if key != "__metadata__"
        and ".blocks." in ("." + key)
        and ".adaln_proj.linear." in key
    }


def infer_hidden(header):
    for key, entry in header.items():
        if key.endswith("blocks.0.norm1.weight"):
            return int(entry["shape"][0])
    for key, entry in header.items():
        if key.endswith("blocks.0.attn.out_proj.weight"):
            return int(entry["shape"][0])
    raise ValueError("could not infer H3 hidden size")


def infer_blocks(header):
    indices = set()
    for key in header:
        marker = "blocks."
        if marker in key and "token_refiner" not in key:
            try:
                indices.add(int(key.split(marker, 1)[1].split(".", 1)[0]))
            except Exception:
                pass
    if not indices:
        raise ValueError("could not infer H3 block count")
    return max(indices) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--distinct-times", type=float, default=2.0,
        help="Average unique timestep values per denoising step (normally 2-4).",
    )
    parser.add_argument("--table-dtype-bytes", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    header = read_header(args.ckpt)
    projections = find_projection_entries(header)
    hidden = infer_hidden(header)
    blocks = infer_blocks(header)
    projection_bytes = sum(tensor_bytes(value) for value in projections.values())
    rows_per_step = float(args.distinct_times) * 3.0
    table_bytes = int(
        blocks
        * int(args.steps)
        * rows_per_step
        * 6
        * hidden
        * int(args.table_dtype_bytes)
    )
    result = {
        "checkpoint": os.path.abspath(args.ckpt),
        "blocks": blocks,
        "hidden": hidden,
        "steps": args.steps,
        "average_distinct_times": args.distinct_times,
        "projection_tensors": len(projections),
        "projection_gib": projection_bytes / 1024**3,
        "table_gib": table_bytes / 1024**3,
        "estimated_saved_gib": (projection_bytes - table_bytes) / 1024**3,
        "table_smaller_than_weights": table_bytes < projection_bytes,
        "note": (
            "The Comfy implementation keeps checkpoint weights offloadable; this "
            "is the geometry ceiling, not a promise that process VRAM falls by the "
            "full estimated_saved_gib."
        ),
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print("%-30s %s" % (key, value))


if __name__ == "__main__":
    main()
