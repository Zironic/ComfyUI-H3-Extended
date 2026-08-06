"""Lay every arm side by side, one image per frame, for flipping through.

Metrics rank arms; they cannot tell you whether an arm *looks* right. Reviewing
that means looking at frames, and looking at frames means not opening one folder
per arm and trying to hold frame 43 of each in your head.

This writes one image per frame index, each a horizontal strip of every arm at
that frame, labelled, in a fixed column order. Flipping through the output
folder is then a scrub through all arms at once, and any arm that diverges shows
up as one column disagreeing with its neighbours.

    python custom_nodes/ComfyUI-H3-Extended/tools/build_arm_contact_strips.py
    python .../build_arm_contact_strips.py --run 20260805_195850_ee6c74a8 --scale 0.5
    python .../build_arm_contact_strips.py --arms baseline_none,aligned_overlap_direct

The monolithic reference and Chunk A are included as leading columns when
present. Chunk A is offset so its column shows the *same global frame* as the
chunked arms: chunk B frame i is global frame S+i, which is chunk A frame S+i
while that exists, and blank afterwards.
"""

import argparse
import os
import re
import sys

DEFAULT_ROOT = r"D:\AI\ComfyUI\Output\h3_ref2v_harness"
LABEL_H = 16


def newest_run(root):
    runs = [d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d, "experiments"))]
    if not runs:
        raise SystemExit("no harness runs under %s" % root)
    return sorted(runs)[-1]


def numeric_sort(files):
    def key(name):
        digits = re.findall(r"\d+", name)
        return (int(digits[-1]) if digits else -1, name)
    return sorted(files, key=key)


def load_arm_frames(directory):
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, f)
            for f in numeric_sort(os.listdir(directory)) if f.endswith(".png")]


def load_tensor_pixels(path):
    from safetensors.torch import load_file
    if not os.path.exists(path):
        return None
    try:
        return load_file(path).get("pixels")
    except Exception:
        return None


def to_image(array, scale):
    import numpy as np
    from PIL import Image
    img = Image.fromarray(
        (array.detach().to("cpu").float().clamp(0, 1).numpy() * 255.0 + 0.5
         ).astype(np.uint8))
    if scale != 1.0:
        img = img.resize((max(1, int(img.width * scale)),
                          max(1, int(img.height * scale))))
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--run", default=None)
    parser.add_argument("--arms", default=None,
                        help="comma-separated subset, default all")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--stride", type=int, default=1,
                        help="every Nth frame, for a quicker skim")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from PIL import Image, ImageDraw

    run = args.run or newest_run(args.root)
    run_dir = os.path.join(args.root, run)
    experiments = os.path.join(run_dir, "experiments")

    arms = sorted(os.listdir(experiments))
    if args.arms:
        wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
        missing = [a for a in wanted if a not in arms]
        if missing:
            raise SystemExit("no such arm(s): %s" % ", ".join(missing))
        arms = wanted

    columns = []
    mono = load_tensor_pixels(os.path.join(run_dir, "common",
                                           "monolithic_output.safetensors"))
    chunk_a = load_tensor_pixels(os.path.join(run_dir, "common",
                                              "chunk_a_output.safetensors"))
    # Chunk B frame i is global frame S+i. Reading the profile from the report
    # keeps this correct if the profile ever changes.
    stride_frames = 51
    try:
        import json
        with open(os.path.join(run_dir, "report.json"), encoding="utf-8") as fh:
            stride_frames = json.load(fh)["profile"]["stride_frames"]
    except Exception:
        pass

    if mono is not None:
        columns.append(("monolithic", ("tensor", mono, stride_frames)))
    if chunk_a is not None:
        columns.append(("chunk A", ("tensor", chunk_a, stride_frames)))
    for arm in arms:
        files = load_arm_frames(os.path.join(experiments, arm, "frames"))
        if files:
            columns.append((arm, ("files", files, 0)))

    if not columns:
        raise SystemExit("nothing to lay out in %s" % run_dir)

    length = max(len(c[1][1]) - c[1][2] for c in columns)
    out = args.out or os.path.join(run_dir, "strips")
    os.makedirs(out, exist_ok=True)

    print("run     %s" % run_dir)
    print("columns %s" % ", ".join(name for name, _ in columns))
    print("frames  %d (stride %d)" % (length, args.stride))
    print("out     %s\n" % out)

    written = 0
    for i in range(0, length, args.stride):
        tiles = []
        for name, (kind, data, offset) in columns:
            index = i + offset
            if kind == "tensor":
                tile = (to_image(data[index], args.scale)
                        if 0 <= index < len(data) else None)
            else:
                tile = (Image.open(data[index]) if 0 <= index < len(data) else None)
                if tile is not None and args.scale != 1.0:
                    tile = tile.resize((max(1, int(tile.width * args.scale)),
                                        max(1, int(tile.height * args.scale))))
            tiles.append((name, tile))

        present = [t for _, t in tiles if t is not None]
        if not present:
            continue
        w, h = present[0].size
        sheet = Image.new("RGB", (w * len(tiles), h + LABEL_H), (16, 16, 16))
        draw = ImageDraw.Draw(sheet)
        # Every column is offset so it shows the same global frame; label it once
        # per column so a cropped screenshot still says which frame it is.
        global_frame = stride_frames + i
        for column, (name, tile) in enumerate(tiles):
            x = column * w
            if tile is not None:
                sheet.paste(tile.convert("RGB"), (x, LABEL_H))
            else:
                draw.rectangle([x, LABEL_H, x + w, LABEL_H + h], fill=(32, 32, 32))
            draw.text((x + 3, 3), "%s  g%03d" % (name[:30], global_frame),
                      fill=(235, 235, 235))
        sheet.save(os.path.join(out, "strip_g%05d.png" % global_frame))
        written += 1

    print("%d strip(s) written" % written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
