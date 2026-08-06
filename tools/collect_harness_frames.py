"""Collect one frame from every harness arm into a single folder for eyeballing.

The harness writes each arm to its own directory, which is right for archiving
and wrong for comparing - judging ten arms means opening ten folders and
remembering which frame you were on. This pulls the same frame out of each and
names it after the arm, so one folder sorts into a comparison.

    python custom_nodes/ComfyUI-H3-Extended/tools/collect_harness_frames.py
    python .../collect_harness_frames.py --run 20260805_195850_ee6c74a8 --frame 0
    python .../collect_harness_frames.py --frame 36 --out D:\\scratch\\midframes

Defaults to the most recent run and its **last** frame - global frame 123 at the
default profile, the point furthest from the carried state and therefore where
the arms differ most.

Chunk A and the monolithic reference are included when present, prefixed `_` so
they sort to the top. They are not arms, but the monolithic run is the ground
truth every arm is scored against and its last frame is the same global
timestamp, so it is the one image worth having next to the others.
"""

import argparse
import os
import re
import sys

DEFAULT_ROOT = r"D:\AI\ComfyUI\Output\h3_ref2v_harness"


def newest_run(root):
    runs = [d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
            and os.path.isdir(os.path.join(root, d, "experiments"))]
    if not runs:
        raise SystemExit("no harness runs with an experiments/ directory under %s" % root)
    return sorted(runs)[-1]


def frame_files(directory):
    if not os.path.isdir(directory):
        return []
    files = [f for f in os.listdir(directory) if f.lower().endswith(".png")]
    # numeric order, not lexical - frame_9 must not follow frame_10
    def key(name):
        digits = re.findall(r"\d+", name)
        return (int(digits[-1]) if digits else -1, name)
    return [os.path.join(directory, f) for f in sorted(files, key=key)]


def pick(files, index):
    if not files:
        return None
    try:
        return files[index]
    except IndexError:
        return None


def save_tensor_frame(path, source, index):
    """Write one frame out of a stored pixel batch, or None if unavailable."""
    try:
        from safetensors.torch import load_file
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        print("  (skipped %s: %s)" % (os.path.basename(source), exc))
        return None
    if not os.path.exists(source):
        return None
    try:
        data = load_file(source)
    except Exception as exc:
        print("  (skipped %s: %s)" % (os.path.basename(source), exc))
        return None
    pixels = data.get("pixels")
    if pixels is None or pixels.ndim != 4:
        return None
    try:
        frame = pixels[index]
    except IndexError:
        return None
    array = frame.detach().to("cpu").float().clamp(0, 1).numpy()
    Image.fromarray((array * 255.0 + 0.5).astype(np.uint8)).save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--run", default=None, help="run id; default is the newest")
    parser.add_argument("--frame", type=int, default=-1,
                        help="frame index within each arm; -1 (default) is the last")
    parser.add_argument("--kind", default="frames", choices=("frames", "boundary"),
                        help="'frames' is chunk B, 'boundary' is the seam clip")
    parser.add_argument("--out", default=None,
                        help="output folder; default <run>/collected_<kind>_<frame>")
    args = parser.parse_args()

    run = args.run or newest_run(args.root)
    run_dir = os.path.join(args.root, run)
    experiments = os.path.join(run_dir, "experiments")
    if not os.path.isdir(experiments):
        raise SystemExit("no experiments/ in %s" % run_dir)

    label = "last" if args.frame == -1 else str(args.frame)
    out = args.out or os.path.join(run_dir, "collected_%s_%s" % (args.kind, label))
    os.makedirs(out, exist_ok=True)

    print("run    %s" % run_dir)
    print("frame  %s of each arm's %s/" % (label, args.kind))
    print("out    %s\n" % out)

    written = 0
    if args.kind == "frames":
        for name, asset in (("_chunk_a", "chunk_a_output.safetensors"),
                            ("_monolithic", "monolithic_output.safetensors")):
            path = save_tensor_frame(
                os.path.join(out, name + ".png"),
                os.path.join(run_dir, "common", asset), args.frame)
            if path:
                print("  %-34s <- common/%s" % (name + ".png", asset))
                written += 1

    for arm in sorted(os.listdir(experiments)):
        files = frame_files(os.path.join(experiments, arm, args.kind))
        chosen = pick(files, args.frame)
        if chosen is None:
            print("  %-34s (no %s frames)" % (arm, args.kind))
            continue
        target = os.path.join(out, arm + ".png")
        with open(chosen, "rb") as src, open(target, "wb") as dst:
            dst.write(src.read())
        print("  %-34s <- %s" % (arm + ".png", os.path.basename(chosen)))
        written += 1

    print("\n%d image(s) in %s" % (written, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
