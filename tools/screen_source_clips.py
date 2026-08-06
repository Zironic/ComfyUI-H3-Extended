"""Score source clips for how well they can discriminate carry strategies.

A chunked-carry experiment can only show a difference between arms if the clip
gives it something to preserve. On a near-static shot every arm reproduces the
same frame and the metrics collapse into the noise floor - which looks like
"the technique does nothing" and is actually "the test could not see".

What matters is not overall motion but motion *in the specific windows the
harness measures*:

  overlap  frames S..C-1   the shared window; this is what the carry transfers,
                           so motion here is what separates a working carry from
                           a broken one
  tail     the last N      furthest from the carry, where drift shows up
  seam     around frame S  a discontinuity here is what a failed carry looks like

A scene cut inside the window disqualifies it: the harness's baseline case
assumes continuous footage, and a cut confounds every metric at once. Cuts are
flagged rather than silently scored.

    python custom_nodes/ComfyUI-H3-Extended/tools/screen_source_clips.py
    python .../screen_source_clips.py --chunk 73 --overlap 22 --top 3
"""

import argparse
import os
import sys

DEFAULT_INPUT = r"D:\AI\ComfyUI\Input"
VIDEO_EXT = (".mp4", ".mkv", ".webm", ".mov", ".avi")
WIDTH = 128          # motion statistics are scale-free enough to downsample hard
CUT_RATIO = 6.0      # max/median frame delta above this reads as a scene cut


def decode(path, fps, width=WIDTH, limit=1200):
    """Grayscale frames at `fps`, downscaled. Returns [N, h, w] float32 in 0..1."""
    import av
    import numpy as np

    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    src_fps = float(stream.average_rate or fps)
    step = max(1.0, src_fps / fps)

    frames, next_index, index = [], 0.0, 0
    for frame in container.decode(stream):
        if index >= next_index:
            img = frame.to_ndarray(format="gray")
            h = max(1, int(round(img.shape[0] * width / img.shape[1])))
            # cheap box resize; exact filtering is irrelevant to a motion statistic
            ys = (np.arange(h) * img.shape[0] // h).clip(0, img.shape[0] - 1)
            xs = (np.arange(width) * img.shape[1] // width).clip(0, img.shape[1] - 1)
            frames.append(img[ys][:, xs].astype("float32") / 255.0)
            next_index += step
            if len(frames) >= limit:
                break
        index += 1
    container.close()
    return np.stack(frames) if frames else None


def frame_deltas(frames):
    import numpy as np
    return np.abs(frames[1:] - frames[:-1]).mean(axis=(1, 2))


def score_windows(deltas, chunk, overlap, tail, stride=17):
    """Score every candidate start offset. `deltas[i]` is motion into frame i+1."""
    import numpy as np

    total = chunk + (chunk - overlap)          # frames the two chunks span
    if len(deltas) + 1 < total:
        return []
    s = chunk - overlap
    out = []
    for start in range(0, len(deltas) + 1 - total + 1, stride):
        window = deltas[start:start + total - 1]
        if window.size == 0:
            continue
        median = float(np.median(window)) or 1e-9
        overlap_motion = float(window[s:chunk].mean()) if chunk > s else 0.0
        tail_motion = float(window[max(0, total - tail - 1):].mean())
        seam = float(window[max(0, s - 2):s + 2].mean())
        out.append({
            "start": start,
            "motion_overall": float(window.mean()),
            "motion_overlap": overlap_motion,
            "motion_tail": tail_motion,
            "motion_seam": seam,
            "cut_ratio": float(window.max()) / median,
            "static_fraction": float((window < median * 0.25).mean()),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--chunk", type=int, default=73)
    parser.add_argument("--overlap", type=int, default=22)
    parser.add_argument("--tail", type=int, default=17)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--top", type=int, default=2, help="windows to show per clip")
    args = parser.parse_args()

    total = args.chunk + (args.chunk - args.overlap)
    print("profile C=%d O=%d S=%d -> %d frames needed per window, tail %d\n"
          % (args.chunk, args.overlap, args.chunk - args.overlap, total, args.tail))

    videos = sorted(f for f in os.listdir(args.input)
                    if f.lower().endswith(VIDEO_EXT))
    if not videos:
        raise SystemExit("no videos in %s" % args.input)

    rows = []
    for name in videos:
        path = os.path.join(args.input, name)
        try:
            frames = decode(path, args.fps)
        except Exception as exc:
            print("  %-58s decode failed: %s" % (name[:58], exc))
            continue
        if frames is None or len(frames) < total:
            print("  %-58s too short (%s frames at %g fps)"
                  % (name[:58], 0 if frames is None else len(frames), args.fps))
            continue
        windows = score_windows(frame_deltas(frames), args.chunk, args.overlap,
                                args.tail)
        clean = [w for w in windows if w["cut_ratio"] < CUT_RATIO]
        for w in sorted(clean, key=lambda w: -w["motion_overlap"])[:args.top]:
            rows.append((name, len(frames), w, len(windows) - len(clean)))

    rows.sort(key=lambda r: -r[2]["motion_overlap"])
    print("%-46s %6s %8s %8s %8s %7s %7s"
          % ("clip  (start frame)", "frames", "overlap", "tail", "seam", "cut", "static"))
    print("-" * 100)
    for name, count, w, cut_windows in rows:
        label = "%s  (@%d)" % (name[:38], w["start"])
        print("%-46s %6d %8.4f %8.4f %8.4f %7.1f %6.0f%%"
              % (label, count, w["motion_overlap"], w["motion_tail"],
                 w["motion_seam"], w["cut_ratio"], w["static_fraction"] * 100))

    print()
    print("overlap = mean frame-to-frame change across the shared window; this is")
    print("the column to rank on - it is the motion a carry has to transfer.")
    print("cut > %.0f suggests a scene change; those windows are excluded above." % CUT_RATIO)
    print("static = share of frames barely moving; high means arms will look alike.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
