"""CLI and packed-layout helpers for the activation-memory VRAM A/B probe."""

import argparse

import _minimax_vram_probe_base as base

MODE_BF16 = "mlp_chunked_bf16"
MODE_NATIVE = "mlp_chunked_native"


def build_parser(description):
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ab-activation-memory", action="store_true",
                   help="compare efficient Sage alone with efficient Sage + chunked MLP")
    p.add_argument("--ab-frames", default="73,90",
                   help="comma-separated profiles, or 'grid' for the complete 17k+5 sweep")
    p.add_argument("--ab-warmup", type=int, default=1)
    p.add_argument("--ab-iterations", type=int, default=3)
    p.add_argument("--activation-chunk-rows", type=int, default=4096)
    p.add_argument("--activation-alignment", type=int, default=256)
    p.add_argument("--activation-mode", choices=(MODE_BF16, MODE_NATIVE), default=MODE_BF16)
    p.add_argument("--no-held-weights", action="store_true")
    p.add_argument("--activation-nonstrict", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--physical-warning-mb",
        type=int,
        default=800,
        help=(
            "mark a variant LOW when its minimum measured free physical VRAM "
            "falls below this many MiB (default 800)"
        ),
    )
    p.add_argument(
        "--physical-poll-ms",
        type=float,
        default=2.0,
        help=(
            "poll cudaMemGetInfo at this interval during an untimed residency "
            "probe; timing iterations run without the poller (default 2 ms)"
        ),
    )

    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--budget", type=float, default=11.0, help="VRAM ceiling in GB")
    p.add_argument("--reserve-gb", type=float, default=None,
                   help="VRAM held outside the measured sampling transient")
    p.add_argument("--text-len", type=int, default=256)
    p.add_argument("--max-frames", type=int, default=396)
    p.add_argument("--ckpt", default=None,
                   help="safetensors checkpoint; only its header is read")
    p.add_argument("--analytic", action="store_true",
                   help="unsupported for A/B; retained for copied-command diagnostics")
    p.add_argument("--sage", action="store_true",
                   help="accepted for compatibility; A/B always uses efficient Sage")

    g = p.add_argument_group("ref2v")
    g.add_argument("--mode", choices=["t2va", "ref2v"], default="t2va")
    g.add_argument("--ref-frames", default="matched")
    g.add_argument("--ref-width", type=int, default=None)
    g.add_argument("--ref-height", type=int, default=None)
    g.add_argument("--ref-audio", action="store_true")
    g.add_argument("--anchor", action="store_true")
    g.add_argument("--static-refs", type=int, default=0)
    g.add_argument("--static-ref-pixels", type=int, default=None)
    g.add_argument("--calibrate-to", type=int, default=None, metavar="FRAMES",
                   help="calibrate reserve from the efficient-Sage baseline")
    g.add_argument("--spill-ratio", type=float, default=1.35)
    g.add_argument("--past-spill", action="store_true")

    for key in base.DEFAULT_ARCH:
        p.add_argument("--" + key.replace("_", "-"), type=int, default=None)
    return p


def parse_profiles(spec, max_frames):
    if spec.strip().lower() == "grid":
        out = []
        n = 5
        while n <= max_frames:
            out.append(n)
            n = base.align_frame_count(n + 1)
        return out, []

    out, adjusted = [], []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        requested = int(raw)
        aligned = base.align_frame_count(max(5, requested))
        if aligned != requested:
            adjusted.append((requested, aligned))
        if aligned <= max_frames and aligned not in out:
            out.append(aligned)
    if not out:
        raise ValueError("--ab-frames selected no profiles at or below --max-frames")
    return out, adjusted


def layout_for(args, frames):
    if args.mode == "t2va":
        return base.build_layout(frames, args.width, args.height, args.text_len)
    ref = frames if args.ref_frames == "matched" else int(args.ref_frames)
    return base.build_layout(
        frames,
        args.width,
        args.height,
        args.text_len,
        ref_frames=ref,
        ref_width=args.ref_width,
        ref_height=args.ref_height,
        ref_audio=args.ref_audio,
        anchor=args.anchor,
        static_refs=args.static_refs,
        static_ref_pixels=args.static_ref_pixels,
    )
