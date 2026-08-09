"""CLI and packed-layout helpers for the MiniMax H3 arm-matrix probe."""

import argparse

import _minimax_vram_probe_base as base

QKV_MODES = ("sage128", "sage128_fused_qkv")
MLP_MODES = ("untiled", "convrot2", "convrot4")


def parse_selector(value, choices, name):
    """Parse a concise comma-list selector, rejecting unknown/duplicate values."""
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("--%s must select at least one value" % name)
    unknown = [item for item in values if item not in choices]
    if unknown:
        raise ValueError("invalid %s %r (choose from %s)" %
                         (name, unknown[0], ", ".join(choices)))
    duplicates = sorted({item for item in values if values.count(item) > 1})
    if duplicates:
        raise ValueError("duplicate %s value %r" % (name, duplicates[0]))
    return tuple(values)


def selected_arms(args):
    qkv = parse_selector(args.ab_qkv, QKV_MODES, "ab-qkv")
    mlp = parse_selector(args.ab_mlp, MLP_MODES, "ab-mlp")
    return tuple("%s/%s" % (qkv_mode, mlp_mode)
                 for qkv_mode in qkv for mlp_mode in mlp)


def build_parser(description):
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ab-activation-memory",
        action="store_true",
        help=(
            "run a selectable QKV x MLP arm matrix (all six arms by default)"
        ),
    )
    p.add_argument("--ab-frames", default="73,90",
                   help="comma-separated profiles, or 'grid' for the complete 17k+5 sweep")
    p.add_argument("--ab-warmup", type=int, default=1)
    p.add_argument("--ab-iterations", type=int, default=3)
    p.add_argument("--activation-chunk-rows", type=int, default=4096)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--physical-warning-mb",
        type=int,
        default=800,
        help="retire a measured arm below this much physical free VRAM",
    )
    p.add_argument(
        "--physical-poll-ms",
        type=float,
        default=2.0,
        help="cudaMemGetInfo sampling interval for the untimed residency probe",
    )

    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--budget", type=float, default=11.0, help="VRAM ceiling in GB")
    p.add_argument("--reserve-gb", type=float, default=None,
                   help="VRAM held outside the measured sampling transient")
    p.add_argument("--text-len", type=int, default=256)
    p.add_argument("--max-frames", type=int, default=396)
    p.add_argument("--ckpt", default=None,
                   help="safetensors checkpoint supplying one real block")
    p.add_argument("--analytic", action="store_true",
                   help="unsupported for the arm matrix; retained for copied-command diagnostics")
    p.add_argument("--sage", action="store_true",
                   help="accepted for compatibility; arm selection controls QKV")
    p.add_argument("--ab-qkv", "--ab-qkv-modes", dest="ab_qkv",
                   default=",".join(QKV_MODES),
                   help="comma-separated QKV modes: sage128,sage128_fused_qkv")
    p.add_argument("--ab-mlp", "--ab-mlp-modes", dest="ab_mlp",
                   default=",".join(MLP_MODES),
                   help="comma-separated MLP modes: untiled,convrot2,convrot4")

    g = p.add_argument_group("ref2v")
    g.add_argument("--mode", choices=["t2va", "ref2v"], default="t2va")
    g.add_argument("--ref-frames", default="matched")
    g.add_argument("--ref-width", type=int, default=None)
    g.add_argument("--ref-height", type=int, default=None)
    g.add_argument("--ref-audio", action="store_true")
    g.add_argument("--anchor", action="store_true")
    g.add_argument("--static-refs", type=int, default=0)
    g.add_argument("--static-ref-pixels", type=int, default=None)
    g.add_argument(
        "--calibrate-to",
        type=int,
        default=None,
        metavar="FRAMES",
        help="calibrate reserve from selected arm (default: first arm)",
    )
    g.add_argument("--calibrate-arm", default=None,
                   help="arm label used by --calibrate-to, e.g. sage128/untiled")
    g.add_argument("--spill-ratio", type=float, default=1.35)
    g.add_argument(
        "--past-spill",
        action="store_true",
        help="accepted for command compatibility; retired arms are never resumed",
    )

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
    return sorted(out), adjusted


def layout_for(args, frames):
    if args.mode == "t2va":
        layout = base.build_layout(frames, args.width, args.height, args.text_len)
    else:
        ref = frames if args.ref_frames == "matched" else int(args.ref_frames)
        layout = base.build_layout(
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
    layout["width"] = args.width
    layout["height"] = args.height
    layout["audio_t"] = sum(
        stop - start for start, stop, kind in layout["segments"] if kind == "audio"
    ) // 2
    return layout
