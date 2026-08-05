"""Calibration, side-by-side table, spill tracking, and summary for the A/B probe."""

import torch

import _minimax_vram_probe_base as base
from _minimax_vram_probe_ab_cli import layout_for, parse_profiles
from _minimax_vram_probe_ab_runtime import (
    fmt_gib,
    fmt_ms,
    safe_measure,
    update_resident_fit,
)


def run(args, parser, arch, dtype, device, reserve_gb, streamed,
        baseline_forward, candidate_forward):
    def shared_cost(frames):
        layout = layout_for(args, frames)
        latent = base.latent_bytes(args.width, args.height, frames) * 3
        condition = base.cond_row_bytes(layout, arch)
        return layout, latent, condition

    def measure_variant(forward_fn, frames, layout):
        return safe_measure(
            forward_fn,
            layout,
            arch,
            dtype,
            device,
            warmup=args.ab_warmup,
            iterations=args.ab_iterations,
            seed=args.seed + frames,
        )

    if args.calibrate_to is not None:
        frames = base.align_frame_count(max(5, args.calibrate_to))
        layout, latent, condition = shared_cost(frames)
        measured, _ = measure_variant(baseline_forward, frames, layout)
        if measured is None:
            print(f"cannot calibrate: efficient-Sage baseline OOMs at {frames} frames. Free the card and retry.")
            return
        transient, _ = measured
        solved = args.budget - (transient + latent + condition) / base.GB
        print(
            f"calibration {frames} frames is known to fit {args.budget:.1f} GB on the "
            f"efficient-Sage baseline, so the shared reserve is {solved:.2f} GB"
        )
        if solved < 0:
            print("            negative reserve: supplied budget/known-good length is inconsistent")
            return
        print(f"            reserve {reserve_gb:.2f} GB -> {solved:.2f} GB\n")
        reserve_gb, streamed = solved, False

    try:
        profiles, adjusted = parse_profiles(args.ab_frames, args.max_frames)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    for requested, aligned in adjusted:
        print(f"profile     {requested} is off-grid; measuring aligned {aligned}")
    if adjusted:
        print()

    print(
        f"{'frames':>7} {'tokens':>9} {'base tr':>8} {'act tr':>8} {'saved':>8} "
        f"{'base tot':>9} {'act tot':>9} {'base ms':>8} {'act ms':>8} {'time':>7}  state"
    )

    best_base = best_candidate = None
    base_resident, candidate_resident, rows = [], [], []

    for frames in profiles:
        layout, latent, condition = shared_cost(frames)
        baseline, _ = measure_variant(baseline_forward, frames, layout)
        candidate, _ = measure_variant(candidate_forward, frames, layout)

        base_trans = baseline[0] if baseline is not None else None
        base_ms = baseline[1] if baseline is not None else None
        act_trans = candidate[0] if candidate is not None else None
        act_ms = candidate[1] if candidate is not None else None
        base_total = (
            reserve_gb + (base_trans + latent + condition) / base.GB
            if base_trans is not None else None
        )
        act_total = (
            reserve_gb + (act_trans + latent + condition) / base.GB
            if act_trans is not None else None
        )
        if base_total is not None and base_total <= args.budget:
            best_base = frames
        if act_total is not None and act_total <= args.budget:
            best_candidate = frames

        saved = (
            base_trans - act_trans
            if base_trans is not None and act_trans is not None else None
        )
        slowdown = (
            (act_ms / base_ms - 1.0) * 100.0
            if base_ms is not None and act_ms is not None else None
        )

        sequence = layout["seq_len"]
        base_spill = act_spill = False
        base_dev = act_dev = None
        if base_ms is not None:
            base_spill, base_dev = update_resident_fit(
                base_resident, sequence, base_ms, args.spill_ratio
            )
        if act_ms is not None:
            act_spill, act_dev = update_resident_fit(
                candidate_resident, sequence, act_ms, args.spill_ratio
            )

        state = []
        if baseline is None:
            state.append("base OOM")
        if candidate is None:
            state.append("act OOM")
        if base_spill:
            state.append("base SPILL %.2fx" % base_dev)
        if act_spill:
            state.append("act SPILL %.2fx" % act_dev)
        if not state:
            base_fits = base_total is not None and base_total <= args.budget
            act_fits = act_total is not None and act_total <= args.budget
            if base_fits and act_fits:
                state.append("both fit")
            elif act_fits and not base_fits:
                state.append("ACT_ONLY")
            elif base_fits and not act_fits:
                state.append("base only")
            else:
                state.append("both OVER")

        saved_text = "    n/a" if saved is None else f"{saved / base.GB:>7.3f}G"
        base_total_text = "      OOM" if base_total is None else f"{base_total:>8.3f}G"
        act_total_text = "      OOM" if act_total is None else f"{act_total:>8.3f}G"
        time_text = "    n/a" if slowdown is None else f"{slowdown:>+6.1f}%"
        print(
            f"{frames:>7} {sequence:>9} {fmt_gib(base_trans)} {fmt_gib(act_trans)} "
            f"{saved_text} {base_total_text} {act_total_text} "
            f"{fmt_ms(base_ms)} {fmt_ms(act_ms)} {time_text}  {', '.join(state)}"
        )

        rows.append({
            "frames": frames,
            "sequence": sequence,
            "base_transient": base_trans,
            "activation_transient": act_trans,
            "saved": saved,
            "base_ms": base_ms,
            "activation_ms": act_ms,
            "slowdown_percent": slowdown,
        })

        if base_spill and act_spill and not args.past_spill:
            print("\nstopping: both variants left their resident timing curves; pass --past-spill to continue")
            break
        if baseline is None and candidate is None:
            print("\nstopping: both variants hard-OOM at this profile")
            break

    print()
    paired = [row for row in rows if row["saved"] is not None]
    if paired:
        target = max(paired, key=lambda row: row["frames"])
        print(
            "largest paired profile: C=%d S=%d saves %.3f GiB (%.1f%% of baseline), time delta %+.1f%%"
            % (
                target["frames"], target["sequence"], target["saved"] / base.GB,
                100.0 * target["saved"] / target["base_transient"],
                target["slowdown_percent"],
            )
        )
    print(
        "largest tested under %.1f GB: baseline=%s, activation=%s"
        % (
            args.budget,
            str(best_base) if best_base is not None else "none",
            str(best_candidate) if best_candidate is not None else "none",
        )
    )
    if args.ab_frames.strip().lower() != "grid":
        print("Selected profiles are not a ceiling search; use --ab-frames grid to sweep the ladder.")
    if streamed and args.calibrate_to is None:
        print(
            "Reserve is a streamed-weight floor, so total-budget lines are upper bounds. "
            "Use --calibrate-to with a known-good efficient-Sage baseline length."
        )
