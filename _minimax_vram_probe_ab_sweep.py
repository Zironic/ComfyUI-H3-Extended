"""Calibration, measured-residency tables, and budget projections for A/B/C."""

import _minimax_vram_probe_base as base
from _minimax_vram_probe_ab_cli import layout_for, parse_profiles
from _minimax_vram_probe_ab_runtime import (
    MIB,
    fmt_gib,
    fmt_mib,
    fmt_ms,
    safe_measure,
    update_resident_fit,
)

VARIANTS = ("A", "B", "C")


def _variant_state(measurement, spill, threshold_bytes):
    if measurement is None:
        return "OOM"
    flags = []
    if measurement.physical_free_min < threshold_bytes:
        flags.append("LOW")
    if spill:
        flags.append("SPILL")
    return "+".join(flags) if flags else "ok"


def _profile_text(value):
    return "none" if value is None else str(value)


def _pct(new, old):
    if new is None or old is None:
        return None
    return (new / old - 1.0) * 100.0


def _fmt_pct(value):
    return "    n/a" if value is None else f"{value:>+6.1f}%"


def _budget_state(total, budget):
    if total is None:
        return "OOM"
    return "fit" if total <= budget else "over"


def run(
    args,
    parser,
    arch,
    dtype,
    device,
    reserve_gb,
    streamed,
    plain_forward,
    efficient_forward,
    activation_forward,
):
    threshold_bytes = int(args.physical_warning_mb) * MIB
    forwards = {
        "A": plain_forward,
        "B": efficient_forward,
        "C": activation_forward,
    }

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
            physical_poll_ms=args.physical_poll_ms,
        )

    if args.calibrate_to is not None:
        frames = base.align_frame_count(max(5, args.calibrate_to))
        layout, latent, condition = shared_cost(frames)
        measured, _ = measure_variant(efficient_forward, frames, layout)
        if measured is None:
            print(
                f"cannot calibrate: variant B efficient Sage OOMs at {frames} "
                "frames. Free the card and retry."
            )
            return
        solved = (
            args.budget
            - (measured.peak_bytes + latent + condition) / base.GB
        )
        print(
            f"calibration {frames} frames is known to fit {args.budget:.1f} GB on "
            f"variant B (efficient Sage), so the shared reserve is {solved:.2f} GB"
        )
        if solved < 0:
            print(
                "            negative reserve: supplied budget/known-good "
                "length is inconsistent"
            )
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
        "measured isolated block memory/residency "
        f"(physical free from cudaMemGetInfo; LOW < {args.physical_warning_mb} MiB)"
    )
    print(
        f"{'frames':>7} {'tokens':>9} {'A tr':>8} {'B tr':>8} {'C tr':>8} "
        f"{'A-B':>8} {'B-C':>8} {'A free':>7} {'B free':>7} {'C free':>7}  residency"
    )

    best_projected = {key: None for key in VARIANTS}
    last_physical_safe = {key: None for key in VARIANTS}
    first_low = {key: None for key in VARIANTS}
    first_spill = {key: None for key in VARIANTS}
    resident_points = {key: [] for key in VARIANTS}
    rows = []

    for frames in profiles:
        layout, latent, condition = shared_cost(frames)
        measurements = {}
        for key in VARIANTS:
            measurements[key], _ = measure_variant(
                forwards[key], frames, layout
            )

        transient = {
            key: (
                measurements[key].peak_bytes
                if measurements[key] is not None
                else None
            )
            for key in VARIANTS
        }
        median_ms = {
            key: (
                measurements[key].median_ms
                if measurements[key] is not None
                else None
            )
            for key in VARIANTS
        }
        totals = {
            key: (
                reserve_gb + (transient[key] + latent + condition) / base.GB
                if transient[key] is not None
                else None
            )
            for key in VARIANTS
        }
        for key in VARIANTS:
            if totals[key] is not None and totals[key] <= args.budget:
                best_projected[key] = frames

        saved_ab = (
            transient["A"] - transient["B"]
            if transient["A"] is not None and transient["B"] is not None
            else None
        )
        saved_bc = (
            transient["B"] - transient["C"]
            if transient["B"] is not None and transient["C"] is not None
            else None
        )
        saved_ac = (
            transient["A"] - transient["C"]
            if transient["A"] is not None and transient["C"] is not None
            else None
        )

        phys_safe = {}
        spills = {}
        states = {}
        sequence = layout["seq_len"]
        for key in VARIANTS:
            measurement = measurements[key]
            phys_safe[key] = bool(
                measurement is not None
                and measurement.physical_free_min >= threshold_bytes
            )
            if phys_safe[key]:
                last_physical_safe[key] = frames
            elif measurement is not None and first_low[key] is None:
                first_low[key] = frames

            spills[key] = False
            if median_ms[key] is not None:
                spills[key], _ = update_resident_fit(
                    resident_points[key],
                    sequence,
                    median_ms[key],
                    args.spill_ratio,
                    include_in_fit=phys_safe[key],
                )
                if spills[key] and first_spill[key] is None:
                    first_spill[key] = frames
            states[key] = _variant_state(
                measurement, spills[key], threshold_bytes
            )

        saved_ab_text = (
            "    n/a" if saved_ab is None else f"{saved_ab / base.GB:>7.3f}G"
        )
        saved_bc_text = (
            "    n/a" if saved_bc is None else f"{saved_bc / base.GB:>7.3f}G"
        )
        print(
            f"{frames:>7} {sequence:>9} "
            f"{fmt_gib(transient['A'])} {fmt_gib(transient['B'])} {fmt_gib(transient['C'])} "
            f"{saved_ab_text} {saved_bc_text} "
            f"{fmt_mib(measurements['A'].physical_free_min if measurements['A'] else None)} "
            f"{fmt_mib(measurements['B'].physical_free_min if measurements['B'] else None)} "
            f"{fmt_mib(measurements['C'].physical_free_min if measurements['C'] else None)}  "
            f"A:{states['A']} B:{states['B']} C:{states['C']}"
        )

        rows.append({
            "frames": frames,
            "sequence": sequence,
            "measurements": measurements,
            "transient": transient,
            "median_ms": median_ms,
            "totals": totals,
            "saved_ab": saved_ab,
            "saved_bc": saved_bc,
            "saved_ac": saved_ac,
            "spills": spills,
            "states": states,
        })

        if all(spills.values()) and not args.past_spill:
            print(
                "\nstopping: all three variants left their resident timing "
                "curves; pass --past-spill to continue"
            )
            break
        if all(measurements[key] is None for key in VARIANTS):
            print("\nstopping: all three variants hard-OOM at this profile")
            break

    print()
    print("measured isolated block timing (same profiles; median of timed forwards)")
    print(
        f"{'frames':>7} {'A ms':>8} {'B ms':>8} {'C ms':>8} "
        f"{'B/A':>7} {'C/B':>7} {'C/A':>7}"
    )
    for row in rows:
        times = row["median_ms"]
        print(
            f"{row['frames']:>7} "
            f"{fmt_ms(times['A'])} {fmt_ms(times['B'])} {fmt_ms(times['C'])} "
            f"{_fmt_pct(_pct(times['B'], times['A']))} "
            f"{_fmt_pct(_pct(times['C'], times['B']))} "
            f"{_fmt_pct(_pct(times['C'], times['A']))}"
        )

    print()
    print(
        "projected production budget "
        "(arithmetic only: calibrated reserve is not allocated in this probe)"
    )
    print(
        f"{'frames':>7} {'A projected':>15} {'B projected':>15} "
        f"{'C projected':>15}  budget state"
    )
    for row in rows:
        totals = row["totals"]
        texts = {
            key: ("OOM" if totals[key] is None else f"{totals[key]:.3f} GiB")
            for key in VARIANTS
        }
        state = " ".join(
            f"{key}:{_budget_state(totals[key], args.budget)}"
            for key in VARIANTS
        )
        print(
            f"{row['frames']:>7} {texts['A']:>15} {texts['B']:>15} "
            f"{texts['C']:>15}  {state}"
        )

    print()
    complete = [
        row for row in rows
        if all(row["measurements"][key] is not None for key in VARIANTS)
    ]
    if complete:
        target = max(complete, key=lambda row: row["frames"])
        a = target["transient"]["A"]
        b = target["transient"]["B"]
        c = target["transient"]["C"]
        print(
            "largest complete profile: C=%d S=%d; "
            "A→B saves %.3f GiB, B→C saves %.3f GiB, "
            "A→C saves %.3f GiB"
            % (
                target["frames"],
                target["sequence"],
                (a - b) / base.GB,
                (b - c) / base.GB,
                (a - c) / base.GB,
            )
        )
        print(
            "timing at that profile: B/A %+.1f%%, C/B %+.1f%%, C/A %+.1f%%"
            % (
                _pct(target["median_ms"]["B"], target["median_ms"]["A"]),
                _pct(target["median_ms"]["C"], target["median_ms"]["B"]),
                _pct(target["median_ms"]["C"], target["median_ms"]["A"]),
            )
        )

    print(f"physical warning threshold: {args.physical_warning_mb} MiB")
    print(
        "last measured at/above threshold: "
        + ", ".join(
            f"{key}={_profile_text(last_physical_safe[key])}"
            for key in VARIANTS
        )
    )
    print(
        "first measured below threshold: "
        + ", ".join(
            f"{key}={_profile_text(first_low[key])}"
            for key in VARIANTS
        )
    )
    print(
        "first timing-curve spill: "
        + ", ".join(
            f"{key}={_profile_text(first_spill[key])}"
            for key in VARIANTS
        )
    )
    print(
        f"largest projected under {args.budget:.1f} GB: "
        + ", ".join(
            f"{key}={_profile_text(best_projected[key])}"
            for key in VARIANTS
        )
    )

    sample_counts = [
        row["measurements"][key].physical_samples
        for row in rows
        for key in VARIANTS
        if row["measurements"][key] is not None
        and row["measurements"][key].physical_samples
    ]
    if sample_counts:
        print(
            "physical sampler: %.2f ms requested interval, %d..%d samples per variant"
            % (
                args.physical_poll_ms,
                min(sample_counts),
                max(sample_counts),
            )
        )

    if args.ab_frames.strip().lower() != "grid":
        print(
            "Selected profiles are not a ceiling search; use --ab-frames grid "
            "to sweep the ladder."
        )
    if streamed and args.calibrate_to is None:
        print(
            "Reserve is a streamed-weight floor, so projected budget lines are "
            "upper bounds. Use --calibrate-to with a known-good variant-B "
            "efficient-Sage length."
        )
