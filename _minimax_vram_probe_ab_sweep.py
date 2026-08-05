"""Calibration, measured-residency tables, and budget projections for the A/B probe."""

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


def run(args, parser, arch, dtype, device, reserve_gb, streamed,
        baseline_forward, candidate_forward):
    threshold_bytes = int(args.physical_warning_mb) * MIB

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
        measured, _ = measure_variant(baseline_forward, frames, layout)
        if measured is None:
            print(
                f"cannot calibrate: efficient-Sage baseline OOMs at {frames} "
                "frames. Free the card and retry."
            )
            return
        solved = (
            args.budget
            - (measured.peak_bytes + latent + condition) / base.GB
        )
        print(
            f"calibration {frames} frames is known to fit {args.budget:.1f} GB on the "
            f"efficient-Sage baseline, so the shared reserve is {solved:.2f} GB"
        )
        if solved < 0:
            print(
                "            negative reserve: supplied budget/known-good "
                "length is inconsistent"
            )
            return
        print(
            f"            reserve {reserve_gb:.2f} GB -> {solved:.2f} GB\n"
        )
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
        "measured isolated block residency "
        f"(physical free from cudaMemGetInfo; LOW < {args.physical_warning_mb} MiB)"
    )
    print(
        f"{'frames':>7} {'tokens':>9} {'base tr':>8} {'act tr':>8} "
        f"{'saved':>8} {'B free':>7} {'A free':>7} "
        f"{'base ms':>8} {'act ms':>8} {'time':>7}  residency"
    )

    best_base_projected = best_candidate_projected = None
    last_base_physical_safe = last_candidate_physical_safe = None
    first_base_low = first_candidate_low = None
    first_base_spill = first_candidate_spill = None
    base_resident, candidate_resident, rows = [], [], []

    for frames in profiles:
        layout, latent, condition = shared_cost(frames)
        baseline, _ = measure_variant(baseline_forward, frames, layout)
        candidate, _ = measure_variant(candidate_forward, frames, layout)

        base_trans = baseline.peak_bytes if baseline is not None else None
        base_ms = baseline.median_ms if baseline is not None else None
        act_trans = candidate.peak_bytes if candidate is not None else None
        act_ms = candidate.median_ms if candidate is not None else None

        base_total = (
            reserve_gb + (base_trans + latent + condition) / base.GB
            if base_trans is not None
            else None
        )
        act_total = (
            reserve_gb + (act_trans + latent + condition) / base.GB
            if act_trans is not None
            else None
        )
        if base_total is not None and base_total <= args.budget:
            best_base_projected = frames
        if act_total is not None and act_total <= args.budget:
            best_candidate_projected = frames

        saved = (
            base_trans - act_trans
            if base_trans is not None and act_trans is not None
            else None
        )
        slowdown = (
            (act_ms / base_ms - 1.0) * 100.0
            if base_ms is not None and act_ms is not None
            else None
        )

        base_phys_safe = bool(
            baseline is not None
            and baseline.physical_free_min >= threshold_bytes
        )
        act_phys_safe = bool(
            candidate is not None
            and candidate.physical_free_min >= threshold_bytes
        )
        if base_phys_safe:
            last_base_physical_safe = frames
        elif baseline is not None and first_base_low is None:
            first_base_low = frames
        if act_phys_safe:
            last_candidate_physical_safe = frames
        elif candidate is not None and first_candidate_low is None:
            first_candidate_low = frames

        sequence = layout["seq_len"]
        base_spill = act_spill = False
        if base_ms is not None:
            base_spill, _ = update_resident_fit(
                base_resident,
                sequence,
                base_ms,
                args.spill_ratio,
                include_in_fit=base_phys_safe,
            )
            if base_spill and first_base_spill is None:
                first_base_spill = frames
        if act_ms is not None:
            act_spill, _ = update_resident_fit(
                candidate_resident,
                sequence,
                act_ms,
                args.spill_ratio,
                include_in_fit=act_phys_safe,
            )
            if act_spill and first_candidate_spill is None:
                first_candidate_spill = frames

        base_state = _variant_state(baseline, base_spill, threshold_bytes)
        act_state = _variant_state(candidate, act_spill, threshold_bytes)
        saved_text = "    n/a" if saved is None else f"{saved / base.GB:>7.3f}G"
        time_text = "    n/a" if slowdown is None else f"{slowdown:>+6.1f}%"
        print(
            f"{frames:>7} {sequence:>9} {fmt_gib(base_trans)} {fmt_gib(act_trans)} "
            f"{saved_text} "
            f"{fmt_mib(baseline.physical_free_min if baseline else None)} "
            f"{fmt_mib(candidate.physical_free_min if candidate else None)} "
            f"{fmt_ms(base_ms)} {fmt_ms(act_ms)} {time_text}  "
            f"B:{base_state} A:{act_state}"
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
            "base_total": base_total,
            "activation_total": act_total,
            "base_physical_free_min": (
                baseline.physical_free_min if baseline else None
            ),
            "activation_physical_free_min": (
                candidate.physical_free_min if candidate else None
            ),
            "base_physical_free_start": (
                baseline.physical_free_start if baseline else None
            ),
            "activation_physical_free_start": (
                candidate.physical_free_start if candidate else None
            ),
            "base_physical_samples": (
                baseline.physical_samples if baseline else 0
            ),
            "activation_physical_samples": (
                candidate.physical_samples if candidate else 0
            ),
            "base_spill": base_spill,
            "activation_spill": act_spill,
        })

        if base_spill and act_spill and not args.past_spill:
            print(
                "\nstopping: both variants left their resident timing curves; "
                "pass --past-spill to continue"
            )
            break
        if baseline is None and candidate is None:
            print("\nstopping: both variants hard-OOM at this profile")
            break

    print()
    print(
        "projected production budget "
        "(arithmetic only: calibrated reserve is not allocated in this probe)"
    )
    print(
        f"{'frames':>7} {'base projected':>15} {'act projected':>15}  budget state"
    )
    for row in rows:
        base_total = row["base_total"]
        act_total = row["activation_total"]
        base_fits = base_total is not None and base_total <= args.budget
        act_fits = act_total is not None and act_total <= args.budget
        if base_fits and act_fits:
            state = "both fit"
        elif act_fits and not base_fits:
            state = "activation only"
        elif base_fits and not act_fits:
            state = "baseline only"
        else:
            state = "both over"
        base_text = "OOM" if base_total is None else f"{base_total:.3f} GiB"
        act_text = "OOM" if act_total is None else f"{act_total:.3f} GiB"
        print(f"{row['frames']:>7} {base_text:>15} {act_text:>15}  {state}")

    print()
    paired = [row for row in rows if row["saved"] is not None]
    if paired:
        target = max(paired, key=lambda row: row["frames"])
        print(
            "largest paired profile: C=%d S=%d saves %.3f GiB "
            "(%.1f%% of baseline), time delta %+.1f%%"
            % (
                target["frames"],
                target["sequence"],
                target["saved"] / base.GB,
                100.0 * target["saved"] / target["base_transient"],
                target["slowdown_percent"],
            )
        )

    last_tested = rows[-1]["frames"] if rows else None
    print(f"physical warning threshold: {args.physical_warning_mb} MiB")
    print(
        "last measured at/above threshold: baseline=%s, activation=%s"
        % (
            _profile_text(last_base_physical_safe),
            _profile_text(last_candidate_physical_safe),
        )
    )
    print(
        "first measured below threshold: baseline=%s, activation=%s"
        % (
            _profile_text(first_base_low),
            _profile_text(first_candidate_low),
        )
    )
    print(
        "first timing-curve spill: baseline=%s, activation=%s"
        % (
            _profile_text(first_base_spill),
            _profile_text(first_candidate_spill),
        )
    )
    print(
        "largest projected under %.1f GB: baseline=%s, activation=%s"
        % (
            args.budget,
            _profile_text(best_base_projected),
            _profile_text(best_candidate_projected),
        )
    )
    if rows:
        base_all_safe = first_base_low is None and all(
            row["base_physical_free_min"] is not None for row in rows
        )
        act_all_safe = first_candidate_low is None and all(
            row["activation_physical_free_min"] is not None for row in rows
        )
        if base_all_safe:
            print(
                "baseline physical-free floor stayed above threshold through "
                f"C={last_tested}"
            )
        if act_all_safe:
            print(
                "activation physical-free floor stayed above threshold through "
                f"C={last_tested}"
            )

    sample_counts = [
        count
        for row in rows
        for count in (
            row["base_physical_samples"],
            row["activation_physical_samples"],
        )
        if count
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
            "upper bounds. Use --calibrate-to with a known-good efficient-Sage "
            "baseline length."
        )
