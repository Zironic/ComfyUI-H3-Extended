"""Calibration, measured-residency tables, and budget projections for arm matrices."""

from pathlib import Path

import _minimax_vram_probe_base as base
from _minimax_vram_probe_ab_cli import layout_for, parse_profiles, selected_arms
from _minimax_vram_probe_ab_runtime import (
    MIB,
    fmt_gib,
    fmt_ms,
    safe_measure,
    update_resident_fit,
)

def resolve_checkpoint(value):
    candidate = Path(value)
    if candidate.is_absolute():
        if candidate.suffix.lower() != ".safetensors" or not candidate.is_file():
            raise FileNotFoundError(str(candidate))
        return str(candidate.resolve())
    import folder_paths
    resolved = Path(folder_paths.get_full_path_or_raise("diffusion_models", value))
    if resolved.suffix.lower() != ".safetensors" or not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return str(resolved.resolve())


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
    forwards,
):
    threshold_bytes = int(args.physical_warning_mb) * MIB
    variants = selected_arms(args)
    variants = tuple(label for label in variants if label in forwards)
    if not variants:
        raise ValueError("arm selector produced no forwards")

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
        calibration_arm = args.calibrate_arm or variants[0]
        if calibration_arm not in forwards:
            parser.error("--calibrate-arm must select one selected arm")
        measured, _ = measure_variant(forwards[calibration_arm], frames, layout)
        if measured is None:
            print(
                f"cannot calibrate: arm {calibration_arm} OOMs at {frames} "
                "frames. Free the card and retry."
            )
            return
        solved = (
            args.budget
            - (measured.peak_bytes + latent + condition) / base.GB
        )
        print(
            f"calibration {frames} frames is known to fit {args.budget:.1f} GB on "
            f"arm {calibration_arm}, so the shared reserve is {solved:.2f} GB"
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
    print("frames  tokens  " + "  ".join("%s tr" % key for key in variants) + "  residency")

    best_projected = {key: None for key in variants}
    last_physical_safe = {key: None for key in variants}
    first_low = {key: None for key in variants}
    first_spill = {key: None for key in variants}
    resident_points = {key: [] for key in variants}
    retired = {}
    rows = []

    for frames in profiles:
        retired_before = set(retired)
        layout, latent, condition = shared_cost(frames)
        measurements = {}
        for key in variants:
            if key in retired:
                continue
            measurements[key], _ = measure_variant(
                forwards[key], frames, layout
            )

        transient = {
            key: (
                measurements.get(key).peak_bytes
                if measurements.get(key) is not None
                else None
            )
            for key in variants
        }
        median_ms = {
            key: (
                measurements.get(key).median_ms
                if measurements.get(key) is not None
                else None
            )
            for key in variants
        }
        totals = {
            key: (
                reserve_gb + (transient[key] + latent + condition) / base.GB
                if transient[key] is not None
                else None
            )
            for key in variants
        }
        for key in variants:
            if key in retired:
                continue
            if totals[key] is not None and totals[key] <= args.budget:
                best_projected[key] = frames

        phys_safe = {}
        spills = {}
        states = {}
        sequence = layout["seq_len"]
        for key in tuple(variants):
            if key in retired_before:
                phys_safe[key] = False
                spills[key] = False
                reason, retired_at = retired[key]
                states[key] = "retired:%s@%d" % (reason, retired_at)
                continue
            measurement = measurements.get(key)
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
            if measurement is None:
                retired.setdefault(key, ("OOM", frames))
            elif not phys_safe[key]:
                retired.setdefault(key, ("LOW", frames))
            elif spills[key]:
                retired.setdefault(key, ("SPILL", frames))

        print(
            f"{frames:>7} {sequence:>9} "
            + " ".join("%s:%s/%s" % (
                key,
                "  SKIP" if key in retired_before else fmt_gib(transient[key]),
                states.get(key, "retired"),
            )
                       for key in variants)
        )

        rows.append({
            "frames": frames,
            "sequence": sequence,
            "measurements": measurements,
            "transient": transient,
            "median_ms": median_ms,
            "totals": totals,
            "spills": spills,
            "states": states,
            "retired_before": retired_before,
        })

        if all(key in retired for key in variants):
            print("\nstopping: all selected arms retired")
            break

    print()
    print("measured isolated block timing (same profiles; median of timed forwards)")
    print(
        f"{'frames':>7} " + " ".join(f"{key + ' ms':>16}" for key in variants)
    )
    for row in rows:
        times = row["median_ms"]
        print(
            f"{row['frames']:>7} " + " ".join(
                f"{('skip' if key in row['retired_before'] else fmt_ms(times[key])):>16}"
                for key in variants
            )
        )

    print()
    print(
        "projected production budget "
        "(arithmetic only: calibrated reserve is not allocated in this probe)"
    )
    print(
        f"{'frames':>7} " + " ".join(f"{key + ' projected':>15}" for key in variants)
    )
    for row in rows:
        totals = row["totals"]
        texts = {
            key: (
                "retired"
                if key in row["retired_before"]
                else "OOM" if totals[key] is None else f"{totals[key]:.3f} GiB"
            )
            for key in variants
        }
        state = " ".join(
            f"{key}:" + (
                "retired"
                if key in row["retired_before"]
                else _budget_state(totals[key], args.budget)
            )
            for key in variants
        )
        print(
            f"{row['frames']:>7} " + " ".join(f"{texts[key]:>15}" for key in variants) + f"  {state}"
        )

    print()
    complete = [
        row for row in rows
        if all(row["measurements"].get(key) is not None for key in variants)
    ]
    if complete:
        target = max(complete, key=lambda row: row["frames"])
        print("largest complete profile: frames=%d tokens=%d" %
              (target["frames"], target["sequence"]))

    print(f"physical warning threshold: {args.physical_warning_mb} MiB")
    print(
        "last measured at/above threshold: "
        + ", ".join(
            f"{key}={_profile_text(last_physical_safe[key])}"
            for key in variants
        )
    )
    print(
        "first measured below threshold: "
        + ", ".join(
            f"{key}={_profile_text(first_low[key])}"
            for key in variants
        )
    )
    print(
        "first timing-curve spill: "
        + ", ".join(
            f"{key}={_profile_text(first_spill[key])}"
            for key in variants
        )
    )
    print(
        f"largest projected under {args.budget:.1f} GB: "
        + ", ".join(
            f"{key}={_profile_text(best_projected[key])}"
            for key in variants
        )
    )

    sample_counts = [
        row["measurements"].get(key).physical_samples
        for row in rows
        for key in variants
        if row["measurements"].get(key) is not None
        and row["measurements"].get(key).physical_samples
    ]
    if sample_counts:
        print(
            "physical sampler: %.2f ms requested interval, %d..%d samples per arm"
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
            "upper bounds. Use --calibrate-to with a known-good selected-arm "
            "length."
        )
