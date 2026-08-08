"""Offline NumPy-only analysis of saved H3 active-mask dynamics."""

import argparse
import json
import os

import numpy as np


THRESHOLDS_PERCENT = (0.5, 1, 2, 3, 5, 7.5, 10)


def _dilate_spatial(mask, halo):
    if halo <= 0:
        return mask
    t, h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-halo, halo + 1):
        for dx in range(-halo, halo + 1):
            ys, ye = max(0, dy), min(h, h + dy)
            xs, xe = max(0, dx), min(w, w + dx)
            out[:, ys:ye, xs:xe] |= mask[:, ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def _dilate_temporal(mask, halo):
    if halo <= 0:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    for dt in range(-halo, halo + 1):
        ts, te = max(0, dt), min(mask.shape[0], mask.shape[0] + dt)
        out[ts:te] |= mask[ts - dt:te - dt]
    return out


def _to_tiles(mask, tile):
    if tile == 1:
        return mask.copy()
    t, h, w = mask.shape
    th, tw = (h + tile - 1) // tile, (w + tile - 1) // tile
    padded = np.pad(mask, ((0, 0), (0, th * tile - h), (0, tw * tile - w)))
    return padded.reshape(t, th, tile, tw, tile).any(axis=(2, 4))


def _from_tiles(mask, tile, height, width):
    if tile == 1:
        return mask[:, :height, :width].copy()
    return np.repeat(np.repeat(mask, tile, axis=1), tile, axis=2)[:, :height, :width]


def _mask(activity, threshold, tile, spatial, temporal):
    active = activity > threshold
    tiled = _to_tiles(active, tile)
    tiled = _dilate_temporal(_dilate_spatial(tiled, spatial), temporal)
    return _from_tiles(tiled, tile, activity.shape[1], activity.shape[2])


def analyze(path):
    data = np.load(path)
    indices = np.asarray(data.get("index", []), dtype=np.int64)
    order = np.argsort(indices)
    activities, energies = [], []
    for i in order:
        label = "%04d_step%d" % (int(indices[i]), int(data["step"][i]))
        ak, ek = "activity_" + label, "energy_" + label
        if ak in data and ek in data:
            activities.append(np.asarray(data[ak], dtype=np.float32))
            energies.append(np.asarray(data[ek], dtype=np.float32))
    rows = []
    for pct in THRESHOLDS_PERCENT:
        threshold = pct / 100.0
        for tile in (1, 2, 4):
            for spatial in (0, 1, 2):
                for temporal in (0, 1, 2):
                    token_fracs, retained = [], []
                    for i in range(max(0, len(activities) - 1)):
                        predicted = _mask(activities[i], threshold, tile, spatial, temporal)
                        energy = energies[i + 1]
                        total = float(energy.sum())
                        token_fracs.append(float(predicted.mean()))
                        retained.append(1.0 if total == 0 else float(energy[predicted].sum()) / total)
                    rows.append({
                        "threshold_percent": float(pct), "tile": tile,
                        "spatial_halo": spatial, "temporal_halo": temporal,
                        "transitions": len(token_fracs),
                        "mean_token_fraction": float(np.mean(token_fracs)) if token_fracs else None,
                        "mean_next_step_energy_retained": float(np.mean(retained)) if retained else None,
                    })
    valid = [r for r in rows if r["mean_token_fraction"] is not None]
    frontier = []
    for row in valid:
        dominated = any(
            other["mean_token_fraction"] <= row["mean_token_fraction"] and
            other["mean_next_step_energy_retained"] >= row["mean_next_step_energy_retained"] and
            (other["mean_token_fraction"] < row["mean_token_fraction"] or
             other["mean_next_step_energy_retained"] > row["mean_next_step_energy_retained"])
            for other in valid
        )
        if not dominated:
            frontier.append(row)
    return {"source": os.path.abspath(path), "configurations": rows,
            "pareto_frontier": sorted(frontier, key=lambda r: r["mean_token_fraction"])}


def write_analysis(path, result=None):
    result = analyze(path) if result is None else result
    output_dir = os.path.dirname(os.path.abspath(path))
    json_path = os.path.join(output_dir, "active_mask_analysis.json")
    text_path = os.path.join(output_dir, "active_mask_analysis.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("H3 ACTIVE MASK ANALYSIS\n")
        f.write("transitions: %d\n\n" % (result["configurations"][0]["transitions"] if result["configurations"] else 0))
        f.write("PARETO FRONTIER (mean token fraction, next-step energy retained)\n")
        for row in result["pareto_frontier"]:
            f.write("  %5g%% tile=%d spatial=%d temporal=%d | tokens %.4f | energy %.4f\n" %
                    (row["threshold_percent"], row["tile"], row["spatial_halo"], row["temporal_halo"],
                     row["mean_token_fraction"], row["mean_next_step_energy_retained"]))
    return json_path, text_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("latent_dynamics", help="Path to latent_dynamics.npz")
    args = ap.parse_args()
    result = analyze(args.latent_dynamics)
    write_analysis(args.latent_dynamics, result)
    return result


if __name__ == "__main__":
    main()
