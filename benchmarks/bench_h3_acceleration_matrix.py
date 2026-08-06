"""Render standalone and marginal H3 acceleration results from JSON files.

Pass entries as ``LABEL=path.json``. Files may be complete benchmark output or a
small object containing ``milliseconds``, ``peak_gib`` and quality fields.
"""

import argparse
import json


def nested(value, *paths):
    for path in paths:
        current = value
        try:
            for key in path.split("."):
                current = current[key]
            return current
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()

    rows = []
    for spec in args.results:
        if "=" not in spec:
            raise SystemExit("result must be LABEL=path.json")
        label, path = spec.split("=", 1)
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        rows.append(
            {
                "label": label,
                "ms": nested(value, "milliseconds", "median_ms", "sol_ms", "custom_ms"),
                "peak_gib": nested(value, "peak_gib", "custom_peak_gib", "sol_peak_gib"),
                "psnr_db": nested(value, "video.psnr_db", "psnr_db"),
                "audio_mse": nested(value, "audio.mse", "audio_mse"),
                "source": path,
            }
        )

    base_ms = rows[0]["ms"]
    for row in rows:
        row["speedup_vs_first"] = (
            None
            if base_ms in (None, 0) or row["ms"] in (None, 0)
            else base_ms / row["ms"]
        )
    print(json.dumps({"baseline": rows[0]["label"], "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
