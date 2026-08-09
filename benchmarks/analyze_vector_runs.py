"""Read-only summaries and invariant checks for H3 vector diagnostics.

The module intentionally uses only the Python standard library so it can be
run from a checkout without importing ComfyUI (or allocating a model).
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_DIAGNOSTICS = Path(r"D:\AI\ComfyUI\Output\h3_vector_accel")
ADAPTIVE_PROFILE_ANCHORS = {
    "adaptive_history_v1": (tuple(range(6)), (17, 18, 19)),
    "adaptive_history_v2": (tuple(range(3)), (18, 19)),
}


def discover_root(explicit: str | os.PathLike[str] | None = None, *, environ=None,
                  checkout: str | os.PathLike[str] | None = None) -> Path:
    """Return the first usable diagnostics root, without creating anything."""
    env = os.environ if environ is None else environ
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if env.get("COMFYUI_OUTPUT_DIR"):
        candidates.append(Path(env["COMFYUI_OUTPUT_DIR"]) / "h3_vector_accel")
    candidates.append(DEFAULT_DIAGNOSTICS)
    base = Path(checkout) if checkout is not None else Path(__file__).resolve().parents[1]
    candidates.append(base / "output" / "h3_vector_accel")
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved.is_dir():
            return resolved
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No diagnostics root found; checked: {tried}")


def diagnostics_files(root: Path) -> list[Path]:
    return sorted((path for path in root.glob("*/diagnostics.json") if path.is_file()),
                  key=lambda path: (path.stat().st_mtime_ns, str(path)))


def select_files(root: Path, *, run: str = "latest", limit: int | None = None) -> list[Path]:
    files = diagnostics_files(root)
    if run != "latest":
        selected = [path for path in files if path.parent.name == run]
        if not selected:
            raise FileNotFoundError(f"Run {run!r} was not found below {root}")
        return selected
    files = list(reversed(files))
    return files[:limit] if limit is not None else files[:1]


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Diagnostics must be a JSON object: {path}")
    return value


def _first(data: dict, *keys):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _finite_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _percent(delta, base):
    base = _finite_number(base)
    if base in (None, 0):
        return None
    return delta / base * 100.0


def check_invariants(data: dict) -> dict:
    """Return named pass/fail checks; malformed fields fail closed."""
    effective = data.get("effective_sigma_sequence") or data.get("effective_schedule") or []
    finite_desc = bool(effective) and all(
        _finite_number(value) is not None for value in effective
    ) and all(float(a) > float(b) for a, b in zip(effective, effective[1:]))
    # The last comparison is intentionally separate: a terminal zero is valid,
    # while an omitted/non-zero terminal is not.
    finite_desc = finite_desc and float(effective[-1]) == 0.0
    steps = data.get("steps") or []
    anchors = data.get("anchors") or []
    actual_steps = [row for row in steps if not row.get("forecast", False)]
    true_nfe = data.get("true_nfe")
    nfe_match = (isinstance(true_nfe, (int, float)) and
                 int(true_nfe) == len(actual_steps) == len(anchors))
    profile = data.get("evaluation_profile") or data.get("configuration", {}).get("evaluation_profile")
    adaptive_anchors = True
    if profile in ADAPTIVE_PROFILE_ANCHORS:
        prefix, tail = ADAPTIVE_PROFILE_ANCHORS[profile]
        anchor_indices = [row.get("source_index") for row in anchors if row.get("actual", True)]
        adaptive_anchors = (
            anchor_indices[:len(prefix)] == list(prefix) and
            anchor_indices[-len(tail):] == list(tail) and
            len(effective) == len(anchors) + 1
        )
        if adaptive_anchors:
            source = data.get("source_sigma_sequence") or data.get("sigma_sequence") or []
            adaptive_anchors = len(source) >= 21
            if adaptive_anchors:
                expected = (list(source[:len(prefix)]) +
                            list(effective[len(prefix):-len(tail) - 1]) +
                            [source[index] for index in tail] + [0.0])
                adaptive_anchors = len(effective) == len(expected) and all(
                    math.isclose(float(actual), float(want), rel_tol=1e-6, abs_tol=1e-7)
                    for actual, want in zip(effective, expected)
                )
    no_forecasts = (int(data.get("forecast_count", 0) or 0) == 0 and
                    not any(row.get("forecast", False) for row in steps) and
                    all(value is True for value in data.get("actual_forecast_mask", [True] * len(steps))))
    no_fallbacks = int(data.get("fallback_count", 0) or 0) == 0 and not any(
        row.get("fallback_reason") for row in steps + anchors
    )
    checks = {
        "finite_strict_descending_effective_ending_zero": finite_desc,
        "true_nfe_matches_actual_steps_and_anchors": nfe_match,
        "adaptive_source_anchors": adaptive_anchors,
        "all_actual_no_hidden_forecasts": no_forecasts,
        "no_fallbacks": no_fallbacks,
    }
    return {"pass": all(checks.values()), "checks": checks}


def _decision_rows(data: dict) -> tuple[list[dict], dict]:
    decisions = data.get("adaptive_decisions") or []
    anchors = data.get("anchors") or []
    rows = []
    source = data.get("source_sigma_sequence") or data.get("sigma_sequence") or []

    def containing_index(sigma):
        value = _finite_number(sigma)
        if value is None or len(source) < 2:
            return None
        for index, (left, right) in enumerate(zip(source, source[1:])):
            if float(left) >= value > float(right):
                return index
        return len(source) - 2

    for index, decision in enumerate(decisions):
        anchor = anchors[index] if index < len(anchors) else {}
        metrics = anchor.get("trajectory_metrics") or anchor.get("metrics") or {}
        video_rate = _first(metrics, "video_rate")
        video_velocity_rate = _first(metrics, "video_velocity_rate")
        video_x0_rate = _first(metrics, "video_x0_rate")
        audio_rate = _first(metrics, "audio_rate", "audio_velocity_rate")
        video_x0 = _first(metrics, "video_x0_change", "video_x0")
        audio_x0 = _first(metrics, "audio_x0_change", "audio_x0")
        video_change = _first(metrics, "video_change", "video_velocity_change")
        audio_change = _first(metrics, "audio_change", "audio_velocity_change")
        video_score = _first(metrics, "video_score")
        audio_score = _first(metrics, "audio_score")
        rows.append({
            "actual_anchor_index": anchor.get("actual_anchor_index", index),
            "logical_step": anchor.get("step", index),
            "source_index": decision.get("source_index", anchor.get("source_index")),
            "containing_source_index": containing_index(decision.get("sigma", anchor.get("sigma"))),
            "sigma": _first(decision, "sigma") or anchor.get("sigma"),
            "next_sigma": decision.get("next_sigma", anchor.get("next_sigma")),
            "video_velocity_change": video_change,
            "audio_velocity_change": audio_change,
            "video_change": video_change,
            "audio_change": audio_change,
            "video_x0_change": video_x0,
            "audio_x0_change": audio_x0,
            "video_rate": video_rate,
            "video_velocity_rate": video_velocity_rate,
            "video_x0_rate": video_x0_rate,
            "reference_video_rate": _first(metrics, "reference_video_rate"),
            "video_rate_ratio": _first(metrics, "video_rate_ratio"),
            "audio_rate": audio_rate,
            "video_score": video_score,
            "audio_score": audio_score,
            "step_scale": decision.get("step_scale", anchor.get("step_scale")),
            "reason": decision.get("reason", anchor.get("policy_reason")),
            "protected_region": decision.get("protected_region", anchor.get("protected_region")),
        })
    constants = data.get("controller_constants") or data.get("configuration", {}).get("adaptive_controller", {}).get("constants", {})
    reference_anchors = int(constants.get(
        "reference_anchors", constants.get("bootstrap_anchors", constants.get("protected_prefix", 6))
    ) or 0)
    rates = [row["video_rate"] for row in rows[:reference_anchors] if row["video_rate"] is not None]
    audio_rates = [row["audio_rate"] for row in rows[:reference_anchors] if row["audio_rate"] is not None]
    window = int(constants.get("reference_intervals", 3) or 3)
    reference = {
        "video_rate": sum(rates[-window:]) / len(rates[-window:]) if rates[-window:] else None,
        "audio_rate": sum(audio_rates[-window:]) / len(audio_rates[-window:]) if audio_rates[-window:] else None,
        "low_ratio": constants.get("low_change_ratio"),
        "high_ratio": constants.get("high_change_ratio"),
    }
    if reference["video_rate"] is not None:
        reference["low_video_threshold"] = reference["video_rate"] * (reference["low_ratio"] or 0)
        reference["high_video_threshold"] = reference["video_rate"] * (reference["high_ratio"] or 0)
    if reference["audio_rate"] is not None:
        reference["audio_emergency_threshold"] = reference["audio_rate"] * (constants.get("audio_emergency_multiplier") or 0)
    return rows, reference


def summarize(path: Path) -> dict:
    data = _read(path)
    configuration = data.get("configuration") if isinstance(data.get("configuration"), dict) else {}
    summary = {
        "run_id": data.get("run_id", path.parent.name),
        "path": str(path),
        "mtime": path.stat().st_mtime,
        "method": data.get("method"),
        "profile": data.get("evaluation_profile") or data.get("profile"),
        "true_nfe": data.get("true_nfe"),
        "nominal_steps": data.get("nominal_steps", len(data.get("steps") or [])),
        "forecast_count": data.get("forecast_count", 0),
        "fallback_count": data.get("fallback_count", 0),
        "source_sequence": data.get("source_sigma_sequence") or data.get("sigma_sequence"),
        "effective_sequence": data.get("effective_sigma_sequence"),
        "source_hash": data.get("source_sigma_hash") or data.get("sigma_hash"),
        "effective_hash": data.get("effective_sigma_hash"),
        "configuration_fingerprint": data.get("configuration_fingerprint"),
        "model_fingerprint": data.get("model_fingerprint"),
        "sampler": _first(data, "sampler_name", "sampler", "scheduler") or
                   _first(configuration, "sampler_name", "sampler", "scheduler"),
        "wall_seconds": _first(data, "wall_seconds", "elapsed_seconds"),
        "elapsed_seconds": data.get("elapsed_seconds"),
        "model_call_seconds": data.get("model_call_seconds"),
        "sampler_overhead_seconds": data.get("sampler_overhead_seconds"),
    }
    summary["source_sigma_sequence"] = summary["source_sequence"]
    summary["effective_sigma_sequence"] = summary["effective_sequence"]
    summary["source_sigma_hash"] = summary["source_hash"]
    summary["effective_sigma_hash"] = summary["effective_hash"]
    summary["invariants"] = check_invariants(data)
    if summary["profile"] in ADAPTIVE_PROFILE_ANCHORS:
        summary["decisions"], summary["decision_reference"] = _decision_rows(data)
    return summary


def compare_runs(target: dict, reference: dict) -> dict:
    result = {"label": "raw/not automatically comparable", "target": target["run_id"], "reference": reference["run_id"]}
    for field in ("true_nfe", "wall_seconds", "elapsed_seconds", "model_call_seconds"):
        a, b = _finite_number(target.get(field)), _finite_number(reference.get(field))
        if a is not None and b is not None:
            result[field] = {"delta": a - b, "percent": _percent(a - b, b)}
    return result


def _loopback(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme in ("http", "https") and parsed.hostname in ("127.0.0.1", "localhost", "::1")


def _get_json(base: str, endpoint: str):
    request = urllib.request.Request(base.rstrip("/") + endpoint, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def correlate_history(summary: dict, server: str, diagnostics_root: Path) -> dict:
    if not _loopback(server):
        raise ValueError("--server must be a loopback http(s) URL")
    stats = _get_json(server, "/api/system_stats")
    system = stats.get("system", stats) if isinstance(stats, dict) else {}
    argv = system.get("argv") if isinstance(system, dict) else None
    output = None
    for key in ("output_directory", "output_dir"):
        output = system.get(key) or (stats.get(key) if isinstance(stats, dict) else None)
        if output:
            break
    expected = diagnostics_root.parent.resolve()
    argv_values = [str(value) for value in (argv or [])]
    argv_output = None
    for index, value in enumerate(argv_values[:-1]):
        if value in ("--output-directory", "--output_dir"):
            argv_output = argv_values[index + 1]
            break
    if argv_output is None:
        for value in argv_values:
            if value.startswith("--output-directory=") or value.startswith("--output_dir="):
                argv_output = value.split("=", 1)[1]
                break
    reported_output = argv_output or output
    if not argv or not reported_output or Path(reported_output).expanduser().resolve() != expected:
        raise RuntimeError("server identity mismatch; diagnostics-only (argv/output directory not verified)")
    queue = _get_json(server, "/api/queue")
    history = _get_json(server, "/api/history?max_items=100")
    mtime = float(summary["mtime"])
    matches = []
    for prompt_id, item in (history.items() if isinstance(history, dict) else []):
        if not isinstance(item, dict):
            continue
        status = item.get("status", {})
        if status.get("status_str") not in ("success", "completed"):
            continue
        times = []
        lifecycle = {}
        for event in (status.get("messages") or []):
            if isinstance(event, (list, tuple)) and len(event) > 1 and isinstance(event[1], dict):
                stamp = event[1].get("timestamp")
                try:
                    stamp = float(stamp)
                    if stamp > 1e11:
                        stamp /= 1000.0
                    lifecycle[str(event[0])] = stamp
                except (TypeError, ValueError):
                    pass
        if "execution_start" in lifecycle and "execution_success" in lifecycle:
            times.extend((lifecycle["execution_start"], lifecycle["execution_success"]))
        for key in ("timestamp", "start_time", "end_time", "completed", "completed_at", "created_at"):
            value = status.get(key, item.get(key))
            try:
                stamp = float(value)
                if stamp > 1e11:
                    stamp /= 1000.0
                times.append(stamp)
            except (TypeError, ValueError):
                if isinstance(value, str):
                    try:
                        parsed = _datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
                        times.append(parsed.timestamp())
                    except ValueError:
                        pass
        if times and min(times) <= mtime <= max(times):
            matches.append((abs(mtime - max(times)), prompt_id, item))
    if not matches:
        raise RuntimeError("no successful history execution contains diagnostics mtime; diagnostics-only")
    _, prompt_id, item = sorted(matches)[0]
    prompt = item.get("prompt")
    nodes = prompt[2] if isinstance(prompt, list) and len(prompt) > 2 else prompt if isinstance(prompt, dict) else {}
    sampler_inputs = {}
    sampler_node = None
    if isinstance(nodes, dict):
        for node_id, node in nodes.items():
            inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
            class_type = str(node.get("class_type", "")) if isinstance(node, dict) else ""
            vector_node = ("vector" in class_type.lower() and "accel" in class_type.lower())
            vector_node = vector_node or ("method" in inputs and "evaluation_profile" in inputs)
            if vector_node:
                sampler_inputs[node_id] = inputs
                sampler_node = {"node_id": node_id, "class_type": class_type, "inputs": inputs}
    return {"prompt_id": prompt_id, "status": item.get("status", {}), "queue": queue,
            "sampler_node": sampler_node,
            "sampler_inputs": sampler_inputs,
            "outputs": item.get("outputs", {})}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-root")
    parser.add_argument("--list", type=int, metavar="N")
    parser.add_argument("--run", default="latest", metavar="latest|RUN_ID")
    parser.add_argument("--compare", metavar="RUN_ID")
    parser.add_argument("--decisions", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--server")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = discover_root(args.diagnostics_root)
        selected = select_files(root, run=args.run, limit=args.list)
        summaries = [summarize(path) for path in selected]
        if args.compare:
            reference_path = select_files(root, run=args.compare)[0]
            summaries[0]["comparison"] = compare_runs(summaries[0], summarize(reference_path))
        if args.server and summaries:
            try:
                summaries[0]["history"] = correlate_history(summaries[0], args.server, root)
            except (ValueError, RuntimeError, urllib.error.URLError) as exc:
                summaries[0]["server_error"] = str(exc)
        if args.format == "json":
            print(json.dumps(summaries if args.list is not None else summaries[0], sort_keys=True, indent=2, allow_nan=False))
        else:
            for summary in summaries:
                print(f"{summary['run_id']} {summary.get('method')} / {summary.get('profile')} NFE={summary.get('true_nfe')} wall={summary.get('wall_seconds')}s")
                print("  invariants: " + ("PASS" if summary["invariants"]["pass"] else "FAIL"))
                if summary.get("comparison"):
                    comparison = summary["comparison"]
                    print(f"  comparison={comparison['reference']} ({comparison['label']})")
                    for field in ("true_nfe", "wall_seconds", "elapsed_seconds", "model_call_seconds"):
                        delta = comparison.get(field)
                        if delta:
                            percent = delta["percent"]
                            percent_text = "n/a" if percent is None else f"{percent:.3f}%"
                            print(f"    {field}: delta={delta['delta']:.6g} percent={percent_text}")
                if summary.get("history"):
                    history = summary["history"]
                    queue = history.get("queue", {})
                    print(f"  prompt={history.get('prompt_id')} queue_running={len(queue.get('queue_running', []))} queue_pending={len(queue.get('queue_pending', []))}")
                    if history.get("sampler_node"):
                        print(f"  sampler_node={history['sampler_node'].get('node_id')} class={history['sampler_node'].get('class_type')}")
                    print(f"  outputs={history.get('outputs', {})}")
                elif summary.get("server_error"):
                    print(f"  server: {summary['server_error']}")
                if args.decisions:
                    reference = summary.get("decision_reference", {})
                    if reference:
                        print(
                            "  reference: "
                            f"video_rate={reference.get('video_rate')} "
                            f"low={reference.get('low_video_threshold')} "
                            f"high={reference.get('high_video_threshold')} "
                            f"audio_rate={reference.get('audio_rate')} "
                            f"audio_emergency={reference.get('audio_emergency_threshold')}"
                        )
                    for row in summary.get("decisions", []):
                        print(
                            f"  anchor {row['actual_anchor_index']} step={row['logical_step']} "
                            f"source={row['source_index']} containing={row['containing_source_index']} "
                            f"{row['sigma']} -> {row['next_sigma']} scale={row['step_scale']} "
                            f"video_rate={row['video_rate']} audio_rate={row['audio_rate']} "
                            f"video_x0={row['video_x0_change']} audio_x0={row['audio_x0_change']} "
                            f"{row['reason']} region={row['protected_region']}"
                        )
        return 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"analyze_vector_runs: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
