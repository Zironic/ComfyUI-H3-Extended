"""CPU tests for the read-only vector-run analyzer."""

import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import h3_test_tempfile as tempfile  # noqa: E402
from benchmarks.analyze_vector_runs import (  # noqa: E402
    check_invariants,
    compare_runs,
    correlate_history,
    discover_root,
    select_files,
    summarize,
)


def _run(root, run_id="run-a", **updates):
    run = {
        "run_id": run_id,
        "method": "res_multistep",
        "evaluation_profile": "adaptive_history_v1",
        "true_nfe": 3,
        "forecast_count": 0,
        "fallback_count": 0,
        "source_sigma_sequence": [1, .8, .6, .4, .2, 0],
        "effective_sigma_sequence": [1, .8, .4, 0],
        "steps": [{"forecast": False}, {"forecast": False}, {"forecast": False}],
        "anchors": [{"actual": True, "source_index": i} for i in [0, 1, 2]],
        "adaptive_decisions": [],
        "elapsed_seconds": 10.0,
        "wall_seconds": 10.0,
    }
    run.update(updates)
    path = Path(root) / run_id / "diagnostics.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(run), encoding="utf-8")
    return path


class AnalyzeVectorRunsTests(unittest.TestCase):
    def test_root_discovery_and_latest_selection(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "output"
            diag = output / "h3_vector_accel"
            first = _run(diag, "old")
            second = _run(diag, "new")
            os.utime(first, (1, 1))
            os.utime(second, (2, 2))
            found = discover_root(environ={"COMFYUI_OUTPUT_DIR": str(output)}, checkout=root)
            self.assertEqual(found, diag.resolve())
            self.assertEqual(select_files(found)[0].parent.name, "new")
            self.assertEqual(select_files(found, run="old")[0], first)

    def test_invariant_pass_and_failure(self):
        good = {
            "true_nfe": 3,
            "effective_sigma_sequence": [1, .8, .2, 0],
            "steps": [{"forecast": False}] * 3,
            "anchors": [{"actual": True, "source_index": i} for i in [0, 1, 2]],
            "fallback_count": 0,
        }
        self.assertTrue(check_invariants(good)["pass"])
        bad = dict(good, effective_sigma_sequence=[1, .8, .8, 0], forecast_count=1)
        bad["steps"] = [{"forecast": True}] + good["steps"][1:]
        self.assertFalse(check_invariants(bad)["pass"])

    def test_adaptive_reference_and_decision_fields(self):
        with tempfile.TemporaryDirectory() as root:
            path = _run(
                root,
                evaluation_profile="adaptive_history_v2",
                effective_sigma_sequence=[1, .8, .4, 0],
                adaptive_decisions=[{"sigma": 1, "next_sigma": .8, "reason": "bootstrap", "step_scale": 1}],
                anchors=[{"actual": True, "source_index": 0, "step": 0, "sigma": 1,
                          "trajectory_metrics": {"video_rate": 2, "audio_rate": 4}}],
                controller_constants={"bootstrap_anchors": 4, "reference_intervals": 1, "low_change_ratio": .75,
                                     "high_change_ratio": 1.5, "audio_emergency_multiplier": 4},
            )
            summary = summarize(path)
            self.assertEqual(summary["profile"], "adaptive_history_v2")
            self.assertEqual(summary["decisions"][0]["video_rate"], 2)
            self.assertEqual(summary["decisions"][0]["audio_rate"], 4)
            self.assertEqual(summary["decision_reference"]["low_video_threshold"], 1.5)

    def test_v2_invariants_require_bootstrap_and_two_anchor_tail(self):
        source = [1.0 - index * 0.04 for index in range(20)] + [0.0]
        effective = source[:4] + [0.7, source[18], source[19], 0.0]
        anchors = [
            {"actual": True, "source_index": index}
            for index in (0, 1, 2, 3)
        ] + [
            {"actual": True, "source_index": None},
            {"actual": True, "source_index": 18},
            {"actual": True, "source_index": 19},
        ]
        data = {
            "evaluation_profile": "adaptive_history_v2",
            "true_nfe": len(anchors),
            "forecast_count": 0,
            "fallback_count": 0,
            "source_sigma_sequence": source,
            "effective_sigma_sequence": effective,
            "steps": [{"forecast": False}] * len(anchors),
            "anchors": anchors,
        }
        self.assertTrue(check_invariants(data)["pass"])
        data["anchors"][-2]["source_index"] = 17
        self.assertFalse(check_invariants(data)["checks"]["adaptive_source_anchors"])

    def test_comparison_is_explicit_and_raw(self):
        result = compare_runs({"run_id": "a", "true_nfe": 12, "wall_seconds": 90},
                              {"run_id": "b", "true_nfe": 10, "wall_seconds": 100})
        self.assertEqual(result["label"], "raw/not automatically comparable")
        self.assertEqual(result["true_nfe"]["delta"], 2)
        self.assertEqual(result["wall_seconds"]["percent"], -10)

    def test_loopback_rejection_and_history_correlation(self):
        with tempfile.TemporaryDirectory() as root:
            path = _run(root)
            os.utime(path, (1000000000, 1000000000))
            summary = summarize(path)
            with self.assertRaises(ValueError):
                correlate_history(summary, "http://example.com:8188", Path(root))
            responses = {
                "/api/system_stats": {"system": {"argv": [str(Path(root) / "main.py"),
                    "--output-directory", str(Path(root).resolve())]}},
                "/api/queue": {"queue_running": [], "queue_pending": []},
                "/api/history?max_items=100": {"prompt-1": {"status": {"status_str": "success",
                    "messages": [["execution_start", {"timestamp": 900000000000}],
                                  ["execution_success", {"timestamp": 1100000000000}]]},
                    "prompt": ["x", "y", {"28": {"class_type": "MiniMaxH3VectorAccelSamplerZi",
                        "inputs": {"method": "res_multistep", "evaluation_profile": "adaptive_history_v1"}}}],
                    "outputs": {"9": {"images": [{"filename": "out.png"}]}}}},
            }
            with mock.patch("benchmarks.analyze_vector_runs._get_json", side_effect=lambda _, endpoint: responses[endpoint]):
                result = correlate_history(summary, "http://127.0.0.1:8188", Path(root) / "h3_vector_accel")
            self.assertEqual(result["prompt_id"], "prompt-1")
            self.assertIn("28", result["sampler_inputs"])
            self.assertEqual(result["sampler_node"]["inputs"]["method"], "res_multistep")


if __name__ == "__main__":
    unittest.main()
