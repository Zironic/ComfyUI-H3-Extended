"""CPU tests for the read-only vector-run analyzer."""

import json
import math
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
                          "trajectory_metrics": {
                              "video_rate": 2, "video_velocity_rate": 3,
                              "video_x0_rate": 1, "reference_video_rate": 4,
                              "video_rate_ratio": .5, "audio_rate": 4,
                          }}],
                controller_constants={"bootstrap_anchors": 3, "reference_anchors": 2,
                                     "reference_intervals": 1, "low_change_ratio": .75,
                                     "high_change_ratio": 1.5, "audio_emergency_multiplier": 4},
            )
            summary = summarize(path)
            self.assertEqual(summary["profile"], "adaptive_history_v2")
            self.assertEqual(summary["decisions"][0]["video_rate"], 2)
            self.assertEqual(summary["decisions"][0]["video_velocity_rate"], 3)
            self.assertEqual(summary["decisions"][0]["video_x0_rate"], 1)
            self.assertEqual(summary["decisions"][0]["reference_video_rate"], 4)
            self.assertEqual(summary["decisions"][0]["video_rate_ratio"], .5)
            self.assertEqual(summary["decisions"][0]["audio_rate"], 4)
            self.assertEqual(summary["decision_reference"]["low_video_threshold"], 1.5)

    def test_v2_invariants_require_bootstrap_and_two_anchor_tail(self):
        source = [1.0 - index * 0.04 for index in range(20)] + [0.0]
        effective = source[:3] + [0.7, source[18], source[19], 0.0]
        anchors = [
            {"actual": True, "source_index": index}
            for index in (0, 1, 2)
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

    def test_v3_analyzer_exposes_interval_residuals_and_action(self):
        with tempfile.TemporaryDirectory() as root:
            residuals = {
                "video_v_error": .01, "video_x0_error": .02, "video_error": .02,
                "audio_v_error": .03, "audio_x0_error": .04, "audio_error": .04,
                "video_error_ratio": .5, "audio_error_ratio": .8,
            }
            path = _run(
                root,
                evaluation_profile="adaptive_history_v3",
                adaptive_decisions=[{
                    "sigma": .8, "next_sigma": .6, "reason": "moderate_residual_hold",
                    "action": "hold", "step_scale": 2.0,
                    "previous_step_scale": 2.0, "actual_delta_t": .25,
                    "reference_video_error": .04, "video_error_ratio": .5,
                    "reference_audio_error": .05, "audio_error_ratio": .8,
                    "residuals": residuals,
                }],
                anchors=[{
                    "actual": True, "source_index": 3, "step": 3, "sigma": .8,
                    "previous_step_scale": 2.0, "actual_delta_t": .25,
                    "action": "hold", "trajectory_metrics": {"residuals": residuals},
                }],
                controller_constants={"bootstrap_anchors": 3, "reference_anchors": 0},
            )
            row = summarize(path)["decisions"][0]
            self.assertEqual(row["previous_step_scale"], 2.0)
            self.assertEqual(row["actual_delta_t"], .25)
            self.assertEqual(row["video_v_error"], .01)
            self.assertEqual(row["video_x0_error"], .02)
            self.assertEqual(row["video_error"], .02)
            self.assertEqual(row["reference_video_error"], .04)
            self.assertEqual(row["video_error_ratio"], .5)
            self.assertEqual(row["audio_v_error"], .03)
            self.assertEqual(row["audio_x0_error"], .04)
            self.assertEqual(row["audio_error"], .04)
            self.assertEqual(row["reference_audio_error"], .05)
            self.assertEqual(row["audio_error_ratio"], .8)
            self.assertEqual(row["action"], "hold")

    def test_v3_invariants_do_not_require_tail_anchors(self):
        source = [1.0 - index * .04 for index in range(20)] + [0.0]
        anchors = [
            {"actual": True, "source_index": index} for index in (0, 1, 2)
        ] + [{"actual": True, "source_index": None}]
        data = {
            "evaluation_profile": "adaptive_history_v3",
            "true_nfe": len(anchors), "forecast_count": 0, "fallback_count": 0,
            "source_sigma_sequence": source,
            "effective_sigma_sequence": source[:3] + [.7, 0.0],
            "steps": [{"forecast": False}] * len(anchors),
            "anchors": anchors,
        }
        self.assertTrue(check_invariants(data)["pass"])

    def test_embedded_analyzer_requires_terminal_floor_and_exposes_defect(self):
        source = [1.0 - index * .03 for index in range(20)] + [0.0]
        with tempfile.TemporaryDirectory() as root:
            decision = {
                "sigma": .8, "next_sigma": source[19], "source_index": None,
                "step_scale": 3.0, "reason": "embedded_res", "action": "accept",
                "tolerance_solution_h": .5, "safety_adjusted_h": .4,
                "accepted_h": .3, "previous_accepted_h": .1, "growth_ratio": 3.0,
                "defect_at_accepted_h": .03, "audio_defect_at_accepted_h": 4.0,
                "video_x0_difference_rms": .2, "audio_x0_difference_rms": 20.0,
                "video_normalization_scale": 1.5, "audio_normalization_scale": 2.5,
                "clamp_selected": "terminal_floor",
            }
            path = _run(
                root, evaluation_profile="adaptive_embedded_res_v1", true_nfe=3,
                source_sigma_sequence=source,
                effective_sigma_sequence=[source[0], .8, source[19], 0.0],
                steps=[{"forecast": False}] * 3,
                anchors=[
                    {"actual": True, "source_index": 0},
                    {"actual": True, "source_index": None, "trajectory_metrics": {}},
                    {"actual": True, "source_index": 19},
                ],
                adaptive_decisions=[decision],
                controller_constants={"bootstrap_intervals": 1},
            )
            summary = summarize(path)
            self.assertTrue(summary["invariants"]["pass"])
            row = summary["decisions"][0]
            self.assertEqual(row["accepted_h"], .3)
            self.assertEqual(row["defect_at_accepted_h"], .03)
            self.assertEqual(row["audio_defect_at_accepted_h"], 4.0)
            self.assertEqual(row["clamp_selected"], "terminal_floor")

            data = json.loads(path.read_text(encoding="utf-8"))
            data["anchors"][-1]["source_index"] = 18
            self.assertFalse(check_invariants(data)["checks"]["adaptive_source_anchors"])

    def test_analyzer_reconstructs_old_v2_interval_fields(self):
        with tempfile.TemporaryDirectory() as root:
            path = _run(
                root,
                evaluation_profile="adaptive_history_v2",
                adaptive_decisions=[
                    {"sigma": 1.0, "next_sigma": .8, "step_scale": 1.5},
                    {"sigma": .8, "next_sigma": .5, "step_scale": 2.25},
                ],
                anchors=[
                    {"actual": True, "sigma": 1.0, "trajectory_metrics": {}},
                    {"actual": True, "sigma": .8, "trajectory_metrics": {
                        "video_change": .35, "video_x0_change": .28,
                    }},
                ],
            )
            row = summarize(path)["decisions"][1]
            self.assertAlmostEqual(row["actual_delta_t"], math.log(1.0 / .8))
            self.assertEqual(row["previous_step_scale"], 1.5)
            self.assertEqual(row["video_velocity_change"], .35)
            self.assertEqual(row["video_x0_change"], .28)

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
