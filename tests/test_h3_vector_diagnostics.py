"""CPU checks for AV diagnostics, scoped metadata, and run JSON."""

import json
import os
import sys
import unittest

import h3_test_tempfile as tempfile
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

_ORIGINAL_ARGV = list(sys.argv)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import comfy.utils  # noqa: E402

from h3_vector_accel.config import SamplerConfig  # noqa: E402
from h3_vector_accel.diagnostics import (  # noqa: E402
    RunDiagnostics,
    callback_metadata_scope,
    current_callback_metadata,
    modality_metrics,
)
from h3_vector_accel.predictor import Prediction  # noqa: E402
from h3_vector_accel.repairability import (  # noqa: E402
    normalized_perturbation,
    per_modality_divergence,
    survival_factor,
)
sys.argv = _ORIGINAL_ARGV


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.shapes = [
            torch.Size((1, 1, 1, 1, 2)),
            torch.Size((1, 1, 2)),
        ]

    def test_audio_is_visible_and_modal_max_is_conservative(self):
        actual = torch.zeros(1, 1, 4)
        predicted = actual.clone()
        predicted[:, :, 2:] = 1
        metrics = modality_metrics(
            actual, predicted, self.shapes, integration_span=-2.0
        )
        self.assertEqual(metrics["video"]["relative_l2"], 0.0)
        self.assertGreater(metrics["audio"]["relative_l2"], 0.0)
        self.assertEqual(metrics["modal_max"], metrics["audio"]["relative_l2"])
        self.assertAlmostEqual(
            metrics["audio"]["integration_error_proxy"],
            2.0 * metrics["audio"]["rms"],
        )

    def test_callback_context_is_copied_and_reset(self):
        metadata = {"h3_vector_forecast": True, "h3_vector_true_nfe": 2}
        self.assertIsNone(current_callback_metadata())
        with callback_metadata_scope(metadata):
            current = current_callback_metadata()
            current["h3_vector_true_nfe"] = 99
            self.assertEqual(current_callback_metadata()["h3_vector_true_nfe"], 2)
        self.assertIsNone(current_callback_metadata())

    def test_full_diagnostics_are_run_scoped_json(self):
        with tempfile.TemporaryDirectory(prefix="h3-vector-diag-") as root:
            config = SamplerConfig(
                method="hold", evaluation_profile="conservative_12",
                diagnostics="full",
            )
            run = RunDiagnostics(
                config=config, output_root=root, run_id="run-a",
                latent_shapes=self.shapes, model_fingerprint="model-a",
            )
            run.start_run(
                sigmas=[2.0, 1.0, 0.0], method="hold",
                evaluation_profile="conservative_12",
                configuration_fingerprint="abc",
            )
            actual = torch.zeros(1, 1, 4)
            predicted = actual.clone()
            predicted[:, :, 2:] = 1
            run.observe_actual_anchor(
                1, 1.0, actual_derivative=actual,
                counterfactual=Prediction(derivative=predicted, valid=True),
                previous_actual_sigma=2.0,
                fallback_reason="direction_cosine",
            )
            run.observe_step(0, 2.0, False, 1)
            run.observe_step(1, 1.0, False, 2, fallback_reason="direction_cosine")
            summary = run.finish_run(model_call_seconds=0.5, sampler_overhead_seconds=0.1)
            path = os.path.join(root, "h3_vector_accel", "run-a", "diagnostics.json")
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding="utf-8") as handle:
                stored = json.load(handle)
            self.assertEqual(stored["true_nfe"], 2)
            self.assertEqual(stored["fallback_count"], 1)
            self.assertEqual(stored["actual_forecast_mask"], [True, True])
            self.assertEqual(summary["maximum_audio_prediction_error"],
                             stored["maximum_audio_prediction_error"])

    def test_repairability_helpers_keep_modalities_separate(self):
        delta = torch.tensor([[[1.0, 1.0, 10.0, 10.0]]])
        video_only = normalized_perturbation(
            delta, self.shapes, target_rms=2.0, modality="video"
        )
        self.assertAlmostEqual(float(video_only[:, :, :2].square().mean().sqrt()), 2.0)
        self.assertEqual(float(video_only[:, :, 2:].abs().max()), 0.0)
        divergence = per_modality_divergence(delta, torch.ones_like(delta), self.shapes)
        self.assertEqual(divergence["modal_max"], max(divergence["video"], divergence["audio"]))
        self.assertGreater(survival_factor(0.25, 0.5), 1.0)


if __name__ == "__main__":
    unittest.main()
