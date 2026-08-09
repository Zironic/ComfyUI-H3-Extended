"""CPU parity, integration, guard, callback, and call-count tests."""

import os
import sys
import unittest
from unittest import mock

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

_ORIGINAL_ARGV = list(sys.argv)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

from comfy.k_diffusion.sampling import sample_euler  # noqa: E402
from h3_vector_accel.config import PROFILES, SamplerConfig  # noqa: E402
from h3_vector_accel.diagnostics import RunDiagnostics, current_callback_metadata  # noqa: E402
from h3_vector_accel.predictor import Prediction  # noqa: E402
from h3_vector_accel.sampler import sample_vector_accel  # noqa: E402
from h3_vector_accel.nodes import MiniMaxH3VectorAccelSampler  # noqa: E402
sys.argv = _ORIGINAL_ARGV


class CONST:
    pass


class ModelSamplingAV(CONST):
    audio_scale = 1.0


class Inner:
    model_sampling = ModelSamplingAV()
    latent_shapes = [torch.Size((1, 1, 1, 1, 2)), torch.Size((1, 1, 2))]


class FakeModel:
    inner_model = Inner()

    def __init__(self, velocity):
        self.velocity = velocity
        self.calls = 0

    def __call__(self, x, sigma, **kwargs):
        self.calls += 1
        value = self.velocity(x, float(sigma[0]))
        return x - sigma.reshape((-1,) + (1,) * (x.ndim - 1)) * value


class GenericInner:
    model_sampling = CONST()
    latent_shapes = [torch.Size((1, 1, 4))]


class GenericModel(FakeModel):
    inner_model = GenericInner()


class GuardPredictor:
    def __init__(self, prediction, ready=True):
        self.prediction = prediction
        self._ready = ready
        self.history = [(20.0, torch.ones(1, 1, 4))]
        self.last_actual_sigma = 20.0

    def reset(self):
        pass

    def ready(self):
        return self._ready

    def predict(self, x, sigma):
        return self.prediction

    def observe_actual(self, x, sigma, derivative):
        self.last_actual_sigma = float(sigma)

    def integrate(self, x, sigma, sigma_next, prediction):
        return x + prediction.derivative * (float(sigma_next) - float(sigma))


class AdaptiveProfile:
    hash = "a" * 64

    def validate_compatibility(self, context):
        return True

    def tolerance(self, preset):
        return 1.0

    def survival(self, progress):
        return {"video": 1.0, "audio": 1.0}


def constant_velocity(value=2.0):
    return lambda x, sigma: torch.full_like(x, value)


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.sigmas = torch.linspace(20.0, 0.0, 21)
        self.x = torch.zeros(1, 1, 4)

    def test_native_matches_stock_euler_and_callbacks(self):
        velocity = lambda x, sigma: x * 0.125 + sigma * 0.01 + 1.0
        stock_model = FakeModel(velocity)
        custom_model = FakeModel(velocity)
        stock_callbacks, custom_callbacks = [], []
        stock = sample_euler(
            stock_model, self.x.clone(), self.sigmas,
            callback=stock_callbacks.append, disable=True,
        )
        custom = sample_vector_accel(
            custom_model, self.x.clone(), self.sigmas,
            callback=custom_callbacks.append,
            config=SamplerConfig(method="native"),
        )
        self.assertTrue(torch.equal(stock, custom))
        self.assertEqual(stock_model.calls, custom_model.calls)
        self.assertEqual(len(stock_callbacks), len(custom_callbacks))
        for expected, actual in zip(stock_callbacks, custom_callbacks):
            for key in ("x", "denoised", "sigma", "sigma_hat"):
                self.assertTrue(torch.equal(torch.as_tensor(expected[key]), torch.as_tensor(actual[key])))
            self.assertEqual(expected["i"], actual["i"])

    def test_console_progress_uses_comfy_model_trange(self):
        calls = []

        def tracked_trange(count, disable=None):
            calls.append((count, disable))
            return range(count)

        with mock.patch("h3_vector_accel.sampler.trange", side_effect=tracked_trange):
            sample_vector_accel(
                FakeModel(constant_velocity()), self.x.clone(), self.sigmas,
                disable=False,
                config=SamplerConfig(method="native"),
            )
        self.assertEqual(calls, [(20, False)])

    def test_node_constructs_fingerprinted_sampler(self):
        output = MiniMaxH3VectorAccelSampler.execute(
            "linear_velocity", "uniform_13", "summary", True, 1.5
        )
        sampler = output.result[0]
        self.assertEqual(sampler.h3_vector_config.method, "linear_velocity")
        self.assertEqual(len(sampler.h3_vector_fingerprint), 64)

    def test_forecasts_reject_generic_const_but_native_allows_it(self):
        native = GenericModel(constant_velocity())
        sample_vector_accel(
            native, self.x.clone(), self.sigmas,
            config=SamplerConfig(method="native"),
        )
        self.assertEqual(native.calls, 20)
        with self.assertRaisesRegex(RuntimeError, "ModelSamplingAV"):
            sample_vector_accel(
                GenericModel(constant_velocity()), self.x.clone(), self.sigmas,
                config=SamplerConfig(method="hold", evaluation_profile="conservative_12"),
            )

    def test_constant_velocity_and_profile_call_counts(self):
        expected_counts = {
            "native_20": 20,
            "conservative_12": 12,
            "early_aggressive_13": 13,
            "uniform_13": 13,
            "late_aggressive_13": 13,
        }
        expected = torch.full_like(self.x, -40.0)
        for method in ("hold", "linear_velocity"):
            for profile in PROFILES:
                model = FakeModel(constant_velocity())
                out = sample_vector_accel(
                    model, self.x.clone(), self.sigmas,
                    config=SamplerConfig(method=method, evaluation_profile=profile),
                )
                self.assertTrue(torch.equal(out, expected), (method, profile))
                self.assertEqual(model.calls, expected_counts[profile], (method, profile))

    def test_dense_hold_equals_sparse_euler_at_same_anchors(self):
        intervals = torch.tensor([
            20.0, 18.5, 17.0, 15.0, 14.5, 13.0, 11.0, 10.0, 8.0, 7.5,
            6.0, 5.0, 4.5, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0,
        ])
        config = SamplerConfig(method="hold", evaluation_profile="early_aggressive_13")
        velocity = lambda x, sigma: 0.2 * x + 0.03 * sigma + 1.0
        dense_model = FakeModel(velocity)
        dense = sample_vector_accel(dense_model, self.x.clone(), intervals, config=config)

        sparse_model = FakeModel(velocity)
        sparse = self.x.clone()
        anchors = list(config.actual_indices)
        for position, step in enumerate(anchors):
            sigma = intervals[step]
            denoised = sparse_model(sparse, sigma.reshape(1))
            derivative = (sparse - denoised) / sigma
            next_step = anchors[position + 1] if position + 1 < len(anchors) else 20
            sparse = sparse + derivative * (intervals[next_step] - sigma)
        self.assertTrue(torch.allclose(dense, sparse, atol=1e-6, rtol=1e-6))

    def test_callback_metadata_and_pre_update_state(self):
        callbacks = []
        contexts = []
        def capture(payload):
            callbacks.append(payload)
            contexts.append(current_callback_metadata())
        sample_vector_accel(
            FakeModel(constant_velocity(1.0)), self.x.clone(), self.sigmas,
            callback=capture,
            config=SamplerConfig(method="hold", evaluation_profile="conservative_12"),
        )
        self.assertEqual(len(callbacks), 20)
        self.assertTrue(torch.equal(callbacks[0]["x"], self.x))
        self.assertEqual([callbacks[i]["h3_vector_true_nfe"] for i in range(3)], [1, 2, 2])
        self.assertFalse(callbacks[1]["h3_vector_forecast"])
        self.assertTrue(callbacks[2]["h3_vector_forecast"])
        self.assertEqual(callbacks[2]["h3_vector_actual_anchor_index"], 1)
        self.assertTrue(all(
            key.startswith("h3_vector_")
            for context in contexts for key in context
        ))
        synthetic = callbacks[2]["x"] - callbacks[2]["sigma"]
        self.assertTrue(torch.equal(callbacks[2]["denoised"], synthetic))

    def test_policy_actual_anchors_receive_counterfactual_metrics(self):
        config = SamplerConfig(method="hold", evaluation_profile="conservative_12")
        diagnostics = RunDiagnostics(config=config, latent_shapes=Inner.latent_shapes)
        sample_vector_accel(
            FakeModel(constant_velocity()), self.x.clone(), self.sigmas,
            config=config, diagnostics=diagnostics,
        )
        anchor_by_step = {row["step"]: row for row in diagnostics._anchors}
        self.assertNotIn("prediction_metrics", anchor_by_step[0])
        self.assertIn("prediction_metrics", anchor_by_step[1])
        self.assertIn("prediction_metrics", anchor_by_step[3])
        self.assertEqual(anchor_by_step[3]["logical_span"], 2)

    def test_invalid_forecasts_fall_back_to_actual(self):
        cases = [
            Prediction(valid=False, failure_reason="insufficient_history"),
            Prediction(valid=False, failure_reason="duplicate_anchor_sigma"),
            Prediction(derivative=torch.full_like(self.x, float("inf")), valid=True),
            Prediction(derivative=torch.ones_like(self.x), slope=torch.full_like(self.x, float("nan")), valid=True),
            Prediction(derivative=torch.full_like(self.x, 100.0), valid=True),
        ]
        for prediction in cases:
            model = FakeModel(constant_velocity())
            predictor = GuardPredictor(prediction)
            with mock.patch("h3_vector_accel.sampler.make_predictor", return_value=predictor):
                sample_vector_accel(
                    model, self.x.clone(), self.sigmas,
                    config=SamplerConfig(method="hold", evaluation_profile="conservative_12"),
                )
            self.assertEqual(model.calls, 20, prediction.failure_reason)

    def test_non_monotonic_schedule_forces_actual_evaluations(self):
        sigmas = self.sigmas.clone()
        sigmas[3] = sigmas[2]
        model = FakeModel(constant_velocity())
        sample_vector_accel(
            model, self.x.clone(), sigmas,
            config=SamplerConfig(method="hold", evaluation_profile="conservative_12"),
        )
        self.assertEqual(model.calls, 20)

    def test_guard_can_fail_strictly_and_runs_are_deterministic(self):
        prediction = Prediction(derivative=torch.full_like(self.x, 100.0), valid=True)
        with mock.patch(
            "h3_vector_accel.sampler.make_predictor",
            return_value=GuardPredictor(prediction),
        ):
            with self.assertRaisesRegex(RuntimeError, "unsafe vector forecast"):
                sample_vector_accel(
                    FakeModel(constant_velocity()), self.x.clone(), self.sigmas,
                    config=SamplerConfig(
                        method="hold", evaluation_profile="conservative_12",
                        fallback_on_guard=False,
                    ),
                )
        config = SamplerConfig(method="linear_velocity", evaluation_profile="uniform_13")
        first = sample_vector_accel(FakeModel(constant_velocity()), self.x.clone(), self.sigmas, config=config)
        second = sample_vector_accel(FakeModel(constant_velocity()), self.x.clone(), self.sigmas, config=config)
        self.assertTrue(torch.equal(first, second))

    def test_adaptive_profile_drives_one_forecast_then_actual(self):
        model = FakeModel(constant_velocity())
        config = SamplerConfig(
            method="linear_velocity", evaluation_profile="native_20",
            policy="adaptive_repair", repairability_profile="measured.json",
        )
        with mock.patch(
            "h3_vector_accel.sampler.RepairabilityProfile.load",
            return_value=AdaptiveProfile(),
        ):
            out = sample_vector_accel(model, self.x.clone(), self.sigmas, config=config)
        self.assertTrue(torch.equal(out, torch.full_like(self.x, -40.0)))
        self.assertEqual(model.calls, 13)


if __name__ == "__main__":
    unittest.main()
