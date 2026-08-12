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

from comfy.k_diffusion.sampling import sample_euler, sample_res_multistep  # noqa: E402
from h3_vector_accel.config import MASK_PROFILES, SamplerConfig  # noqa: E402
from h3_vector_accel.adaptive_res import IncrementalRES  # noqa: E402
from h3_vector_accel.diagnostics import RunDiagnostics, current_callback_metadata  # noqa: E402
from h3_vector_accel.predictor import Prediction  # noqa: E402
from h3_vector_accel.sampler import sample_vector_accel  # noqa: E402
from h3_vector_accel.schedules import CONTINUOUS_SCHEDULE_FAMILIES, continuous_schedule  # noqa: E402
from h3_vector_accel.nodes import MiniMaxH3SamplerScheduler, MiniMaxH3VectorAccelSampler  # noqa: E402
sys.argv = _ORIGINAL_ARGV


class CONST:
    pass


class ModelSamplingAV(CONST):
    audio_scale = 1.0


class Inner:
    model_sampling = ModelSamplingAV()
    latent_shapes = [torch.Size((1, 1, 1, 1, 2)), torch.Size((1, 1, 2))]


class ModelPatcher:
    def __init__(self, sampling):
        self.sampling = sampling

    def get_model_object(self, name):
        if name != "model_sampling":
            raise KeyError(name)
        return self.sampling


class ResInner:
    model_sampling = ModelSamplingAV()
    model_patcher = ModelPatcher(model_sampling)
    latent_shapes = [torch.Size((1, 1, 4)), torch.Size((1, 1, 2))]


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


class ResModel(FakeModel):
    inner_model = ResInner()


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

    def test_full_schedule_euler_matches_core_and_callbacks(self):
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
            config=SamplerConfig(method="euler", evaluation_profile="full_20"),
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

        with mock.patch("comfy.k_diffusion.sampling.trange", side_effect=tracked_trange):
            sample_vector_accel(
                FakeModel(constant_velocity()), self.x.clone(), self.sigmas,
                disable=False,
                config=SamplerConfig(method="euler", evaluation_profile="full_20"),
            )
        self.assertEqual(calls, [(20, False)])

    def test_node_constructs_fingerprinted_sampler(self):
        output = MiniMaxH3VectorAccelSampler.execute(
            "linear_velocity", "uniform_13", "summary", True, 1.5
        )
        sampler = output.result[0]
        self.assertEqual(sampler.h3_vector_config.method, "linear_velocity")
        self.assertEqual(len(sampler.h3_vector_fingerprint), 64)

    def test_combined_sampler_scheduler_uses_comfy_registries_and_requested_steps(self):
        schema = MiniMaxH3SamplerScheduler.define_schema()
        inputs = {value.id: value for value in schema.inputs}
        self.assertEqual(
            [value.id for value in schema.inputs],
            ["model", "sampler_name", "scheduler", "steps", "denoise"],
        )
        self.assertEqual(inputs["sampler_name"].options, comfy.samplers.SAMPLER_NAMES)
        self.assertEqual(
            inputs["scheduler"].options,
            [*comfy.samplers.SCHEDULER_NAMES, *CONTINUOUS_SCHEDULE_FAMILIES],
        )
        self.assertEqual(inputs["sampler_name"].default, "res_multistep")
        self.assertEqual(inputs["scheduler"].default, "simple")

        model_sampling = object()
        model = mock.Mock()
        model.get_model_object.return_value = model_sampling
        selected_sampler = object()
        source = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        with mock.patch.object(comfy.samplers, "sampler_object", return_value=selected_sampler) as sampler_object, \
                mock.patch.object(comfy.samplers, "calculate_sigmas", return_value=source) as calculate_sigmas:
            output = MiniMaxH3SamplerScheduler.execute(
                model, "res_multistep", "beta", 2, 0.5,
            )

        self.assertIs(output.result[0], selected_sampler)
        self.assertTrue(torch.equal(output.result[1], source[-3:]))
        sampler_object.assert_called_once_with("res_multistep")
        calculate_sigmas.assert_called_once_with(model_sampling, "beta", 4)

    def test_combined_sampler_scheduler_builds_arbitrary_custom_steps(self):
        model_sampling = object()
        model = mock.Mock()
        model.get_model_object.return_value = model_sampling
        source = torch.linspace(1.0, 0.0, 21)
        with mock.patch.object(comfy.samplers, "sampler_object", return_value=object()), \
                mock.patch.object(comfy.samplers, "calculate_sigmas", return_value=source) as calculate_sigmas:
            output = MiniMaxH3SamplerScheduler.execute(
                model, "res_multistep", "multiplicative_stride", 13, 1.0,
            )
        self.assertEqual(output.result[1].shape, (14,))
        self.assertTrue(torch.all(output.result[1][:-1] > output.result[1][1:]))
        self.assertEqual(float(output.result[1][-1]), 0.0)
        calculate_sigmas.assert_called_once_with(model_sampling, "simple", 20)

    def test_combined_sampler_scheduler_zero_denoise_returns_empty_sigmas(self):
        with mock.patch.object(comfy.samplers, "sampler_object", return_value=object()), \
                mock.patch.object(comfy.samplers, "calculate_sigmas") as calculate_sigmas:
            output = MiniMaxH3SamplerScheduler.execute(
                mock.Mock(), "euler", "simple", 20, 0.0,
            )
        self.assertEqual(output.result[1].numel(), 0)
        calculate_sigmas.assert_not_called()

    def test_node_exposes_res_benchmark_without_changing_widget_order(self):
        schema = MiniMaxH3VectorAccelSampler.define_schema()
        inputs = {value.id: value for value in schema.inputs}
        self.assertEqual(
            [value.id for value in schema.inputs],
            [
                "method", "evaluation_profile", "diagnostics", "policy",
                "quality_preset", "repairability_profile", "conditioning_mode",
                "fallback_on_guard", "max_extrapolation_ratio",
                "max_adaptive_step_scale", "embedded_video_tolerance",
                "adaptive_safety_factor", "max_adaptive_growth_ratio",
            ],
        )
        self.assertEqual(inputs["method"].options, [
            "euler", "res_multistep", "hold", "linear_velocity", "vde",
        ])
        self.assertIn("full_20", inputs["evaluation_profile"].options)
        self.assertIn("late_aggressive_13", inputs["evaluation_profile"].options)
        self.assertIn("adaptive_history_v1", inputs["evaluation_profile"].options)
        self.assertIn("adaptive_history_v2", inputs["evaluation_profile"].options)
        self.assertIn("adaptive_history_v3", inputs["evaluation_profile"].options)
        self.assertIn("adaptive_embedded_res_v1", inputs["evaluation_profile"].options)
        self.assertIn("geometric_11", inputs["evaluation_profile"].options)
        self.assertIn("geometric_linear_ends_11", inputs["evaluation_profile"].options)
        self.assertIn("multiplicative_stride_11", inputs["evaluation_profile"].options)
        self.assertIn("multiplicative_stride_linear_ends_11", inputs["evaluation_profile"].options)
        self.assertEqual(inputs["method"].display_name, "solver / forecast mode")
        self.assertEqual(inputs["evaluation_profile"].display_name, "actual-evaluation schedule")
        self.assertEqual(inputs["max_adaptive_step_scale"].default, 3.0)
        self.assertEqual(inputs["max_adaptive_step_scale"].min, 1.0)
        self.assertEqual(inputs["embedded_video_tolerance"].default, 0.05)
        self.assertEqual(inputs["adaptive_safety_factor"].default, 0.8)
        self.assertEqual(inputs["max_adaptive_growth_ratio"].default, 2.0)
        output = MiniMaxH3VectorAccelSampler.execute(
            "res_multistep", "late_aggressive_13", "off"
        )
        config = output.result[0].h3_vector_config
        self.assertEqual(config.method, "res_multistep")
        self.assertEqual(config.evaluation_profile, "late_aggressive_13")
        self.assertEqual(len(config.actual_indices), 13)
        output = MiniMaxH3VectorAccelSampler.execute(
            "res_multistep", "adaptive_history_v1", "off"
        )
        self.assertEqual(
            output.result[0].h3_vector_config.evaluation_profile,
            "adaptive_history_v1",
        )
        self.assertIn(
            "requires the res_multistep method",
            MiniMaxH3VectorAccelSampler.validate_inputs("euler", "adaptive_history_v1"),
        )
        output = MiniMaxH3VectorAccelSampler.execute(
            "res_multistep", "adaptive_history_v2", "off",
            max_adaptive_step_scale=5.0,
        )
        self.assertEqual(
            output.result[0].h3_vector_config.evaluation_profile,
            "adaptive_history_v2",
        )
        self.assertEqual(output.result[0].h3_vector_config.max_adaptive_step_scale, 5.0)
        self.assertIn(
            "requires the res_multistep method",
            MiniMaxH3VectorAccelSampler.validate_inputs("euler", "adaptive_history_v2"),
        )
        output = MiniMaxH3VectorAccelSampler.execute(
            "res_multistep", "adaptive_history_v3", "off",
            max_adaptive_step_scale=5.0,
        )
        self.assertEqual(output.result[0].h3_vector_config.evaluation_profile,
                         "adaptive_history_v3")
        self.assertIn(
            "requires the res_multistep method",
            MiniMaxH3VectorAccelSampler.validate_inputs("euler", "adaptive_history_v3"),
        )
        output = MiniMaxH3VectorAccelSampler.execute(
            "res_multistep", "adaptive_embedded_res_v1", "off",
            max_adaptive_step_scale=7.0, embedded_video_tolerance=.03,
            adaptive_safety_factor=.75, max_adaptive_growth_ratio=1.75,
        )
        embedded = output.result[0].h3_vector_config
        self.assertEqual(embedded.evaluation_profile, "adaptive_embedded_res_v1")
        self.assertEqual(embedded.max_adaptive_step_scale, 7.0)
        self.assertEqual(embedded.embedded_video_tolerance, .03)
        self.assertEqual(embedded.adaptive_safety_factor, .75)
        self.assertEqual(embedded.max_adaptive_growth_ratio, 1.75)
        self.assertIn(
            "requires the res_multistep method",
            MiniMaxH3VectorAccelSampler.validate_inputs("euler", "adaptive_embedded_res_v1"),
        )

    def test_node_hides_unavailable_adaptive_controls_without_removing_them(self):
        with mock.patch("h3_vector_accel.nodes._profile_names", return_value=[]):
            schema = MiniMaxH3VectorAccelSampler.define_schema()
        inputs = {value.id: value for value in schema.inputs}
        self.assertEqual(inputs["policy"].options, ["fixed"])
        for name in ("policy", "quality_preset", "repairability_profile", "conditioning_mode"):
            self.assertTrue(inputs[name].extra_dict["hidden"])
        self.assertEqual(inputs["quality_preset"].display_name, "adaptive quality tolerance")
        self.assertIn("Forecast-only", inputs["max_extrapolation_ratio"].tooltip)
        self.assertIn("Adaptive RES only", inputs["max_adaptive_step_scale"].tooltip)

    def test_node_reveals_adaptive_controls_when_a_profile_exists(self):
        with mock.patch("h3_vector_accel.nodes._profile_names", return_value=["measured.json"]):
            schema = MiniMaxH3VectorAccelSampler.define_schema()
        inputs = {value.id: value for value in schema.inputs}
        self.assertEqual(inputs["policy"].options, ["fixed", "adaptive_repair"])
        for name in ("policy", "quality_preset", "repairability_profile", "conditioning_mode"):
            self.assertNotIn("hidden", inputs[name].extra_dict)
        self.assertEqual(inputs["repairability_profile"].options, ["measured.json"])

    def test_h3_modes_reject_generic_const(self):
        for method in ("euler", "res_multistep", "hold"):
            with self.assertRaisesRegex(RuntimeError, "ModelSamplingAV"):
                sample_vector_accel(
                    GenericModel(constant_velocity()), self.x.clone(), self.sigmas,
                    config=SamplerConfig(method=method, evaluation_profile="conservative_12"),
                )

    def test_constant_velocity_and_profile_call_counts(self):
        expected_counts = {
            "full_20": 20,
            "conservative_12": 12,
            "early_aggressive_13": 13,
            "uniform_13": 13,
            "late_cautious_14": 14,
            "late_aggressive_13": 13,
            "late_aggressive_12": 12,
            "late_max_11": 11,
        }
        expected = torch.full_like(self.x, -40.0)
        for method in ("hold", "linear_velocity"):
            for profile in MASK_PROFILES:
                model = FakeModel(constant_velocity())
                out = sample_vector_accel(
                    model, self.x.clone(), self.sigmas,
                    config=SamplerConfig(method=method, evaluation_profile=profile),
                )
                self.assertTrue(torch.equal(out, expected), (method, profile))
                self.assertEqual(model.calls, expected_counts[profile], (method, profile))

    def test_geometric_profiles_have_expected_ratios_and_end_contracts(self):
        geometric, _, geometric_ratio = continuous_schedule(self.sigmas, "geometric_11")
        self.assertEqual(geometric.numel(), 12)
        self.assertEqual(float(geometric[-1]), 0.0)
        self.assertEqual(float(geometric[0]), float(self.sigmas[0]))
        geometric_delta = geometric[:-1].double() - geometric[1:].double()
        geometric_ratios = geometric_delta[1:] / geometric_delta[:-1]
        self.assertTrue(torch.allclose(
            geometric_ratios,
            torch.full_like(geometric_ratios, geometric_ratio),
            atol=2e-4, rtol=2e-4,
        ))
        base = 1.0 / geometric_ratio
        self.assertAlmostEqual(
            sum(base ** power for power in range(1, 12)), 1.0, places=12,
        )
        self.assertLess(float(geometric_delta[0]), float(self.sigmas[0] - self.sigmas[1]))
        self.assertGreater(float(geometric_delta[-1]), float(self.sigmas[-2] - self.sigmas[-1]))

        linear_ends, _, interior_ratio = continuous_schedule(
            self.sigmas, "geometric_linear_ends_11",
        )
        self.assertEqual(linear_ends.numel(), 12)
        self.assertTrue(torch.equal(linear_ends[:3], self.sigmas[:3]))
        self.assertTrue(torch.equal(linear_ends[-3:], self.sigmas[-3:]))
        linear_h = torch.diff(-torch.log(linear_ends[:-1].double()))
        interior_ratios = linear_h[2:-1] / linear_h[1:-2]
        self.assertTrue(torch.allclose(
            interior_ratios,
            torch.full_like(interior_ratios, interior_ratio),
            atol=2e-5, rtol=2e-5,
        ))

    def test_multiplicative_stride_profiles_have_expected_logical_coordinates(self):
        full, coordinates, ratio = continuous_schedule(
            self.sigmas, "multiplicative_stride_11",
        )
        full_coordinates = torch.tensor((*coordinates, 20.0), dtype=torch.float64)
        full_strides = torch.diff(full_coordinates)
        self.assertEqual(full.numel(), 12)
        self.assertEqual(float(full[0]), float(self.sigmas[0]))
        self.assertEqual(float(full[-1]), 0.0)
        self.assertAlmostEqual(float(full_strides[0]), 1.0, places=12)
        self.assertTrue(torch.allclose(
            full_strides[1:] / full_strides[:-1],
            torch.full_like(full_strides[1:], ratio),
            atol=1e-10, rtol=1e-10,
        ))
        self.assertAlmostEqual(float(full_strides.sum()), 20.0, places=12)

        linear, coordinates, interior_ratio = continuous_schedule(
            self.sigmas, "multiplicative_stride_linear_ends_11",
        )
        linear_coordinates = torch.tensor((*coordinates, 20.0), dtype=torch.float64)
        linear_strides = torch.diff(linear_coordinates)
        self.assertEqual(linear.numel(), 12)
        self.assertEqual(tuple(linear_coordinates[:3].tolist()), (0.0, 1.0, 2.0))
        self.assertEqual(tuple(linear_coordinates[-3:].tolist()), (18.0, 19.0, 20.0))
        self.assertTrue(torch.equal(linear[:3], self.sigmas[:3]))
        self.assertTrue(torch.equal(linear[-3:], self.sigmas[-3:]))
        self.assertTrue(torch.allclose(
            linear_strides[3:9] / linear_strides[2:8],
            torch.full_like(linear_strides[3:9], interior_ratio),
            atol=1e-10, rtol=1e-10,
        ))

    def test_continuous_profiles_require_res_and_match_direct_core(self):
        profiles = (
            "geometric_11", "geometric_linear_ends_11",
            "multiplicative_stride_11", "multiplicative_stride_linear_ends_11",
        )
        for profile in profiles:
            with self.assertRaisesRegex(ValueError, "continuous schedules require"):
                SamplerConfig(method="euler", evaluation_profile=profile)
            config = SamplerConfig(method="res_multistep", evaluation_profile=profile)
            effective, coordinates, ratio = continuous_schedule(self.sigmas, profile)
            velocity = lambda x, sigma: 0.2 * x + 0.03 * sigma + 1.0
            direct_model = ResModel(velocity)
            configured_model = ResModel(velocity)
            callbacks = []
            diagnostics = RunDiagnostics(config=config, latent_shapes=ResInner.latent_shapes)
            direct = sample_res_multistep(
                direct_model, self.x.clone(), effective, disable=True,
            )
            configured = sample_vector_accel(
                configured_model, self.x.clone(), self.sigmas,
                callback=callbacks.append, disable=True,
                config=config, diagnostics=diagnostics,
            )
            self.assertTrue(torch.equal(direct, configured))
            self.assertEqual(direct_model.calls, 11)
            self.assertEqual(configured_model.calls, 11)
            self.assertEqual(len(callbacks), 11)
            self.assertEqual(
                [float(row["sigma"]) for row in callbacks],
                [float(value) for value in effective[:-1]],
            )
            self.assertEqual(
                [row["h3_vector_logical_coordinate"] for row in callbacks],
                list(coordinates),
            )
            self.assertTrue(all(row["h3_vector_actual_indices"] is None for row in callbacks))
            self.assertTrue(all(row["h3_vector_geometric_ratio"] == ratio for row in callbacks))
            self.assertIsNone(diagnostics._run_metadata["actual_indices"])
            self.assertEqual(
                diagnostics._run_metadata["effective_sigma_sequence"], effective.tolist(),
            )

    def test_dense_hold_equals_reduced_schedule_euler_at_same_anchors(self):
        intervals = torch.tensor([
            20.0, 18.5, 17.0, 15.0, 14.5, 13.0, 11.0, 10.0, 8.0, 7.5,
            6.0, 5.0, 4.5, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0,
        ])
        config = SamplerConfig(method="hold", evaluation_profile="early_aggressive_13")
        velocity = lambda x, sigma: 0.2 * x + 0.03 * sigma + 1.0
        dense_model = FakeModel(velocity)
        dense = sample_vector_accel(dense_model, self.x.clone(), intervals, config=config)

        actual_model = FakeModel(velocity)
        actual_only = self.x.clone()
        anchors = list(config.actual_indices)
        for position, step in enumerate(anchors):
            sigma = intervals[step]
            denoised = actual_model(actual_only, sigma.reshape(1))
            derivative = (actual_only - denoised) / sigma
            next_step = anchors[position + 1] if position + 1 < len(anchors) else 20
            actual_only = actual_only + derivative * (intervals[next_step] - sigma)
        self.assertTrue(torch.allclose(dense, actual_only, atol=1e-6, rtol=1e-6))

    def test_euler_matches_core_on_reduced_schedule_and_maps_callbacks(self):
        config = SamplerConfig(method="euler", evaluation_profile="late_aggressive_13")
        effective = torch.cat((self.sigmas[list(config.actual_indices)], self.sigmas[-1:]))
        velocity = lambda x, sigma: 0.2 * x + 0.03 * sigma + 1.0
        direct = sample_euler(FakeModel(velocity), self.x.clone(), effective, disable=True)
        callbacks, contexts = [], []
        diagnostics = RunDiagnostics(config=config, latent_shapes=Inner.latent_shapes)
        actual_only = sample_vector_accel(
            FakeModel(velocity), self.x.clone(), self.sigmas,
            callback=lambda payload: (callbacks.append(payload), contexts.append(current_callback_metadata())),
            config=config, diagnostics=diagnostics,
        )
        self.assertTrue(torch.equal(direct, actual_only))
        self.assertEqual(len(callbacks), len(config.actual_indices))
        self.assertEqual(callbacks[-1]["i"], 19)
        self.assertEqual(callbacks[-1]["h3_vector_true_nfe"], len(config.actual_indices))
        self.assertTrue(contexts[-1]["h3_vector_actual_only"])
        self.assertEqual(contexts[-1]["h3_vector_core_solver"], "euler")
        self.assertEqual(diagnostics._run_metadata["actual_indices"], list(config.actual_indices))
        self.assertEqual(diagnostics._run_metadata["effective_sigma_sequence"], effective.tolist())
        self.assertIn("source_sigma_hash", diagnostics._run_metadata)
        self.assertIn("effective_sigma_hash", diagnostics._run_metadata)
        self.assertGreaterEqual(
            diagnostics._run_metadata["wall_seconds"],
            diagnostics._run_metadata["model_call_seconds"],
        )

    def test_res_matches_core_on_nonuniform_reduced_schedule(self):
        sigmas = torch.tensor([
            20.0, 18.5, 17.0, 15.0, 14.5, 13.0, 11.0, 10.0, 8.0, 7.5,
            6.0, 5.0, 4.5, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0,
        ])
        config = SamplerConfig(method="res_multistep", evaluation_profile="late_aggressive_13")
        effective = torch.cat((sigmas[list(config.actual_indices)], sigmas[-1:]))
        velocity = lambda x, sigma: 0.2 * x + 0.03 * sigma + 1.0
        direct_model = ResModel(velocity)
        configured_model = ResModel(velocity)
        callbacks = []
        direct = sample_res_multistep(direct_model, self.x.clone(), effective, disable=True)
        configured = sample_vector_accel(
            configured_model, self.x.clone(), sigmas, callback=callbacks.append,
            disable=True, config=config,
        )
        self.assertTrue(torch.equal(direct, configured))
        self.assertEqual(direct_model.calls, 13)
        self.assertEqual(configured_model.calls, 13)
        self.assertEqual([row["i"] for row in callbacks], list(config.actual_indices))
        self.assertEqual(
            [float(row["sigma"]) for row in callbacks],
            [float(value) for value in effective[:-1]],
        )
        self.assertEqual(float(effective[-1]), 0.0)

    def test_res_full_profile_matches_full_core_res(self):
        config = SamplerConfig(method="res_multistep", evaluation_profile="full_20")
        velocity = lambda x, sigma: 0.2 * x + 0.03 * sigma + 1.0
        direct_model = ResModel(velocity)
        configured_model = ResModel(velocity)
        direct = sample_res_multistep(
            direct_model, self.x.clone(), self.sigmas, disable=True,
        )
        configured = sample_vector_accel(
            configured_model, self.x.clone(), self.sigmas, disable=True,
            config=config,
        )
        self.assertTrue(torch.equal(direct, configured))
        self.assertEqual(direct_model.calls, 20)
        self.assertEqual(configured_model.calls, 20)

    def test_incremental_res_matches_stock_on_full_and_irregular_schedules(self):
        schedules = (
            self.sigmas,
            torch.tensor([20.0, 17.0, 13.0, 8.0, 4.0, 1.0, 0.0]),
        )
        velocity = lambda x, sigma: 0.2 * x + 0.03 * sigma + 1.0
        for sigmas in schedules:
            direct = sample_res_multistep(
                ResModel(velocity), self.x.clone(), sigmas, disable=True,
            )
            incremental_model = ResModel(velocity)
            incremental = self.x.clone()
            stepper = IncrementalRES()
            for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
                denoised = incremental_model(incremental, sigma.reshape(1))
                incremental = stepper.step(incremental, sigma, denoised, sigma_next)
            self.assertTrue(torch.equal(direct, incremental))

    def test_adaptive_res_reports_causal_effective_schedule_and_honest_nfe(self):
        sigmas = torch.tensor([
            1.0, 0.9956331849, 0.9908256531, 0.9855073094, 0.9795918465,
            0.9729729891, 0.9655172229, 0.9570552111, 0.9473683834,
            0.9361702204, 0.9230769277, 0.9075629711, 0.8888888955,
            0.8659793735, 0.8372092843, 0.8000000119, 0.75,
            0.6792452931, 0.5714285970, 0.3870967925, 0.0,
        ])
        config = SamplerConfig(
            method="res_multistep", evaluation_profile="adaptive_history_v1",
        )
        diagnostics = RunDiagnostics(config=config, latent_shapes=ResInner.latent_shapes)
        callbacks, contexts = [], []
        model = ResModel(lambda x, sigma: torch.full_like(
            x, 1.0 + 100.0 * max(sigma - 0.98, 0.0)
        ))
        sample_vector_accel(
            model, torch.zeros(1, 1, 6), sigmas, disable=True, config=config,
            diagnostics=diagnostics,
            callback=lambda row: (callbacks.append(row), contexts.append(current_callback_metadata())),
        )
        effective = diagnostics._run_metadata["effective_sigma_sequence"]
        self.assertEqual(effective[:6], sigmas[:6].tolist())
        self.assertEqual(effective[-4:], sigmas[17:20].tolist() + [0.0])
        self.assertTrue(all(a > b for a, b in zip(effective, effective[1:])))
        self.assertLess(model.calls, 20, diagnostics._run_metadata["adaptive_decisions"])
        self.assertEqual(model.calls, len(callbacks))
        self.assertEqual(model.calls, diagnostics.true_nfe)
        self.assertEqual(
            [row["h3_vector_actual_anchor_index"] for row in callbacks],
            list(range(model.calls)),
        )
        self.assertNotEqual(callbacks[5]["h3_vector_policy_reason"], "protected_prefix")
        self.assertEqual(callbacks[-1]["i"], 19)
        self.assertTrue(all(not row["h3_vector_forecast"] for row in callbacks))
        self.assertTrue(all(context["h3_vector_callback_context"] == "h3_vector_adaptive_actual_only"
                            for context in contexts))
        self.assertIn("effective_sigma_hash", diagnostics._run_metadata)
        self.assertIn("adaptive_controller", diagnostics._run_metadata["configuration"])
        with self.assertRaisesRegex(ValueError, "does not have fixed actual indices"):
            _ = config.actual_indices

    def test_adaptive_res_v2_has_bootstrap_two_anchor_tail_and_honest_nfe(self):
        sigmas = torch.tensor([
            1.0, 0.9956331849, 0.9908256531, 0.9855073094, 0.9795918465,
            0.9729729891, 0.9655172229, 0.9570552111, 0.9473683834,
            0.9361702204, 0.9230769277, 0.9075629711, 0.8888888955,
            0.8659793735, 0.8372092843, 0.8000000119, 0.75,
            0.6792452931, 0.5714285970, 0.3870967925, 0.0,
        ])
        config = SamplerConfig(
            method="res_multistep", evaluation_profile="adaptive_history_v2",
        )
        diagnostics = RunDiagnostics(config=config, latent_shapes=ResInner.latent_shapes)
        callbacks = []
        model = ResModel(lambda x, sigma: torch.full_like(
            x, 1.0 + 100.0 * max(sigma - 0.98, 0.0)
        ))
        with self.assertLogs(level="INFO") as captured_logs:
            sample_vector_accel(
                model, torch.zeros(1, 1, 6), sigmas, disable=True, config=config,
                diagnostics=diagnostics, callback=callbacks.append,
            )
        effective = diagnostics._run_metadata["effective_sigma_sequence"]
        decisions = diagnostics._run_metadata["adaptive_decisions"]
        self.assertEqual(effective[:3], sigmas[:3].tolist())
        self.assertEqual(effective[-3:], sigmas[18:20].tolist() + [0.0])
        self.assertEqual([row["reason"] for row in decisions[:2]], ["bootstrap"] * 2)
        self.assertEqual([row["protected_region"] for row in decisions[:3]], ["bootstrap"] * 3)
        self.assertNotEqual(decisions[2]["reason"], "bootstrap")
        first_growth = next(
            index for index, row in enumerate(decisions)
            if row["reason"] == "low_video_change_grow"
        )
        self.assertGreaterEqual(first_growth, 3)
        self.assertEqual(decisions[first_growth - 1]["reason"], "low_video_change_wait")
        self.assertTrue(all(row["reason"] != "protected_prefix" for row in decisions))
        self.assertTrue(all(a > b for a, b in zip(effective, effective[1:])))
        self.assertLessEqual(model.calls, 20)
        self.assertEqual(model.calls, len(callbacks))
        self.assertEqual(model.calls, diagnostics.true_nfe)
        self.assertEqual(callbacks[-1]["i"], 19)
        self.assertTrue(all(not row["h3_vector_forecast"] for row in callbacks))
        self.assertEqual(
            diagnostics._run_metadata["configuration"]["adaptive_controller"]["version"],
            "adaptive_history_v2",
        )
        progress = [line for line in captured_logs.output if "[H3 Adaptive RES v2]" in line]
        self.assertEqual(len(progress), model.calls)
        self.assertIn("NFE 1/~", progress[0])
        self.assertIn("est. compute", progress[0])
        self.assertIn("schedule 0.00/20", progress[0])
        self.assertIn("scale 1.00x", progress[0])
        self.assertIn("delta_t=", progress[0])
        self.assertIn("video raw v=", progress[0])
        self.assertIn("video rate v=", progress[0])
        self.assertIn("x0=", progress[0])
        self.assertIn("combined=", progress[0])
        self.assertIn("ref=", progress[0])
        self.assertIn("ratio=", progress[0])
        self.assertIn("-> 20.00", progress[-1])

    def test_adaptive_res_v3_reports_predictive_residuals_and_honest_nfe(self):
        sigmas = torch.tensor([
            1.0, 0.9956331849, 0.9908256531, 0.9855073094, 0.9795918465,
            0.9729729891, 0.9655172229, 0.9570552111, 0.9473683834,
            0.9361702204, 0.9230769277, 0.9075629711, 0.8888888955,
            0.8659793735, 0.8372092843, 0.8000000119, 0.75,
            0.6792452931, 0.5714285970, 0.3870967925, 0.0,
        ])
        config = SamplerConfig(
            method="res_multistep", evaluation_profile="adaptive_history_v3",
            max_adaptive_step_scale=5.0,
        )
        diagnostics = RunDiagnostics(config=config, latent_shapes=ResInner.latent_shapes)
        callbacks = []
        model = ResModel(lambda x, sigma: torch.full_like(
            x, 1.0 + .1 * sigma + .02 * sigma * sigma
        ))
        with self.assertLogs(level="INFO") as captured_logs:
            sample_vector_accel(
                model, torch.zeros(1, 1, 6), sigmas, disable=True, config=config,
                diagnostics=diagnostics, callback=callbacks.append,
            )
        effective = diagnostics._run_metadata["effective_sigma_sequence"]
        decisions = diagnostics._run_metadata["adaptive_decisions"]
        self.assertEqual(effective[:4], sigmas[:4].tolist())
        self.assertEqual(effective[-1], 0.0)
        self.assertEqual(diagnostics._run_metadata["controller_constants"]["protected_tail"], ())
        self.assertTrue(all(row["reason"] != "protected_tail" for row in decisions))
        self.assertTrue(all(row["reason"] != "reserve_protected_tail" for row in decisions))
        self.assertEqual([row["action"] for row in decisions[:3]], ["bootstrap"] * 3)
        self.assertEqual(decisions[2]["reason"], "bootstrap_predict")
        self.assertIsNotNone(decisions[3]["residuals"])
        self.assertEqual(decisions[3]["action"], "reference_calibration")
        for name in (
            "video_v_error", "video_x0_error", "video_error",
            "audio_v_error", "audio_x0_error", "audio_error",
        ):
            self.assertIn(name, decisions[3]["residuals"])
        self.assertGreater(decisions[3]["actual_delta_t"], 0.0)
        self.assertEqual(model.calls, diagnostics.true_nfe)
        self.assertEqual(model.calls, len(callbacks))
        self.assertLessEqual(model.calls, 20)
        self.assertTrue(all(not row["h3_vector_forecast"] for row in callbacks))
        self.assertIn("h3_vector_video_error", callbacks[3])
        self.assertIn("h3_vector_reference_video_error", callbacks[3])
        self.assertIn("h3_vector_video_error_ratio", callbacks[3])
        self.assertIn("h3_vector_reference_audio_error", callbacks[3])
        self.assertIn("h3_vector_audio_error_ratio", callbacks[3])
        self.assertIn("h3_vector_actual_delta_t", callbacks[3])
        self.assertIsNotNone(callbacks[3]["h3_vector_reference_video_error"])
        self.assertEqual(callbacks[3]["h3_vector_video_error_ratio"], 1.0)
        self.assertEqual(
            diagnostics._anchors[3]["trajectory_metrics"]["reference_video_error"],
            callbacks[3]["h3_vector_reference_video_error"],
        )
        self.assertEqual(
            diagnostics._run_metadata["configuration"]["adaptive_controller"]["version"],
            "adaptive_history_v3",
        )
        progress = [line for line in captured_logs.output if "[H3 Adaptive RES v3]" in line]
        self.assertEqual(len(progress), model.calls)
        self.assertIn("previous scale=", progress[3])
        self.assertIn("delta_t=", progress[3])
        self.assertIn("error video v=", progress[3])
        self.assertIn("audio v=", progress[3])
        self.assertIn("action=", progress[3])
        self.assertIn("next scale=", progress[3])

    def test_adaptive_embedded_res_reserves_terminal_anchor_and_reports_defect(self):
        sigmas = torch.tensor([
            1.0, 0.9956331849, 0.9908256531, 0.9855073094, 0.9795918465,
            0.9729729891, 0.9655172229, 0.9570552111, 0.9473683834,
            0.9361702204, 0.9230769277, 0.9075629711, 0.8888888955,
            0.8659793735, 0.8372092843, 0.8000000119, 0.75,
            0.6792452931, 0.5714285970, 0.3870967925, 0.0,
        ])
        config = SamplerConfig(
            method="res_multistep", evaluation_profile="adaptive_embedded_res_v1",
            embedded_video_tolerance=1e-8, adaptive_safety_factor=.8,
            max_adaptive_step_scale=1.0, max_adaptive_growth_ratio=2.0,
        )
        diagnostics = RunDiagnostics(config=config, latent_shapes=ResInner.latent_shapes)
        callbacks = []
        model = ResModel(lambda x, sigma: torch.full_like(x, 1.0 + .1 * sigma))
        with self.assertLogs(level="INFO") as captured_logs:
            sample_vector_accel(
                model, torch.zeros(1, 1, 6), sigmas, disable=True, config=config,
                diagnostics=diagnostics, callback=callbacks.append,
            )
        effective = diagnostics._run_metadata["effective_sigma_sequence"]
        decisions = diagnostics._run_metadata["adaptive_decisions"]
        self.assertEqual(model.calls, 20)
        self.assertEqual(model.calls, diagnostics.true_nfe)
        self.assertEqual(model.calls, len(callbacks))
        self.assertEqual(effective[-2:], [float(sigmas[19]), 0.0])
        self.assertEqual(decisions[0]["reason"], "bootstrap")
        self.assertEqual(decisions[-2]["reason"], "max_nfe_terminal_floor")
        self.assertEqual(decisions[-2]["clamp_selected"], "terminal_floor")
        self.assertEqual(decisions[-1]["reason"], "terminal_zero")
        self.assertEqual(callbacks[-1]["i"], 19)
        self.assertTrue(all(not row["h3_vector_forecast"] for row in callbacks))
        self.assertIn("h3_vector_tolerance_solution_h", callbacks[1])
        self.assertIn("h3_vector_defect_at_accepted_h", callbacks[1])
        self.assertIn("h3_vector_audio_defect_at_accepted_h", callbacks[1])
        self.assertEqual(
            diagnostics._run_metadata["configuration"]["adaptive_controller"]["version"],
            "adaptive_embedded_res_v1",
        )
        progress = [line for line in captured_logs.output
                    if "[H3 Adaptive RES embedded v1]" in line]
        self.assertEqual(len(progress), model.calls)
        self.assertIn("NFE 1/~", progress[0])
        self.assertIn("schedule 0.00/20", progress[0])
        self.assertIn("defect=", progress[1])
        self.assertIn("clamp=", progress[1])

    def test_core_solvers_reject_adaptive_and_invalid_source_schedule(self):
        with self.assertRaisesRegex(ValueError, "core solver methods require the fixed policy"):
            SamplerConfig(method="euler", policy="adaptive_repair", repairability_profile="measured.json")
        with self.assertRaisesRegex(ValueError, "exactly 20 logical steps"):
            sample_vector_accel(
                FakeModel(constant_velocity()), self.x.clone(), self.sigmas[:-1],
                config=SamplerConfig(method="euler"),
            )
        invalid = self.sigmas.clone()
        invalid[4] = invalid[3]
        with self.assertRaisesRegex(ValueError, "strictly descending"):
            sample_vector_accel(
                FakeModel(constant_velocity()), self.x.clone(), invalid,
                config=SamplerConfig(method="euler"),
            )

    def test_legacy_solver_and_profile_names_normalize(self):
        self.assertEqual(SamplerConfig(method="native").method, "euler")
        self.assertEqual(SamplerConfig(method="native").evaluation_profile, "full_20")
        self.assertEqual(SamplerConfig(method="sparse_euler").method, "euler")
        self.assertEqual(SamplerConfig(method="sparse_res_multistep").method, "res_multistep")
        self.assertEqual(
            SamplerConfig(evaluation_profile="native_20").evaluation_profile,
            "full_20",
        )
        self.assertIs(
            MiniMaxH3VectorAccelSampler.validate_inputs("native", "native_20"),
            True,
        )
        self.assertIs(
            MiniMaxH3VectorAccelSampler.validate_inputs(
                "sparse_res_multistep", "late_aggressive_13"
            ),
            True,
        )
        self.assertIn(
            "unknown vector acceleration method",
            MiniMaxH3VectorAccelSampler.validate_inputs("magic", "full_20"),
        )

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
        self.assertIsNotNone(callbacks[2]["h3_vector_guard_predicted_derivative_ratio"])
        self.assertIsNotNone(contexts[2]["h3_vector_guard_predicted_derivative_ratio"])
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
            method="linear_velocity", evaluation_profile="full_20",
            policy="adaptive_repair", repairability_profile="measured.json",
        )
        with mock.patch(
            "h3_vector_accel.sampler.RepairabilityProfile.load",
            return_value=AdaptiveProfile(),
        ):
            out = sample_vector_accel(model, self.x.clone(), self.sigmas, config=config)
        self.assertTrue(torch.equal(out, torch.full_like(self.x, -40.0)))
        self.assertEqual(model.calls, 14)


if __name__ == "__main__":
    unittest.main()
