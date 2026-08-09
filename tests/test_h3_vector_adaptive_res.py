"""CPU-only checks for the adaptive RES controller and stateful stepper."""

import importlib.util
import math
import os
import unittest

import torch


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "h3_vector_adaptive_res", os.path.join(_ROOT, "h3_vector_accel", "adaptive_res.py")
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


class AdaptiveResTests(unittest.TestCase):
    def _reference_res(self, sigmas):
        x = torch.zeros(1, 1, 4)
        old_denoised = old_sigma_down = previous_sigma = None
        for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
            denoised = x - sigma * (1.0 + 0.03 * sigma + 0.2 * x)
            if sigma_next == 0 or old_denoised is None:
                x = x + (x - denoised) / sigma * (sigma_next - sigma)
            else:
                t, t_old = -sigma.log(), -old_sigma_down.log()
                t_next, t_prev = -sigma_next.log(), -previous_sigma.log()
                h = t_next - t
                c2 = (t_prev - t_old) / h
                p1 = torch.expm1(-h) / (-h)
                p2 = (p1 - 1) / (-h)
                b1 = torch.nan_to_num(p1 - p2 / c2, nan=0.0)
                b2 = torch.nan_to_num(p2 / c2, nan=0.0)
                x = torch.exp(-h) * x + h * (b1 * denoised + b2 * old_denoised)
            old_denoised, old_sigma_down, previous_sigma = denoised, sigma_next, sigma
        return x

    def test_incremental_euler_and_second_order_are_deterministic(self):
        sigmas = torch.tensor([20.0, 17.0, 13.0, 8.0, 4.0, 1.0, 0.0])
        x = torch.zeros(1, 1, 4)
        stepper = _MOD.IncrementalRES()
        for index in range(len(sigmas) - 1):
            sigma = sigmas[index]
            denoised = x - sigma * (1.0 + 0.03 * sigma + 0.2 * x)
            x = stepper.step(x, sigma, denoised, sigmas[index + 1])
        self.assertTrue(torch.isfinite(x).all())
        self.assertEqual(stepper.previous_sigma, 1.0)

    def test_incremental_matches_reference_full_and_irregular(self):
        for sigmas in (torch.linspace(20.0, 0.0, 21), torch.tensor([20., 17., 13., 8., 4., 1., 0.])):
            x = torch.zeros(1, 1, 4)
            stepper = _MOD.IncrementalRES()
            for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
                denoised = x - sigma * (1.0 + 0.03 * sigma + 0.2 * x)
                x = stepper.step(x, sigma, denoised, sigma_next)
            self.assertTrue(torch.allclose(x, self._reference_res(sigmas), atol=1e-6, rtol=1e-6))

    def test_schedule_protects_prefix_tail_and_terminal(self):
        source = torch.linspace(20.0, 0.0, 21)
        controller = _MOD.AdaptiveHistoryController(source)
        sigma = float(source[0])
        schedule = []
        while sigma > 0 and len(schedule) < 20:
            schedule.append(sigma)
            sigma, _ = controller.propose(sigma)
        schedule.append(0.0)
        self.assertEqual(schedule[:6], [float(v) for v in source[:6]])
        for actual, expected in zip(schedule[-4:], [float(v) for v in source[17:20]] + [0.0]):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertTrue(all(a > b for a, b in zip(schedule, schedule[1:-1])))
        self.assertLessEqual(len(schedule) - 1, 20)

    def test_low_and_high_change_update_scale_causally(self):
        source = torch.linspace(20.0, 0.0, 21)
        controller = _MOD.AdaptiveHistoryController(source)
        for i in range(6):
            controller.observe(float(source[i]), torch.ones(1), torch.ones(1))
        controller.reference_video_rate = 1.0
        controller.reference_audio_rate = 1.0
        obs = _MOD.AnchorObservation(
            15.0, 5, 1.0, 0.1, 1.0, 1.0, 0.1, 0.1,
            video_x0_change=0.0, audio_x0_change=0.1,
        )
        controller.propose(15.0, obs)
        self.assertEqual(controller.step_scale, 1.5)
        self.assertNotEqual(controller.decisions[-1]["reason"], "protected_prefix")
        obs = _MOD.AnchorObservation(
            7.0, 7, 0.1, 0.1, 1.0, 1.0, 2.05, 0.1,
            video_x0_change=4.0, audio_x0_change=0.1,
        )
        controller.propose(7.0, obs)
        self.assertAlmostEqual(controller.step_scale, 1.05)
        obs = _MOD.AnchorObservation(
            6.0, 8, 1.0, 0.1, 1.0, 1.0, 1.0, 5.05,
            video_x0_change=1.0, audio_x0_change=10.0,
        )
        controller.propose(6.0, obs)
        self.assertEqual(controller.decisions[-1]["reason"], "audio_emergency_shrink")

    def test_observation_splits_video_and_audio(self):
        shapes = [torch.Size((1, 1, 2)), torch.Size((1, 1, 2))]
        controller = _MOD.AdaptiveHistoryController(torch.linspace(20.0, 0.0, 21), shapes)
        old = torch.ones(1, 1, 4)
        new = old.clone()
        new[:, :, 2:] = 3.0
        observation = controller.observe(19.0, new, new, old, old, 20.0)
        self.assertEqual(observation.video_change, 0.0)
        self.assertGreater(observation.audio_change, 0.0)
        x0 = old.clone()
        x0[:, :, :2] = 2.0
        observation = controller.observe(18.0, old, x0, old, old, 19.0)
        self.assertEqual(observation.video_change, 0.0)
        self.assertGreater(observation.video_x0_change, 0.0)
        self.assertGreater(observation.video_rate, 0.0)
        self.assertEqual(observation.video_velocity_rate, 0.0)
        self.assertGreater(observation.video_x0_rate, 0.0)

    def test_v2_bootstrap_gate_and_tail(self):
        source = torch.linspace(20.0, 0.0, 21)
        controller = _MOD.AdaptiveHistoryControllerV2(source)
        self.assertEqual(controller.constants["protected_prefix"], 0)
        self.assertEqual(controller.constants["bootstrap_anchors"], 3)
        self.assertEqual(controller.constants["reference_anchors"], 2)
        for i in range(2):
            nxt, decision = controller.propose(float(source[i]))
            self.assertEqual(nxt, float(source[i + 1]))
            self.assertEqual(decision["reason"], "bootstrap")
            self.assertEqual(decision["protected_region"], "bootstrap")
            self.assertNotEqual(decision["reason"], "protected_prefix")
        controller.observe(float(source[0]), torch.ones(1), torch.ones(1))
        controller.observe(
            float(source[1]), torch.full((1,), 2.0), torch.full((1,), 2.0),
            torch.ones(1), torch.ones(1), float(source[0]),
        )
        self.assertIsNotNone(controller.reference_video_rate)
        controller.reference_video_rate = 1.0
        controller.reference_audio_rate = 1.0
        low = _MOD.AnchorObservation(
            float(source[2]), 2, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1,
        )
        _, decision = controller.propose(float(source[2]), low)
        self.assertEqual(controller.step_scale, 1.0)
        self.assertEqual(decision["reason"], "low_video_change_wait")
        low = _MOD.AnchorObservation(
            float(source[3]), 3, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1,
        )
        _, decision = controller.propose(float(source[3]), low)
        self.assertEqual(controller.step_scale, 1.5)
        self.assertEqual(decision["reason"], "low_video_change_grow")
        high = _MOD.AnchorObservation(6.0, 6, 2.0, 0.1, 1.0, 1.0, 2.0, 0.1)
        _, decision = controller.propose(6.0, high)
        self.assertEqual(decision["reason"], "high_video_change_shrink")
        self.assertAlmostEqual(controller.step_scale, 1.05)
        self.assertEqual(controller.low_change_streak, 0)
        controller.low_change_streak = 1
        audio_emergency = _MOD.AnchorObservation(
            5.0, 15, None, 0.1, None, 1.0, None, 4.1,
        )
        _, decision = controller.propose(5.0, audio_emergency)
        self.assertEqual(decision["reason"], "audio_emergency_shrink")
        self.assertEqual(controller.low_change_streak, 0)
        next_sigma, decision = controller.propose(float(source[18]))
        self.assertEqual(next_sigma, float(source[19]))
        self.assertEqual(decision["protected_region"], "tail")

    def test_v2_run_owned_max_step_scale(self):
        source = torch.linspace(20.0, 0.0, 21)
        controller = _MOD.AdaptiveHistoryControllerV2(source, max_step_scale=5.0)
        self.assertEqual(controller.max_step_scale, 5.0)
        self.assertEqual(controller.constants["step_scale_max"], 5.0)
        controller.reference_video_rate = 1.0
        controller.reference_audio_rate = 1.0
        controller.low_change_streak = 1
        controller.step_scale = 4.0
        low = _MOD.AnchorObservation(8.0, 12, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1)
        _, decision = controller.propose(8.0, low)
        self.assertEqual(controller.step_scale, 5.0)
        self.assertEqual(decision["reason"], "low_video_change_capped")
        next_sigma, decision = controller.propose(float(source[19]))
        self.assertEqual(next_sigma, 0.0)
        self.assertEqual(decision["protected_region"], "tail")

    def test_max_step_scale_must_be_finite_and_at_least_one(self):
        source = torch.linspace(20.0, 0.0, 21)
        for value in (0.99, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "max_step_scale must be finite and at least one"):
                _MOD.AdaptiveHistoryControllerV2(source, max_step_scale=value)

    def test_v3_linear_t_prediction_has_zero_residual_after_baseline_interval(self):
        source = torch.tensor([
            1.0, .95, .89, .82, .74, .65, .56, .48, .40, .33,
            .27, .22, .18, .14, .11, .08, .06, .04, .025, .01, 0.0,
        ])
        shapes = [torch.Size((1, 1, 2)), torch.Size((1, 1, 2))]
        controller = _MOD.AdaptiveHistoryControllerV3(source, shapes, max_step_scale=5.0)
        previous_d = previous_x0 = previous_sigma = None
        decisions = []
        for index in range(4):
            sigma = float(source[index])
            t = -math.log(sigma)
            derivative = torch.tensor([[[1.0 + 2.0 * t, 2.0 - t,
                                         3.0 + .5 * t, 4.0 - .25 * t]]])
            denoised = torch.tensor([[[5.0 - t, 6.0 + .75 * t,
                                      7.0 + .2 * t, 8.0 - .4 * t]]])
            observation = controller.observe(
                sigma, derivative, denoised,
                previous_d, previous_x0, previous_sigma,
            )
            _, decision = controller.propose(sigma, observation)
            decisions.append(decision)
            previous_d, previous_x0, previous_sigma = derivative, denoised, sigma
        self.assertEqual([row["reason"] for row in decisions[:2]], ["bootstrap"] * 2)
        self.assertEqual(decisions[2]["reason"], "bootstrap_predict")
        self.assertEqual(decisions[3]["action"], "reference_calibration")
        self.assertAlmostEqual(decisions[3]["actual_delta_t"],
                               -math.log(float(source[3])) + math.log(float(source[2])))
        for name, value in decisions[3]["residuals"].items():
            if name.endswith("_ratio"):
                continue
            if value is not None:
                self.assertLess(value, 1e-6)
        self.assertEqual(controller.step_scale, 1.0)
        self.assertIsNotNone(controller.reference_video_error)
        self.assertIsNotNone(controller.reference_audio_error)
        self.assertFalse(hasattr(controller, "_history"))

    def test_v3_ratio_bands_minimum_hold_and_recovery(self):
        source = torch.linspace(1.0, 0.0, 21)

        def observation(video_error, audio_error=.0, previous_scale=1.0):
            residuals = {
                "video_v_error": video_error,
                "video_x0_error": video_error / 2,
                "video_error": video_error,
                "audio_v_error": audio_error,
                "audio_x0_error": audio_error / 2,
                "audio_error": audio_error,
                "video_error_ratio": video_error,
                "audio_error_ratio": audio_error,
            }
            return _MOD.AnchorObservation(
                .75, 5, None, None, None, None, None, None,
                previous_step_scale=previous_scale, actual_delta_t=.1,
                residuals=residuals,
            )

        cases = (
            (.01, 1.0, "grow", 1.5),
            (.03, 2.0, "grow", 3.0),
            (.70, 2.0, "shrink", 1.4),
            (1.20, 2.0, "reset", 1.0),
        )
        for error, initial_scale, action, expected in cases:
            with self.subTest(error=error):
                controller = _MOD.AdaptiveHistoryControllerV3(source, max_step_scale=5.0)
                controller.step_scale = initial_scale
                _, decision = controller.propose(.75, observation(error, previous_scale=initial_scale))
                self.assertEqual(decision["action"], action)
                self.assertAlmostEqual(decision["step_scale"], expected)

        boundaries = (
            (.40, "hold", 2.0),
            (.70, "shrink", 1.4),
            (1.00, "reset", 1.0),
            (1.30, "critical_recovery", 1.0),
        )
        for ratio, action, expected in boundaries:
            with self.subTest(boundary=ratio):
                controller = _MOD.AdaptiveHistoryControllerV3(source, max_step_scale=5.0)
                controller.step_scale = 2.0
                _, decision = controller.propose(.75, observation(ratio, previous_scale=2.0))
                self.assertEqual(decision["action"], action)
                self.assertAlmostEqual(decision["step_scale"], expected)

        minimum = _MOD.AdaptiveHistoryControllerV3(source, max_step_scale=5.0)
        _, decision = minimum.propose(.75, observation(float("inf")))
        self.assertEqual(decision["action"], "minimum_step_hold")
        self.assertEqual(decision["recovery_remaining"], 0)

        capped = _MOD.AdaptiveHistoryControllerV3(source, max_step_scale=5.0)
        capped.step_scale = 4.0
        _, decision = capped.propose(.75, observation(.01, previous_scale=4.0))
        self.assertEqual(decision["action"], "capped")
        self.assertEqual(decision["step_scale"], 5.0)

        audio = _MOD.AdaptiveHistoryControllerV3(source, max_step_scale=5.0)
        audio.step_scale = 4.0
        _, decision = audio.propose(.75, observation(.01, 2.0, 4.0))
        self.assertEqual(decision["action"], "capped")
        self.assertFalse(decision["audio_emergency"])
        self.assertEqual(decision["step_scale"], 5.0)

        audio_nonfinite = _MOD.AdaptiveHistoryControllerV3(source, max_step_scale=5.0)
        audio_nonfinite.step_scale = 4.0
        _, decision = audio_nonfinite.propose(
            .75, observation(.01, float("inf"), previous_scale=4.0)
        )
        self.assertEqual(decision["action"], "capped")
        self.assertFalse(decision["audio_emergency"])

        critical = _MOD.AdaptiveHistoryControllerV3(source, max_step_scale=5.0)
        critical.step_scale = 4.0
        _, first = critical.propose(.75, observation(1.30, previous_scale=4.0))
        _, recovery = critical.propose(.70, observation(.01))
        _, resumed = critical.propose(.65, observation(.01))
        self.assertEqual(first["action"], "critical_recovery")
        self.assertEqual(recovery["action"], "forced_recovery")
        self.assertEqual(recovery["step_scale"], 1.0)
        self.assertEqual(resumed["action"], "grow")

        terminal = _MOD.AdaptiveHistoryControllerV3(source, max_step_scale=10.0)
        self.assertEqual(terminal.constants["protected_tail"], ())
        terminal.step_scale = 10.0
        _, decision = terminal.propose(float(source[16]), observation(.01, previous_scale=10.0))
        self.assertEqual(decision["next_sigma"], 0.0)
        self.assertEqual(decision["action"], "capped")

    def test_embedded_defect_is_monotonic_and_tolerance_selects_interval(self):
        values = torch.exp(-torch.linspace(0.0, 2.0, 20))
        source = torch.cat((values, torch.zeros(1)))
        shapes = [torch.Size((1, 1, 2)), torch.Size((1, 1, 2))]
        controller = _MOD.AdaptiveEmbeddedRESController(
            source, shapes, max_step_scale=10.0, video_tolerance=.05,
            safety_factor=.8, max_growth_ratio=10.0,
        )
        h_previous = -math.log(float(source[1])) + math.log(float(source[0]))
        controller.previous_accepted_h = h_previous
        defects = [controller.embedded_defect(h, h_previous, .2, 1.0)
                   for h in (.01, .05, .1, .2, .4)]
        self.assertTrue(all(a < b for a, b in zip(defects, defects[1:])))

        observation = _MOD.AnchorObservation(
            float(source[1]), 1, None, None, None, None, None, None,
            video_x0_difference_rms=.2, audio_x0_difference_rms=.4,
        )
        next_sigma, decision = controller.propose(
            float(source[1]), observation,
            current_x=torch.ones(1, 1, 4), current_x0=torch.ones(1, 1, 4),
        )
        self.assertEqual(decision["clamp_selected"], "tolerance")
        self.assertAlmostEqual(
            controller.embedded_defect(
                decision["tolerance_solution_h"], h_previous, .2, 1.0
            ), .05, places=6,
        )
        self.assertAlmostEqual(
            decision["accepted_h"], decision["tolerance_solution_h"] * .8, places=7
        )
        self.assertLess(decision["defect_at_accepted_h"], .05)
        self.assertGreater(next_sigma, float(source[19]))
        self.assertLess(next_sigma, float(source[1]))

    def test_embedded_clamps_and_audio_is_diagnostic_only(self):
        values = torch.exp(-torch.linspace(0.0, 2.0, 20))
        source = torch.cat((values, torch.zeros(1)))
        shapes = [torch.Size((1, 1, 2)), torch.Size((1, 1, 2))]
        h_previous = -math.log(float(source[1])) + math.log(float(source[0]))

        def propose(index, *, max_scale, growth, video=.1, audio=.1):
            controller = _MOD.AdaptiveEmbeddedRESController(
                source, shapes, max_step_scale=max_scale, video_tolerance=100.0,
                safety_factor=1.0, max_growth_ratio=growth,
            )
            controller.previous_accepted_h = h_previous
            observation = _MOD.AnchorObservation(
                float(source[index]), index, None, None, None, None, None, None,
                video_x0_difference_rms=video, audio_x0_difference_rms=audio,
            )
            _, decision = controller.propose(
                float(source[index]), observation,
                current_x=torch.ones(1, 1, 4), current_x0=torch.ones(1, 1, 4),
            )
            return decision

        absolute = propose(5, max_scale=1.0, growth=100.0)
        self.assertEqual(absolute["clamp_selected"], "absolute")
        self.assertAlmostEqual(absolute["step_scale"], 1.0, places=6)
        growth = propose(5, max_scale=100.0, growth=1.5)
        self.assertEqual(growth["clamp_selected"], "growth")
        self.assertAlmostEqual(growth["growth_ratio"], 1.5, places=6)
        terminal = propose(18, max_scale=100.0, growth=100.0)
        self.assertEqual(terminal["clamp_selected"], "terminal_floor")
        self.assertAlmostEqual(terminal["next_sigma"], float(source[19]), places=7)

        quiet_audio = propose(5, max_scale=10.0, growth=10.0, audio=.01)
        loud_audio = propose(5, max_scale=10.0, growth=10.0, audio=100.0)
        self.assertAlmostEqual(quiet_audio["accepted_h"], loud_audio["accepted_h"], places=12)
        self.assertGreater(loud_audio["audio_defect_at_accepted_h"],
                           quiet_audio["audio_defect_at_accepted_h"])

    def test_embedded_bootstraps_once_and_terminal_zero_is_special(self):
        values = torch.exp(-torch.linspace(0.0, 2.0, 20))
        source = torch.cat((values, torch.zeros(1)))
        controller = _MOD.AdaptiveEmbeddedRESController(source)
        next_sigma, first = controller.propose(float(source[0]))
        self.assertEqual(first["reason"], "bootstrap")
        self.assertAlmostEqual(next_sigma, float(source[1]))
        next_sigma, terminal = controller.propose(float(source[19]))
        self.assertEqual(next_sigma, 0.0)
        self.assertEqual(terminal["reason"], "terminal_zero")
        self.assertEqual(controller.constants["protected_tail"], ())


if __name__ == "__main__":
    unittest.main()
