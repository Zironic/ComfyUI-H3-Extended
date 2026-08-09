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


if __name__ == "__main__":
    unittest.main()
