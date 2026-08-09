"""CPU checks for exact named fixed masks."""
import os, sys, unittest
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

_ORIGINAL_ARGV = list(sys.argv)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

from h3_vector_accel.config import SamplerConfig, actual_mask  # noqa: E402
from h3_vector_accel.policy import AdaptiveRepairPolicy, make_policy  # noqa: E402
sys.argv = _ORIGINAL_ARGV


class Profile:
    def __init__(self, survival=0.1, tolerance=0.03):
        self.value = survival
        self._tolerance = tolerance

    def survival(self, progress):
        return {"video": self.value, "audio": self.value}

    def tolerance(self, preset):
        return self._tolerance


def metrics(video, audio):
    return {
        "video": {"integration_error_proxy": video},
        "audio": {"integration_error_proxy": audio},
    }


def _runs(values):
    runs = []
    current = []
    for value in values:
        if value:
            current.append(value)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs

class PolicyTests(unittest.TestCase):
    def test_masks(self):
        expected = {
            "conservative_12": (0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19),
            "early_aggressive_13": (0, 1, 4, 7, 8, 10, 12, 14, 15, 16, 17, 18, 19),
            "uniform_13": (0, 1, 3, 5, 7, 9, 11, 13, 15, 16, 17, 18, 19),
            "late_cautious_14": (0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 18, 19),
            "late_aggressive_13": (0, 1, 2, 3, 4, 5, 7, 9, 12, 15, 17, 18, 19),
            "late_aggressive_12": (0, 1, 2, 3, 4, 5, 9, 13, 16, 17, 18, 19),
            "late_max_11": (0, 1, 2, 3, 4, 5, 9, 13, 17, 18, 19),
        }
        for profile, indices in expected.items():
            config = SamplerConfig(method="hold", evaluation_profile=profile)
            self.assertEqual(actual_mask(profile), indices)
            self.assertFalse(make_policy(config).decide(0).is_forecast)
            self.assertTrue(make_policy(config).decide(next(i for i, a in enumerate(config.mask) if not a)).is_forecast)
        for profile, maximum_run in {
            "late_cautious_14": 1,
            "late_aggressive_13": 2,
            "late_aggressive_12": 3,
            "late_max_11": 3,
        }.items():
            forecasts = [not actual for actual in SamplerConfig(
                method="hold", evaluation_profile=profile,
            ).mask]
            self.assertEqual(max(map(len, _runs(forecasts))), maximum_run)


    def test_schedule_and_guard_validation(self):
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            SamplerConfig(method="euler").validate_schedule_length(19)
        for value in (float("nan"), float("inf"), 0.0):
            with self.assertRaises(ValueError):
                SamplerConfig(max_extrapolation_ratio=value)
        for value in (float("nan"), float("inf"), 0.99):
            with self.assertRaises(ValueError):
                SamplerConfig(max_adaptive_step_scale=value)

    def test_adaptive_uses_video_surviving_risk_and_one_forecast_limit(self):
        policy = AdaptiveRepairPolicy(
            Profile(), 0.03, 20, safety_factor=1.25,
            protected_prefix_steps=0,
        )
        policy.observe_actual(0, metrics(0.10, 0.05))
        policy.observe_actual(1, metrics(0.20, 0.10))
        decision = policy.decide(2, predictor_ready=True)
        self.assertTrue(decision.is_forecast)
        self.assertAlmostEqual(decision.risk, 0.025)
        policy.observe_step(True)
        self.assertEqual(policy.decide(3, predictor_ready=True).reason, "adaptive_consecutive_limit")
        policy.observe_step(False)
        self.assertTrue(policy.decide(4, predictor_ready=True).is_forecast)

    def test_adaptive_protects_six_step_prefix(self):
        policy = AdaptiveRepairPolicy(Profile(), 0.03, 20)
        for step in range(6):
            self.assertFalse(policy.decide(step, predictor_ready=True).is_forecast)

    def test_audio_only_emergency_veto(self):
        policy = AdaptiveRepairPolicy(
            Profile(), 0.03, 20, protected_prefix_steps=0,
        )
        policy._local_errors = {"video": [0.01, 0.01], "audio": [0.40, 0.40]}
        decision = policy.decide(6, predictor_ready=True)
        self.assertTrue(decision.is_forecast)
        self.assertGreater(decision.audio_risk, policy.tolerance)

        policy._local_errors = {"video": [0.01, 0.01], "audio": [0.01, 1.0]}
        decision = policy.decide(6, predictor_ready=True)
        self.assertFalse(decision.is_forecast)
        self.assertEqual(decision.reason, "adaptive_audio_emergency")
        self.assertAlmostEqual(decision.video_risk, 0.00125)
        self.assertAlmostEqual(decision.audio_risk, 0.125)

    def test_adaptive_forces_two_recovery_steps_after_high_error(self):
        policy = AdaptiveRepairPolicy(
            Profile(survival=1.0, tolerance=0.01), 0.01, 20,
            protected_prefix_steps=0,
        )
        policy.observe_actual(0, metrics(0.10, 0.10))
        policy.observe_actual(1, metrics(0.20, 0.20))
        self.assertEqual(policy.decide(2, predictor_ready=True).reason, "adaptive_recovery")
        self.assertEqual(policy.decide(3, predictor_ready=True).reason, "adaptive_recovery")
        self.assertEqual(policy.decide(4, predictor_ready=True).reason, "adaptive_risk_actual")

    def test_adaptive_config_requires_measured_profile(self):
        with self.assertRaisesRegex(ValueError, "repairability profile"):
            SamplerConfig(method="linear_velocity", policy="adaptive_repair")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            SamplerConfig(max_consecutive_forecasts=2)

if __name__ == "__main__": unittest.main()
