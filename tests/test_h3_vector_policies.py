"""CPU checks for exact named fixed masks."""
import os, sys, unittest
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from h3_vector_accel.config import SamplerConfig, actual_mask
from h3_vector_accel.policy import make_policy

class PolicyTests(unittest.TestCase):
    def test_masks(self):
        expected = {
            "conservative_12": (0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19),
            "early_aggressive_13": (0, 1, 4, 7, 8, 10, 12, 14, 15, 16, 17, 18, 19),
            "uniform_13": (0, 1, 3, 5, 7, 9, 11, 13, 15, 16, 17, 18, 19),
            "late_aggressive_13": (0, 1, 2, 3, 4, 5, 7, 9, 12, 15, 17, 18, 19),
        }
        for profile, indices in expected.items():
            config = SamplerConfig(method="hold", evaluation_profile=profile)
            self.assertEqual(actual_mask(profile), indices)
            self.assertFalse(make_policy(config).decide(0).is_forecast)
            self.assertTrue(make_policy(config).decide(next(i for i, a in enumerate(config.mask) if not a)).is_forecast)

    def test_schedule_and_guard_validation(self):
        SamplerConfig(method="native").validate_schedule_length(7)
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            SamplerConfig(method="hold").validate_schedule_length(19)
        for value in (float("nan"), float("inf"), 0.0):
            with self.assertRaises(ValueError):
                SamplerConfig(max_extrapolation_ratio=value)

if __name__ == "__main__": unittest.main()
