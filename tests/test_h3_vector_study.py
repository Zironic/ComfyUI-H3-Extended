"""CPU checks for the ordered fixed-policy experiment matrix."""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_vector_accel.study import (
    adaptive_comparison_arms,
    fixed_policy_arms,
    run_fixed_policy_study,
)


class StudyTests(unittest.TestCase):
    def test_matrix_keeps_predictor_and_placement_controls_comparable(self):
        arms = fixed_policy_arms(include_vde=True)
        by_label = {arm.label: arm for arm in arms}
        self.assertEqual(by_label["hold_conservative_12"].expected_true_nfe, 12)
        self.assertEqual(by_label["linear_conservative_12"].expected_true_nfe, 12)
        placement = [arm for arm in arms if arm.phase == "placement"]
        self.assertEqual({arm.expected_true_nfe for arm in placement}, {13})
        self.assertEqual({arm.method for arm in placement}, {"linear_velocity"})
        self.assertEqual(by_label["vde_conservative_12"].evaluation_profile, "conservative_12")

    def test_results_require_separate_av_metrics_and_explicit_quality_pass(self):
        def runner(arm):
            return {
                "true_nfe": arm.expected_true_nfe,
                "wall_seconds": float(arm.expected_true_nfe),
                "quality_pass": arm.label == "linear_conservative_12",
                "video": {"psnr": 40.0},
                "audio": {"mse": 0.001},
            }

        result = run_fixed_policy_study(runner)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["arm"]["label"], "linear_conservative_12")

        with self.assertRaisesRegex(ValueError, "audio"):
            run_fixed_policy_study(lambda arm: {
                "true_nfe": arm.expected_true_nfe,
                "video": {},
            })

    def test_adaptive_comparison_keeps_selected_fixed_control(self):
        arms = adaptive_comparison_arms(
            "linear_velocity", "conservative_12", "measured.json",
            "linear_velocity", "continuation",
        )
        self.assertEqual(arms[0].expected_true_nfe, 12)
        self.assertTrue(all(arm.policy == "adaptive_repair" for arm in arms[1:]))
        self.assertEqual({arm.quality_preset for arm in arms[1:]}, {
            "conservative", "balanced", "aggressive",
        })


if __name__ == "__main__":
    unittest.main()
