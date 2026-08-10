"""CPU checks for the ordered fixed-policy experiment matrix."""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

_ORIGINAL_ARGV = list(sys.argv)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

from h3_vector_accel.study import (
    adaptive_comparison_arms,
    adaptive_history_arm,
    adaptive_history_v2_arm,
    adaptive_history_v3_arm,
    adaptive_embedded_res_v1_arm,
    fixed_policy_arms,
    four_arm_study_arms,
    geometric_schedule_arms,
    multiplicative_stride_arms,
    run_fixed_policy_study,
)
sys.argv = _ORIGINAL_ARGV


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
        late_tail = [arm for arm in arms if arm.phase == "late_tail_pace"]
        self.assertEqual(
            [(arm.evaluation_profile, arm.expected_true_nfe) for arm in late_tail],
            [("late_cautious_14", 14), ("late_aggressive_12", 12),
             ("late_max_11", 11)],
        )

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

    def test_four_arm_study_isolates_anchor_placement_and_integrator(self):
        arms = four_arm_study_arms()
        self.assertEqual(
            [(arm.method, arm.evaluation_profile, arm.expected_true_nfe) for arm in arms],
            [
                ("res_multistep", "full_20", 20),
                ("res_multistep", "late_aggressive_13", 13),
                ("linear_velocity", "late_aggressive_13", 13),
                ("euler", "late_aggressive_13", 13),
            ],
        )
        self.assertEqual({arm.phase for arm in arms}, {"solver_comparison"})

    def test_adaptive_history_arm_has_variable_nfe_and_no_fixed_indices(self):
        for arm, profile in (
            (adaptive_history_arm(), "adaptive_history_v1"),
            (adaptive_history_v2_arm(), "adaptive_history_v2"),
            (adaptive_history_v3_arm(), "adaptive_history_v3"),
            (adaptive_embedded_res_v1_arm(), "adaptive_embedded_res_v1"),
        ):
            self.assertEqual(arm.method, "res_multistep")
            self.assertEqual(arm.evaluation_profile, profile)
            self.assertIsNone(arm.expected_true_nfe)
            self.assertIsNone(arm.as_dict()["actual_indices"])

    def test_geometric_schedule_arms_have_equal_nfe_and_no_source_grid_indices(self):
        arms = geometric_schedule_arms()
        self.assertEqual(
            [(arm.evaluation_profile, arm.expected_true_nfe) for arm in arms],
            [("geometric_11", 11), ("geometric_linear_ends_11", 11)],
        )
        self.assertTrue(all(arm.method == "res_multistep" for arm in arms))
        self.assertTrue(all(arm.as_dict()["actual_indices"] is None for arm in arms))

    def test_multiplicative_stride_arms_have_equal_nfe_and_no_source_grid_indices(self):
        arms = multiplicative_stride_arms()
        self.assertEqual(
            [(arm.evaluation_profile, arm.expected_true_nfe) for arm in arms],
            [
                ("multiplicative_stride_11", 11),
                ("multiplicative_stride_linear_ends_11", 11),
            ],
        )
        self.assertTrue(all(arm.method == "res_multistep" for arm in arms))
        self.assertTrue(all(arm.as_dict()["actual_indices"] is None for arm in arms))


if __name__ == "__main__":
    unittest.main()
