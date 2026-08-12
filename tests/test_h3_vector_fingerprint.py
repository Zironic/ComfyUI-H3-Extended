"""CPU checks for canonical vector sampler identities."""
import os, sys, unittest, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

_ORIGINAL_ARGV = list(sys.argv)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

from h3_vector_accel.config import SamplerConfig  # noqa: E402
from h3_vector_accel.fingerprint import configuration_fingerprint, configuration_payload  # noqa: E402
from h3_vector_accel.schedules import continuous_schedule, continuous_schedule_family  # noqa: E402
sys.argv = _ORIGINAL_ARGV

class FingerprintTests(unittest.TestCase):
    def test_changes_with_method_profile_and_sigma(self):
        s = torch.linspace(20., 0., 21)
        base = configuration_fingerprint(SamplerConfig(), s)
        variants = [
            SamplerConfig(method="hold"),
            SamplerConfig(evaluation_profile="uniform_13"),
            SamplerConfig(max_extrapolation_ratio=1.25),
            SamplerConfig(fallback_on_guard=False),
            SamplerConfig(protected_prefix_steps=7),
            SamplerConfig(audio_emergency_multiplier=5.0),
            SamplerConfig(max_adaptive_step_scale=5.0),
            SamplerConfig(embedded_video_tolerance=.03),
            SamplerConfig(adaptive_safety_factor=.7),
            SamplerConfig(max_adaptive_growth_ratio=1.5),
        ]
        for config in variants:
            self.assertNotEqual(base, configuration_fingerprint(config, s))
        altered = s.clone(); altered[5] -= 0.125
        self.assertNotEqual(base, configuration_fingerprint(SamplerConfig(), altered))

    def test_fingerprint_is_deterministic(self):
        config = SamplerConfig(method="linear_velocity", evaluation_profile="late_aggressive_13")
        sigmas = torch.linspace(20., 0., 21)
        self.assertEqual(configuration_fingerprint(config, sigmas),
                         configuration_fingerprint(config, sigmas.clone()))

    def test_core_solver_fingerprint_includes_effective_schedule(self):
        sigmas = torch.linspace(20., 0., 21)
        config = SamplerConfig(
            method="res_multistep",
            evaluation_profile="late_aggressive_13",
        )
        indices = config.actual_indices
        effective = torch.cat((sigmas[list(indices)], sigmas[-1:]))
        base = configuration_fingerprint(
            config, sigmas, effective_sigmas=effective, actual_indices=indices,
        )
        changed = effective.clone()
        changed[6] -= 0.125
        self.assertNotEqual(base, configuration_fingerprint(
            config, sigmas, effective_sigmas=changed, actual_indices=indices,
        ))

    def test_adaptive_controller_identity_is_fingerprinted_without_a_fixed_mask(self):
        sigmas = torch.linspace(1.0, 0.0, 21)
        v1 = SamplerConfig(
            method="res_multistep", evaluation_profile="adaptive_history_v1",
        )
        v2 = SamplerConfig(
            method="res_multistep", evaluation_profile="adaptive_history_v2",
        )
        v3 = SamplerConfig(
            method="res_multistep", evaluation_profile="adaptive_history_v3",
        )
        embedded = SamplerConfig(
            method="res_multistep", evaluation_profile="adaptive_embedded_res_v1",
        )
        v1_payload = configuration_payload(v1, sigmas)
        v2_payload = configuration_payload(v2, sigmas)
        v3_payload = configuration_payload(v3, sigmas)
        embedded_payload = configuration_payload(embedded, sigmas)
        self.assertIsNone(v1_payload["actual_mask"])
        self.assertIsNone(v2_payload["actual_mask"])
        self.assertEqual(v1_payload["adaptive_controller"]["version"], "adaptive_history_v1")
        self.assertEqual(v2_payload["adaptive_controller"]["version"], "adaptive_history_v2")
        self.assertEqual(v1_payload["adaptive_controller"]["constants"]["high_change_ratio"], 1.5)
        self.assertEqual(v2_payload["adaptive_controller"]["constants"]["bootstrap_anchors"], 3)
        self.assertEqual(v2_payload["adaptive_controller"]["constants"]["reference_anchors"], 2)
        self.assertEqual(v2_payload["adaptive_controller"]["constants"]["protected_prefix"], 0)
        self.assertEqual(v3_payload["adaptive_controller"]["version"], "adaptive_history_v3")
        self.assertEqual(v3_payload["adaptive_controller"]["constants"]["predictor"],
                         "linear_secant_negative_log_sigma")
        self.assertEqual(v3_payload["adaptive_controller"]["constants"]["grow_below"], .40)
        self.assertEqual(v3_payload["adaptive_controller"]["constants"]["reset_below"], 1.30)
        self.assertEqual(v3_payload["adaptive_controller"]["constants"]["protected_tail"], ())
        self.assertEqual(embedded_payload["adaptive_controller"]["version"],
                         "adaptive_embedded_res_v1")
        self.assertEqual(embedded_payload["adaptive_controller"]["constants"]["defect"],
                         "eta_zero_incremental_res")
        self.assertEqual(embedded_payload["adaptive_controller"]["constants"]["control"],
                         "video_only")
        self.assertEqual(embedded_payload["embedded_video_tolerance"], .05)
        v2_5x = SamplerConfig(
            method="res_multistep", evaluation_profile="adaptive_history_v2",
            max_adaptive_step_scale=5.0,
        )
        v2_5x_payload = configuration_payload(v2_5x, sigmas)
        self.assertEqual(v2_5x_payload["adaptive_controller"]["constants"]["step_scale_max"], 5.0)
        self.assertNotEqual(configuration_fingerprint(v2, sigmas),
                            configuration_fingerprint(v2_5x, sigmas))
        self.assertNotEqual(configuration_fingerprint(v1, sigmas),
                            configuration_fingerprint(v2, sigmas))
        self.assertNotEqual(configuration_fingerprint(v2, sigmas),
                            configuration_fingerprint(v3, sigmas))
        self.assertNotEqual(configuration_fingerprint(v3, sigmas),
                            configuration_fingerprint(embedded, sigmas))
        self.assertNotEqual(
            configuration_fingerprint(v1, sigmas),
            configuration_fingerprint(SamplerConfig(method="res_multistep"), sigmas),
        )

    def test_geometric_schedule_identity_and_effective_sigmas_are_fingerprinted(self):
        sigmas = torch.linspace(20.0, 0.0, 21)
        geometric = SamplerConfig(
            method="res_multistep", evaluation_profile="geometric_11",
        )
        linear_ends = SamplerConfig(
            method="res_multistep", evaluation_profile="geometric_linear_ends_11",
        )
        geometric_sigmas, _, _ = continuous_schedule(sigmas, geometric.evaluation_profile)
        linear_sigmas, _, _ = continuous_schedule(sigmas, linear_ends.evaluation_profile)
        payload = configuration_payload(
            geometric, sigmas, effective_sigmas=geometric_sigmas,
        )
        self.assertIsNone(payload["actual_mask"])
        self.assertEqual(payload["continuous_schedule"]["true_nfe"], 11)
        self.assertEqual(payload["continuous_schedule"]["time_coordinate"], "sigma")
        self.assertEqual(
            payload["continuous_schedule"]["interval_rule"],
            "normalized_reverse_powers_sum_to_sigma_span",
        )
        self.assertNotEqual(
            configuration_fingerprint(
                geometric, sigmas, effective_sigmas=geometric_sigmas,
            ),
            configuration_fingerprint(
                linear_ends, sigmas, effective_sigmas=linear_sigmas,
            ),
        )

    def test_arbitrary_continuous_schedule_families_preserve_contract(self):
        sigmas = torch.linspace(20.0, 0.0, 21, dtype=torch.float64)
        families = (
            "geometric", "geometric_linear_ends",
            "multiplicative_stride", "multiplicative_stride_linear_ends",
        )
        for family in families:
            for steps in (7, 11, 15):
                effective, coordinates, ratio = continuous_schedule_family(
                    sigmas, family, steps,
                )
                self.assertEqual(effective.shape, (steps + 1,), (family, steps))
                self.assertEqual(len(coordinates), steps, (family, steps))
                self.assertEqual(effective.dtype, sigmas.dtype)
                self.assertEqual(effective.device, sigmas.device)
                self.assertTrue(torch.isfinite(effective).all())
                self.assertTrue(torch.all(effective[:-1] > 0.0))
                self.assertTrue(torch.all(effective[:-1] > effective[1:]))
                self.assertEqual(float(effective[0]), float(sigmas[0]))
                self.assertEqual(float(effective[-1]), 0.0)
                self.assertTrue(torch.isfinite(torch.tensor(ratio)))
                if family.endswith("_linear_ends"):
                    self.assertTrue(torch.equal(effective[:3], sigmas[:3]))
                    self.assertTrue(torch.equal(effective[-3:], sigmas[-3:]))

        _, _, ratio = continuous_schedule_family(sigmas, "multiplicative_stride", 21)
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 1.0)

    def test_linear_end_families_reject_overlapping_protected_endpoints(self):
        sigmas = torch.linspace(20.0, 0.0, 21)
        for family in ("geometric_linear_ends", "multiplicative_stride_linear_ends"):
            for steps in (1, 2, 3, 4):
                with self.assertRaisesRegex(ValueError, "at least 5 steps"):
                    continuous_schedule_family(sigmas, family, steps)
        with self.assertRaisesRegex(ValueError, "at least 2 steps"):
            continuous_schedule_family(sigmas, "multiplicative_stride", 1)

    def test_legacy_continuous_profiles_are_exact_11_step_aliases(self):
        sigmas = torch.linspace(20.0, 0.0, 21, dtype=torch.float64)
        for family in (
            "geometric", "geometric_linear_ends",
            "multiplicative_stride", "multiplicative_stride_linear_ends",
        ):
            legacy = continuous_schedule(sigmas, f"{family}_11")
            arbitrary = continuous_schedule_family(sigmas, family, 11)
            self.assertTrue(torch.equal(legacy[0], arbitrary[0]), family)
            self.assertEqual(legacy[1:], arbitrary[1:], family)

    def test_multiplicative_stride_profiles_have_distinct_schedule_identities(self):
        sigmas = torch.linspace(20.0, 0.0, 21)
        full = SamplerConfig(
            method="res_multistep", evaluation_profile="multiplicative_stride_11",
        )
        linear = SamplerConfig(
            method="res_multistep",
            evaluation_profile="multiplicative_stride_linear_ends_11",
        )
        full_sigmas, _, _ = continuous_schedule(sigmas, full.evaluation_profile)
        linear_sigmas, _, _ = continuous_schedule(sigmas, linear.evaluation_profile)
        full_payload = configuration_payload(full, sigmas, effective_sigmas=full_sigmas)
        linear_payload = configuration_payload(linear, sigmas, effective_sigmas=linear_sigmas)
        self.assertEqual(
            full_payload["continuous_schedule"]["interval_rule"],
            "unit_first_multiplicative_stride_sum_to_20",
        )
        self.assertEqual(
            linear_payload["continuous_schedule"]["interval_rule"],
            "native_0_1_2_multiplicative_interior_native_18_19_20",
        )
        self.assertNotEqual(
            configuration_fingerprint(full, sigmas, effective_sigmas=full_sigmas),
            configuration_fingerprint(linear, sigmas, effective_sigmas=linear_sigmas),
        )

if __name__ == "__main__": unittest.main()
