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
        v1_payload = configuration_payload(v1, sigmas)
        v2_payload = configuration_payload(v2, sigmas)
        self.assertIsNone(v1_payload["actual_mask"])
        self.assertIsNone(v2_payload["actual_mask"])
        self.assertEqual(v1_payload["adaptive_controller"]["version"], "adaptive_history_v1")
        self.assertEqual(v2_payload["adaptive_controller"]["version"], "adaptive_history_v2")
        self.assertEqual(v1_payload["adaptive_controller"]["constants"]["high_change_ratio"], 1.5)
        self.assertEqual(v2_payload["adaptive_controller"]["constants"]["bootstrap_anchors"], 3)
        self.assertEqual(v2_payload["adaptive_controller"]["constants"]["reference_anchors"], 2)
        self.assertEqual(v2_payload["adaptive_controller"]["constants"]["protected_prefix"], 0)
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
        self.assertNotEqual(
            configuration_fingerprint(v1, sigmas),
            configuration_fingerprint(SamplerConfig(method="res_multistep"), sigmas),
        )

if __name__ == "__main__": unittest.main()
