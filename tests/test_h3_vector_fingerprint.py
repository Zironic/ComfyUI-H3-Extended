"""CPU checks for canonical vector sampler identities."""
import os, sys, unittest, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from h3_vector_accel.config import SamplerConfig
from h3_vector_accel.fingerprint import configuration_fingerprint

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

if __name__ == "__main__": unittest.main()
