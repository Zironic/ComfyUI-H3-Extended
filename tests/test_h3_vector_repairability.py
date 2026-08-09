"""CPU trajectory replay and repairability-profile checks."""

import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_vector_accel.fingerprint import sigma_hash
from h3_vector_accel.repairability import (
    ProfileCompatibility,
    RepairabilityProfile,
    build_repairability_profile,
    capture_native_trajectory,
    run_natural_omission,
    run_normalized_perturbation,
)


class FakeModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, x, sigma, **kwargs):
        self.calls += 1
        value = 0.05 * x + 0.01 * float(sigma[0]) ** 2 + 1.0
        return x - sigma.reshape((-1,) + (1,) * (x.ndim - 1)) * value


class RepairabilityTests(unittest.TestCase):
    def setUp(self):
        self.shapes = [torch.Size((1, 1, 1, 1, 2)), torch.Size((1, 1, 2))]
        self.sigmas = torch.linspace(20.0, 0.0, 21)
        self.x = torch.zeros(1, 1, 4)

    def test_capture_restores_anchor_history_and_continues_natively(self):
        reference_model = FakeModel()
        trajectory = capture_native_trajectory(
            reference_model, self.x.clone(), self.sigmas, latent_shapes=self.shapes
        )
        self.assertEqual(trajectory.logical_steps, 20)
        self.assertEqual(trajectory.snapshots[2].x.dtype, torch.float32)
        self.assertEqual(len(trajectory.snapshots[2].predictor_states["linear_velocity"]), 2)
        branch_model = FakeModel()
        natural = run_natural_omission(branch_model, trajectory, 2, "linear_velocity")
        self.assertEqual(natural.branch_type, "natural_omission")
        self.assertEqual(natural.divergence_curve[0]["state_index"], 3)
        self.assertEqual(natural.divergence_curve[-1]["state_index"], 20)
        self.assertGreater(float(natural.natural_delta.abs().max()), 0.0)
        self.assertEqual(branch_model.calls, 17)

        normalized = run_normalized_perturbation(
            FakeModel(), trajectory, natural, "video", target_rms=0.01
        )
        self.assertEqual(normalized.modality, "video")
        self.assertTrue(torch.isfinite(torch.tensor(list(normalized.survival.values()))).all())

    def test_profile_matches_exact_run_identity_and_owns_tolerances(self):
        entries = [
            {"progress": 0.1, "survival": {"video": 0.2, "audio": 0.3}},
            {"progress": 0.5, "survival": {"video": 0.6, "audio": 0.4}},
        ]
        payload = build_repairability_profile(
            entries, sigmas=self.sigmas, model_fingerprint="model-a",
            video_shift=1.0, audio_shift=2.0, nominal_steps=20,
            predictor_method="linear_velocity", conditioning_mode="continuation",
            quality_presets={"conservative": 0.01, "balanced": 0.02, "aggressive": 0.04},
            adaptive_methods=["linear_velocity"],
        )
        profile = RepairabilityProfile(payload)
        context = ProfileCompatibility(
            "model-a", sigma_hash(self.sigmas), 1.0, 2.0, 20,
            "linear_velocity", "continuation",
        )
        self.assertTrue(profile.validate_compatibility(context))
        self.assertEqual(profile.tolerance("balanced"), 0.02)
        self.assertEqual(profile.survival(0.3), {"video": 0.6, "audio": 0.4})
        with self.assertRaisesRegex(ValueError, "sigma_hash"):
            profile.validate_compatibility(ProfileCompatibility(
                "model-a", "wrong", 1.0, 2.0, 20, "linear_velocity", "continuation"
            ))

    def test_vde_adaptive_requires_explicit_profile_approval(self):
        payload = build_repairability_profile(
            [{"progress": 0.5, "survival": {"video": 0.2, "audio": 0.2}}],
            sigmas=self.sigmas, model_fingerprint="model-a", video_shift=1.0,
            audio_shift=2.0, nominal_steps=20, predictor_method="vde",
            conditioning_mode="default", quality_presets={"balanced": 0.02},
            adaptive_methods=[],
        )
        with self.assertRaisesRegex(ValueError, "not approved"):
            RepairabilityProfile(payload).validate_compatibility(ProfileCompatibility(
                "model-a", sigma_hash(self.sigmas), 1.0, 2.0, 20, "vde", "default"
            ))


if __name__ == "__main__":
    unittest.main()
