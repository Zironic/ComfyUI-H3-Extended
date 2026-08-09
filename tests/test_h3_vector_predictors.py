"""CPU checks for genuine-anchor vector predictors."""
import os, sys, unittest
import torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
from h3_vector_accel.predictor import HoldPredictor, LinearVelocityPredictor, VDEPredictor

class PredictorTests(unittest.TestCase):
    def test_hold_uses_actual_only(self):
        p = HoldPredictor(); p.observe_actual(None, 3., torch.ones(2))
        first = p.predict(torch.zeros(2), 2.)
        p.observe_actual(None, 1., torch.full((2,), 4.))
        self.assertTrue(torch.equal(first.derivative, torch.ones(2)))
        self.assertTrue(torch.equal(p.predict(torch.zeros(2), 0.).derivative, torch.full((2,), 4.)))
        self.assertEqual(len(p.history), 2)

    def test_linear_integrates_linear_sigma_velocity(self):
        p = LinearVelocityPredictor(); a, b = 2., 3.
        for sigma in (5., 4.):
            d = a + b * sigma
            p.observe_actual(None, sigma, torch.tensor([d]))
        pred = p.predict(torch.zeros(1), 3.)
        out = p.integrate(torch.zeros(1), 3., 2., pred)
        expected = a * (2. - 3.) + .5 * b * (2. ** 2 - 3. ** 2)
        self.assertAlmostEqual(float(out), expected, places=5)

    def test_linear_rejects_duplicate_and_non_finite_anchors(self):
        p = LinearVelocityPredictor()
        p.observe_actual(None, 2.0, torch.ones(1))
        p.observe_actual(None, 2.0, torch.ones(1))
        self.assertEqual(p.predict(torch.zeros(1), 1.0).failure_reason,
                         "duplicate_anchor_sigma")
        p.reset()
        p.observe_actual(None, 2.0, torch.ones(1))
        p.observe_actual(None, 1.0, torch.full((1,), float("inf")))
        self.assertFalse(p.ready())

    def test_predictions_never_become_history(self):
        p = LinearVelocityPredictor()
        p.observe_actual(None, 3.0, torch.tensor([3.0]))
        p.observe_actual(None, 2.0, torch.tensor([2.0]))
        before = p.history
        p.predict(torch.zeros(1), 1.0)
        p.predict(torch.zeros(1), 0.0)
        self.assertEqual(len(p.history), len(before))
        for old, new in zip(before, p.history):
            self.assertEqual(old[0], new[0])
            self.assertTrue(torch.equal(old[1], new[1]))

    def test_predictor_snapshot_restores_actual_anchors(self):
        first = LinearVelocityPredictor()
        first.observe_actual(None, 3.0, torch.tensor([3.0]))
        first.observe_actual(None, 2.0, torch.tensor([2.0]))
        second = LinearVelocityPredictor()
        second.restore(first.snapshot())
        self.assertTrue(torch.equal(
            first.predict(torch.zeros(1), 1.0).derivative,
            second.predict(torch.zeros(1), 1.0).derivative,
        ))

    def test_vde_extrapolates_coefficients_and_reuses_direction(self):
        predictor = VDEPredictor()
        x = torch.tensor([[[1.0, 0.0]]])
        predictor.observe_actual(x, 3.0, torch.tensor([[[4.0, 7.0]]]))
        predictor.observe_actual(x, 2.0, torch.tensor([[[3.0, 5.0]]]))
        prediction = predictor.predict(x, 1.0)
        self.assertTrue(prediction.valid)
        self.assertTrue(torch.allclose(prediction.derivative, torch.tensor([[[2.0, 3.0]]]), atol=1e-6))
        restored = VDEPredictor()
        restored.restore(predictor.snapshot())
        self.assertTrue(torch.equal(restored.predict(x, 1.0).derivative, prediction.derivative))

    def test_vde_decomposes_video_and_audio_separately(self):
        shapes = [torch.Size((1, 1, 1, 1, 2)), torch.Size((1, 1, 2))]
        predictor = VDEPredictor(latent_shapes=shapes)
        x = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
        predictor.observe_actual(x, 3.0, torch.tensor([[[2.0, 1.0, 20.0, 10.0]]]))
        predictor.observe_actual(x, 2.0, torch.tensor([[[2.0, 1.0, 20.0, 10.0]]]))
        self.assertTrue(torch.equal(
            predictor.predict(x, 1.0).derivative,
            torch.tensor([[[2.0, 1.0, 20.0, 10.0]]]),
        ))

if __name__ == "__main__": unittest.main()
