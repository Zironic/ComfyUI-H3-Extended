"""CPU-only tests for the offline H3 temporal descriptor."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(PACK_ROOT / "h3_vector_accel"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from safetensors.torch import save_file

from benchmarks.h3_temporal_descriptor_probe import render_output, run_probe
from h3_test_tempfile import TemporaryDirectory
from temporal_descriptor import (
    compare_temporal_descriptors,
    descriptor_summary,
    extract_temporal_descriptor,
    transition_summary,
)


def _base_frame(height: int = 16, width: int = 16) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height).view(height, 1)
    x = torch.linspace(-1.0, 1.0, width).view(1, width)
    return torch.exp(-5.0 * (x.square() + y.square())) + 0.25 * x - 0.15 * y


def _static_pattern(temporal: int = 8) -> torch.Tensor:
    return _base_frame().view(1, 1, 1, 16, 16).repeat(1, 2, temporal, 1, 1)


def _moving_pattern(temporal: int = 8) -> torch.Tensor:
    base = _base_frame()
    frames = torch.stack([
        torch.roll(base, shifts=(index, index // 2), dims=(0, 1))
        for index in range(temporal)
    ], dim=0)
    return frames.view(1, 1, temporal, 16, 16).repeat(1, 2, 1, 1, 1)


class TemporalDescriptorTests(unittest.TestCase):
    def test_featureless_and_structured_static_paths(self):
        featureless = torch.ones(1, 2, 5, 16, 16)
        structured = _static_pattern(5)
        featureless_descriptor = extract_temporal_descriptor(featureless)
        structured_descriptor = extract_temporal_descriptor(structured)
        self.assertTrue(torch.equal(featureless_descriptor.spatial_structure_score, torch.zeros(5)))
        self.assertTrue(torch.all(structured_descriptor.spatial_structure_score > 0))
        for descriptor in (featureless_descriptor, structured_descriptor):
            transition = compare_temporal_descriptors(descriptor, descriptor)
            summary = transition_summary(transition)
            self.assertEqual(summary["motion_observable_fraction"], 0.0)
            self.assertIsNone(summary["motion_alignment_margin"])
            self.assertIsNone(summary["motion_normalized_argmax_warp"])
            json.dumps(summary, allow_nan=False)
        pool_one = descriptor_summary(extract_temporal_descriptor(structured, pool_size=1))
        self.assertEqual(pool_one["spatial_gradient_energy"]["max"], 0.0)
        json.dumps(pool_one, allow_nan=False)

    def test_coherent_spatial_translation_beats_temporal_permutation(self):
        source = _moving_pattern(8)
        coherent = torch.roll(source, shifts=1, dims=3)
        permuted = coherent.flip(2)
        base = extract_temporal_descriptor(source)
        coherent_transition = transition_summary(compare_temporal_descriptors(
            base, extract_temporal_descriptor(coherent), motion_energy_floor=1e-8
        ))
        permuted_transition = transition_summary(compare_temporal_descriptors(
            base, extract_temporal_descriptor(permuted), motion_energy_floor=1e-8
        ))
        self.assertGreater(
            coherent_transition["frame_same_position_score"],
            permuted_transition["frame_same_position_score"],
        )
        self.assertLess(
            coherent_transition["frame_normalized_argmax_warp"],
            permuted_transition["frame_normalized_argmax_warp"],
        )
        self.assertGreater(
            coherent_transition["motion_same_position_score"],
            permuted_transition["motion_same_position_score"],
        )

    def test_localized_break_reaches_upper_quantile(self):
        source = _moving_pattern(12)
        broken = source.clone()
        broken[:, :, 7] = torch.flip(broken[:, :, 7], dims=(2, 3)) * -3.0
        summary = transition_summary(compare_temporal_descriptors(
            extract_temporal_descriptor(source), extract_temporal_descriptor(broken)
        ))
        first = summary["localized_first_difference_change"]
        second = summary["localized_second_difference_change"]
        self.assertGreater(first["p95"], first["p50"])
        self.assertGreater(second["p95"], second["p50"])
        self.assertGreater(first["max"], 0.0)

    def test_normalized_measurements_are_scale_invariant(self):
        source = _moving_pattern(8)
        original = extract_temporal_descriptor(source)
        scaled = extract_temporal_descriptor(source * 17.0)
        for left, right in (
            (original.spatial_contrast, scaled.spatial_contrast),
            (original.spatial_gradient_energy, scaled.spatial_gradient_energy),
            (original.spatial_structure_score, scaled.spatial_structure_score),
            (original.first_difference_relative_energy, scaled.first_difference_relative_energy),
            (original.second_difference_relative_energy, scaled.second_difference_relative_energy),
        ):
            self.assertTrue(torch.allclose(left, right, rtol=1e-4, atol=1e-5))

    def test_explicit_coverage_detects_unstructured_back_half(self):
        structured = _static_pattern(8)
        full_descriptor = extract_temporal_descriptor(structured)
        threshold = float(full_descriptor.spatial_structure_score.min().item() * 0.5)
        front_only = structured.clone()
        front_only[:, :, 4:] = 0.0
        full_summary = descriptor_summary(full_descriptor, structure_threshold=threshold)
        front_summary = descriptor_summary(
            extract_temporal_descriptor(front_only), structure_threshold=threshold
        )
        self.assertEqual(full_summary["structure_coverage"], 1.0)
        self.assertEqual(front_summary["structure_coverage"], 0.5)

    def test_probe_manifest_order_shape_and_deterministic_output(self):
        with TemporaryDirectory(prefix="descriptor_probe_") as temporary:
            root = Path(temporary)
            tensors = [_moving_pattern(5), _moving_pattern(5) * 1.2]
            callbacks = []
            for file_index, (order, tensor) in enumerate(((3, tensors[0]), (1, tensors[1]))):
                artifact = root / f"callback_{file_index}.safetensors"
                save_file({"x0.video": tensor.contiguous(), "ignored": torch.ones(2)}, str(artifact))
                callbacks.append({
                    "callback": file_index,
                    "order": order,
                    "step": order,
                    "artifact": artifact.name,
                    "sigma": 1.0 - file_index * 0.1,
                    "tensors": {
                        "x0.video": {"shape": list(tensor.shape), "dtype": "float32"},
                        "ignored": {"shape": [2], "dtype": "float32"},
                    },
                    "metadata": {"h3_vector_true_nfe": file_index + 1},
                })
            manifest = {
                "run_id": "synthetic",
                "status": "complete",
                "seed": 7,
                "source_sigmas": [1.0, 0.9, 0.0],
                "callbacks": callbacks,
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            payload = run_probe(root)
            self.assertEqual([row["order"] for row in payload["callbacks"]], [3, 1])
            first_json = render_output(payload, "json")
            second_json = render_output(run_probe(root), "json")
            self.assertEqual(first_json, second_json)
            self.assertIn("callbacks.0.order", render_output(payload, "csv"))
            self.assertNotIn("NaN", first_json)
            self.assertNotIn("Infinity", first_json)

            manifest["callbacks"][1]["tensors"]["x0.video"]["shape"][-1] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                run_probe(root)


if __name__ == "__main__":
    unittest.main()
