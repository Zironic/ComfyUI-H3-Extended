import json
import os
import sys
import tempfile
import unittest

import comfy.options
comfy.options.enable_args_parsing()
_ARGV, sys.argv = list(sys.argv), [sys.argv[0], "--cpu"]

import torch

from chunked_ref2v.longform.chunk_stream import (
    actual_frames_for_chunk,
    chunk_count_for,
    frames_needed_for,
    plan_chunks,
)
from chunked_ref2v.longform.manifest import (
    RunManifest,
    SCHEMA_VERSION,
    identity_hash,
    object_fingerprint,
)
from chunked_ref2v.longform.reference_runner import (
    _pack_blocks,
    _paired_audio,
    _reference_identity,
    _unpack_blocks,
)
from chunked_ref2v.longform.runner import (
    _pack_conditioning,
    _unpack_conditioning,
    decode_chunk,
)
from chunked_ref2v.longform.writer import _partial_path


class LongFormGeometryTests(unittest.TestCase):
    def test_short_video_still_gets_one_chunk(self):
        self.assertEqual(plan_chunks(1, 90, 68), [0])
        self.assertEqual(chunk_count_for(1, 90, 68), 1)

    def test_exact_three_minute_plan(self):
        total = 180 * 24
        starts = plan_chunks(total, 90, 68)
        self.assertEqual(len(starts), 64)
        self.assertEqual(starts[-1], 63 * 68)
        self.assertEqual(actual_frames_for_chunk(
            63, total_frames=total, chunk_frames=90, stride_frames=68
        ), 36)
        self.assertEqual(min(total, starts[-1] + 36), total)

    def test_frames_needed_is_capacity_not_requested_duration(self):
        count = chunk_count_for(4320, 90, 68)
        self.assertGreater(frames_needed_for(count, 90, 68), 4320)


class ManifestTests(unittest.TestCase):
    def test_same_identity_resumes(self):
        with tempfile.TemporaryDirectory() as root:
            RunManifest(root, {"source": "a", "seed": 1}).ensure()
            RunManifest(root, {"source": "a", "seed": 1}).ensure()

    def test_changed_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            RunManifest(root, {"source": "a", "seed": 1}).ensure()
            with self.assertRaises(RuntimeError):
                RunManifest(root, {"source": "a", "seed": 2}).ensure()

    def test_hash_is_order_independent(self):
        self.assertEqual(identity_hash({"a": 1, "b": 2}), identity_hash({"b": 2, "a": 1}))

    def test_schema_two_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(root, exist_ok=True)
            with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as fh:
                json.dump({"schema_version": 2, "identity_hash": "old", "identity": {}}, fh)
            with self.assertRaises(RuntimeError):
                RunManifest(root, {"source": "a"}).ensure()

    def test_model_sampling_identity_includes_effective_class_and_shifts(self):
        class SamplingA:
            def __init__(self, shift=12.0, audio_shift=3.0):
                self.shift = shift
                self.audio_shift = audio_shift

        class SamplingB(SamplingA):
            pass

        class Model:
            def __init__(self, sampling):
                self.sampling = sampling
                self.model_options = {"transformer_options": {
                    "minimax_h3_sigma_shift_video": 12.0,
                    "minimax_h3_sigma_shift_audio": sampling.audio_shift,
                }}

            def get_model_object(self, name):
                assert name == "model_sampling"
                return self.sampling

        first = object_fingerprint(Model(SamplingA()))
        video_shift = object_fingerprint(Model(SamplingA(10.0, 3.0)))
        audio_shift = object_fingerprint(Model(SamplingA(12.0, 2.0)))
        sampling_class = object_fingerprint(Model(SamplingB()))
        self.assertEqual(first["model_sampling"]["shift"], 12.0)
        self.assertEqual(first["model_sampling"]["audio_shift"], 3.0)
        self.assertEqual(first["minimax_h3_sigma_shift_video"], 12.0)
        self.assertEqual(first["minimax_h3_sigma_shift_audio"], 3.0)
        self.assertNotIn("h3_video_shift", first)
        self.assertNotIn("h3_audio_shift", first)
        self.assertNotEqual(identity_hash(first), identity_hash(video_shift))
        self.assertNotEqual(identity_hash(first), identity_hash(audio_shift))
        self.assertNotEqual(identity_hash(first), identity_hash(sampling_class))
        self.assertEqual(SCHEMA_VERSION, 3)


class ConditioningPersistenceTests(unittest.TestCase):
    def test_all_supported_extra_types_round_trip(self):
        conditioning = [[
            torch.arange(4).reshape(1, 4),
            {
                "pooled_output": torch.ones(1, 2),
                "mask": None,
                "strength": 0.5,
                "labels": ["a", "b"],
            },
        ]]
        tensors, meta = _pack_conditioning(conditioning)
        rebuilt = _unpack_conditioning(tensors, meta)
        self.assertTrue(torch.equal(rebuilt[0][0], conditioning[0][0]))
        self.assertTrue(torch.equal(
            rebuilt[0][1]["pooled_output"], conditioning[0][1]["pooled_output"]
        ))
        self.assertIsNone(rebuilt[0][1]["mask"])
        self.assertEqual(rebuilt[0][1]["strength"], 0.5)
        self.assertEqual(rebuilt[0][1]["labels"], ["a", "b"])


class ReferencePersistenceTests(unittest.TestCase):
    def test_reference_blocks_round_trip_without_interpreting_kinds(self):
        blocks = [
            {
                "kind": "image",
                "latent_h": 4,
                "latent_w": 5,
                "latent": torch.arange(20).reshape(1, 1, 4, 5),
            },
            {
                "kind": "audio",
                "ref_audio_t": 7,
                "audio_latent": torch.ones(1, 2, 3),
            },
        ]
        tensors, metadata = _pack_blocks(blocks)
        rebuilt = _unpack_blocks(tensors, metadata)
        self.assertEqual(rebuilt[0]["kind"], "image")
        self.assertEqual(rebuilt[1]["kind"], "audio")
        self.assertTrue(torch.equal(rebuilt[0]["latent"], blocks[0]["latent"]))
        self.assertTrue(torch.equal(
            rebuilt[1]["audio_latent"], blocks[1]["audio_latent"]
        ))

    def test_reference_identity_is_order_independent_and_content_sensitive(self):
        a = torch.zeros(1, 2, 2, 3)
        b = torch.ones(1, 2, 2, 3)
        first = _reference_identity(
            {"ref_image_1": b, "ref_image_0": a}, {}, {}, {}
        )
        reordered = _reference_identity(
            {"ref_image_0": a, "ref_image_1": b}, {}, {}, {}
        )
        changed = _reference_identity(
            {"ref_image_0": a, "ref_image_1": b + 1}, {}, {}, {}
        )
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed)

    def test_reference_video_audio_pairs_by_autogrow_suffix(self):
        audio = {"waveform": torch.zeros(1, 1, 8), "sample_rate": 32000}
        self.assertIs(
            _paired_audio({"ref_video_audio_2": audio}, "ref_video_2"), audio
        )
        self.assertIsNone(
            _paired_audio({"ref_video_audio_1": audio}, "ref_video_2")
        )


class DecodeTests(unittest.TestCase):
    def test_decode_chunk_disables_autograd(self):
        class RecordingVAE:
            grad_enabled = None

            def decode(self, latent):
                self.grad_enabled = torch.is_grad_enabled()
                return latent

        vae = RecordingVAE()
        latent = torch.zeros(1, 2, 3, 4)
        with torch.enable_grad():
            output = decode_chunk(vae, latent)
        self.assertFalse(vae.grad_enabled)
        self.assertIs(output, latent)


class WriterTests(unittest.TestCase):
    def test_partial_path_keeps_container_extension(self):
        self.assertEqual(_partial_path("x/final.mp4"), "x/final.partial.mp4")
        self.assertEqual(_partial_path("x/video.mkv"), "x/video.partial.mkv")


if __name__ == "__main__":
    sys.argv = _ARGV
    unittest.main()
