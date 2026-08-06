import json
import os
import tempfile
import unittest

import torch

from chunked_ref2v.longform.chunk_stream import (
    actual_frames_for_chunk,
    chunk_count_for,
    frames_needed_for,
    plan_chunks,
)
from chunked_ref2v.longform.manifest import RunManifest, identity_hash
from chunked_ref2v.longform.runner import _pack_conditioning, _unpack_conditioning
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


class WriterTests(unittest.TestCase):
    def test_partial_path_keeps_container_extension(self):
        self.assertEqual(_partial_path("x/final.mp4"), "x/final.partial.mp4")
        self.assertEqual(_partial_path("x/video.mkv"), "x/video.partial.mkv")


if __name__ == "__main__":
    unittest.main()
