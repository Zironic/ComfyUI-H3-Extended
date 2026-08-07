"""CPU checks for same-time source AV conditioning on N+1 continuation."""

from __future__ import annotations

import unittest

import torch

from chunked_ref2v.longform import aligned_source_runtime as aligned
from chunked_ref2v.longform.av_continuation_nodes import (
    MiniMaxH3LongFormAVContinuation,
)


class NPlusOneAlignedSourceTests(unittest.TestCase):
    def test_video_slice_is_chronological(self):
        frames = torch.arange(300, dtype=torch.float32).reshape(300, 1, 1, 1)
        sliced = aligned.slice_video_reference(
            frames, start_frame=141, frame_count=141
        )
        self.assertEqual(sliced.shape[0], 141)
        self.assertEqual(float(sliced[0, 0, 0, 0]), 141.0)
        self.assertEqual(float(sliced[-1, 0, 0, 0]), 281.0)

    def test_final_video_slice_holds_last_frame_in_unused_tail(self):
        frames = torch.arange(250, dtype=torch.float32).reshape(250, 1, 1, 1)
        sliced = aligned.slice_video_reference(
            frames, start_frame=141, frame_count=141
        )
        self.assertEqual(sliced.shape[0], 141)
        self.assertEqual(float(sliced[0, 0, 0, 0]), 141.0)
        self.assertEqual(float(sliced[108, 0, 0, 0]), 249.0)
        self.assertTrue(torch.equal(sliced[109:], sliced[108:109].expand(32, 1, 1, 1)))

    def test_video_slice_never_wraps(self):
        frames = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1)
        sliced = aligned.slice_video_reference(
            frames, start_frame=8, frame_count=5
        )
        self.assertEqual(
            sliced[:, 0, 0, 0].tolist(),
            [8.0, 9.0, 9.0, 9.0, 9.0],
        )

    def test_video_slice_rejects_start_after_source(self):
        frames = torch.zeros(10, 1, 1, 1)
        with self.assertRaises(ValueError):
            aligned.slice_video_reference(frames, start_frame=10, frame_count=5)

    def test_source_canvas_is_independent_of_target(self):
        frames = torch.zeros(5, 768, 1344, 3)
        self.assertEqual(
            aligned._reference_canvas(frames, (2048, 1152), "source"),
            (1344, 768),
        )
        self.assertEqual(
            aligned._reference_canvas(frames, (2048, 1152), "match"),
            (2048, 1152),
        )

    def test_native_reference_canvas_uses_reference_aspect(self):
        frames = torch.zeros(5, 720, 1280, 3)
        canvas = aligned._reference_canvas(frames, (2048, 1152), "native")
        self.assertNotEqual(canvas, (2048, 1152))
        self.assertEqual(canvas[0] % 32, 0)
        self.assertEqual(canvas[1] % 32, 0)

    def test_aligned_prompt_is_same_time_not_continuation(self):
        text = aligned.aligned_source_prompt("preserve detail", 2, 3)
        self.assertIn("<Video 2>", text)
        self.assertIn("<Audio 3>", text)
        self.assertIn("same chronological interval", text)
        self.assertIn("target resolution", text)
        self.assertNotIn("immediately preceding", text)
        self.assertTrue(text.endswith("preserve detail"))

    def test_video_only_prompt_does_not_invent_audio_reference(self):
        text = aligned.aligned_source_prompt("same action", 1, 0)
        self.assertIn("<Video 1>", text)
        self.assertNotIn("<Audio", text)

    def test_static_reference_numbering_is_independent_by_type(self):
        videos = {
            "ref_video_1": torch.zeros(5, 32, 32, 3),
            "ref_video_2": torch.zeros(5, 32, 32, 3),
        }
        paired = {
            "ref_video_audio_1": {"waveform": torch.zeros(1, 1, 100), "sample_rate": 32000},
        }
        audios = {
            "ref_audio_1": {"waveform": torch.zeros(1, 1, 100), "sample_rate": 32000},
        }
        video_count, audio_count = aligned._static_reference_numbers(
            videos, paired, audios
        )
        self.assertEqual(video_count, 2)
        self.assertEqual(audio_count, 2)

    def test_node_appends_aligned_inputs_without_removing_existing_inputs(self):
        aligned.install()
        schema = MiniMaxH3LongFormAVContinuation.define_schema()
        names = [
            getattr(item, "id", getattr(item, "name", None))
            for item in schema.inputs
        ]
        self.assertIn("reference_frames", names)
        self.assertIn("aligned_source_video", names)
        self.assertIn("aligned_source_audio", names)
        self.assertIn("aligned_source_video_size", names)
        self.assertGreater(
            names.index("aligned_source_video"),
            names.index("reference_frames"),
        )


if __name__ == "__main__":
    unittest.main()
