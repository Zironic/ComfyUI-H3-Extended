"""CPU checks for LongFormReferenceVideo audio-reference slicing."""

from __future__ import annotations

import unittest

import torch

from chunked_ref2v.longform.chunk_aligned_audio_refs import (
    slice_audio_reference,
)


class ChunkAlignedAudioReferenceTests(unittest.TestCase):
    def test_slice_uses_chunk_stride_position_and_keeps_overlap(self):
        sample_rate = 240
        waveform = torch.arange(2400, dtype=torch.float32).reshape(1, 1, -1)
        audio = {"waveform": waveform, "sample_rate": sample_rate}

        # C=90, O=4 at 24 fps gives S=86. Chunk 1 therefore begins at
        # 86 frames, while its 90-frame slice includes the four-frame overlap.
        sliced = slice_audio_reference(
            audio,
            start_frame=86,
            frame_count=90,
            fps=24,
        )

        self.assertEqual(sliced["waveform"].shape[-1], 900)
        self.assertEqual(sliced["waveform"][0, 0, 0].item(), 860.0)
        self.assertEqual(sliced["waveform"][0, 0, -1].item(), 1759.0)

    def test_slice_pads_past_end_instead_of_wrapping(self):
        sample_rate = 24
        waveform = torch.arange(100, dtype=torch.float32).reshape(1, 1, -1)
        audio = {"waveform": waveform, "sample_rate": sample_rate}

        sliced = slice_audio_reference(
            audio,
            start_frame=90,
            frame_count=20,
            fps=24,
        )

        self.assertEqual(sliced["waveform"].shape[-1], 20)
        self.assertEqual(sliced["waveform"][0, 0, :10].tolist(), list(range(90, 100)))
        self.assertTrue(torch.equal(
            sliced["waveform"][0, 0, 10:], torch.zeros(10)
        ))

    def test_slice_entirely_after_end_is_silence(self):
        audio = {
            "waveform": torch.ones(1, 2, 32),
            "sample_rate": 24,
        }
        sliced = slice_audio_reference(
            audio,
            start_frame=100,
            frame_count=12,
            fps=24,
        )
        self.assertEqual(sliced["waveform"].shape, (1, 2, 12))
        self.assertEqual(torch.count_nonzero(sliced["waveform"]).item(), 0)

    def test_invalid_start_is_rejected(self):
        audio = {"waveform": torch.zeros(1, 1, 8), "sample_rate": 24}
        with self.assertRaises(ValueError):
            slice_audio_reference(
                audio,
                start_frame=-1,
                frame_count=8,
                fps=24,
            )


if __name__ == "__main__":
    unittest.main()
