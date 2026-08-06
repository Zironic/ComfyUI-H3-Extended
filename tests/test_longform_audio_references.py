"""CPU checks for LongFormReferenceVideo audio-reference modes."""

from __future__ import annotations

import unittest

import torch

from chunked_ref2v.longform.chunk_aligned_audio_refs import (
    slice_audio_reference,
)


class AudioReferenceSliceTests(unittest.TestCase):
    def test_second_chunk_starts_at_stride_not_zero(self):
        sample_rate = 24000
        waveform = torch.arange(
            sample_rate * 10, dtype=torch.float32
        ).reshape(1, 1, -1)
        audio = {"waveform": waveform, "sample_rate": sample_rate}

        sliced = slice_audio_reference(
            audio,
            start_frame=86,
            frame_count=90,
            fps=24,
        )

        expected_start = round(86 / 24 * sample_rate)
        expected_stop = round((86 + 90) / 24 * sample_rate)
        self.assertEqual(
            sliced["waveform"][0, 0, 0].item(),
            waveform[0, 0, expected_start].item(),
        )
        self.assertEqual(
            sliced["waveform"].shape[-1],
            expected_stop - expected_start,
        )

    def test_video_overlap_is_present_at_next_chunk_start(self):
        sample_rate = 24000
        waveform = torch.arange(
            sample_rate * 10, dtype=torch.float32
        ).reshape(1, 1, -1)
        audio = {"waveform": waveform, "sample_rate": sample_rate}

        first = slice_audio_reference(
            audio, start_frame=0, frame_count=90, fps=24
        )["waveform"]
        second = slice_audio_reference(
            audio, start_frame=86, frame_count=90, fps=24
        )["waveform"]

        overlap_samples = round(90 / 24 * sample_rate) - round(
            86 / 24 * sample_rate
        )
        self.assertTrue(
            torch.equal(first[..., -overlap_samples:], second[..., :overlap_samples])
        )

    def test_end_of_track_pads_silence_without_wrapping(self):
        sample_rate = 24
        waveform = torch.arange(100, dtype=torch.float32).reshape(1, 1, -1)
        audio = {"waveform": waveform, "sample_rate": sample_rate}

        sliced = slice_audio_reference(
            audio, start_frame=96, frame_count=24, fps=24
        )["waveform"]

        self.assertEqual(sliced.shape[-1], 24)
        self.assertTrue(torch.equal(sliced[..., :4], waveform[..., 96:100]))
        self.assertTrue(torch.equal(sliced[..., 4:], torch.zeros(1, 1, 20)))

    def test_invalid_ranges_are_rejected(self):
        audio = {
            "waveform": torch.zeros(1, 1, 24),
            "sample_rate": 24,
        }
        with self.assertRaises(ValueError):
            slice_audio_reference(
                audio, start_frame=-1, frame_count=24, fps=24
            )
        with self.assertRaises(ValueError):
            slice_audio_reference(
                audio, start_frame=0, frame_count=0, fps=24
            )


if __name__ == "__main__":
    unittest.main()
