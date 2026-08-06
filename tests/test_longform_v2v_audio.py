"""CPU checks for long-form Ref2V source conditioning and track muxing."""

from __future__ import annotations

import unittest

import torch

from chunked_ref2v.longform.audio_source import _missing_audio_error
from chunked_ref2v.longform.dual_audio import build_dual_audio_mux_args
from chunked_ref2v.longform.v2v_audio_runtime import _source_reference_block


class SourceAudioConditioningTests(unittest.TestCase):
    def test_source_block_carries_video_and_audio_together(self):
        video = torch.zeros(1, 24, 27, 8, 12)
        audio = torch.zeros(1, 32, 2, 150)
        block = _source_reference_block(
            {
                "source_latent": video,
                "source_audio_latent": audio,
            },
            (192, 128),
        )
        self.assertEqual(block["kind"], "video_audio")
        self.assertIs(block["latent"], video)
        self.assertIs(block["audio_latent"], audio)
        self.assertEqual(block["latent_t"], 27)
        self.assertEqual(block["ref_audio_t"], 150)
        self.assertEqual(block["latent_h"], 8)
        self.assertEqual(block["latent_w"], 12)

    def test_video_only_source_remains_valid(self):
        video = torch.zeros(1, 24, 27, 8, 12)
        block = _source_reference_block(
            {"source_latent": video}, (192, 128)
        )
        self.assertEqual(block["kind"], "video")
        self.assertIsNone(block["audio_latent"])
        self.assertEqual(block["ref_audio_t"], 0)

    def test_missing_audio_errors_are_recognized(self):
        self.assertTrue(
            _missing_audio_error(
                "Output file #0 does not contain any stream"
            )
        )
        self.assertTrue(
            _missing_audio_error(
                "Stream map '0:a:0' matches no streams"
            )
        )
        self.assertFalse(_missing_audio_error("Invalid data found"))


class DualTrackMuxTests(unittest.TestCase):
    def test_generated_track_is_first_and_default(self):
        args = build_dual_audio_mux_args(
            "ffmpeg",
            "video.mkv",
            "generated.wav",
            "source.mp4",
            "final.mp4",
            start_frame=48,
            frame_count=240,
            fps=24,
        )
        maps = [
            args[index + 1]
            for index, value in enumerate(args[:-1])
            if value == "-map"
        ]
        self.assertEqual(maps, ["0:v:0", "1:a:0", "2:a:0"])
        self.assertIn("title=Generated audio", args)
        self.assertIn("title=Source audio", args)
        generated_disposition = args.index("-disposition:a:0")
        source_disposition = args.index("-disposition:a:1")
        self.assertEqual(args[generated_disposition + 1], "default")
        self.assertEqual(args[source_disposition + 1], "0")

    def test_short_source_track_cannot_truncate_output(self):
        args = build_dual_audio_mux_args(
            "ffmpeg",
            "video.mkv",
            "generated.wav",
            "source.mp4",
            "final.mp4",
            start_frame=0,
            frame_count=90,
            fps=24,
        )
        self.assertNotIn("-shortest", args)
        duration_index = args.index("-t")
        self.assertEqual(args[duration_index + 1], "3.750000000")


if __name__ == "__main__":
    unittest.main()
