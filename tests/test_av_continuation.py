"""CPU checks for native long-form audiovisual continuation."""

from __future__ import annotations

import unittest

import torch

from chunked_ref2v.longform.av_continuation_nodes import (
    _chunk_count,
    _dynamic_av_reference,
    continuation_prompt,
)


class LongFormAVContinuationTests(unittest.TestCase):
    def test_chunk_count_has_no_overlap_stride(self):
        self.assertEqual(_chunk_count(240, 141), 2)
        self.assertEqual(_chunk_count(282, 141), 2)
        self.assertEqual(_chunk_count(283, 141), 3)

    def test_continuation_prompt_uses_independent_video_audio_numbers(self):
        text = continuation_prompt("continue dancing", 3, 5)
        self.assertIn("[video continuation + audio reference]", text)
        self.assertIn("<Video 3>", text)
        self.assertIn("<Audio 5>", text)
        self.assertIn("begins immediately after its end", text)
        self.assertTrue(text.endswith("continue dancing"))

    def test_dynamic_reference_reuses_generated_latents_directly(self):
        pixels = torch.zeros(141, 64, 96, 3)
        video = torch.zeros(1, 24, 42, 4, 6)
        audio = torch.zeros(1, 32, 2, 235)
        items, block = _dynamic_av_reference(
            pixels, video, audio, (96, 64)
        )

        self.assertEqual([item["type"] for item in items], ["audio", "video"])
        self.assertIs(block["latent"], video)
        self.assertIs(block["audio_latent"], audio)
        self.assertEqual(block["kind"], "video_audio")
        self.assertEqual(block["latent_t"], 42)
        self.assertEqual(block["latent_h"], 4)
        self.assertEqual(block["latent_w"], 6)
        self.assertEqual(block["ref_audio_t"], 235)

    def test_dynamic_reference_requires_complete_av_source(self):
        pixels = torch.zeros(141, 64, 96, 3)
        video = torch.zeros(1, 24, 42, 4, 6)
        with self.assertRaises(ValueError):
            _dynamic_av_reference(pixels, video, None, (96, 64))


if __name__ == "__main__":
    unittest.main()
