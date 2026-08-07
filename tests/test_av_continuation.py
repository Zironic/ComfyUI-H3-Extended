"""CPU checks for native long-form audiovisual continuation."""

from __future__ import annotations

import unittest

import torch

from chunked_ref2v.longform.av_continuation_nodes import (
    MiniMaxH3LongFormAVContinuation,
    _chunk_count,
    _dynamic_av_reference,
    _slice_dynamic_av_reference,
    _resolve_reference_frames,
    continuation_prompt,
)
from chunked_ref2v.longform.nplusone_chunk_prompt_timeline import (
    build_nplusone_chunk_prompt_plan,
    prompts_for_av_continuation_plan,
)
from chunked_ref2v.geometry import HarnessGeometry
from chunked_ref2v.longform.audio_runtime import audio_overlap_slice


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

    def test_dynamic_reference_tail_is_sliced_to_reference_frames(self):
        geometry = HarnessGeometry(chunk_frames=141, overlap_frames=81).validate()
        _, overlap_count = geometry.overlap_slice()
        pixels = torch.arange(141 * 64 * 96 * 3, dtype=torch.float32).reshape(
            141, 64, 96, 3,
        )
        video = torch.arange(
            1 * 24 * 42 * 4 * 6,
            dtype=torch.float32,
        ).reshape(1, 24, 42, 4, 6)
        audio = torch.arange(1 * 32 * 2 * 235, dtype=torch.float32).reshape(
            1, 32, 2, 235
        )

        sliced_pixels, sliced_video, sliced_audio = _slice_dynamic_av_reference(
            pixels,
            video,
            audio,
            reference_frames=81,
            geometry=geometry,
        )
        self.assertEqual(sliced_pixels.shape, (81, 64, 96, 3))
        self.assertTrue(torch.equal(sliced_pixels, pixels[-81:]))
        self.assertEqual(sliced_video.shape, (1, 24, overlap_count, 4, 6))
        self.assertEqual(sliced_audio.shape, (1, 32, 2, 135))
        self.assertEqual(int(sliced_video.shape[2]), int(overlap_count))

    def test_dynamic_reference_tail_can_use_whole_chunk(self):
        geometry = HarnessGeometry(chunk_frames=141, overlap_frames=81).validate()
        pixels = torch.arange(141 * 64 * 96 * 3, dtype=torch.float32).reshape(
            141, 64, 96, 3,
        )
        video = torch.arange(
            1 * 24 * 42 * 4 * 6,
            dtype=torch.float32,
        ).reshape(1, 24, 42, 4, 6)
        audio = torch.arange(1 * 32 * 2 * 235, dtype=torch.float32).reshape(
            1, 32, 2, 235
        )

        sliced_pixels, sliced_video, sliced_audio = _slice_dynamic_av_reference(
            pixels,
            video,
            audio,
            reference_frames=141,
            geometry=geometry,
        )
        self.assertEqual(sliced_pixels.shape, pixels.shape)
        self.assertTrue(torch.equal(sliced_pixels, pixels))
        self.assertIs(sliced_video, video)
        self.assertIs(sliced_audio, audio)

    def test_reference_frames_are_snapped_to_legal_values(self):
        self.assertEqual(_resolve_reference_frames(141, 80), 81)

    def test_reference_frames_respect_reference_input_range(self):
        self.assertEqual(_resolve_reference_frames(141, 40), 51)

    def test_reference_frames_can_snap_to_whole_chunk(self):
        self.assertEqual(_resolve_reference_frames(141, 141), 141)

    def test_reference_frames_reject_illegal_chunk(self):
        with self.assertRaises(ValueError):
            _resolve_reference_frames(124, 60)

    def test_reference_tail_slice_preserves_shared_intervals(self):
        geometry = HarnessGeometry(chunk_frames=141, overlap_frames=60).validate()
        pixels = torch.arange(141 * 64 * 96 * 3, dtype=torch.float32).reshape(
            141, 64, 96, 3,
        )
        video = torch.arange(
            24 * 42 * 4 * 6,
            dtype=torch.float32,
        ).reshape(1, 24, 42, 4, 6)
        audio = torch.arange(1 * 32 * 2 * 235, dtype=torch.float32).reshape(
            1, 32, 2, 235
        )

        sliced_pixels, sliced_video, sliced_audio = _slice_dynamic_av_reference(
            pixels,
            video,
            audio,
            reference_frames=60,
            geometry=geometry,
        )
        self.assertEqual(sliced_pixels.shape, (60, 64, 96, 3))
        _, audio_overlap_count = audio_overlap_slice(geometry)
        self.assertEqual(int(audio_overlap_count), 100)
        self.assertEqual(sliced_audio.shape[-1], 100)

    def test_reference_frames_cannot_exceed_previous_chunk(self):
        geometry = HarnessGeometry(chunk_frames=141, overlap_frames=81).validate()
        pixels = torch.zeros(141, 64, 96, 3)
        video = torch.zeros(1, 24, 42, 4, 6)
        audio = torch.zeros(1, 32, 2, 235)
        with self.assertRaises(ValueError):
            _slice_dynamic_av_reference(
                pixels,
                video,
                audio,
                reference_frames=200,
                geometry=geometry,
            )

    def test_dynamic_reference_requires_complete_av_source(self):
        pixels = torch.zeros(141, 64, 96, 3)
        video = torch.zeros(1, 24, 42, 4, 6)
        with self.assertRaises(ValueError):
            _dynamic_av_reference(pixels, video, None, (96, 64))

    def test_node_exposes_nplusone_plan_input(self):
        schema = MiniMaxH3LongFormAVContinuation.define_schema()
        names = [getattr(item, "id", getattr(item, "name", None)) for item in schema.inputs]
        self.assertIn("n_plus_one_prompt_plan", names)

    def test_plan_resolves_different_base_prompt_for_each_chunk(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=12,
            chunk_frames=141,
            global_prompt="same subject",
            chunk_prompts_json='{"prompts":["first action","second action","third action"]}',
        )
        prompts = prompts_for_av_continuation_plan(
            plan,
            "fallback",
            output_seconds=12,
            chunk_frames=141,
        )
        self.assertEqual(
            prompts,
            [
                "same subject\n\nfirst action",
                "same subject\n\nsecond action",
                "same subject\n\nthird action",
            ],
        )
        for prompt in prompts:
            self.assertNotIn("<Video", prompt)
            self.assertNotIn("<Audio", prompt)
            self.assertNotIn("video continuation", prompt)

    def test_plan_geometry_mismatch_fails_before_sampling(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=12,
            chunk_frames=141,
        )
        with self.assertRaises(ValueError):
            prompts_for_av_continuation_plan(
                plan,
                "fallback",
                output_seconds=12,
                chunk_frames=90,
            )


if __name__ == "__main__":
    unittest.main()
