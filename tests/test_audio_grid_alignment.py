"""CPU checks for conservative carry and experimental grid diagnostics."""

from __future__ import annotations

import os
import sys
import unittest
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chunked_ref2v.geometry import (
    AUDIO_LATENT_FPS,
    FPS,
    UnalignedProfileError,
    aligned_overlap_frames,
    audio_boundary_is_exact,
)
from chunked_ref2v.longform.audio_runtime import (
    audio_latent_boundary,
    audio_carry_latents_for_video_frames,
)


class AudioBoundaryTests(unittest.TestCase):
    def test_only_multiples_of_three_frames_land_on_both_grids(self):
        exact = [n for n in range(1, 25) if audio_boundary_is_exact(n)]
        self.assertEqual(exact, list(range(3, 25, 3)))

    def test_rounding_error_is_bounded_by_a_third_of_a_latent(self):
        for stride in range(1, 200):
            used = audio_latent_boundary(stride)
            exact = Fraction(stride * AUDIO_LATENT_FPS, FPS)
            error_ms = abs(float(Fraction(used) - exact)) / AUDIO_LATENT_FPS * 1000
            self.assertLessEqual(round(error_ms, 3), 8.334)
            if audio_boundary_is_exact(stride):
                self.assertEqual(error_ms, 0.0)

    def test_carry_count_is_the_maximum_whole_audio_latent_count(self):
        for video_frames, expected in ((4, 6), (8, 13), (12, 20)):
            count = audio_carry_latents_for_video_frames(video_frames)
            self.assertEqual(count, expected)
        for video_frames in range(1, 200):
            count = audio_carry_latents_for_video_frames(video_frames)
            self.assertLessEqual(count * FPS, video_frames * AUDIO_LATENT_FPS)
            self.assertGreater(
                (count + 1) * FPS, video_frames * AUDIO_LATENT_FPS
            )


class ExperimentalAlignedOverlapTests(unittest.TestCase):
    def test_known_aligned_profiles_for_the_default_chunk_length(self):
        aligned = [
            overlap
            for overlap in range(1, 90)
            if aligned_overlap_frames(90, overlap) == overlap
        ]
        self.assertEqual(aligned, [9, 21, 30, 39, 51, 60, 72, 81])
        for overlap in aligned:
            self.assertTrue(audio_boundary_is_exact(90 - overlap))

    def test_an_aligned_overlap_is_left_alone(self):
        self.assertEqual(aligned_overlap_frames(90, 9), 9)
        self.assertEqual(aligned_overlap_frames(90, 21), 21)

    def test_the_diagnostic_helper_finds_the_nearer_aligned_overlap(self):
        # O=17 (stride 73) rounds its carry up by a third of a latent.
        self.assertFalse(audio_boundary_is_exact(90 - 17))
        self.assertEqual(aligned_overlap_frames(90, 17), 21)

    def test_the_diagnostic_helper_can_snap_the_node_default(self):
        self.assertFalse(audio_boundary_is_exact(90 - 4))
        self.assertEqual(aligned_overlap_frames(90, 4), 9)

    def test_ties_prefer_the_larger_overlap(self):
        # 15 is equidistant from 9 and 21; more carried context wins.
        self.assertEqual(aligned_overlap_frames(90, 15), 21)

    def test_alignment_is_a_property_of_the_profile_not_the_overlap_alone(self):
        # The same O=4 default is already exact at C=22 (stride 18) and off-grid
        # at C=90 (stride 86), which is why this cannot be fixed by changing the
        # default overlap.
        self.assertEqual(aligned_overlap_frames(22, 4), 4)
        self.assertEqual(aligned_overlap_frames(90, 4), 9)

    def test_a_chunk_length_with_no_aligned_overlap_raises(self):
        # Not a legal profile, but the helper must fail loudly rather than
        # return an overlap that only satisfies one of the two grids.
        with self.assertRaises(UnalignedProfileError):
            aligned_overlap_frames(5, 1)


class PreviewWidgetOrderTests(unittest.TestCase):
    """Preview widgets remain ahead of autogrow inputs without an AV toggle."""

    def build(self, base_names, preview_names):
        # Mirrors reference_preview_nodes.define_schema's insertion rule without
        # importing the node module, which needs the whole Comfy runtime.
        inputs = list(base_names)
        anchors = ("ref_images",)
        insert_at = next(
            (i for i, name in enumerate(inputs) if name in anchors),
            len(inputs),
        )
        inputs[insert_at:insert_at] = preview_names
        return inputs

    def test_preview_widgets_anchor_on_reference_images(self):
        base = ["ffmpeg_location", "save_frames", "ref_images"]
        preview = ["chunk_align_audio_references", "live_preview_width"]
        merged = self.build(base, preview)

        self.assertEqual(
            merged,
            [
                "ffmpeg_location",
                "save_frames",
                "chunk_align_audio_references",
                "live_preview_width",
                "ref_images",
            ],
        )
        self.assertNotIn("align_audio_chunks", merged)

    def test_shorter_schema_still_anchors_on_ref_images(self):
        base = ["ffmpeg_location", "save_frames", "ref_images"]
        merged = self.build(base, ["live_preview_width"])
        self.assertEqual(merged[-1], "ref_images")
        self.assertEqual(merged[-2], "live_preview_width")


if __name__ == "__main__":
    unittest.main()
