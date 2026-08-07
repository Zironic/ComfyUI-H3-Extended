"""CPU checks for the shared video/audio chunk grid.

Video frames are 1/24 s and H3 audio latents 1/40 s, so a chunk boundary sits on
both grids only when the stride is a multiple of three frames. The video path
already refuses an unaligned profile; these pin the audio half, which used to
round silently.
"""

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
    resolve_audio_aligned_overlap,
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


class AlignedOverlapTests(unittest.TestCase):
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

    def test_the_workflow_profile_snaps_to_the_nearer_aligned_overlap(self):
        # O=17 (stride 73) rounds its carry up by a third of a latent.
        self.assertFalse(audio_boundary_is_exact(90 - 17))
        self.assertEqual(aligned_overlap_frames(90, 17), 21)

    def test_the_node_default_snaps_up_as_well(self):
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


class ResolveOverlapTests(unittest.TestCase):
    def test_disabled_keeps_the_overlap_but_reports_the_defect(self):
        overlap, note = resolve_audio_aligned_overlap(90, 17, False)
        self.assertEqual(overlap, 17)
        self.assertIn("off-grid", note)
        self.assertIn("align_audio_chunks", note)

    def test_disabled_and_already_aligned_says_nothing(self):
        self.assertEqual(resolve_audio_aligned_overlap(90, 9, False), (9, None))

    def test_enabled_snaps_and_warns_that_the_result_changes(self):
        overlap, note = resolve_audio_aligned_overlap(90, 17, True)
        self.assertEqual(overlap, 21)
        self.assertIn("17 -> 21", note)
        self.assertIn("run directory", note)

    def test_enabled_on_an_aligned_profile_is_a_silent_no_op(self):
        self.assertEqual(resolve_audio_aligned_overlap(90, 21, True), (21, None))


class WidgetOrderTests(unittest.TestCase):
    """Comfy maps widgets_values by position, so a new widget must be last."""

    def build(self, base_names, preview_names):
        # Mirrors reference_preview_nodes.define_schema's insertion rule without
        # importing the node module, which needs the whole Comfy runtime.
        inputs = list(base_names)
        anchors = ("align_audio_chunks", "ref_images")
        insert_at = next(
            (i for i, name in enumerate(inputs) if name in anchors),
            len(inputs),
        )
        inputs[insert_at:insert_at] = preview_names
        return inputs

    def test_align_toggle_stays_after_every_previously_saved_widget(self):
        base = ["ffmpeg_location", "save_frames", "align_audio_chunks", "ref_images"]
        preview = ["chunk_align_audio_references", "live_preview_width"]
        merged = self.build(base, preview)

        self.assertEqual(
            merged,
            [
                "ffmpeg_location",
                "save_frames",
                "chunk_align_audio_references",
                "live_preview_width",
                "align_audio_chunks",
                "ref_images",
            ],
        )
        widgets = [name for name in merged if name != "ref_images"]
        self.assertEqual(widgets[-1], "align_audio_chunks")

    def test_older_schema_without_the_toggle_still_anchors_on_ref_images(self):
        base = ["ffmpeg_location", "save_frames", "ref_images"]
        merged = self.build(base, ["live_preview_width"])
        self.assertEqual(merged[-1], "ref_images")
        self.assertEqual(merged[-2], "live_preview_width")


if __name__ == "__main__":
    unittest.main()
