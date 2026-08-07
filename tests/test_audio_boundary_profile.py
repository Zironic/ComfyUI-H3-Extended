"""CPU checks for exact long-form H3 video/audio chunk boundaries."""

from __future__ import annotations

import unittest

from chunked_ref2v.audio_boundary_profile import (
    legal_reference_tail_frames,
    audio_aligned_chunk_frames,
    profile_audio_boundaries_are_exact,
    resolve_audio_boundary_profile,
)
from chunked_ref2v.geometry import HarnessGeometry, audio_boundary_is_exact


class AudioBoundaryProfileTests(unittest.TestCase):
    def test_joint_chunk_lengths_are_h3_legal_and_audio_exact(self):
        self.assertEqual(
            audio_aligned_chunk_frames(),
            [39, 90, 141, 192, 243, 294, 345],
        )
        for frames in audio_aligned_chunk_frames():
            self.assertEqual(frames % 17, 5)
            self.assertTrue(audio_boundary_is_exact(frames))

    def test_profile_requires_chunk_overlap_and_stride_boundaries(self):
        self.assertTrue(profile_audio_boundaries_are_exact(90, 21))
        self.assertTrue(profile_audio_boundaries_are_exact(141, 21))
        self.assertFalse(profile_audio_boundaries_are_exact(90, 17))
        self.assertFalse(profile_audio_boundaries_are_exact(73, 22))

    def test_legal_reference_tail_frames(self):
        self.assertEqual(
            legal_reference_tail_frames(90),
            [9, 21, 30, 39, 51, 60, 72, 81],
        )
        self.assertEqual(
            legal_reference_tail_frames(141),
            [9, 21, 30, 39, 51, 60, 72, 81, 90, 102, 111, 123, 132],
        )

    def test_default_90_4_profile_snaps_to_90_9(self):
        chunk, overlap, note = resolve_audio_boundary_profile(90, 4, True)
        self.assertEqual((chunk, overlap), (90, 9))
        self.assertIsNotNone(note)
        self.assertEqual(chunk - overlap, 81)

    def test_existing_90_21_profile_is_unchanged(self):
        chunk, overlap, note = resolve_audio_boundary_profile(90, 21, True)
        self.assertEqual((chunk, overlap), (90, 21))
        self.assertIsNone(note)

    def test_unaligned_chunk_and_overlap_are_both_corrected(self):
        chunk, overlap, note = resolve_audio_boundary_profile(73, 22, True)
        self.assertEqual((chunk, overlap), (90, 21))
        self.assertIsNotNone(note)
        self.assertIn("C=73 O=22 S=51", note)
        self.assertIn("C=90 O=21 S=69", note)

    def test_disabled_preserves_legacy_profile_exactly(self):
        self.assertEqual(
            resolve_audio_boundary_profile(73, 22, False),
            (73, 22, None),
        )
        self.assertEqual(
            resolve_audio_boundary_profile(90, 4, False),
            (90, 4, None),
        )

    def test_resolved_profiles_are_exact_on_all_three_audio_boundaries(self):
        for requested_chunk in (22, 39, 73, 90, 107, 141, 192, 362):
            for requested_overlap in (4, 9, 17, 21, 30, 39):
                if requested_overlap >= requested_chunk:
                    continue
                chunk, overlap, _ = resolve_audio_boundary_profile(
                    requested_chunk,
                    requested_overlap,
                    True,
                )
                geometry = HarnessGeometry(
                    chunk_frames=chunk,
                    overlap_frames=overlap,
                ).validate()
                self.assertTrue(
                    profile_audio_boundaries_are_exact(chunk, overlap),
                    (requested_chunk, requested_overlap, chunk, overlap),
                )
                self.assertTrue(audio_boundary_is_exact(geometry.chunk_frames))
                self.assertTrue(audio_boundary_is_exact(geometry.overlap_frames))
                self.assertTrue(audio_boundary_is_exact(geometry.stride_frames))


if __name__ == "__main__":
    unittest.main()
