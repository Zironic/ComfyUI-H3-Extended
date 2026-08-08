"""The emit path must keep every contract with opening-picture injection on.

Opening-picture continuity used to install its own copy of ``_emit_chunk``. The
copy published completed previews directly instead of staging them for the audio
runtime, and skipped the diagnostics dump, so enabling the toggle silently cost
every completed preview its sound and every diagnostics chunk its frames. These
tests drive the real emitter with the hook registered, both ways.
"""

from __future__ import annotations

import os
import sys
import h3_test_tempfile as tempfile
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chunked_ref2v.longform import opening_picture_runtime, preview, runner


CHUNK_FRAMES = 8
STRIDE_FRAMES = 5
CHUNK_COUNT = 3


class FakeGeometry:
    chunk_frames = CHUNK_FRAMES
    stride_frames = STRIDE_FRAMES
    fps = 24


class RecordingPublisher:
    """Records which completed-preview entry point the emitter chose."""

    def __init__(self, *, audio_expected):
        self.audio_expected = bool(audio_expected)
        self.staged = []
        self.published = []

    def stage_completed_chunk(self, *, chunk_index, frames_u8, completed_frames):
        self.staged.append((chunk_index, int(frames_u8.shape[0]), completed_frames))

    def publish_completed_chunk(self, *, chunk_index, frames_u8, completed_frames,
                               audio=None):
        self.published.append((chunk_index, int(frames_u8.shape[0]), completed_frames))


class FakeRun:
    def __init__(self, root, *, diagnostics_enabled):
        self.root = root
        self.geometry = FakeGeometry()
        self.target_frames = STRIDE_FRAMES * CHUNK_COUNT
        self.manifest = None
        self.carry = runner.CARRY_OVERLAP
        self.diagnostic_dump_chunks = bool(diagnostics_enabled)
        self.events = []

    def event(self, **kwargs):
        self.events.append(kwargs)


class FakeVAE:
    """Decodes a whole chunk; each frame carries its own index as its value."""

    def decode(self, latent):
        return (
            torch.arange(CHUNK_FRAMES, dtype=torch.float32)
            .div(255.0)
            .view(CHUNK_FRAMES, 1, 1, 1)
            .expand(CHUNK_FRAMES, 8, 8, 3)
            .contiguous()
        )


def emit_all(run, *, publisher):
    token = preview.activate(publisher)
    try:
        written = 0
        for index in range(CHUNK_COUNT):
            written += preview._emit_chunk_with_preview(
                run,
                index,
                torch.zeros(1, 4, 3, 8, 8),
                video_vae=FakeVAE(),
                chunk_count=CHUNK_COUNT,
                writer=None,
                save_frames=False,
                written=written,
                out_dir=run.root,
                pbar=None,
            )
        return written
    finally:
        preview.deactivate(token)


class EmitChunkContractTests(unittest.TestCase):
    def setUp(self):
        # install() is idempotent and normally runs from the package __init__.
        opening_picture_runtime.install()
        self.assertIn(
            opening_picture_runtime._retain_opening_picture,
            runner._DECODED_PIXELS_HOOKS,
        )

    def run_case(self, *, injection, audio_expected, diagnostics_enabled=True):
        publisher = RecordingPublisher(audio_expected=audio_expected)
        token = opening_picture_runtime._ACTIVE.set(injection)
        with tempfile.TemporaryDirectory() as temp:
            run = FakeRun(temp, diagnostics_enabled=diagnostics_enabled)
            try:
                written = emit_all(run, publisher=publisher)
            finally:
                opening_picture_runtime._ACTIVE.reset(token)
            dumped = [
                len(
                    os.listdir(
                        os.path.join(
                            temp, "diagnostics", "chunk_%06d" % i, "frames"
                        )
                    )
                )
                for i in range(CHUNK_COUNT)
                if os.path.isdir(
                    os.path.join(temp, "diagnostics", "chunk_%06d" % i, "frames")
                )
            ]
        return run, publisher, written, dumped

    def test_audio_chunks_are_staged_with_injection_enabled(self):
        _, publisher, _, _ = self.run_case(injection=True, audio_expected=True)
        self.assertEqual(
            publisher.staged,
            [(0, 5, 5), (1, 5, 10), (2, 5, 15)],
        )
        # Publishing here would emit the segment before its audio was decoded.
        self.assertEqual(publisher.published, [])

    def test_silent_chunks_publish_directly_with_injection_enabled(self):
        _, publisher, _, _ = self.run_case(injection=True, audio_expected=False)
        self.assertEqual(publisher.staged, [])
        self.assertEqual(
            publisher.published,
            [(0, 5, 5), (1, 5, 10), (2, 5, 15)],
        )

    def test_injection_does_not_change_the_staging_decision(self):
        _, with_injection, _, _ = self.run_case(injection=True, audio_expected=True)
        _, without, _, _ = self.run_case(injection=False, audio_expected=True)
        self.assertEqual(with_injection.staged, without.staged)
        self.assertEqual(with_injection.published, without.published)

    def test_diagnostics_dump_untrimmed_frames_with_injection_enabled(self):
        _, _, written, dumped = self.run_case(injection=True, audio_expected=True)
        self.assertEqual(written, STRIDE_FRAMES * CHUNK_COUNT)
        # The dump is the whole decode, not the stride the output keeps.
        self.assertEqual(dumped, [CHUNK_FRAMES] * CHUNK_COUNT)

    def test_diagnostics_match_with_injection_disabled(self):
        _, _, _, enabled = self.run_case(injection=True, audio_expected=True)
        _, _, _, disabled = self.run_case(injection=False, audio_expected=True)
        self.assertEqual(enabled, disabled)

    def test_injection_retains_the_next_chunk_frame_zero(self):
        run, _, _, _ = self.run_case(injection=True, audio_expected=True)
        frame = run._h3_next_opening_picture
        self.assertEqual(tuple(frame.shape), (1, 8, 8, 3))
        # Last chunk has no successor, so chunk 1's stride frame is what remains.
        self.assertAlmostEqual(
            float(frame.flatten()[0]), STRIDE_FRAMES / 255.0, places=6
        )

    def test_no_frame_is_retained_when_injection_is_disabled(self):
        run, _, _, _ = self.run_case(injection=False, audio_expected=True)
        self.assertIsNone(getattr(run, "_h3_next_opening_picture", None))


if __name__ == "__main__":
    unittest.main()
