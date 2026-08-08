"""A long-form AV run that fails must not publish its partial output.

The raw MKV/WAV are written incrementally and only renamed into place on
``close(commit=True)``. Committing from an unconditional ``finally`` therefore
promotes a truncated video to the name that means "this run finished".

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_longform_output_commit.py
"""

import os
import sys
import h3_test_tempfile as tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
_ARGV, sys.argv = list(sys.argv), [sys.argv[0], "--cpu"]

import torch  # noqa: E402

from chunked_ref2v.geometry import HarnessGeometry  # noqa: E402
from chunked_ref2v.longform import audio_runtime  # noqa: E402
from chunked_ref2v.longform.audio_output import (  # noqa: E402
    audio_samples_for_frames,
)
from chunked_ref2v.longform.writer import close_writers  # noqa: E402

CHUNK_FRAMES = 90
OVERLAP_FRAMES = 4
STRIDE_FRAMES = CHUNK_FRAMES - OVERLAP_FRAMES
SAMPLE_RATE = 32000


class Boom(RuntimeError):
    """The failure injected mid-run."""


class FakeWriter:
    """Same partial-then-rename contract as the ffmpeg writers, no ffmpeg."""

    def __init__(self, path, registry, **kwargs):
        self.path = path
        root, ext = os.path.splitext(path)
        self.partial = "%s.partial%s" % (root, ext)
        self.committed = None
        self.close_error = None
        registry.append(self)

    def open(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.partial, "wb"):
            pass
        return self

    def write(self, payload):
        with open(self.partial, "ab") as handle:
            handle.write(b"x")

    def close(self, *, commit=True):
        self.committed = commit
        if self.close_error is not None:
            raise self.close_error
        if commit:
            os.replace(self.partial, self.path)
        else:
            try:
                os.remove(self.partial)
            except FileNotFoundError:
                pass


class FakeRun:
    """The parts of LongFormReferenceRun that _sample_and_write_av touches."""

    def __init__(self, root, *, chunk_count, fail_after=None, short_by=0):
        self.root = root
        self.canvas = (64, 64)
        self.geometry = HarnessGeometry(CHUNK_FRAMES, OVERLAP_FRAMES).validate()
        self.chunk_count = chunk_count
        self.target_frames = STRIDE_FRAMES * chunk_count
        self.manifest = None
        self.fail_after = fail_after
        # Chunks the sampler stops short of emitting, so the totals check fails
        # without anything raising inside the loop.
        self.short_by = short_by
        self.events = []

    def path(self, kind, index, suffix=".safetensors"):
        return os.path.join(self.root, kind, "%06d%s" % (index, suffix))

    def _first_invalid(self, kind, count):
        return 0

    def event(self, **fields):
        self.events.append(fields)

    def _emit_chunk(self, index, video_latent, *, writer=None, **kwargs):
        if writer is not None:
            writer.write(video_latent)
        return STRIDE_FRAMES

    def sample_chunks(self, *, chunk_count, on_sampled=None, **kwargs):
        for index in range(chunk_count - self.short_by):
            if self.fail_after is not None and index > self.fail_after:
                raise Boom("sampling failed at chunk %d" % index)
            on_sampled(index, torch.zeros(1, 4, 2, 8, 8), torch.zeros(1, 8, 64))
        return chunk_count


class LongFormAVCommitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        self.writers = []

        self.raw_video = os.path.join(self.root, "output", "video_only.mkv")
        self.raw_audio = os.path.join(
            self.root, "output", "generated_audio.wav"
        )
        self.final_video = os.path.join(self.root, "output", "final.mp4")

        def video_writer(path, **kwargs):
            return FakeWriter(path, self.writers, **kwargs)

        def audio_writer(path, **kwargs):
            return FakeWriter(path, self.writers, **kwargs)

        def mux(video_path, audio_path, output_path, **kwargs):
            with open(output_path, "wb"):
                pass
            return output_path

        patches = {
            "FFmpegVideoWriter": video_writer,
            "FFmpegAudioWriter": audio_writer,
            "mux_generated_audio": mux,
            # One chunk of audio, with room to spare for the trim.
            "decode_audio_chunk": lambda vae, latent: torch.zeros(
                1, 2, 2 * audio_samples_for_frames(
                    CHUNK_FRAMES, SAMPLE_RATE, fps=24)
            ),
            "audio_sample_rate": lambda vae: SAMPLE_RATE,
        }
        for name, replacement in patches.items():
            original = getattr(audio_runtime, name)
            setattr(audio_runtime, name, replacement)
            self.addCleanup(setattr, audio_runtime, name, original)

        token = audio_runtime._ACTIVE_AUDIO_VAE.set(object())
        self.addCleanup(audio_runtime._ACTIVE_AUDIO_VAE.reset, token)

    def run_assembly(self, run):
        return audio_runtime._sample_and_write_av(
            run,
            model=None,
            conditioning=None,
            sampler=None,
            sigmas=None,
            video_vae=None,
            chunk_count=run.chunk_count,
            save_frames=False,
            ffmpeg_location=None,
        )

    def assert_nothing_committed(self):
        self.assertEqual(len(self.writers), 2, "both writers were opened")
        for writer in self.writers:
            self.assertIs(
                writer.committed, False,
                "%s was closed with commit=%r" % (
                    os.path.basename(writer.path), writer.committed),
            )
            self.assertFalse(
                os.path.exists(writer.path),
                "%s exists after a failed run" % writer.path,
            )
            self.assertFalse(
                os.path.exists(writer.partial),
                "%s was left behind" % writer.partial,
            )

    def test_completed_run_commits_both_artifacts(self):
        run = FakeRun(self.root, chunk_count=2)
        frames, output_path = self.run_assembly(run)

        self.assertEqual(frames, run.target_frames)
        self.assertEqual(output_path, self.final_video)
        self.assertEqual(len(self.writers), 2)
        self.assertTrue(all(w.committed for w in self.writers))
        self.assertTrue(os.path.exists(self.raw_video))
        self.assertTrue(os.path.exists(self.raw_audio))
        self.assertEqual(
            run._h3_audio_output["samples"],
            audio_samples_for_frames(run.target_frames, SAMPLE_RATE, fps=24),
        )

    def test_failure_after_first_chunk_discards_partials(self):
        run = FakeRun(self.root, chunk_count=3, fail_after=0)
        with self.assertRaises(Boom):
            self.run_assembly(run)
        self.assert_nothing_committed()

    def test_short_run_fails_validation_before_committing(self):
        # Nothing raises inside the loop; the frame total simply does not add
        # up, which used to be checked only after the writers had committed.
        run = FakeRun(self.root, chunk_count=3, short_by=1)
        with self.assertRaises(RuntimeError) as caught:
            self.run_assembly(run)
        self.assertIn("expected exactly", str(caught.exception))
        self.assert_nothing_committed()

    def test_video_close_failure_still_closes_the_audio_writer(self):
        run = FakeRun(self.root, chunk_count=3, fail_after=0)

        def failing_video_writer(path, **kwargs):
            writer = FakeWriter(path, self.writers, **kwargs)
            writer.close_error = OSError("encoder died")
            return writer

        audio_runtime.FFmpegVideoWriter = failing_video_writer

        with self.assertRaises(Boom):
            self.run_assembly(run)

        # The close failure is secondary: it must not replace Boom, and the
        # audio writer must still have been shut down.
        self.assertEqual(len(self.writers), 2)
        self.assertIs(self.writers[1].committed, False)
        self.assertFalse(os.path.exists(self.raw_audio))
        self.assertFalse(os.path.exists(self.writers[1].partial))


class CloseWritersTests(unittest.TestCase):
    class Recorder:
        def __init__(self, error=None):
            self.error = error
            self.committed = None

        def close(self, *, commit=True):
            self.committed = commit
            if self.error is not None:
                raise self.error

    def test_skips_none_and_forwards_commit(self):
        writer = self.Recorder()
        close_writers(None, writer, None, commit=True)
        self.assertIs(writer.committed, True)

    def test_commit_closes_all_then_raises_the_first_error(self):
        first = self.Recorder(error=OSError("first"))
        second = self.Recorder(error=OSError("second"))
        third = self.Recorder()

        with self.assertRaises(OSError) as caught:
            close_writers(first, second, third, commit=True)

        self.assertEqual(str(caught.exception), "first")
        self.assertIs(third.committed, True)

    def test_discard_swallows_close_errors(self):
        first = self.Recorder(error=OSError("first"))
        second = self.Recorder()

        # A run is already unwinding here; raising would hide its cause.
        close_writers(first, second, commit=False)

        self.assertIs(first.committed, False)
        self.assertIs(second.committed, False)


if __name__ == "__main__":
    # comfy's arg parsing ran on a rewritten argv; give unittest the real one.
    sys.argv = _ARGV
    unittest.main()
