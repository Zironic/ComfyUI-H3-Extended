"""CPU-only checks for the long-form live preview plumbing."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chunked_ref2v.longform import preview


class FakeNested:
    def __init__(self, *tensors):
        self.tensors = tensors


class FakeVAE:
    def decode(self, latent):
        frames = 17 if latent.shape[2] >= 5 else max(1, latent.shape[2])
        return (
            torch.linspace(0, 1, frames)
            .view(frames, 1, 1, 1)
            .expand(frames, 24, 32, 3)
            .contiguous()
        )


class FailingVAE:
    def decode(self, latent):
        raise RuntimeError("VAE cannot load during sampling")


class FakeLatentPreviewer:
    def decode_latent_to_preview(self, latent):
        value = int(latent.shape[2])
        return Image.new("RGB", (32, 24), (value, 32, 64))


def make_publisher(temp, *, video_vae=None, latent_previewer=None):
    publisher = object.__new__(preview.LongFormPreviewPublisher)
    publisher.node_id = "12"
    publisher.video_vae = video_vae
    publisher.latent_previewer = latent_previewer
    publisher.latent_process_out = None
    publisher.root = temp
    publisher.fps = 24
    publisher.ffmpeg_location = None
    publisher.options = preview.PreviewOptions(
        current_enabled=True,
        completed_enabled=True,
        every_steps=2,
        current_frames=17,
        width=128,
    )
    publisher.revision = 0
    publisher.completed_frames = 0
    publisher.temp_root = temp
    publisher.audio_channels = None
    publisher.audio_rate = None
    publisher.audio_path = os.path.join(temp, "completed_audio.f32le")
    return publisher


class LongFormPreviewTests(unittest.TestCase):
    def test_default_two_step_cadence_and_final_step(self):
        emitted = [
            step
            for step in range(1, 8)
            if preview.should_emit_step(step, 7, 2)
        ]
        self.assertEqual(emitted, [2, 4, 6, 7])

    def test_video_stream_is_selected_from_nested_callback(self):
        video = torch.zeros(1, 24, 7, 4, 4)
        audio = torch.zeros(1, 32, 2, 40)
        self.assertIs(preview._video_latent(FakeNested(video, audio)), video)

    def test_resize_is_bounded_even_and_preserves_frame_count(self):
        frames = torch.zeros(17, 101, 301, 3, dtype=torch.uint8)
        resized = preview._resize_frames_u8(frames, 128)
        self.assertEqual(resized.shape[0], 17)
        self.assertLessEqual(resized.shape[2], 128)
        self.assertEqual(resized.shape[1] % 2, 0)
        self.assertEqual(resized.shape[2] % 2, 0)
        self.assertEqual(resized.dtype, torch.uint8)

    def test_sampler_callback_emits_only_requested_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            publisher = make_publisher(
                temp,
                latent_previewer=FakeLatentPreviewer(),
            )
            calls = []
            publisher.publish_current_chunk = lambda **kw: calls.append(kw)
            callback = publisher.sampler_callback(3)
            nested = FakeNested(
                torch.zeros(1, 24, 7, 4, 4),
                torch.zeros(1, 32, 2, 40),
            )
            for step in range(5):
                callback(step, nested, nested, 5)
            self.assertEqual(
                [call["step"] for call in calls],
                [2, 4, 5],
            )
            self.assertTrue(
                all(call["chunk_index"] == 3 for call in calls)
            )

    def test_callback_uses_current_when_denoised_is_none(self):
        with tempfile.TemporaryDirectory() as temp:
            publisher = make_publisher(
                temp,
                latent_previewer=FakeLatentPreviewer(),
            )
            calls = []
            publisher.publish_current_chunk = lambda **kw: calls.append(kw)
            callback = publisher.sampler_callback(0)
            current = FakeNested(
                torch.zeros(1, 24, 7, 4, 4),
                torch.zeros(1, 32, 2, 40),
            )
            callback(0, None, current, 1)
            self.assertIs(calls[0]["current"], current)

    def test_default_current_preview_uses_latent_path_without_vae(self):
        with tempfile.TemporaryDirectory() as temp:
            publisher = make_publisher(
                temp,
                video_vae=None,
                latent_previewer=FakeLatentPreviewer(),
            )
            events = []
            publisher._announce = (
                lambda kind, **fields: events.append((kind, fields))
            )
            publisher._write_animation = (
                lambda path, images, **kwargs: open(path, "wb").write(
                    bytes([len(images)])
                )
            )
            nested = FakeNested(
                torch.zeros(1, 24, 7, 4, 4),
                torch.zeros(1, 32, 2, 40),
            )
            with mock.patch.object(
                preview,
                "_asset_payload",
                return_value={
                    "filename": "current.gif",
                    "subfolder": "",
                    "type": "temp",
                },
            ):
                publisher.publish_current_chunk(
                    chunk_index=0,
                    step=2,
                    total_steps=20,
                    denoised=nested,
                )

            self.assertEqual(events[0][0], "current_chunk")
            self.assertEqual(events[0][1]["mode"], "latent")
            self.assertEqual(events[0][1]["frames"], 5)

    def test_explicit_preview_vae_still_produces_exact_animation(self):
        with tempfile.TemporaryDirectory() as temp:
            publisher = make_publisher(
                temp,
                video_vae=FakeVAE(),
                latent_previewer=FakeLatentPreviewer(),
            )
            events = []
            publisher._announce = (
                lambda kind, **fields: events.append((kind, fields))
            )
            publisher._write_animation = (
                lambda path, images, **kwargs: open(path, "wb").write(
                    bytes([len(images)])
                )
            )
            nested = FakeNested(
                torch.zeros(1, 24, 7, 4, 4),
                torch.zeros(1, 32, 2, 40),
            )
            with mock.patch.object(
                preview,
                "_asset_payload",
                return_value={
                    "filename": "current.gif",
                    "subfolder": "",
                    "type": "temp",
                },
            ):
                publisher.publish_current_chunk(
                    chunk_index=0,
                    step=2,
                    total_steps=20,
                    denoised=nested,
                )

            self.assertEqual(events[0][1]["mode"], "vae")
            self.assertEqual(events[0][1]["frames"], 17)

    def test_failed_explicit_vae_falls_back_to_latent_preview(self):
        with tempfile.TemporaryDirectory() as temp:
            publisher = make_publisher(
                temp,
                video_vae=FailingVAE(),
                latent_previewer=FakeLatentPreviewer(),
            )
            events = []
            publisher._announce = (
                lambda kind, **fields: events.append((kind, fields))
            )
            publisher._write_animation = (
                lambda path, images, **kwargs: open(path, "wb").write(
                    bytes([len(images)])
                )
            )
            nested = FakeNested(
                torch.zeros(1, 24, 7, 4, 4),
                torch.zeros(1, 32, 2, 40),
            )
            with mock.patch.object(
                preview,
                "_asset_payload",
                return_value={
                    "filename": "current.gif",
                    "subfolder": "",
                    "type": "temp",
                },
            ):
                publisher.publish_current_chunk(
                    chunk_index=0,
                    step=2,
                    total_steps=20,
                    denoised=nested,
                )

            self.assertEqual(events[0][1]["mode"], "latent")
            self.assertIn("fallback_reason", events[0][1])

    def test_preview_failure_announces_error_without_escaping_callback(self):
        with tempfile.TemporaryDirectory() as temp:
            publisher = make_publisher(temp)
            publisher.options.every_steps = 1
            publisher.publish_current_chunk = mock.Mock(
                side_effect=RuntimeError("preview failed")
            )
            events = []
            publisher._announce = (
                lambda kind, **fields: events.append((kind, fields))
            )
            callback = publisher.sampler_callback(0)
            nested = FakeNested(
                torch.zeros(1, 24, 7, 4, 4),
                torch.zeros(1, 32, 2, 40),
            )
            callback(0, nested, nested, 1)
            self.assertEqual(events[0][0], "current_chunk_error")
            self.assertIn("preview failed", events[0][1]["message"])


class PreviewRateTests(unittest.TestCase):
    """A latent image covers four pixel frames; a decoded one covers one."""

    def publish(self, *, video_vae, latent_previewer):
        with tempfile.TemporaryDirectory() as temp:
            publisher = make_publisher(
                temp,
                video_vae=video_vae,
                latent_previewer=latent_previewer,
            )
            events = []
            publisher._announce = (
                lambda kind, **fields: events.append((kind, fields))
            )
            nested = FakeNested(
                torch.zeros(1, 24, 7, 4, 4),
                torch.zeros(1, 32, 2, 40),
            )
            with mock.patch.object(
                preview,
                "_asset_payload",
                return_value={
                    "filename": "current.gif",
                    "subfolder": "",
                    "type": "temp",
                },
            ):
                publisher.publish_current_chunk(
                    chunk_index=0,
                    step=2,
                    total_steps=20,
                    denoised=nested,
                )
            return events[0][1]

    def test_latent_preview_runs_at_a_quarter_of_the_production_rate(self):
        fields = self.publish(
            video_vae=None,
            latent_previewer=FakeLatentPreviewer(),
        )
        self.assertEqual(fields["mode"], "latent")
        self.assertEqual(fields["preview_fps"], 6)

    def test_decoded_preview_keeps_the_production_rate(self):
        fields = self.publish(
            video_vae=FakeVAE(),
            latent_previewer=FakeLatentPreviewer(),
        )
        self.assertEqual(fields["mode"], "vae")
        self.assertEqual(fields["preview_fps"], 24)

    def test_requested_rate_becomes_the_gif_frame_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            publisher = make_publisher(temp)
            images = [Image.new("RGB", (8, 8), (i * 10, 0, 0)) for i in range(3)]
            durations = []

            def capture(self, path, **kwargs):
                durations.append(kwargs["duration"])
                open(path, "wb").close()

            with mock.patch.object(
                Image.Image, "save", autospec=True, side_effect=capture
            ):
                publisher._write_animation(
                    os.path.join(temp, "a.gif"), images, fps=6
                )
                publisher._write_animation(
                    os.path.join(temp, "b.gif"), images, fps=24
                )
            self.assertEqual(durations, [167, 42])


class FakeSegmentWriter:
    """Stand in for the real encoder and just create the segment file."""

    def __init__(self, path, **kwargs):
        self.path = path

    def open(self):
        return self

    def write(self, frames_u8):
        with open(self.path, "wb") as handle:
            handle.write(b"segment")

    def close(self, *, commit=True):
        pass


class CompletedAudioTests(unittest.TestCase):
    """The completed pane's segments must carry the committed audio."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.publisher = make_publisher(self.temp.name)
        self.publisher.segment_paths = []
        self.publisher.stitched_path = None
        self.publisher.stitch_index = 0
        self.publisher.pending_chunk = None
        self.publisher.audio_expected = True
        self.publisher.audio_channels = None
        self.publisher.audio_rate = None
        self.publisher.audio_path = os.path.join(
            self.temp.name, "completed_audio.f32le"
        )
        self.events = []
        self.publisher._announce = (
            lambda kind, **fields: self.events.append((kind, fields))
        )

    def test_staging_waits_and_publishes_with_audio(self):
        frames = torch.zeros(4, 24, 32, 3, dtype=torch.uint8)
        self.publisher.stage_completed_chunk(
            chunk_index=0, frames_u8=frames, completed_frames=86
        )
        # Nothing may reach the browser until the waveform arrives.
        self.assertEqual(self.events, [])
        self.assertIsNotNone(self.publisher.pending_chunk)

        published = {}
        self.publisher.publish_completed_chunk = (
            lambda **kwargs: published.update(kwargs)
        )
        waveform = torch.zeros(1, 2, 1024)
        self.publisher.flush_completed_chunk(waveform=waveform, sample_rate=32000)

        self.assertEqual(published["chunk_index"], 0)
        self.assertEqual(published["completed_frames"], 86)
        self.assertIs(published["audio"][0], waveform)
        self.assertEqual(published["audio"][1], 32000)
        self.assertIsNone(self.publisher.pending_chunk)

    def test_flush_without_audio_still_publishes_the_chunk(self):
        self.publisher.stage_completed_chunk(
            chunk_index=2,
            frames_u8=torch.zeros(4, 24, 32, 3, dtype=torch.uint8),
            completed_frames=200,
        )
        published = {}
        self.publisher.publish_completed_chunk = (
            lambda **kwargs: published.update(kwargs)
        )
        self.publisher.flush_completed_chunk()
        self.assertEqual(published["chunk_index"], 2)
        self.assertIsNone(published["audio"])

    def test_staging_a_second_chunk_flushes_the_first(self):
        calls = []
        self.publisher.publish_completed_chunk = (
            lambda **kwargs: calls.append(kwargs["chunk_index"])
        )
        for index in (0, 1):
            self.publisher.stage_completed_chunk(
                chunk_index=index,
                frames_u8=torch.zeros(4, 24, 32, 3, dtype=torch.uint8),
                completed_frames=86 * (index + 1),
            )
        self.assertEqual(calls, [0])

    def test_flush_with_nothing_staged_is_a_no_op(self):
        self.publisher.publish_completed_chunk = mock.Mock(
            side_effect=AssertionError("must not publish")
        )
        self.publisher.flush_completed_chunk()

    def test_audio_accumulates_interleaved_across_chunks(self):
        # Planar [channels, samples]; ffmpeg needs it interleaved.
        first = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
        second = torch.tensor([[3.0], [-3.0]])
        self.publisher._append_preview_audio(first, 32000)
        self.publisher._append_preview_audio(second.unsqueeze(0), 32000)

        with open(self.publisher.audio_path, "rb") as handle:
            samples = torch.frombuffer(
                bytearray(handle.read()), dtype=torch.float32
            )
        self.assertEqual(
            samples.tolist(), [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
        )
        self.assertEqual(self.publisher.audio_channels, 2)
        self.assertEqual(self.publisher.audio_rate, 32000)
        self.assertTrue(self.publisher._has_preview_audio())

    def test_changed_audio_format_is_rejected_not_silently_appended(self):
        self.publisher._append_preview_audio(torch.zeros(2, 8), 32000)
        with self.assertRaises(ValueError):
            self.publisher._append_preview_audio(torch.zeros(1, 8), 32000)
        with self.assertRaises(ValueError):
            self.publisher._append_preview_audio(torch.zeros(2, 8), 44100)

    def test_stitch_muxes_the_whole_track_once(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            open(args[args.index("-y") + 1], "wb").write(b"out")
            return mock.Mock(returncode=0, stderr=b"")

        self.publisher._append_preview_audio(torch.zeros(2, 512), 32000)
        with mock.patch.object(preview, "resolve_ffmpeg", return_value="ffmpeg"), \
                mock.patch.object(preview.subprocess, "run", side_effect=fake_run):
            self.publisher._stitch_segments(
                [os.path.join(self.temp.name, "completed_000000.mp4")]
            )

        self.assertEqual(len(calls), 2, "one concat, one audio mux")
        concat, mux = calls
        self.assertIn("concat", concat)
        self.assertEqual(mux[mux.index("-ar") + 1], "32000")
        self.assertEqual(mux[mux.index("-ac") + 1], "2")
        self.assertEqual(mux[mux.index("-c:a") + 1], "aac")
        self.assertEqual(mux[mux.index("-c:v") + 1], "copy")

    def test_mux_failure_leaves_the_stitch_silent_but_playable(self):
        def fake_run(args, **kwargs):
            if "concat" in args:
                open(args[args.index("-y") + 1], "wb").write(b"stitched")
                return mock.Mock(returncode=0, stderr=b"")
            return mock.Mock(returncode=1, stderr=b"aac encoder missing")

        with mock.patch.object(preview, "FFmpegVideoWriter", FakeSegmentWriter), \
                mock.patch.object(preview, "resolve_ffmpeg", return_value="ffmpeg"), \
                mock.patch.object(preview.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(
                    preview,
                    "_asset_payload",
                    side_effect=lambda path, kind: {
                        "filename": os.path.basename(path),
                        "subfolder": "",
                        "type": kind,
                    },
                ):
            self.publisher.publish_completed_chunk(
                chunk_index=0,
                frames_u8=torch.zeros(4, 24, 32, 3, dtype=torch.uint8),
                completed_frames=86,
                audio=(torch.zeros(2, 512), 32000),
            )

        kind, fields = self.events[0]
        self.assertEqual(kind, "completed_chunk")
        self.assertTrue(fields["stitched"], "video must survive an audio failure")
        self.assertEqual(
            open(self.publisher.stitched_path, "rb").read(), b"stitched"
        )


class CompletedStitchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.publisher = make_publisher(self.temp.name)
        self.publisher.segment_paths = []
        self.publisher.stitched_path = None
        self.publisher.pending_chunk = None
        self.publisher.audio_expected = False
        self.events = []
        self.publisher._announce = (
            lambda kind, **fields: self.events.append((kind, fields))
        )

    def publish(self, chunk_index, completed_frames, *, concat_returncode=0):
        def fake_run(args, **kwargs):
            if concat_returncode == 0:
                # ffmpeg would write the concat target named after "-y".
                with open(args[args.index("-y") + 1], "wb") as handle:
                    handle.write(b"stitched")
            return mock.Mock(
                returncode=concat_returncode,
                stderr=b"concat demuxer refused the inputs",
            )

        with mock.patch.object(preview, "FFmpegVideoWriter", FakeSegmentWriter), \
                mock.patch.object(preview, "resolve_ffmpeg", return_value="ffmpeg"), \
                mock.patch.object(preview.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(
                    preview,
                    "_asset_payload",
                    side_effect=lambda path, kind: {
                        "filename": os.path.basename(path),
                        "subfolder": "",
                        "type": kind,
                    },
                ):
            self.publisher.publish_completed_chunk(
                chunk_index=chunk_index,
                frames_u8=torch.zeros(4, 24, 32, 3, dtype=torch.uint8),
                completed_frames=completed_frames,
            )

    def test_every_completed_chunk_is_stitched_into_one_asset(self):
        self.publish(0, 86)
        self.publish(1, 172)

        kinds = [kind for kind, _ in self.events]
        self.assertEqual(kinds, ["completed_chunk", "completed_chunk"])
        second = self.events[1][1]
        self.assertTrue(second["stitched"])
        self.assertEqual(second["segments"], 2)
        self.assertEqual(second["completed_frames"], 172)
        self.assertTrue(second["asset"]["filename"].startswith("completed_all_"))
        self.assertEqual(
            second["segment_asset"]["filename"],
            "completed_000001.mp4",
        )

        listing = os.path.join(self.temp.name, "completed_segments.txt")
        with open(listing, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].endswith("completed_000000.mp4'"))
        self.assertTrue(lines[1].endswith("completed_000001.mp4'"))

    def test_stitched_asset_url_changes_between_chunks(self):
        self.publish(0, 86)
        self.publish(1, 172)
        self.assertNotEqual(
            self.events[0][1]["asset"]["filename"],
            self.events[1][1]["asset"]["filename"],
        )

    def test_concat_failure_falls_back_to_the_single_segment(self):
        self.publish(0, 86, concat_returncode=1)
        fields = self.events[0][1]
        self.assertFalse(fields["stitched"])
        self.assertEqual(fields["asset"], fields["segment_asset"])
        self.assertIn("concat demuxer refused", fields["stitch_error"])


if __name__ == "__main__":
    unittest.main()
