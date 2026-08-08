"""CPU checks for failure-safe completed-chunk preview publishing."""

from __future__ import annotations

import os
import sys
import h3_test_tempfile as tempfile
import unittest
from unittest import mock

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chunked_ref2v.longform import completed_preview_runtime as runtime
from chunked_ref2v.longform import preview


def make_publisher(temp):
    publisher = object.__new__(preview.LongFormPreviewPublisher)
    publisher.node_id = "12"
    publisher.root = temp
    publisher.temp_root = temp
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
    return publisher


class CompletedPreviewTests(unittest.TestCase):
    def setUp(self):
        self.original = runtime._ORIGINAL_PUBLISH

    def tearDown(self):
        runtime._ORIGINAL_PUBLISH = self.original

    def test_primary_mp4_path_is_left_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            publisher = make_publisher(temp)
            calls = []

            def success(self, **kwargs):
                calls.append(kwargs)
                return "mp4"

            runtime._ORIGINAL_PUBLISH = success
            result = runtime._publish_completed_resilient(
                publisher,
                chunk_index=2,
                frames_u8=torch.zeros(4, 24, 32, 3, dtype=torch.uint8),
                completed_frames=100,
            )
            self.assertEqual(result, "mp4")
            self.assertEqual(calls[0]["chunk_index"], 2)

    def test_mp4_failure_publishes_gif_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            publisher = make_publisher(temp)
            events = []
            publisher._announce = (
                lambda kind, **fields: events.append((kind, fields))
            )

            def fail(self, **kwargs):
                raise RuntimeError("h264 unavailable")

            runtime._ORIGINAL_PUBLISH = fail
            with mock.patch.object(
                preview,
                "_asset_payload",
                return_value={
                    "filename": "completed_000001.gif",
                    "subfolder": "",
                    "type": "temp",
                },
            ):
                runtime._publish_completed_resilient(
                    publisher,
                    chunk_index=1,
                    frames_u8=torch.zeros(
                        4, 24, 32, 3, dtype=torch.uint8
                    ),
                    completed_frames=90,
                )

            self.assertEqual(events[0][0], "completed_chunk_fallback")
            self.assertEqual(events[0][1]["mode"], "gif")
            self.assertEqual(events[0][1]["completed_frames"], 90)
            self.assertIn("h264 unavailable", events[0][1]["fallback_reason"])
            self.assertTrue(
                os.path.isfile(os.path.join(temp, "completed_000001.gif"))
            )

    def test_both_failures_publish_visible_error_without_raising(self):
        with tempfile.TemporaryDirectory() as temp:
            publisher = make_publisher(temp)
            events = []
            publisher._announce = (
                lambda kind, **fields: events.append((kind, fields))
            )
            publisher._write_animation = mock.Mock(
                side_effect=OSError("temp directory is read-only")
            )

            def fail(self, **kwargs):
                raise RuntimeError("ffmpeg exited")

            runtime._ORIGINAL_PUBLISH = fail
            runtime._publish_completed_resilient(
                publisher,
                chunk_index=3,
                frames_u8=torch.zeros(4, 24, 32, 3, dtype=torch.uint8),
                completed_frames=180,
            )

            self.assertEqual(events[0][0], "completed_chunk_error")
            self.assertIn("ffmpeg exited", events[0][1]["message"])
            self.assertIn("read-only", events[0][1]["message"])

    def test_disabled_completed_preview_does_no_work(self):
        with tempfile.TemporaryDirectory() as temp:
            publisher = make_publisher(temp)
            publisher.options.completed_enabled = False
            original = mock.Mock(side_effect=AssertionError("must not run"))
            runtime._ORIGINAL_PUBLISH = original
            runtime._publish_completed_resilient(
                publisher,
                chunk_index=0,
                frames_u8=torch.zeros(4, 24, 32, 3, dtype=torch.uint8),
                completed_frames=4,
            )
            original.assert_not_called()


if __name__ == "__main__":
    unittest.main()
