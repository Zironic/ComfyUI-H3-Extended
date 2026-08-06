"""CPU-only checks for the long-form live preview plumbing."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chunked_ref2v.longform import preview


class FakeNested:
    def __init__(self, *tensors):
        self.tensors = tensors


class FakeVAE:
    def decode(self, latent):
        # Five temporal latents are sufficient for the behavioral test; the
        # production MiniMax VAE supplies the actual 17-frame mapping.
        frames = 17 if latent.shape[2] >= 5 else max(1, latent.shape[2])
        return torch.linspace(0, 1, frames).view(frames, 1, 1, 1).expand(
            frames, 24, 32, 3).contiguous()


class LongFormPreviewTests(unittest.TestCase):
    def test_default_two_step_cadence_and_final_step(self):
        emitted = [
            step for step in range(1, 8)
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
        options = preview.PreviewOptions(every_steps=2)
        publisher = object.__new__(preview.LongFormPreviewPublisher)
        publisher.options = options
        calls = []
        publisher.publish_current_chunk = lambda **kw: calls.append(kw)
        callback = publisher.sampler_callback(3)
        nested = FakeNested(torch.zeros(1, 24, 7, 4, 4),
                            torch.zeros(1, 32, 2, 40))
        for step in range(5):
            callback(step, nested, nested, 5)
        self.assertEqual([call["step"] for call in calls], [2, 4, 5])
        self.assertTrue(all(call["chunk_index"] == 3 for call in calls))

    def test_current_preview_writes_seventeen_frames_and_event(self):
        with tempfile.TemporaryDirectory() as temp:
            publisher = object.__new__(preview.LongFormPreviewPublisher)
            publisher.node_id = "12"
            publisher.video_vae = FakeVAE()
            publisher.root = temp
            publisher.fps = 24
            publisher.ffmpeg_location = None
            publisher.options = preview.PreviewOptions(
                current_enabled=True, completed_enabled=True,
                every_steps=2, current_frames=17, width=128)
            publisher.revision = 0
            publisher.completed_frames = 0
            publisher.temp_root = temp
            events = []
            publisher._announce = lambda kind, **fields: events.append((kind, fields))
            publisher._write_webp = lambda path, frames: open(path, "wb").write(
                bytes([frames.shape[0]]))

            nested = FakeNested(torch.zeros(1, 24, 7, 4, 4),
                                torch.zeros(1, 32, 2, 40))
            with mock.patch.object(preview, "_asset_payload",
                                   return_value={"filename": "current.webp",
                                                 "subfolder": "", "type": "temp"}):
                publisher.publish_current_chunk(
                    chunk_index=0, step=2, total_steps=20, denoised=nested)

            self.assertEqual(events[0][0], "current_chunk")
            self.assertEqual(events[0][1]["frames"], 17)

    def test_preview_failure_does_not_escape_sampler_callback(self):
        publisher = object.__new__(preview.LongFormPreviewPublisher)
        publisher.options = preview.PreviewOptions(every_steps=1)
        publisher.publish_current_chunk = mock.Mock(
            side_effect=RuntimeError("preview failed"))
        callback = publisher.sampler_callback(0)
        nested = FakeNested(torch.zeros(1, 24, 7, 4, 4),
                            torch.zeros(1, 32, 2, 40))
        callback(0, nested, nested, 1)  # must not raise


if __name__ == "__main__":
    unittest.main()
