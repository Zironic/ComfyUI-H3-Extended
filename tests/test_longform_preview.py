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
                lambda path, images: open(path, "wb").write(
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
                lambda path, images: open(path, "wb").write(
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
                lambda path, images: open(path, "wb").write(
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


if __name__ == "__main__":
    unittest.main()
