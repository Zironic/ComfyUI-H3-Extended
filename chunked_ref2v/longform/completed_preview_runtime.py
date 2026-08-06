"""Failure-safe publishing for completed long-form preview chunks.

The primary completed-preview path remains a finalized H.264 MP4 segment so the
browser can play committed chunks as a playlist.  This wrapper adds two missing
failure semantics:

* if FFmpeg cannot create the MP4 segment, publish the same committed frames as
  an animated GIF instead;
* if both encoders fail, announce an explicit error event rather than leaving
  the browser pane at ``waiting`` forever.

Preview failures remain non-fatal to the actual long-form generation.
"""

from __future__ import annotations

import logging
import os

from . import preview

LOG = "[H3 Extended] longform completed preview"
_INSTALLED = False
_ORIGINAL_PUBLISH = None


def _message(exc):
    return "%s: %s" % (type(exc).__name__, exc)


def _gif_images(frames_u8):
    from PIL import Image

    return [Image.fromarray(frame.numpy(), "RGB") for frame in frames_u8]


def _publish_completed_resilient(
    self,
    *,
    chunk_index,
    frames_u8,
    completed_frames,
):
    """Publish MP4 first, then GIF, then a visible error event.

    The method deliberately consumes only the already-decoded committed frame
    batch.  It never loads a model or VAE and cannot affect the sampled result.
    """

    if not self.options.completed_enabled or int(frames_u8.shape[0]) == 0:
        return

    try:
        return _ORIGINAL_PUBLISH(
            self,
            chunk_index=chunk_index,
            frames_u8=frames_u8,
            completed_frames=completed_frames,
        )
    except Exception as primary_exc:
        primary_message = _message(primary_exc)
        logging.warning(
            "%s MP4 failed for chunk %d; trying GIF fallback: %s",
            LOG,
            int(chunk_index),
            primary_message,
        )

    resized = None
    images = None
    try:
        resized = preview._resize_frames_u8(
            frames_u8,
            self.options.width,
        )
        images = _gif_images(resized)
        path = os.path.join(
            self.temp_root,
            "completed_%06d.gif" % int(chunk_index),
        )
        self._write_animation(path, images)
        self.completed_frames = int(completed_frames)
        self._announce(
            "completed_chunk_fallback",
            chunk_index=int(chunk_index),
            chunk_frames=int(resized.shape[0]),
            completed_frames=self.completed_frames,
            fps=int(self.fps),
            mode="gif",
            fallback_reason=primary_message[:500],
            asset=preview._asset_payload(path, "temp"),
        )
    except Exception as fallback_exc:
        fallback_message = _message(fallback_exc)
        combined = (
            "MP4 preview failed (%s); GIF fallback failed (%s)"
            % (primary_message, fallback_message)
        )
        logging.warning(
            "%s all encoders failed for chunk %d: %s",
            LOG,
            int(chunk_index),
            combined,
        )
        self._announce(
            "completed_chunk_error",
            chunk_index=int(chunk_index),
            completed_frames=int(completed_frames),
            message=combined[:1000],
        )
    finally:
        del resized, images


def install():
    """Install once after :mod:`preview` has defined its publisher class."""

    global _INSTALLED, _ORIGINAL_PUBLISH
    if _INSTALLED:
        return
    _ORIGINAL_PUBLISH = (
        preview.LongFormPreviewPublisher.publish_completed_chunk
    )
    preview.LongFormPreviewPublisher.publish_completed_chunk = (
        _publish_completed_resilient
    )
    _INSTALLED = True


__all__ = ["install"]
