"""Generated-audio carry and assembly for LongFormReferenceVideo.

This module installs the audiovisual behavior on the generic reference runner
without changing the source-timeline Ref2V runner. The existing reference run
still owns reference persistence, manifests, seeds, and video emission.
"""

from __future__ import annotations

import contextvars
import gc
import logging
import os

import torch

from .. import harness
from ..geometry import AUDIO_LATENT_FPS, HarnessGeometry, latent_frame_spans
from ..layout_ops import TargetAlignedCondition
from ..model_patch import patch_target_conditions
from . import reference_runner
from .audio_conditions import (
    TargetAlignedAudioCondition,
    patch_target_audio_conditions,
)
from .audio_output import (
    FFmpegAudioWriter,
    audio_sample_rate,
    audio_samples_for_frames,
    decode_audio_chunk,
    mux_generated_audio,
)
from .runner import CARRY_FRAME, CARRY_NONE
from .writer import FFmpegVideoWriter

LOG = "[H3 Extended] longform reference audio"
_ACTIVE_AUDIO_VAE = contextvars.ContextVar(
    "h3_longform_reference_audio_vae", default=None
)
_INSTALLED = False


def audio_latent_boundary(
    frame_index, *, fps=24, audio_latent_fps=AUDIO_LATENT_FPS
):
    """Audio-latent boundary matching H3 target construction."""

    return round(int(frame_index) / float(fps) * int(audio_latent_fps))


def audio_overlap_slice(geometry):
    """Tail audio slice for the same pixel-frame overlap used by video."""

    start = audio_latent_boundary(
        geometry.stride_frames,
        fps=geometry.fps,
        audio_latent_fps=AUDIO_LATENT_FPS,
    )
    count = geometry.audio_latent_t - start
    if start < 0 or count <= 0:
        raise ValueError(
            "invalid audio overlap for C=%d O=%d: start=%d count=%d"
            % (
                geometry.chunk_frames,
                geometry.overlap_frames,
                start,
                count,
            )
        )
    return start, count


def _sample_chunks_av(
    self,
    *,
    model,
    conditioning,
    sampler,
    sigmas,
    chunk_count,
    on_sampled=None,
):
    geometry = self.geometry
    video_overlap_start = video_overlap_count = None
    audio_overlap_start = audio_overlap_count = None
    if self.carry != CARRY_NONE:
        video_overlap_start, video_overlap_count = geometry.overlap_slice()
        audio_overlap_start, audio_overlap_count = audio_overlap_slice(
            geometry
        )

    prefix = self._first_invalid("samples", chunk_count)
    self._remove_suffix("samples", prefix, chunk_count)
    carry_video = carry_audio = None
    if prefix:
        previous = reference_runner._load(
            self.path("samples", prefix - 1)
        )
        carry_video = previous["video_latent"]
        carry_audio = previous["audio_latent"]
        del previous

    for index in range(prefix, chunk_count):
        chunk_conditioning = harness.attach_refs(
            conditioning, self.static_blocks
        )
        video_conditions = []
        audio_conditions = []
        if (
            carry_video is not None
            and carry_audio is not None
            and self.carry != CARRY_NONE
        ):
            if self.carry == CARRY_FRAME:
                video_count = 1
                shared_frames = latent_frame_spans(
                    geometry.target_latent_t
                )[video_overlap_start]
                audio_start = audio_overlap_start
                audio_stop = audio_latent_boundary(
                    geometry.stride_frames + shared_frames,
                    fps=geometry.fps,
                    audio_latent_fps=AUDIO_LATENT_FPS,
                )
                audio_count = audio_stop - audio_start
            else:
                video_count = video_overlap_count
                audio_start = audio_overlap_start
                audio_count = audio_overlap_count

            video_conditions.append(
                TargetAlignedCondition(
                    latent=carry_video[
                        :,
                        :,
                        video_overlap_start:
                        video_overlap_start + video_count,
                    ],
                    target_latent_start=0,
                    label="video carry from chunk %d" % (index - 1),
                )
            )
            audio_conditions.append(
                TargetAlignedAudioCondition(
                    latent=carry_audio[
                        ...,
                        audio_start:audio_start + audio_count,
                    ],
                    target_latent_start=0,
                    label="audio carry from chunk %d" % (index - 1),
                )
            )

        armed = patch_target_conditions(model, video_conditions)
        armed = patch_target_audio_conditions(armed, audio_conditions)
        latent = harness.empty_av_latent(self.canvas, geometry)

        callback = None
        try:
            from .preview import current_publisher

            publisher = current_publisher()
            if publisher is not None:
                callback = publisher.sampler_callback(index)
        except Exception:
            callback = None

        sample_kwargs = {
            "model": armed,
            "conditioning": chunk_conditioning,
            "latent": latent,
            "sampler": sampler,
            "sigmas": sigmas,
            "seed": self.chunk_seed(index),
        }
        if callback is not None:
            sample_kwargs["callback"] = callback
        video_latent, audio_latent = harness.sample(**sample_kwargs)
        video_latent = video_latent.to("cpu", torch.float32)
        audio_latent = audio_latent.to("cpu", torch.float32)
        reference_runner._save(
            self.path("samples", index),
            {
                "video_latent": video_latent,
                "audio_latent": audio_latent,
            },
        )
        reference_runner._atomic_json(
            self.path("samples", index, ".json"),
            {
                "index": index,
                "seed": self.chunk_seed(index),
                "previous_sample": index - 1 if index else None,
                "carry": self.carry,
                "video_conditions": len(video_conditions),
                "audio_conditions": len(audio_conditions),
            },
        )
        carry_video = video_latent
        carry_audio = audio_latent
        if on_sampled is not None:
            on_sampled(index, video_latent, audio_latent)
        if self.manifest:
            self.manifest.update_state(pass_c_chunks=index + 1)
        self.event(
            pass_="C",
            chunk=index,
            carry=self.carry,
            video_conditions=len(video_conditions),
            audio_conditions=len(audio_conditions),
        )
        del chunk_conditioning, armed, latent
        gc.collect()
        logging.info(
            "%s chunk %d/%d sampled",
            LOG,
            index + 1,
            chunk_count,
        )
    return chunk_count


def _sample_and_write_av(
    self,
    *,
    model,
    conditioning,
    sampler,
    sigmas,
    video_vae,
    chunk_count,
    output_video,
    save_frames,
    ffmpeg_location,
):
    audio_vae = _ACTIVE_AUDIO_VAE.get()
    if output_video and audio_vae is None:
        raise RuntimeError("LongFormReferenceVideo audio VAE is not active")

    out_dir = os.path.join(self.root, "frames")
    raw_video = os.path.join(self.root, "output", "video_only.mkv")
    raw_audio = os.path.join(
        self.root, "output", "generated_audio.wav"
    )
    final_video = os.path.join(self.root, "output", "final.mp4")

    video_writer = audio_writer = None
    state = {
        "frames": 0,
        "audio_samples": 0,
        "audio_sample_rate": None,
        "audio_channels": None,
    }
    pbar = None
    try:
        from .runner import _progress_bar

        pbar = _progress_bar(chunk_count)
    except Exception:
        pass

    if output_video:
        video_writer = FFmpegVideoWriter(
            raw_video,
            width=self.canvas[0],
            height=self.canvas[1],
            fps=self.geometry.fps,
            ffmpeg_location=ffmpeg_location,
        ).open()

    def emit(index, video_latent, audio_latent):
        nonlocal audio_writer
        take_frames = self._emit_chunk(
            index,
            video_latent,
            video_vae=video_vae,
            chunk_count=chunk_count,
            writer=video_writer,
            save_frames=save_frames,
            written=state["frames"],
            out_dir=out_dir,
            pbar=pbar,
        )
        if take_frames <= 0:
            return

        if output_video:
            waveform = decode_audio_chunk(audio_vae, audio_latent)
            sample_rate = audio_sample_rate(audio_vae)
            channels = int(waveform.shape[1])
            if state["audio_sample_rate"] is None:
                state["audio_sample_rate"] = sample_rate
                state["audio_channels"] = channels
                audio_writer = FFmpegAudioWriter(
                    raw_audio,
                    sample_rate=sample_rate,
                    channels=channels,
                    ffmpeg_location=ffmpeg_location,
                ).open()
            elif (
                sample_rate != state["audio_sample_rate"]
                or channels != state["audio_channels"]
            ):
                raise RuntimeError(
                    "audio VAE output format changed between chunks"
                )

            desired_total = audio_samples_for_frames(
                state["frames"] + take_frames,
                sample_rate,
                fps=self.geometry.fps,
            )
            take_samples = desired_total - state["audio_samples"]
            if waveform.shape[-1] < take_samples:
                raise RuntimeError(
                    "audio chunk %d decoded %d samples, need %d for "
                    "%d committed video frames"
                    % (
                        index,
                        waveform.shape[-1],
                        take_samples,
                        take_frames,
                    )
                )
            audio_writer.write(waveform[..., :take_samples])
            state["audio_samples"] += take_samples
            del waveform

        state["frames"] += take_frames

    try:
        resume_from = self._first_invalid("samples", chunk_count)
        for index in range(resume_from):
            stored = reference_runner._load(
                self.path("samples", index)
            )
            if stored is None:
                raise RuntimeError(
                    "chunk %d has no stored sample" % index
                )
            emit(
                index,
                stored["video_latent"],
                stored["audio_latent"],
            )
            del stored

        self.sample_chunks(
            model=model,
            conditioning=conditioning,
            sampler=sampler,
            sigmas=sigmas,
            chunk_count=chunk_count,
            on_sampled=emit,
        )
    finally:
        if video_writer is not None:
            video_writer.close(commit=True)
        if audio_writer is not None:
            audio_writer.close(commit=True)

    if state["frames"] != self.target_frames:
        raise RuntimeError(
            "assembled %d frames, expected exactly %d"
            % (state["frames"], self.target_frames)
        )

    output_path = ""
    if output_video:
        expected_audio = audio_samples_for_frames(
            self.target_frames,
            state["audio_sample_rate"],
            fps=self.geometry.fps,
        )
        if state["audio_samples"] != expected_audio:
            raise RuntimeError(
                "assembled %d audio samples, expected exactly %d"
                % (state["audio_samples"], expected_audio)
            )
        output_path = mux_generated_audio(
            raw_video,
            raw_audio,
            final_video,
            frame_count=self.target_frames,
            fps=self.geometry.fps,
            ffmpeg_location=ffmpeg_location,
        )

    self._h3_audio_output = {
        "samples": state["audio_samples"],
        "sample_rate": state["audio_sample_rate"],
    }
    if self.manifest:
        self.manifest.update_state(
            pass_d_chunks=chunk_count,
            frames_written=state["frames"],
            audio_samples_written=state["audio_samples"],
            audio_sample_rate=state["audio_sample_rate"],
        )
    return state["frames"], output_path


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    reference_runner.LongFormReferenceRun.sample_chunks = (
        _sample_chunks_av
    )
    reference_runner.LongFormReferenceRun.sample_and_write = (
        _sample_and_write_av
    )
    _INSTALLED = True


def run(**kwargs):
    """Run the existing generic engine with audiovisual carry installed."""

    install()
    audio_vae = kwargs["audio_vae"]
    geometry = HarnessGeometry(
        chunk_frames=kwargs["chunk_frames"],
        overlap_frames=kwargs["overlap_frames"],
    ).validate()
    audio_start, audio_count = audio_overlap_slice(geometry)
    token = _ACTIVE_AUDIO_VAE.set(audio_vae)
    try:
        summary = reference_runner.run(**kwargs)
    finally:
        _ACTIVE_AUDIO_VAE.reset(token)

    if kwargs.get("output_video", True):
        sample_rate = audio_sample_rate(audio_vae)
        audio_samples = audio_samples_for_frames(
            kwargs["target_frames"],
            sample_rate,
            fps=geometry.fps,
        )
    else:
        sample_rate = None
        audio_samples = 0
    summary.update(
        {
            "audio_samples": audio_samples,
            "audio_sample_rate": sample_rate,
            "audio_overlap_latent": (
                audio_start,
                audio_count,
            ),
        }
    )
    return summary


install()


__all__ = [
    "audio_latent_boundary",
    "audio_overlap_slice",
    "install",
    "run",
]
