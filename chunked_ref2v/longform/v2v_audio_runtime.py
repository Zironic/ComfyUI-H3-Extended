"""Audiovisual runtime for disk-backed long-form Ref2V.

Every source chunk is encoded as a synchronized ``video_audio`` reference when
its source file has audio. The generated video and audio streams then carry the
same pixel-frame overlap into the next chunk. Final output stores generated
audio as the default track and the original source soundtrack as a second,
selectable track when requested.
"""

from __future__ import annotations

import contextvars
import gc
import json
import logging
import os
import sys
import time

import torch

from .. import harness, ref_builder
from ..geometry import HarnessGeometry, latent_frame_spans
from ..layout_ops import TargetAlignedCondition
from ..model_patch import patch_target_conditions
from . import runner
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
from .audio_runtime import audio_carry_timing, audio_overlap_slice, log_audio_carry
from .audio_source import read_audio_window
from .chunk_stream import iter_source_chunks
from .dual_audio import mux_generated_and_source_audio
from .writer import FFmpegVideoWriter, close_writers

LOG = "[H3 Extended] longform Ref2V audio"
_SOURCE_AUDIO_VERSION = 1
_ACTIVE_AUDIO_VAE = contextvars.ContextVar(
    "h3_longform_ref2v_audio_vae", default=None
)
_ORIGINAL_RUN = runner.run


def _preview_callback(chunk_index):
    """Use the dual-preview publisher without importing it as a side effect."""

    module = sys.modules.get(__package__ + ".preview")
    if module is None:
        return None
    publisher = module.current_publisher()
    if publisher is None:
        return None
    return publisher.sampler_callback(chunk_index)


def _source_reference_block(stored, canvas):
    """Rebuild the persisted synchronized source reference block."""

    audio_latent = stored.get("source_audio_latent")
    return {
        "kind": "video_audio" if audio_latent is not None else "video",
        "latent_t": int(stored["source_latent"].shape[2]),
        "latent_h": int(canvas[1]) // 16,
        "latent_w": int(canvas[0]) // 16,
        "ref_audio_t": (
            int(audio_latent.shape[-1]) if audio_latent is not None else 0
        ),
        "latent": stored["source_latent"],
        "audio_latent": audio_latent,
    }


def _pass_a_av(
    self,
    *,
    video_path,
    start_frame,
    chunk_count,
    video_vae,
    audio_vae,
    ref_images,
    ref_image_size,
    cond_cache,
    fps=24,
    ffmpeg_location=None,
):
    geometry = self.geometry

    static_items, static_blocks, notes = [], [], []
    for name, image in sorted((ref_images or {}).items()):
        if image is None:
            continue
        item, block, note = ref_builder.encode_image_ref(
            video_vae,
            image,
            self.canvas,
            ref_image_size,
            cond_cache=cond_cache,
        )
        static_items.append(item)
        static_blocks.append(block)
        notes.append("%s: %s" % (name, note))
    self.static_blocks = static_blocks
    self.static_items = static_items
    self.source_audio_available = False

    source_sample_rate = int(
        getattr(audio_vae, "audio_sample_rate", 32000)
    )
    started = time.time()
    done = 0
    for chunk in iter_source_chunks(
        video_path,
        chunk_frames=geometry.chunk_frames,
        stride_frames=geometry.stride_frames,
        canvas=self.canvas,
        start_frame=start_frame,
        total_frames=self.target_frames,
        fps=fps,
        ffmpeg_location=ffmpeg_location,
    ):
        if chunk.index >= chunk_count:
            break
        target = self.path("precompute", chunk.index)
        meta_path = self.path("precompute", chunk.index, ".json")
        existing = runner._load(target)
        existing_meta = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as fh:
                    existing_meta = json.load(fh)
            except (OSError, ValueError):
                existing_meta = None
        valid_existing = bool(
            existing is not None
            and existing_meta is not None
            and existing_meta.get("source_audio_conditioning_version")
            == _SOURCE_AUDIO_VERSION
            and (
                not existing_meta.get("has_source_audio")
                or "source_audio_latent" in existing
            )
        )
        if valid_existing:
            self.source_audio_available = bool(
                self.source_audio_available
                or existing_meta.get("has_source_audio")
            )
            done += 1
            continue

        frames = chunk.frames_u8.to(torch.float32).div_(255.0)
        source_audio = read_audio_window(
            video_path,
            start_frame=int(start_frame) + int(chunk.global_start),
            frame_count=geometry.chunk_frames,
            fps=fps,
            sample_rate=source_sample_rate,
            ffmpeg_location=ffmpeg_location,
            missing_ok=True,
        )
        items, block, note = ref_builder.encode_video_ref(
            video_vae,
            frames,
            self.canvas,
            audio=source_audio,
            audio_vae=audio_vae,
            cond_cache=cond_cache,
        )
        qwen_frames = next(
            item["data"] for item in items if item.get("type") == "video"
        )
        payload = {
            "source_latent": block["latent"],
            "qwen_frames": (
                (qwen_frames * 255.0)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
            ),
        }
        if block.get("audio_latent") is not None:
            payload["source_audio_latent"] = block["audio_latent"]
            self.source_audio_available = True
        runner._save(target, payload)
        runner._atomic_json(
            meta_path,
            {
                "index": chunk.index,
                "global_start": chunk.global_start,
                "actual_frames": chunk.actual_frames,
                "model_frames": chunk.model_frames,
                "is_final": chunk.is_final,
                "latent_t": int(block["latent"].shape[2]),
                "ref_audio_t": int(block.get("ref_audio_t") or 0),
                "has_source_audio": block.get("audio_latent") is not None,
                "source_audio_conditioning_version": _SOURCE_AUDIO_VERSION,
                "note": note,
            },
        )
        self.event(
            pass_="A",
            chunk=chunk.index,
            actual=chunk.actual_frames,
            source_audio=block.get("audio_latent") is not None,
        )
        done += 1
        del frames, source_audio, items, block, qwen_frames, payload, chunk

    if done != chunk_count:
        raise RuntimeError("source ended after %d/%d chunks" % (done, chunk_count))
    if self.manifest:
        self.manifest.update_state(
            pass_a_chunks=done,
            source_audio_conditioning=True,
            source_audio_available=bool(self.source_audio_available),
        )
    logging.info(
        "%s pass A complete: %d chunks in %.1f s; source audio=%s; %s",
        LOG,
        done,
        time.time() - started,
        "present" if self.source_audio_available else "absent",
        "; ".join(notes) or "no static refs",
    )
    return done


def _pass_b_av(self, *, clip, prompt, chunk_count, cond_cache):
    started = time.time()
    done = 0
    for index in range(chunk_count):
        target = self.path("conditioning", index)
        meta_path = self.path("conditioning", index, ".json")
        if runner._load(target) is not None and os.path.exists(meta_path):
            done += 1
            continue
        stored = runner._load(self.path("precompute", index))
        if stored is None:
            raise RuntimeError("pass B: missing precompute for chunk %d" % index)
        qwen_frames = stored["qwen_frames"].to(torch.float32).div_(255.0)
        items = list(self.static_items)
        if stored.get("source_audio_latent") is not None:
            items.append({"type": "audio"})
        items.append(
            {
                "type": "video",
                "data": qwen_frames,
                "timestamps": [i / 2.0 for i in range(qwen_frames.shape[0])],
            }
        )
        conditioning = harness._encode(clip, prompt, items, cond_cache)
        payload, metadata = runner._pack_conditioning(conditioning)
        runner._save(target, payload)
        runner._atomic_json(meta_path, metadata)
        self.event(
            pass_="B",
            chunk=index,
            source_audio=stored.get("source_audio_latent") is not None,
        )
        done += 1
        del stored, qwen_frames, items, conditioning
    if self.manifest:
        self.manifest.update_state(pass_b_chunks=done)
    logging.info(
        "%s pass B complete: %d chunks in %.1f s",
        LOG,
        done,
        time.time() - started,
    )
    return done


def _pass_c_av(
    self,
    *,
    model,
    sampler,
    sigmas,
    chunk_count,
    on_sampled=None,
):
    geometry = self.geometry
    started = time.time()
    video_overlap_start = video_overlap_count = None
    audio_overlap_start = audio_overlap_count = None
    if self.carry != runner.CARRY_NONE:
        video_overlap_start, video_overlap_count = geometry.overlap_slice()
        video_carry_frames = geometry.overlap_frames
        if self.carry == runner.CARRY_FRAME:
            video_carry_frames = latent_frame_spans(
                geometry.target_latent_t
            )[video_overlap_start]
        audio_overlap_start, audio_overlap_count = audio_overlap_slice(
            geometry, video_carry_frames
        )
        log_audio_carry(
            geometry,
            self.carry,
            video_carry_frames,
            1 if self.carry == runner.CARRY_FRAME else video_overlap_count,
            audio_overlap_start,
            audio_overlap_count,
        )

    prefix = self._first_invalid("samples", chunk_count)
    self._remove_suffix("samples", prefix, chunk_count)
    carry_video = carry_audio = None
    if prefix:
        previous = runner._load(self.path("samples", prefix - 1))
        carry_video = previous["video_latent"]
        carry_audio = previous["audio_latent"]
        del previous

    for index in range(prefix, chunk_count):
        stored = runner._load(self.path("precompute", index))
        cond_store = runner._load(self.path("conditioning", index))
        cond_meta_path = self.path("conditioning", index, ".json")
        if stored is None or cond_store is None or not os.path.exists(cond_meta_path):
            raise RuntimeError("pass C: chunk %d missing inputs" % index)
        with open(cond_meta_path, encoding="utf-8") as fh:
            cond_meta = json.load(fh)

        source_block = _source_reference_block(stored, self.canvas)
        conditioning = runner._unpack_conditioning(cond_store, cond_meta)
        conditioning = harness.attach_refs(
            conditioning, self.static_blocks + [source_block]
        )

        video_conditions = []
        audio_conditions = []
        if (
            carry_video is not None
            and carry_audio is not None
            and self.carry != runner.CARRY_NONE
        ):
            if self.carry == runner.CARRY_FRAME:
                video_count = 1
                audio_start = audio_overlap_start
                audio_count = audio_overlap_count
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
        callback = _preview_callback(index)
        sample_kwargs = {
            "model": armed,
            "conditioning": conditioning,
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
        runner._save(
            self.path("samples", index),
            {
                "video_latent": video_latent,
                "audio_latent": audio_latent,
            },
        )
        runner._atomic_json(
            self.path("samples", index, ".json"),
            {
                "index": index,
                "seed": self.chunk_seed(index),
                "previous_sample": index - 1 if index else None,
                "carry": self.carry,
                "video_conditions": len(video_conditions),
                "audio_conditions": len(audio_conditions),
                "source_audio_reference": (
                    source_block.get("audio_latent") is not None
                ),
            },
        )
        carry_video = video_latent
        carry_audio = audio_latent
        self.event(
            pass_="C",
            chunk=index,
            carry=self.carry,
            video_conditions=len(video_conditions),
            audio_conditions=len(audio_conditions),
            source_audio_reference=(
                source_block.get("audio_latent") is not None
            ),
        )
        if on_sampled is not None:
            on_sampled(index, video_latent, audio_latent)
        if self.manifest:
            self.manifest.update_state(pass_c_chunks=index + 1)
        del stored, cond_store, conditioning, source_block, armed, latent
        gc.collect()
        logging.info(
            "%s pass C: chunk %d/%d done (%.1f s)",
            LOG,
            index + 1,
            chunk_count,
            time.time() - started,
        )
    return chunk_count


class _AVAssembler:
    def __init__(
        self,
        run_obj,
        *,
        video_vae,
        audio_vae,
        chunk_count,
        save_frames,
        source_video,
        start_frame,
        preserve_source_audio,
        ffmpeg_location,
    ):
        self.run = run_obj
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.chunk_count = int(chunk_count)
        self.save_frames = bool(save_frames)
        self.source_video = source_video
        self.start_frame = int(start_frame)
        self.preserve_source_audio = bool(preserve_source_audio)
        self.ffmpeg_location = ffmpeg_location
        self.out_dir = os.path.join(run_obj.root, "frames")
        self.raw_video = os.path.join(
            run_obj.root, "output", "video_only.mkv"
        )
        self.raw_audio = os.path.join(
            run_obj.root, "output", "generated_audio.wav"
        )
        self.final_video = os.path.join(
            run_obj.root, "output", "final.mp4"
        )
        self.video_writer = None
        self.audio_writer = None
        self.frames = 0
        self.audio_samples = 0
        self.sample_rate = None
        self.channels = None
        self.pbar = runner._progress_bar(self.chunk_count)

    def open(self):
        self.video_writer = FFmpegVideoWriter(
            self.raw_video,
            width=self.run.canvas[0],
            height=self.run.canvas[1],
            fps=self.run.geometry.fps,
            ffmpeg_location=self.ffmpeg_location,
        ).open()
        return self

    def emit(self, index, video_latent, audio_latent):
        take_frames = self.run._emit_chunk(
            index,
            video_latent,
            video_vae=self.video_vae,
            chunk_count=self.chunk_count,
            writer=self.video_writer,
            save_frames=self.save_frames,
            written=self.frames,
            out_dir=self.out_dir,
            pbar=self.pbar,
        )
        if take_frames <= 0:
            return

        waveform = decode_audio_chunk(self.audio_vae, audio_latent)
        sample_rate = audio_sample_rate(self.audio_vae)
        channels = int(waveform.shape[1])
        if self.sample_rate is None:
            self.sample_rate = sample_rate
            self.channels = channels
            self.audio_writer = FFmpegAudioWriter(
                self.raw_audio,
                sample_rate=sample_rate,
                channels=channels,
                ffmpeg_location=self.ffmpeg_location,
            ).open()
        elif sample_rate != self.sample_rate or channels != self.channels:
            raise RuntimeError(
                "audio VAE output format changed between chunks"
            )

        desired_total = audio_samples_for_frames(
            self.frames + take_frames,
            sample_rate,
            fps=self.run.geometry.fps,
        )
        take_samples = desired_total - self.audio_samples
        if int(waveform.shape[-1]) < take_samples:
            raise RuntimeError(
                "audio chunk %d decoded %d samples, need %d for %d "
                "committed video frames"
                % (
                    index,
                    waveform.shape[-1],
                    take_samples,
                    take_frames,
                )
            )
        self.audio_writer.write(waveform[..., :take_samples])
        self.audio_samples += take_samples
        del waveform

        self.frames += take_frames

    def close(self, *, commit):
        close_writers(self.video_writer, self.audio_writer, commit=commit)

    def finalize(self):
        if self.frames != self.run.target_frames:
            raise RuntimeError(
                "assembled %d frames, expected exactly %d"
                % (self.frames, self.run.target_frames)
            )
        expected_audio = audio_samples_for_frames(
            self.run.target_frames,
            self.sample_rate,
            fps=self.run.geometry.fps,
        )
        if self.audio_samples != expected_audio:
            raise RuntimeError(
                "assembled %d generated-audio samples, expected exactly %d"
                % (self.audio_samples, expected_audio)
            )

        source_track = bool(
            self.preserve_source_audio
            and self.source_video
            and getattr(self.run, "source_audio_available", False)
        )
        if source_track:
            try:
                output_path = mux_generated_and_source_audio(
                    self.raw_video,
                    self.raw_audio,
                    self.source_video,
                    self.final_video,
                    start_frame=self.start_frame,
                    frame_count=self.run.target_frames,
                    fps=self.run.geometry.fps,
                    ffmpeg_location=self.ffmpeg_location,
                )
            except Exception as exc:
                logging.warning(
                    "%s source-track mux failed; retaining generated audio only: %s",
                    LOG,
                    exc,
                )
                source_track = False
                output_path = mux_generated_audio(
                    self.raw_video,
                    self.raw_audio,
                    self.final_video,
                    frame_count=self.run.target_frames,
                    fps=self.run.geometry.fps,
                    ffmpeg_location=self.ffmpeg_location,
                )
        else:
            output_path = mux_generated_audio(
                self.raw_video,
                self.raw_audio,
                self.final_video,
                frame_count=self.run.target_frames,
                fps=self.run.geometry.fps,
                ffmpeg_location=self.ffmpeg_location,
            )

        self.run._h3_audio_output = {
            "generated_samples": self.audio_samples,
            "sample_rate": self.sample_rate,
            "source_audio_conditioned": bool(
                getattr(self.run, "source_audio_available", False)
            ),
            "source_track": source_track,
            "audio_tracks": 2 if source_track else 1,
        }
        if self.run.manifest:
            self.run.manifest.update_state(
                pass_d_chunks=self.chunk_count,
                frames_written=self.frames,
                generated_audio_samples_written=self.audio_samples,
                generated_audio_sample_rate=self.sample_rate,
                source_audio_conditioned=bool(
                    getattr(self.run, "source_audio_available", False)
                ),
                source_audio_track=source_track,
                audio_tracks=2 if source_track else 1,
            )
        return self.frames, output_path


def _pass_cd_av(
    self,
    *,
    model,
    sampler,
    sigmas,
    video_vae,
    chunk_count,
    save_frames=True,
    source_video=None,
    start_frame=0,
    preserve_audio=True,
    ffmpeg_location=None,
):
    audio_vae = _ACTIVE_AUDIO_VAE.get()
    if audio_vae is None:
        raise RuntimeError("long-form Ref2V audio VAE is not active")
    assembler = _AVAssembler(
        self,
        video_vae=video_vae,
        audio_vae=audio_vae,
        chunk_count=chunk_count,
        save_frames=save_frames,
        source_video=source_video,
        start_frame=start_frame,
        preserve_source_audio=preserve_audio,
        ffmpeg_location=ffmpeg_location,
    ).open()
    completed = False
    started = time.time()
    try:
        resume_from = self._first_invalid("samples", chunk_count)
        if resume_from:
            logging.info(
                "%s resuming: decoding %d already-sampled chunk(s)",
                LOG,
                resume_from,
            )
        for index in range(resume_from):
            stored = runner._load(self.path("samples", index))
            if stored is None:
                raise RuntimeError(
                    "pass CD: chunk %d has no sampled latent" % index
                )
            assembler.emit(
                index,
                stored["video_latent"],
                stored["audio_latent"],
            )
            del stored
        self.pass_c(
            model=model,
            sampler=sampler,
            sigmas=sigmas,
            chunk_count=chunk_count,
            on_sampled=assembler.emit,
        )
        completed = True
    finally:
        assembler.close(commit=completed)
    frames, output_path = assembler.finalize()
    logging.info(
        "%s passes C+D complete: %d frames, %d generated audio samples in %.1f s",
        LOG,
        frames,
        assembler.audio_samples,
        time.time() - started,
    )
    return frames, output_path


def _pass_d_av(
    self,
    *,
    video_vae,
    chunk_count,
    save_frames=True,
    source_video=None,
    start_frame=0,
    preserve_audio=True,
    ffmpeg_location=None,
):
    audio_vae = _ACTIVE_AUDIO_VAE.get()
    if audio_vae is None:
        raise RuntimeError("long-form Ref2V audio VAE is not active")
    assembler = _AVAssembler(
        self,
        video_vae=video_vae,
        audio_vae=audio_vae,
        chunk_count=chunk_count,
        save_frames=save_frames,
        source_video=source_video,
        start_frame=start_frame,
        preserve_source_audio=preserve_audio,
        ffmpeg_location=ffmpeg_location,
    ).open()
    completed = False
    started = time.time()
    try:
        for index in range(chunk_count):
            stored = runner._load(self.path("samples", index))
            if stored is None:
                raise RuntimeError(
                    "pass D: chunk %d has no sampled latent" % index
                )
            assembler.emit(
                index,
                stored["video_latent"],
                stored["audio_latent"],
            )
            del stored
        completed = True
    finally:
        assembler.close(commit=completed)
    frames, output_path = assembler.finalize()
    logging.info(
        "%s pass D complete: %d frames, %d generated audio samples in %.1f s",
        LOG,
        frames,
        assembler.audio_samples,
        time.time() - started,
    )
    return frames, output_path


def install():
    """Install AV methods after the dual-preview patch has loaded."""

    runner.LongFormRun.pass_a = _pass_a_av
    runner.LongFormRun.pass_b = _pass_b_av
    runner.LongFormRun.pass_c = _pass_c_av
    runner.LongFormRun.pass_cd = _pass_cd_av
    runner.LongFormRun.pass_d = _pass_d_av
    runner.run = run


def _read_state(root):
    path = os.path.join(root, "state.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def run(**kwargs):
    """Run the original engine with source-conditioned generated audio enabled."""

    install()
    runtime = dict(kwargs.get("runtime_config") or {})
    runtime.update(
        {
            "source_audio_conditioning": "per_chunk_video_audio_v1",
            "audio_carry_policy": "video_floor_v1",
            "audio_output": "generated_default_source_optional_v1",
        }
    )
    kwargs["runtime_config"] = runtime
    token = _ACTIVE_AUDIO_VAE.set(kwargs["audio_vae"])
    try:
        summary = _ORIGINAL_RUN(**kwargs)
    finally:
        _ACTIVE_AUDIO_VAE.reset(token)

    geometry = HarnessGeometry(
        kwargs["chunk_frames"], kwargs["overlap_frames"]
    ).validate()
    video_overlap_start, video_overlap_count = geometry.overlap_slice()
    video_carry_frames = geometry.overlap_frames
    if kwargs.get("carry") == runner.CARRY_FRAME:
        video_carry_frames = latent_frame_spans(
            geometry.target_latent_t
        )[video_overlap_start]
    audio_start, audio_count = audio_overlap_slice(
        geometry, video_carry_frames
    )
    state = _read_state(kwargs["root"])
    summary.update(
        {
            "source_audio_conditioning": True,
            "source_audio_available": state.get(
                "source_audio_available", False
            ),
            "generated_audio_samples": state.get(
                "generated_audio_samples_written", 0
            ),
            "generated_audio_sample_rate": state.get(
                "generated_audio_sample_rate"
            ),
            "source_audio_track": state.get("source_audio_track", False),
            "audio_tracks": state.get("audio_tracks", 0),
            "audio_carry": {
                **audio_carry_timing(video_carry_frames, geometry.fps),
                "video_latents": (
                    1 if kwargs.get("carry") == runner.CARRY_FRAME
                    else video_overlap_count
                ),
                "audio_start": audio_start,
            },
            "audio_carry_policy": "video_floor_v1",
            "audio_profile": "C=%s O=%s" % (
                geometry.chunk_frames, geometry.overlap_frames
            ),
        }
    )
    return summary


__all__ = [
    "_source_reference_block",
    "install",
    "run",
]
