"""CPU checks for native long-form audiovisual continuation."""

from __future__ import annotations

import sys
import unittest

import comfy.options
comfy.options.enable_args_parsing()
_ARGV, sys.argv = list(sys.argv), [sys.argv[0], "--cpu"]

import torch
from comfy.ldm.minimax.model import PackedLayout, _video_t_spans

from chunked_ref2v.longform.av_continuation_nodes import (
    MiniMaxH3LongFormAVContinuation,
    _chunk_count,
    _dynamic_av_reference,
    _resolve_execution_plan,
    _run_identity,
    _slice_dynamic_av_reference,
    _resolve_video_reference_frames,
    continuation_prompt,
)
from chunked_ref2v.longform.nplusone_chunk_prompt_timeline import (
    build_nplusone_chunk_prompt_plan,
    prompts_for_av_continuation_plan,
)
from chunked_ref2v.geometry import HarnessGeometry


class LongFormAVContinuationTests(unittest.TestCase):
    def test_chunk_count_has_no_overlap_stride(self):
        self.assertEqual(_chunk_count(240, 141), 2)
        self.assertEqual(_chunk_count(282, 141), 2)
        self.assertEqual(_chunk_count(283, 141), 3)

    def test_continuation_prompt_uses_independent_video_audio_numbers(self):
        text = continuation_prompt("continue dancing", 3, 5)
        self.assertIn("[video continuation + audio reference]", text)
        self.assertIn("<Video 3>", text)
        self.assertIn("<Audio 5>", text)
        self.assertIn("begins immediately after its end", text)
        self.assertIn("longer history than the video tail", text)
        self.assertTrue(text.endswith("continue dancing"))

    def test_dynamic_reference_reuses_generated_latents_directly(self):
        pixels = torch.zeros(141, 64, 96, 3)
        video = torch.zeros(1, 24, 42, 4, 6)
        audio = torch.zeros(1, 32, 2, 235)
        items, block = _dynamic_av_reference(
            pixels, video, audio, (96, 64)
        )

        self.assertEqual([item["type"] for item in items], ["audio", "video"])
        self.assertIs(block["latent"], video)
        self.assertIs(block["audio_latent"], audio)
        self.assertEqual(block["kind"], "video_audio")
        self.assertEqual(block["latent_t"], 42)
        self.assertEqual(block["latent_h"], 4)
        self.assertEqual(block["latent_w"], 6)
        self.assertEqual(block["ref_audio_t"], 235)
        self.assertEqual(block["temporal_alignment"], "end")

    def test_dynamic_reference_tails_have_independent_lengths(self):
        geometry = HarnessGeometry(chunk_frames=141, overlap_frames=4).validate()
        pixels = torch.arange(141 * 2 * 2 * 3, dtype=torch.float32).reshape(141, 2, 2, 3)
        video = torch.arange(1 * 24 * 42 * 2 * 2, dtype=torch.float32).reshape(1, 24, 42, 2, 2)
        audio = torch.arange(1 * 32 * 2 * 235, dtype=torch.float32).reshape(1, 32, 2, 235)
        sliced_pixels, sliced_video, sliced_audio = _slice_dynamic_av_reference(
            pixels,
            video,
            audio,
            video_reference_frames=22,
            audio_reference_latents=160,
            geometry=geometry,
        )
        self.assertEqual(sliced_pixels.shape[0], 22)
        self.assertEqual(sliced_video.shape[2], 7)
        self.assertEqual(sliced_audio.shape[-1], 160)
        self.assertTrue(torch.equal(sliced_pixels, pixels[-22:]))
        self.assertTrue(torch.equal(sliced_audio, audio[..., -160:]))

    def test_packed_layout_static_start_and_dynamic_end_alignment(self):
        base = {
            "kind": "video_audio",
            "latent_t": 2,
            "latent_h": 4,
            "latent_w": 6,
            "ref_audio_t": 160,
        }
        start = PackedLayout(3, 42, 4, 6, 235, refs=[base])
        end = PackedLayout(3, 42, 4, 6, 235, refs=[dict(base, temporal_alignment="end")])
        start_audio, start_audio_stop, _ = next(s for s in start.segments if s[2] == "ref_audio")
        start_video, start_video_stop, _ = next(s for s in start.segments if s[2] == "ref_img")
        end_audio, end_audio_stop, _ = next(s for s in end.segments if s[2] == "ref_audio")
        end_video, end_video_stop, _ = next(s for s in end.segments if s[2] == "ref_img")
        self.assertEqual(
            float(start.position_ids[start_audio, 0]),
            float(start.position_ids[start_video, 0]),
        )
        audio_end = float(end.position_ids[end_audio_stop - 1, 0]) + 1.0
        video_end = float(end.position_ids[end_video_stop - 1, 0]) + _video_t_spans(2)[-1]
        self.assertAlmostEqual(audio_end, video_end)

    def test_reference_frames_are_snapped_to_legal_values(self):
        self.assertEqual(_resolve_video_reference_frames(141, 80), 73)

    def test_reference_frames_respect_reference_input_range(self):
        self.assertEqual(_resolve_video_reference_frames(141, 40), 39)

    def test_reference_frames_can_snap_to_whole_chunk(self):
        self.assertEqual(_resolve_video_reference_frames(141, 141), 141)

    def test_reference_frames_reject_illegal_chunk(self):
        with self.assertRaises(ValueError):
            _resolve_video_reference_frames(120, 60)

    def test_reference_frames_cannot_exceed_previous_chunk(self):
        geometry = HarnessGeometry(chunk_frames=141, overlap_frames=81).validate()
        pixels = torch.zeros(141, 64, 96, 3)
        video = torch.zeros(1, 24, 42, 4, 6)
        audio = torch.zeros(1, 32, 2, 235)
        with self.assertRaises(ValueError):
            _slice_dynamic_av_reference(
                pixels,
                video,
                audio,
                video_reference_frames=200,
                audio_reference_latents=160,
                geometry=geometry,
            )

    def test_audio_reference_cannot_exceed_previous_chunk(self):
        geometry = HarnessGeometry(chunk_frames=141, overlap_frames=81).validate()
        with self.assertRaisesRegex(ValueError, "exceeds previous audio length"):
            _slice_dynamic_av_reference(
                torch.zeros(141, 64, 96, 3),
                torch.zeros(1, 24, 42, 4, 6),
                torch.zeros(1, 32, 2, 100),
                video_reference_frames=22,
                audio_reference_latents=160,
                geometry=geometry,
            )

    def test_dynamic_reference_requires_complete_av_source(self):
        pixels = torch.zeros(141, 64, 96, 3)
        video = torch.zeros(1, 24, 42, 4, 6)
        with self.assertRaises(ValueError):
            _dynamic_av_reference(pixels, video, None, (96, 64))

    def test_node_exposes_nplusone_plan_input(self):
        schema = MiniMaxH3LongFormAVContinuation.define_schema()
        names = [getattr(item, "id", getattr(item, "name", None)) for item in schema.inputs]
        self.assertIn("n_plus_one_prompt_plan", names)

    def test_plan_resolves_different_base_prompt_for_each_chunk(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=12,
            chunk_frames=141,
            global_prompt="same subject",
            chunk_prompts_json='{"prompts":["first action","second action","third action"]}',
        )
        prompts = prompts_for_av_continuation_plan(
            plan,
            "fallback",
            output_seconds=12,
            chunk_frames=141,
        )
        self.assertEqual(
            prompts,
            [
                "same subject\n\nfirst action",
                "same subject\n\nsecond action",
                "same subject\n\nthird action",
            ],
        )
        for prompt in prompts:
            self.assertNotIn("<Video", prompt)
            self.assertNotIn("<Audio", prompt)
            self.assertNotIn("video continuation", prompt)

    def test_connected_plan_owns_geometry_and_seed(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=12,
            chunk_frames=141,
            seed=123,
        )
        effective, prompts, source, overrides = _resolve_execution_plan(
            plan,
            "fallback",
            output_seconds=1,
            chunk_frames=90,
            video_reference_frames=39,
            seed=999,
        )
        self.assertEqual(effective["output_seconds"], 12)
        self.assertEqual(effective["chunk_frames"], 141)
        self.assertEqual(effective["video_reference_frames"], 90)
        self.assertEqual(effective["seed"], 123)
        self.assertEqual(source, "N+1 prompt plan")
        self.assertEqual(
            overrides,
            ["output_seconds", "chunk_frames", "video_reference_frames", "seed"],
        )
        self.assertEqual(prompts, ["fallback", "fallback", "fallback"])

    def test_plan_fallback_prompt_updates_effective_digest(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=1,
            chunk_frames=141,
        )
        effective, prompts, _, _ = _resolve_execution_plan(
            plan,
            "fallback",
            output_seconds=1,
            chunk_frames=141,
            video_reference_frames=90,
            seed=0,
        )
        self.assertEqual(prompts, ["fallback"])
        self.assertNotEqual(effective["chunk_digests"], plan["chunk_digests"])

    def test_run_manifest_allows_prompt_and_seed_edits(self):
        class SamplingA:
            def __init__(self, shift=12.0, audio_shift=3.0):
                self.shift = shift
                self.audio_shift = audio_shift

        class SamplingB(SamplingA):
            pass

        class Model:
            def __init__(self, sampling):
                self.sampling = sampling
                self.model_options = {"transformer_options": {
                    "minimax_h3_sigma_shift_video": sampling.shift,
                    "minimax_h3_sigma_shift_audio": sampling.audio_shift,
                }}

            def get_model_object(self, name):
                if name != "model_sampling":
                    raise KeyError(name)
                return self.sampling

        before = build_nplusone_chunk_prompt_plan(
            output_seconds=12,
            chunk_frames=141,
            global_prompt="before",
            seed=1,
        )
        after = build_nplusone_chunk_prompt_plan(
            output_seconds=12,
            chunk_frames=141,
            global_prompt="after",
            seed=2,
        )
        component = object()
        kwargs = dict(
            canvas=(96, 64),
            model=Model(SamplingA()),
            clip=component,
            video_vae=component,
            audio_vae=component,
            sampler=component,
            sigmas=torch.tensor([1.0, 0.0]),
            ref_images=None,
            ref_videos=None,
            ref_video_audios=None,
            ref_audios=None,
            ref_image_size="native",
            cond_cache="auto",
            attention="auto",
            activation="mlp_chunked_native",
        )
        baseline = _run_identity(before, **kwargs)
        self.assertEqual(baseline, _run_identity(after, **kwargs))
        for sampling in (SamplingA(10.0, 3.0), SamplingA(12.0, 2.0), SamplingB()):
            changed = dict(kwargs, model=Model(sampling))
            self.assertNotEqual(baseline, _run_identity(before, **changed))


if __name__ == "__main__":
    sys.argv = _ARGV
    unittest.main()
