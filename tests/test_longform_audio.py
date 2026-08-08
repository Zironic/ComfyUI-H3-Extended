"""CPU checks for long-form generated-audio carry and assembly."""

from __future__ import annotations

import unittest

import torch

from chunked_ref2v.geometry import HarnessGeometry, latent_frame_spans
from chunked_ref2v.layout_ops import (
    TargetAlignedCondition,
    insert_target_conditions,
)
from chunked_ref2v.longform.audio_conditions import (
    TargetAlignedAudioCondition,
    insert_target_audio_conditions,
    patch_target_audio_conditions,
)
from chunked_ref2v.longform.audio_output import (
    audio_samples_for_frames,
    decode_audio_chunk,
)
from chunked_ref2v.longform.audio_runtime import (
    audio_latent_boundary,
    audio_carry_timing,
    audio_overlap_slice,
)


class AudioOverlapGeometryTests(unittest.TestCase):
    def test_default_o4_uses_six_audio_latent_positions(self):
        geometry = HarnessGeometry(90, 4).validate()
        self.assertEqual(geometry.audio_latent_t, 150)
        self.assertEqual(audio_overlap_slice(geometry), (144, 6))
        self.assertEqual(audio_latent_boundary(86), 143)
        self.assertEqual(audio_latent_boundary(90), 150)

    def test_single_carried_video_latent_takes_last_six_audio_positions(self):
        geometry = HarnessGeometry(141, 4).validate()
        video_start, video_count = geometry.overlap_slice()
        self.assertEqual(video_count, 1)
        video_frames = latent_frame_spans(geometry.target_latent_t)[video_start]
        self.assertEqual(video_frames, 4)
        audio_start, audio_count = audio_overlap_slice(geometry, video_frames)
        self.assertEqual((audio_start, audio_count), (229, 6))

        carry = torch.arange(geometry.audio_latent_t).reshape(1, 1, 1, -1)
        selected = carry[..., audio_start:audio_start + audio_count]
        self.assertEqual(selected.flatten().tolist(), [229, 230, 231, 232, 233, 234])

    def test_carry_timing_reports_the_unrepresented_video_residual(self):
        timing = audio_carry_timing(4)
        self.assertEqual(timing["video_frames"], 4)
        self.assertEqual(timing["audio_latents"], 6)
        self.assertAlmostEqual(timing["video_ms"], 1000 / 6)
        self.assertEqual(timing["audio_ms"], 150.0)
        self.assertAlmostEqual(timing["residual_ms"], 1000 / 60)

    def test_cumulative_sample_boundaries_do_not_drift(self):
        sample_rate = 32000
        fps = 24
        stride = 86
        boundaries = [
            audio_samples_for_frames(index * stride, sample_rate, fps)
            for index in range(8)
        ]
        increments = [
            boundaries[index + 1] - boundaries[index]
            for index in range(len(boundaries) - 1)
        ]
        self.assertEqual(sum(increments), boundaries[-1])
        self.assertTrue(
            all(value in (114666, 114667) for value in increments)
        )


class AudioLayoutTests(unittest.TestCase):
    def test_patched_payload_receives_the_last_six_audio_positions(self):
        import comfy.conds
        from comfy.ldm.minimax.model import PackedLayout

        geometry = HarnessGeometry(141, 4).validate()
        audio_start, audio_count = audio_overlap_slice(geometry)
        previous = torch.arange(
            1 * 32 * 2 * geometry.audio_latent_t
        ).reshape(1, 32, 2, geometry.audio_latent_t)
        carry = previous[..., audio_start:audio_start + audio_count]
        layout = PackedLayout(
            40,
            geometry.target_latent_t,
            16,
            24,
            geometry.audio_latent_t,
        )

        class FakeModel:
            def clone(self):
                return self

            def get_model_object(self, name):
                self.assert_name = name

                def extra_conds(**kwargs):
                    return {
                        "minimax_payload": comfy.conds.CONDConstant(
                            {"layout": layout, "refs": []}
                        )
                    }

                return extra_conds

            def add_object_patch(self, name, patch):
                self.patch_name = name
                self.patch = patch

        model = patch_target_audio_conditions(
            FakeModel(),
            [TargetAlignedAudioCondition(carry, 0, "audio carry")],
        )
        payload = model.patch()["minimax_payload"].cond

        self.assertEqual(model.assert_name, "extra_conds")
        self.assertEqual(model.patch_name, "extra_conds")
        self.assertTrue(
            torch.equal(
                payload["cond_audio_latents"][0],
                previous[..., -6:],
            )
        )
        self.assertEqual(
            payload["layout"].audio_condition_segments[0][0:2],
            (40, 52),
        )

    def test_audio_condition_composes_with_visual_condition_and_refs(self):
        from comfy.ldm.minimax.model import PackedLayout

        text_len = 40
        latent_t = 27
        latent_h = 16
        latent_w = 24
        audio_t = 150
        refs = [
            {
                "kind": "video_audio",
                "latent_t": 7,
                "latent_h": latent_h,
                "latent_w": latent_w,
                "ref_audio_t": 12,
            }
        ]
        base = PackedLayout(
            text_len,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            refs=refs,
        )
        visual = TargetAlignedCondition(
            latent=torch.zeros(
                1, 24, 1, latent_h, latent_w
            ),
            target_latent_start=0,
            label="visual carry",
        )
        with_visual = insert_target_conditions(base, [visual])
        audio = TargetAlignedAudioCondition(
            latent=torch.arange(1 * 32 * 2 * 6).reshape(1, 32, 2, 6),
            target_latent_start=0,
            label="audio carry",
        )
        out = insert_target_audio_conditions(
            with_visual, [audio]
        )

        self.assertEqual(
            int((~out.audio_update).sum()),
            int((~with_visual.audio_update).sum()) + 12,
        )
        self.assertEqual(
            out.audio_pos.shape[0],
            out.audio_update.shape[0],
        )
        self.assertEqual(
            out.img_pos.shape[0],
            out.img_update.shape[0],
        )
        audio_target = next(
            start
            for start, _, kind in out.segments
            if kind == "audio"
        )
        condition_start, condition_stop, _ = (
            out.audio_condition_segments[0]
        )
        self.assertTrue(
            torch.equal(
                out.position_ids[
                    condition_start:condition_stop
                ],
                out.position_ids[
                    audio_target:audio_target + 12
                ],
            )
        )
        visual_start, visual_stop, _ = (
            out.condition_segments[0]
        )
        video_target = next(
            start
            for start, _, kind in out.segments
            if kind == "video"
        )
        frame_rows = (latent_h // 2) * (latent_w // 2)
        self.assertTrue(
            torch.equal(
                out.position_ids[visual_start:visual_stop],
                out.position_ids[
                    video_target:video_target + frame_rows
                ],
            )
        )

    def test_audio_condition_rejects_past_target(self):
        from comfy.ldm.minimax.model import PackedLayout

        layout = PackedLayout(10, 2, 4, 4, 5)
        condition = TargetAlignedAudioCondition(
            latent=torch.zeros(1, 32, 2, 3),
            target_latent_start=4,
        )
        with self.assertRaises(ValueError):
            insert_target_audio_conditions(
                layout, [condition]
            )


class AudioDecodeTests(unittest.TestCase):
    def test_decode_disables_autograd_and_returns_bcl(self):
        class RecordingVAE:
            grad_enabled = None

            def decode(self, latent):
                self.grad_enabled = torch.is_grad_enabled()
                return torch.zeros(1, 1600, 2)

        vae = RecordingVAE()
        with torch.enable_grad():
            waveform = decode_audio_chunk(
                vae, torch.zeros(1, 32, 2, 7)
            )
        self.assertFalse(vae.grad_enabled)
        self.assertEqual(waveform.shape, (1, 2, 1600))


if __name__ == "__main__":
    unittest.main()
