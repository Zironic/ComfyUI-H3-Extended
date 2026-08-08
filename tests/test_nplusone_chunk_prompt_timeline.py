"""CPU checks for the N+1-aware chunk prompt timeline."""

from __future__ import annotations

import unittest

from chunked_ref2v.longform.nplusone_chunk_prompt_timeline import (
    POLICY_AV_CONTINUATION,
    build_nplusone_chunk_prompt_plan,
    MiniMaxH3NPlusOneChunkPromptTimeline,
    compile_nplusone_chunk_prompts,
    prompts_for_av_continuation_plan,
    validate_nplusone_chunk_prompt_plan,
)


class NPlusOneChunkPromptTimelineTests(unittest.TestCase):
    def test_schedule_uses_full_chunks_without_overlap(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=12,
            chunk_frames=141,
            chunk_prompts_json='{"version":2,"prompts":["a","b","c"]}',
        )
        self.assertEqual(plan["target_frames"], 288)
        self.assertEqual(plan["chunk_count"], 3)
        self.assertEqual(plan["chunk_prompts"], ["a", "b", "c"])
        self.assertIn("reference_frames", plan)
        self.assertEqual(plan["schedule"], "full_chunks")
        self.assertEqual(plan["continuation_policy"], POLICY_AV_CONTINUATION)
        self.assertEqual(plan["reference_frames"], 90)
        self.assertNotIn("overlap_frames", plan)
        self.assertNotIn("stride_frames", plan)

    def test_reference_frames_snap_to_legal_if_not_exact(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=10,
            chunk_frames=141,
            reference_frames=80,
        )
        self.assertEqual(plan["reference_frames"], 90)

    def test_reference_frames_respects_reference_input_range(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=10,
            chunk_frames=141,
            reference_frames=40,
        )
        self.assertEqual(plan["reference_frames"], 90)

    def test_strict_chunk_rejects_illegal_length(self):
        with self.assertRaises(ValueError):
            build_nplusone_chunk_prompt_plan(
                output_seconds=10,
                chunk_frames=124,
                reference_frames=60,
            )

    def test_reference_frames_full_chunk_and_partial_records(self):
        self.assertEqual(
            build_nplusone_chunk_prompt_plan(
                output_seconds=10,
                chunk_frames=141,
                reference_frames=60,
            )["reference_frames"],
            90,
        )
        self.assertEqual(
            build_nplusone_chunk_prompt_plan(
                output_seconds=10,
                chunk_frames=141,
                reference_frames=141,
            )["reference_frames"],
            141,
        )

    def test_global_prompt_is_compiled_with_each_local_instruction(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=6,
            chunk_frames=141,
            global_prompt="same character and bedroom",
            chunk_prompts_json='{"prompts":["remove glove","remove other glove"]}',
        )
        self.assertEqual(
            compile_nplusone_chunk_prompts(plan, "fallback"),
            [
                "same character and bedroom\n\nremove glove",
                "same character and bedroom\n\nremove other glove",
            ],
        )

    def test_blank_local_uses_global_then_fallback(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=1,
            chunk_frames=141,
            global_prompt="persistent",
            chunk_prompts_json='{"prompts":[""]}',
        )
        self.assertEqual(compile_nplusone_chunk_prompts(plan, "fallback"), ["persistent"])
        plan["global_prompt"] = ""
        self.assertEqual(compile_nplusone_chunk_prompts(plan, "fallback"), ["fallback"])

    def test_validator_uses_geometry_from_the_plan(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=12,
            chunk_frames=141,
        )
        normalized = validate_nplusone_chunk_prompt_plan(plan)
        self.assertEqual(normalized["output_seconds"], 12)
        self.assertEqual(normalized["chunk_frames"], 141)

    def test_invalid_internal_reference_geometry_is_rejected(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=12,
            chunk_frames=141,
            reference_frames=90,
        )
        plan["reference_frames"] = 81
        with self.assertRaises(ValueError):
            validate_nplusone_chunk_prompt_plan(plan)

    def test_prompts_function_prefers_plan_geometry(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=12,
            chunk_frames=141,
            global_prompt="plan",
            reference_frames=90,
        )
        prompts = prompts_for_av_continuation_plan(
            plan,
            "fallback",
            output_seconds=1,
            chunk_frames=90,
            reference_frames=39,
        )
        self.assertEqual(prompts, ["plan", "plan", "plan"])

    def test_av_consumer_receives_compiled_prompts(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=6,
            chunk_frames=141,
            global_prompt="global",
            chunk_prompts_json='{"prompts":["one","two"]}',
        )
        self.assertEqual(
            prompts_for_av_continuation_plan(
                plan,
                "fallback",
                output_seconds=6,
                chunk_frames=141,
            ),
            ["global\n\none", "global\n\ntwo"],
        )

    def test_none_plan_repeats_fallback_for_required_chunks(self):
        prompts = prompts_for_av_continuation_plan(
            None,
            "fallback",
            output_seconds=12,
            chunk_frames=141,
        )
        self.assertEqual(prompts, ["fallback", "fallback", "fallback"])

    def test_schema_orders_reference_widget_after_chunk_prompts_json(self):
        # Order is pinned because widget positions are stored positionally in
        # saved workflows. New widgets append; they never insert.
        schema = MiniMaxH3NPlusOneChunkPromptTimeline.define_schema()
        names = [getattr(item, "id", getattr(item, "name", None)) for item in schema.inputs]
        self.assertEqual(
            names,
            [
                "output_seconds",
                "chunk_frames",
                "global_prompt",
                "chunk_prompts_json",
                "reference_frames",
                "seed",
            ],
        )

    def test_schema_outputs_append_reference_frames(self):
        schema = MiniMaxH3NPlusOneChunkPromptTimeline.define_schema()
        names = [getattr(item, "id", getattr(item, "name", None)) for item in schema.outputs]
        self.assertEqual(
            names,
            [
                "n_plus_one_prompt_plan",
                "output_seconds",
                "chunk_frames",
                "report",
                "reference_frames",
                "seed",
            ],
        )


if __name__ == "__main__":
    unittest.main()
