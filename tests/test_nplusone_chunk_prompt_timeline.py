"""CPU checks for the N+1-aware chunk prompt timeline."""

from __future__ import annotations

import unittest

from chunked_ref2v.longform.nplusone_chunk_prompt_timeline import (
    POLICY_AV_CONTINUATION,
    build_nplusone_chunk_prompt_plan,
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
        self.assertEqual(plan["schedule"], "full_chunks")
        self.assertEqual(plan["continuation_policy"], POLICY_AV_CONTINUATION)
        self.assertNotIn("overlap_frames", plan)
        self.assertNotIn("stride_frames", plan)

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

    def test_downstream_geometry_mismatch_is_rejected(self):
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=12,
            chunk_frames=141,
        )
        with self.assertRaises(ValueError):
            validate_nplusone_chunk_prompt_plan(
                plan,
                output_seconds=12,
                chunk_frames=90,
            )

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


if __name__ == "__main__":
    unittest.main()
