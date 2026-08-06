"""CPU checks for the H3 chunk-aware prompt timeline."""

from __future__ import annotations

import unittest

from chunked_ref2v.longform.chunk_prompt_timeline import (
    build_chunk_prompt_plan,
    compile_chunk_prompts,
    pack_chunk_prompt_plan,
    unpack_chunk_prompt_envelope,
    validate_chunk_prompt_plan,
)


class ChunkPromptTimelineTests(unittest.TestCase):
    def test_plan_matches_longform_chunk_geometry(self):
        plan = build_chunk_prompt_plan(
            output_seconds=10,
            chunk_frames=90,
            overlap_frames=4,
            chunk_prompts_json='{"version":1,"prompts":["a","b","c"]}',
        )
        self.assertEqual(plan["target_frames"], 240)
        self.assertEqual(plan["stride_frames"], 86)
        self.assertEqual(plan["chunk_count"], 3)
        self.assertEqual(plan["chunk_prompts"], ["a", "b", "c"])

    def test_global_prompt_is_prepended_to_each_local_prompt(self):
        plan = build_chunk_prompt_plan(
            output_seconds=5,
            chunk_frames=90,
            overlap_frames=4,
            global_prompt="persistent identity",
            chunk_prompts_json='{"prompts":["remove gloves","remove top"]}',
        )
        self.assertEqual(
            compile_chunk_prompts(plan, "fallback"),
            [
                "persistent identity\n\nremove gloves",
                "persistent identity\n\nremove top",
            ],
        )

    def test_blank_chunk_uses_global_or_fallback_prompt(self):
        with_global = build_chunk_prompt_plan(
            output_seconds=1,
            chunk_frames=90,
            overlap_frames=4,
            global_prompt="persistent",
            chunk_prompts_json='{"prompts":[""]}',
        )
        self.assertEqual(compile_chunk_prompts(with_global, "fallback"), ["persistent"])

        without_global = dict(with_global)
        without_global["global_prompt"] = ""
        self.assertEqual(compile_chunk_prompts(without_global, "fallback"), ["fallback"])

    def test_envelope_round_trip_preserves_per_chunk_prompts(self):
        plan = build_chunk_prompt_plan(
            output_seconds=10,
            chunk_frames=90,
            overlap_frames=4,
            global_prompt="global",
            chunk_prompts_json='{"prompts":["one","two","three"]}',
        )
        packed = pack_chunk_prompt_plan(
            "fallback",
            plan,
            output_seconds=10,
            chunk_frames=90,
            overlap_frames=4,
        )
        unpacked = unpack_chunk_prompt_envelope(packed)
        self.assertEqual(
            unpacked["prompts"],
            ["global\n\none", "global\n\ntwo", "global\n\nthree"],
        )
        self.assertEqual(unpacked["plan"]["chunk_count"], 3)

    def test_geometry_mismatch_is_rejected(self):
        plan = build_chunk_prompt_plan(
            output_seconds=10,
            chunk_frames=90,
            overlap_frames=4,
        )
        with self.assertRaises(ValueError):
            validate_chunk_prompt_plan(
                plan,
                output_seconds=20,
                chunk_frames=90,
                overlap_frames=4,
            )


if __name__ == "__main__":
    unittest.main()
