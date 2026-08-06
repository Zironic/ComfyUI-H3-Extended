import unittest

from chunked_ref2v.longform import runner
from chunked_ref2v.longform.dual_prompt_audio_compat import _select_conditioning
from chunked_ref2v.longform.dual_prompt_runtime import (
    pack_prompts,
    prompt_for_index,
    unpack_prompts,
)


class DualPromptEnvelopeTests(unittest.TestCase):
    def test_round_trip_preserves_both_prompts(self):
        packed = pack_prompts("begin here", "continue from visible state")
        self.assertEqual(
            unpack_prompts(packed),
            ("begin here", "continue from visible state"),
        )

    def test_blank_continuation_falls_back_to_initial(self):
        packed = pack_prompts("same prompt", "   ")
        self.assertEqual(unpack_prompts(packed), ("same prompt", "same prompt"))

    def test_legacy_prompt_is_accepted(self):
        self.assertEqual(unpack_prompts("legacy"), ("legacy", "legacy"))


class DualPromptSelectionTests(unittest.TestCase):
    def setUp(self):
        self.packed = pack_prompts("initial", "continuation")

    def test_first_invocation_uses_initial(self):
        self.assertEqual(
            prompt_for_index(self.packed, 0, runner.CARRY_OVERLAP),
            "initial",
        )

    def test_later_carried_invocation_uses_continuation(self):
        self.assertEqual(
            prompt_for_index(self.packed, 1, runner.CARRY_OVERLAP),
            "continuation",
        )

    def test_no_carry_never_uses_continuation(self):
        self.assertEqual(
            prompt_for_index(self.packed, 3, runner.CARRY_NONE),
            "initial",
        )

    def test_av_conditioning_selects_matching_role(self):
        conditioning = {"initial": object(), "continuation": object()}
        first, first_role = _select_conditioning(
            conditioning, 0, runner.CARRY_OVERLAP
        )
        later, later_role = _select_conditioning(
            conditioning, 1, runner.CARRY_OVERLAP
        )
        self.assertIs(first, conditioning["initial"])
        self.assertEqual(first_role, "initial")
        self.assertIs(later, conditioning["continuation"])
        self.assertEqual(later_role, "continuation")


if __name__ == "__main__":
    unittest.main()
