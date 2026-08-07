"""CPU checks for generated opening-picture long-form continuity."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chunked_ref2v.longform.opening_picture_runtime import (
    DynamicOpeningPictureConditionings,
    _OpeningPictureBlocks,
    _picture_number,
    opening_picture_prompt,
)


class OpeningPictureRuntimeTests(unittest.TestCase):
    def test_prompt_aligns_generated_picture_to_frame_zero(self):
        prompt = opening_picture_prompt("continue dancing", 3)
        self.assertTrue(
            prompt.startswith(
                "For the target video, at 0.00 seconds into the target video, "
                "<Picture 3> is fully referenced."
            )
        )
        self.assertIn("<Picture 3> is the opening frame", prompt)
        self.assertTrue(prompt.endswith("continue dancing"))

    def test_picture_number_counts_only_image_items(self):
        items = [
            {"type": "image"},
            {"type": "audio"},
            {"type": "video"},
            {"type": "image"},
        ]
        self.assertEqual(_picture_number(items), 3)
        self.assertEqual(_picture_number([]), 1)

    def test_picture_block_is_appended_after_existing_dynamic_blocks(self):
        class BaseBlocks:
            def __init__(self):
                self.calls = 0

            def __iter__(self):
                self.calls += 1
                return iter([{"kind": "image", "name": "static"}])

            def __len__(self):
                return 1

        class Conditionings:
            current_picture_block = {"kind": "image", "name": "generated"}

        base = BaseBlocks()
        blocks = _OpeningPictureBlocks(base, Conditionings())
        self.assertEqual(
            list(blocks),
            [
                {"kind": "image", "name": "static"},
                {"kind": "image", "name": "generated"},
            ],
        )
        self.assertEqual(base.calls, 1)

    def test_first_chunk_has_no_generated_picture_block(self):
        class BaseBlocks:
            def __iter__(self):
                return iter([{"kind": "audio"}])

            def __len__(self):
                return 1

        class Conditionings:
            current_picture_block = None

        self.assertEqual(
            list(_OpeningPictureBlocks(BaseBlocks(), Conditionings())),
            [{"kind": "audio"}],
        )


if __name__ == "__main__":
    unittest.main()
