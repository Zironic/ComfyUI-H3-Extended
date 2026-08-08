"""Plan v3: the run seed and the derived per-chunk seeds.

These are the fields the resume scan keys on, so they are tested without torch
or Comfy - the geometry module is deliberately import-light for this reason.
"""

import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    """Import the planner and geometry without pulling in the Comfy runtime."""
    pkg = types.ModuleType("h3pkg")
    pkg.__path__ = [os.path.join(ROOT, "chunked_ref2v")]
    sys.modules.setdefault("h3pkg", pkg)
    sub = types.ModuleType("h3pkg.longform")
    sub.__path__ = [os.path.join(ROOT, "chunked_ref2v", "longform")]
    sys.modules.setdefault("h3pkg.longform", sub)
    import importlib

    geometry = importlib.import_module("h3pkg.geometry")
    try:
        planner = importlib.import_module("h3pkg.longform.nplusone_chunk_prompt_timeline")
    except ImportError as exc:  # comfy_api is unavailable outside the server
        raise unittest.SkipTest("comfy_api not importable: %s" % exc)
    return geometry, planner


class ChunkSeedTest(unittest.TestCase):
    def setUp(self):
        self.geometry, self.planner = _load()

    def test_matches_the_runners_derivation(self):
        # The literal the runners use: splitmix64(seed + 1000 + index).
        for seed in (0, 1, 413146064173853):
            for index in (0, 1, 9):
                self.assertEqual(
                    self.geometry.chunk_seed(seed, index),
                    self.geometry.splitmix64(seed + 1000 + index),
                )

    def test_distinct_per_chunk_and_per_run(self):
        a = [self.geometry.chunk_seed(7, i) for i in range(10)]
        b = [self.geometry.chunk_seed(8, i) for i in range(10)]
        self.assertEqual(len(set(a)), 10, "chunks must not share a seed")
        self.assertTrue(all(x != y for x, y in zip(a, b)),
                        "a different run seed must reroll every chunk")

    def test_stays_in_64_bit_range(self):
        value = self.geometry.chunk_seed(0xFFFFFFFFFFFFFFFF, 5)
        self.assertGreaterEqual(value, 0)
        self.assertLess(value, 1 << 64)


class PlanSeedTest(unittest.TestCase):
    def setUp(self):
        self.geometry, self.planner = _load()

    def build(self, **kwargs):
        params = dict(output_seconds=58, chunk_frames=141, reference_frames=90,
                      seed=413146064173853)
        params.update(kwargs)
        return self.planner.build_nplusone_chunk_prompt_plan(**params)

    def test_plan_carries_seed_and_one_seed_per_chunk(self):
        plan = self.build()
        self.assertEqual(plan["version"], 3)
        self.assertEqual(plan["seed"], 413146064173853)
        self.assertEqual(len(plan["chunk_seeds"]), plan["chunk_count"])
        self.assertEqual(
            plan["chunk_seeds"],
            [self.geometry.chunk_seed(plan["seed"], i)
             for i in range(plan["chunk_count"])],
        )

    def test_ten_chunks_for_the_58_second_reference_run(self):
        plan = self.build()
        self.assertEqual(plan["chunk_count"], 10)
        self.assertEqual(plan["target_frames"], 1392)

    def test_validator_accepts_a_freshly_built_plan(self):
        plan = self.build()
        normalized = self.planner.validate_nplusone_chunk_prompt_plan(plan)
        self.assertEqual(normalized["seed"], plan["seed"])
        self.assertEqual(normalized["chunk_seeds"], plan["chunk_seeds"])

    def test_validator_rejects_tampered_chunk_seeds(self):
        plan = self.build()
        plan["chunk_seeds"] = list(plan["chunk_seeds"])
        plan["chunk_seeds"][3] += 1
        with self.assertRaises(ValueError):
            self.planner.validate_nplusone_chunk_prompt_plan(plan)

    def test_validator_rejects_wrong_length_seed_list(self):
        plan = self.build()
        plan["chunk_seeds"] = plan["chunk_seeds"][:-1]
        with self.assertRaises(ValueError):
            self.planner.validate_nplusone_chunk_prompt_plan(plan)

    def test_plan_carries_one_compiled_prompt_digest_per_chunk(self):
        plan = self.build(global_prompt="global", chunk_prompts_json='{"prompts":["a","b"]}')
        compiled = self.planner.compile_nplusone_chunk_prompts(plan)
        self.assertEqual(
            plan["chunk_digests"],
            [self.planner.prompt_digest(text) for text in compiled],
        )

    def test_validator_rejects_tampered_chunk_digest(self):
        plan = self.build()
        plan["chunk_digests"] = list(plan["chunk_digests"])
        plan["chunk_digests"][0] = "0" * 64
        with self.assertRaises(ValueError):
            self.planner.validate_nplusone_chunk_prompt_plan(plan)

    def test_editing_a_chunk_prompt_leaves_every_seed_alone(self):
        """The whole point: a prompt edit must not reroll anything."""
        import json

        before = self.build(chunk_prompts_json=json.dumps(
            {"version": 2, "prompts": ["a", "b", "c"]}))
        after = self.build(chunk_prompts_json=json.dumps(
            {"version": 2, "prompts": ["a", "EDITED", "c"]}))
        self.assertEqual(before["chunk_seeds"], after["chunk_seeds"])
        self.assertNotEqual(before["chunk_prompts"], after["chunk_prompts"])


if __name__ == "__main__":
    unittest.main()
