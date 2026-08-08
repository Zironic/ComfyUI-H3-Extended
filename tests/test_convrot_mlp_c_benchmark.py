import importlib.util
import json
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "bench_convrot_mlp_c.py"
SPEC = importlib.util.spec_from_file_location("bench_convrot_mlp_c", MODULE_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def fake_handle(tensors):
    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return tensors.keys()

        def get_tensor(self, key):
            return tensors[key]

    return Handle


class ConvRotMlpBenchmarkTests(unittest.TestCase):
    def test_parser_help_and_defaults_are_cpu_safe(self):
        args = BENCH.build_parser().parse_args([])
        self.assertEqual(args.rows, BENCH.DEFAULT_ROWS)
        self.assertEqual(args.feature_tiles, (7168, 3584))
        self.assertFalse(args.i_understand_this_uses_gpu)

    def test_loader_requires_a_complete_swiglu_pair(self):
        config = json.dumps({
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 4,
        }).encode("utf-8")
        tensors = {
            "model.diffusion_model.blocks.2.mlp.fc1.weight": torch.empty((8, 4), dtype=torch.int8),
            "model.diffusion_model.blocks.2.mlp.fc1.weight_scale": torch.ones((8, 1)),
            "model.diffusion_model.blocks.2.mlp.fc1.comfy_quant": torch.tensor(list(config), dtype=torch.uint8),
            "model.diffusion_model.blocks.2.mlp.fc2.weight": torch.empty((4, 4), dtype=torch.int8),
            "model.diffusion_model.blocks.2.mlp.fc2.weight_scale": torch.ones((4, 1)),
            "model.diffusion_model.blocks.2.mlp.fc2.comfy_quant": torch.tensor(list(config), dtype=torch.uint8),
        }

        loaded = BENCH.load_convrot_mlp(
            "unused.safetensors", block_index=2,
            safe_open_fn=lambda *args, **kwargs: fake_handle(tensors)(),
        )
        self.assertEqual(loaded["expanded_width"], 8)
        self.assertEqual(loaded["ffn_width"], 4)
        self.assertEqual(loaded["hidden_width"], 4)

    def test_loader_rejects_non_convrot_layout(self):
        config = json.dumps({"format": "int8_tensorwise", "convrot": False}).encode("utf-8")
        tensors = {
            "blocks.0.mlp.fc1.weight": torch.empty((8, 4), dtype=torch.int8),
            "blocks.0.mlp.fc1.weight_scale": torch.ones((8, 1)),
            "blocks.0.mlp.fc1.comfy_quant": torch.tensor(list(config), dtype=torch.uint8),
            "blocks.0.mlp.fc2.weight": torch.empty((4, 4), dtype=torch.int8),
            "blocks.0.mlp.fc2.weight_scale": torch.ones((4, 1)),
            "blocks.0.mlp.fc2.comfy_quant": torch.tensor(list(config), dtype=torch.uint8),
        }

        with self.assertRaises(ValueError):
            BENCH.load_convrot_mlp(
                "unused.safetensors",
                safe_open_fn=lambda *args, **kwargs: fake_handle(tensors)(),
            )

    def test_feature_ranges_are_convrot_aligned(self):
        self.assertEqual(
            BENCH.feature_ranges(14336, 3584),
            ((0, 3584), (3584, 7168), (7168, 10752), (10752, 14336)),
        )
        self.assertEqual(
            BENCH.feature_ranges(14336, 4096)[-1],
            (12288, 14336),
        )
        with self.assertRaises(ValueError):
            BENCH.feature_ranges(14336, 1000)


if __name__ == "__main__":
    unittest.main()
