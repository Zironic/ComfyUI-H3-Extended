import importlib.util
import json
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "bench_te_fp8_mlp.py"
SPEC = importlib.util.spec_from_file_location("bench_te_fp8_mlp", MODULE_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class BenchmarkTests(unittest.TestCase):
    def test_parser_help_and_defaults_are_cpu_safe(self):
        parser = BENCH.build_parser()
        self.assertIn("i-understand-this-uses-gpu", parser.format_help())
        args = parser.parse_args([])
        self.assertEqual(args.rows, BENCH.DEFAULT_ROWS)
        self.assertEqual(args.recipes, BENCH.DEFAULT_RECIPES)

    def test_parse_and_validate_dimensions(self):
        self.assertEqual(BENCH.parse_rows("2048, 8192"), (2048, 8192))
        self.assertEqual(BENCH.parse_recipes("delayed_e4m3,current_hybrid"), ("delayed_e4m3", "current_hybrid"))
        with self.assertRaises(ValueError):
            BENCH.validate_dimensions(32, 12, 16, (2,))
        with self.assertRaises(ValueError):
            BENCH.validate_dimensions(30, 15, 16, (2,))

    def test_numerical_metrics_and_serialization(self):
        reference = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        actual = torch.tensor([[2.0, 2.0]], dtype=torch.bfloat16)
        self.assertAlmostEqual(BENCH.max_absolute_error(actual, reference), 1.0)
        self.assertAlmostEqual(BENCH.relative_l2(actual, reference), 1.0 / (5.0 ** 0.5))
        encoded = BENCH.serialize_result({"shape": (2, 3), "value": 1})
        self.assertEqual(json.loads(encoded), {"shape": [2, 3], "value": 1})

        self.assertEqual(
            BENCH.expected_swiglu_quantization_path("delayed_e4m3"),
            "direct_fp8_output",
        )
        self.assertEqual(
            BENCH.expected_swiglu_quantization_path("current_e4m3"),
            "high_precision_temporary_then_fp8",
        )

    def test_carrier_inspection_uses_fake_te_quantized_tensor(self):
        class QuantizedTensor(torch.Tensor):
            pass

        backing = torch.empty((2, 4), dtype=torch.uint8)
        carrier = torch.Tensor._make_subclass(QuantizedTensor, backing, require_grad=False)
        carrier._data = backing
        carrier.logical_shape = (2, 4)
        carrier.logical_dtype = torch.bfloat16
        details = BENCH.inspect_carrier(carrier, QuantizedTensor)
        self.assertEqual(details["logical_shape"], (2, 4))
        self.assertEqual(details["backing_data_bytes"], 8)

    def test_carrier_inspection_rejects_non_te_tensor(self):
        with self.assertRaises(RuntimeError):
            BENCH.inspect_carrier(torch.empty((2, 4)), type("Other", (), {}))

    def test_checkpoint_loader_requires_convrot_int8(self):
        config = json.dumps({
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 256,
        }).encode("utf-8")
        tensors = {
            "blocks.3.mlp.fc2.weight": torch.empty((8, 16), dtype=torch.int8),
            "blocks.3.mlp.fc2.weight_scale": torch.ones((8, 1), dtype=torch.float32),
            "blocks.3.mlp.fc2.comfy_quant": torch.tensor(list(config), dtype=torch.uint8),
        }

        class Handle:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def keys(self):
                return tensors.keys()

            def get_tensor(self, key):
                return tensors[key]

        loaded = BENCH.load_convrot_fc2(
            "unused.safetensors",
            block_index=3,
            safe_open_fn=lambda *args, **kwargs: Handle(),
        )
        self.assertEqual(loaded["group_size"], 256)
        self.assertEqual(tuple(loaded["weight"].shape), (8, 16))

if __name__ == "__main__":
    unittest.main()
