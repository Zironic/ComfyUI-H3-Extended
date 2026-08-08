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


if __name__ == "__main__":
    unittest.main()
