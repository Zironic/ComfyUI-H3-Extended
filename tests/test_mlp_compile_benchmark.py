"""CPU-safe contract checks for the H3 MLP compile benchmark."""

import os
import sys
from pathlib import Path

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from benchmarks import bench_mlp_compile as bench  # noqa: E402


def test_default_case_plan_has_full_and_tail_shape():
    shapes = bench.plan_slab_shapes()
    assert len(shapes) == 31
    assert shapes[:-1] == (bench.DEFAULT_CHUNK_ROWS,) * 30
    assert shapes[-1] == 2008


def test_compile_kwargs_are_static_fullgraph():
    assert bench.compile_kwargs() == {"fullgraph": True, "dynamic": False}


def test_custom_op_fake_contract_without_cuda():
    class FakeKitchen:
        def int8_linear(self, x, weight, scale, **kwargs):
            del scale, kwargs
            return torch.empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)

    op = bench.make_convrot_adapter(
        FakeKitchen(), torch.empty((8, 4), dtype=torch.int8), torch.ones(8), 256, 8, label="test_fake"
    )
    result = op(torch.empty((3, 4)), torch.empty((8, 4), dtype=torch.int8), torch.ones(8))
    assert tuple(result.shape) == (3, 8)
    assert result.dtype == torch.bfloat16


def test_measurement_warmup_is_excluded():
    calls = []

    def fn():
        calls.append(len(calls))
        return torch.empty((1, 1))

    result = bench.measure_path(fn, torch.device("cpu"), warmup=1, iterations=2)
    assert len(calls) == 3
    assert len(result["samples_ms"]) == 2


def test_parser_requires_explicit_checkpoint_and_ack():
    parser = bench.build_parser()
    args = parser.parse_args(["--checkpoint", "weights.safetensors", "--i-understand-this-uses-gpu"])
    assert args.seq == 63448
    assert args.chunk_rows == 2048
    assert args.feature_tile_width == 7168
    assert args.warmup == 1
    assert args.iterations == 3


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_"):
            value()
    print("MLP compile benchmark tests passed")
