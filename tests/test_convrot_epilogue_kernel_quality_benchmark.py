"""CPU contracts for the ConvRot epilogue kernel-quality benchmark."""

import os
import sys

import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

_ORIGINAL_ARGV = list(sys.argv)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
from benchmarks import bench_convrot_epilogue_kernel_quality as bench  # noqa: E402
sys.argv = _ORIGINAL_ARGV


def test_rows_require_positive_integers():
    assert bench.parse_rows("2048,4096,8192") == (2048, 4096, 8192)
    for value in ("", "0,4096", "-1"):
        try:
            bench.parse_rows(value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid rows were accepted: %r" % value)


def test_exact_h3_temporary_contract_at_production_chunk():
    details = bench.temporary_contract(4096)
    assert details["production_fc1_tile_bf16"]["shape"] == [4096, 14336]
    assert details["production_fc1_tile_bf16"]["bytes"] == 112 * 2**20
    assert details["fused_fc1_activated_tile_bf16"]["shape"] == [4096, 7168]
    assert details["fused_fc1_activated_tile_bf16"]["bytes"] == 56 * 2**20
    assert details["production_fc2_partial_bf16"]["bytes"] == 42 * 2**20
    assert details["fc1_input_int8"]["bytes"] == 21 * 2**20


def test_launch_contract_distinguishes_launches_from_dot_streams():
    details = bench.gemm_contract(4096)
    assert details["production"]["gemm_kernel_launches"] == 4
    assert details["production"]["fc1"] == {
        "count": 2,
        "m": 4096,
        "n": 14336,
        "k": 5376,
    }
    assert details["fused"]["gemm_kernel_launches"] == 4
    assert details["fused"]["fc1"]["dot_streams_per_launch"] == 2
    assert details["fused"]["fc1_input_quantizations"] == 1
    assert details["production"]["fc1_input_quantizations"] == 2


def test_traffic_model_counts_only_explicit_intermediate_carriers():
    details = bench.intermediate_traffic_model(4096)
    assert details["fc1_bf16_round_trip_saved_bytes"] == 224 * 2**20
    assert details["fc2_bf16_output_write_bytes_eliminated"] == 84 * 2**20
    assert details["fc1_int8_carrier_write_bytes_eliminated"] == 21 * 2**20
    assert details["full_2f_bf16_intermediate_eliminated"]
    assert details["activated_bf16_intermediate_remains"]


def test_full_width_contract_names_control_and_candidate_boundaries():
    details = bench.full_width_gemm_contract(4096)
    assert details["kitchen_control"]["gemm_kernel_launches"] == 2
    assert details["kitchen_control"]["fc1"] == {
        "count": 1,
        "m": 4096,
        "n": 28672,
        "k": 5376,
    }
    assert details["kitchen_control"]["fc2"] == {
        "count": 1,
        "m": 4096,
        "n": 5376,
        "k": 14336,
    }
    candidate = details["candidate_kernel_contract"]
    assert candidate["fc1"]["output"] == {
        "shape": [4096, 14336],
        "dtype": "bfloat16",
    }
    assert candidate["fc2"]["output"] == {
        "shape": [4096, 5376],
        "dtype": "bfloat16",
    }


def test_full_width_weight_reconstruction_restores_gate_up_order():
    tiles = (
        {
            "fc1_weight": torch.tensor([[0], [1], [10], [11]], dtype=torch.int8),
            "fc1_scale": torch.tensor([0.0, 1.0, 10.0, 11.0]),
            "fc2_weight": torch.tensor([[0, 1], [2, 3]], dtype=torch.int8),
            "fc2_scale": torch.tensor([0.5, 1.5]),
        },
        {
            "fc1_weight": torch.tensor([[2], [3], [12], [13]], dtype=torch.int8),
            "fc1_scale": torch.tensor([2.0, 3.0, 12.0, 13.0]),
            "fc2_weight": torch.tensor([[4, 5], [6, 7]], dtype=torch.int8),
            "fc2_scale": torch.tensor([0.5, 1.5]),
        },
    )
    fc1_weight, fc1_scale = bench.merge_fc1_tiles(tiles)
    fc2_weight, fc2_scale = bench.merge_fc2_tiles(tiles)
    assert fc1_weight[:, 0].tolist() == [0, 1, 2, 3, 10, 11, 12, 13]
    assert fc1_scale.tolist() == [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]
    assert fc2_weight.tolist() == [[0, 1, 4, 5], [2, 3, 6, 7]]
    assert torch.equal(fc2_scale, torch.tensor([0.5, 1.5]))


def test_full_width_weight_reconstruction_preserves_scalar_scale():
    tiles = tuple(
        {
            "fc1_weight": torch.zeros((4, 1), dtype=torch.int8),
            "fc1_scale": torch.tensor([0.25]),
            "fc2_weight": torch.zeros((1, 2), dtype=torch.int8),
            "fc2_scale": torch.tensor([0.5]),
        }
        for _ in range(2)
    )
    _weight, scale = bench.merge_fc1_tiles(tiles)
    assert scale.tolist() == [0.25]


def test_segmented_error_uses_logical_fc1_slices():
    reference = (torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0]]))
    actual = (torch.tensor([[1.0, 2.0]]), torch.tensor([[4.0]]))
    error = bench.tensor_error_segments(reference, actual)
    assert not error["exact"]
    assert error["max_abs"] == 1.0


def main():
    tests = tuple(
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for test in tests:
        test()
    print("%d ConvRot epilogue benchmark contract tests passed" % len(tests))


if __name__ == "__main__":
    main()
