"""CPU contracts for the paired-domain CUTLASS FC1 geometry benchmark."""

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
from benchmarks import bench_convrot_fc1_cutlass_geometry as bench  # noqa: E402
sys.argv = _ORIGINAL_ARGV


def test_candidate_geometry_preserves_primary_accumulator_area():
    copied = bench.config_contract(bench.CONFIGS[0])
    wide = bench.config_contract(bench.CONFIGS[1])
    assert copied["raw_accumulator_outputs_per_cta"] == 32768
    assert wide["raw_accumulator_outputs_per_cta"] == 32768
    assert copied["warps_per_cta"] == wide["warps_per_cta"] == 8
    assert copied["raw_accumulator_outputs_per_thread"] == 128
    assert wide["raw_accumulator_outputs_per_thread"] == 128


def test_gate_up_interleave_round_trip():
    gate = torch.arange(bench.FFN, dtype=torch.int32)
    up = gate + 100000
    ordered = torch.cat((gate, up)).reshape(bench.EXPANDED, 1)
    packed = bench.interleave_gate_up(ordered)
    assert packed[: bench.DOMAIN, 0].tolist() == gate[: bench.DOMAIN].tolist()
    assert packed[bench.DOMAIN : 2 * bench.DOMAIN, 0].tolist() == up[: bench.DOMAIN].tolist()

    output = ordered.reshape(1, bench.EXPANDED)
    packed_output = bench.interleave_gate_up_output(output)
    restored = bench.deinterleave_gate_up_output(packed_output)
    assert torch.equal(restored, output)


def test_carrier_contract_reports_eliminated_bf16_round_trip():
    contract = bench.carrier_contract(4096)
    assert contract["raw_fc1_output"]["bytes"] == 224 * 2**20
    assert contract["carrier"]["bytes"] == 56 * 2**20
    assert contract["scales"]["shape"] == [4096, 56]
    assert contract["bf16_round_trip_bytes_removed_by_real_fusion"] == 448 * 2**20


def test_config_parser_fails_closed():
    assert bench.parse_configs("copied_128x256x64_s3,wide_64x512x64_s2") == (
        "copied_128x256x64_s3",
        "wide_64x512x64_s2",
    )
    for value in ("", "missing", "wide_64x512x64_s2,wide_64x512x64_s2"):
        try:
            bench.parse_configs(value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid configuration selection was accepted")


def test_candidate_classification_prioritizes_spilling():
    kitchen = {"median_ms": 10.0}
    candidate = {"median_ms": 9.0}
    attributes = {"local_memory_bytes_per_thread": 8}
    assert bench.classify_candidate(candidate, kitchen, attributes)["classification"] == "SPILL-LIMITED"


def test_candidate_classification_prioritizes_correctness():
    kitchen = {"median_ms": 10.0}
    candidate = {"median_ms": 9.0}
    attributes = {"local_memory_bytes_per_thread": 0}
    assert bench.classify_candidate(candidate, kitchen, attributes, correct=False)["classification"] == "INVALID-CORRECTNESS"


def main():
    tests = tuple(
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for test in tests:
        test()
    print("%d CUTLASS FC1 geometry benchmark contract tests passed" % len(tests))


if __name__ == "__main__":
    main()
