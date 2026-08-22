"""CPU contracts for the ConvRot activation scale-domain benchmark."""

import os
import sys

import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from benchmarks import bench_convrot_scale_domains as bench  # noqa: E402


def test_default_domains_cover_block_production_and_full_width():
    assert bench.validate_scale_domains(
        bench.FFN, bench.DEFAULT_SCALE_DOMAINS
    ) == (256, 512, 1024, 2048, 3584, 7168, 14336)


def test_invalid_domains_fail_closed():
    for domains in ((128,), (300,), (256, 256), (4096,)):
        try:
            bench.validate_scale_domains(bench.FFN, domains)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid domains were accepted: %r" % (domains,))


def test_domain_ranges_cover_ffn_once():
    ranges = bench.domain_ranges(bench.FFN, 2048)
    assert len(ranges) == 7
    assert ranges[0] == (0, 2048)
    assert ranges[-1] == (12288, 14336)
    assert sum(stop - start for start, stop in ranges) == bench.FFN


def test_scale_domain_contract_counts_carrier_and_scale_bytes():
    block = bench.scale_domain_contract(4096, 256)
    assert block["scale_domains_per_row"] == 56
    assert block["int8_carrier_bytes"] == 56 * 2**20
    assert block["fp32_scale_bytes"] == 4096 * 56 * 4
    production = bench.scale_domain_contract(4096, 7168)
    assert production["scale_domains_per_row"] == 2
    assert production["fp32_scale_bytes"] == 4096 * 2 * 4


def test_production_tile_packing_preserves_gate_up_order():
    fc1 = {
        "weight": torch.arange(8, dtype=torch.int8).reshape(8, 1),
        "weight_scale": torch.arange(8, dtype=torch.float32),
    }
    fc2 = {
        "weight": torch.arange(4, dtype=torch.int8).reshape(1, 4),
    }
    tiles = bench.prepare_production_tiles(fc1, fc2)
    assert len(tiles) == 2
    assert tiles[0]["fc1_weight"][:, 0].tolist() == [0, 1, 4, 5]
    assert tiles[1]["fc1_weight"][:, 0].tolist() == [2, 3, 6, 7]
    assert tiles[0]["fc1_scale"].tolist() == [0.0, 1.0, 4.0, 5.0]
    assert tiles[0]["fc2_weight"].tolist() == [[0, 1]]
    assert tiles[1]["fc2_weight"].tolist() == [[2, 3]]


def test_production_tile_packing_preserves_scalar_scale():
    fc1 = {
        "weight": torch.zeros((8, 1), dtype=torch.int8),
        "weight_scale": torch.tensor([0.25]),
    }
    fc2 = {"weight": torch.zeros((1, 4), dtype=torch.int8)}
    tiles = bench.prepare_production_tiles(fc1, fc2)
    assert [tile["fc1_scale"].tolist() for tile in tiles] == [[0.25], [0.25]]


def main():
    tests = tuple(
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for test in tests:
        test()
    print("%d ConvRot scale-domain benchmark contract tests passed" % len(tests))


if __name__ == "__main__":
    main()
