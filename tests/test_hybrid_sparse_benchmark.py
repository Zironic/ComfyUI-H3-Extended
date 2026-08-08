"""CPU-only contracts for the headless H3 hybrid benchmark."""

import csv
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
SCRIPT = os.path.join(PACK, "benchmarks", "bench_hybrid_sparse.py")
SPEC = importlib.util.spec_from_file_location("bench_hybrid_sparse", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def layout(sequence=640, video_start=192):
    return SimpleNamespace(
        seq_len=sequence,
        video_range=(video_start, sequence),
        video_shape=(1, 1, sequence - video_start),
        audio_t=16,
        segments=[
            (0, video_start - 32, "text"),
            (video_start - 32, video_start, "audio"),
            (video_start, sequence, "video"),
        ],
    )


def test_fraction_parsing():
    assert bench.parse_fraction("25%") == 0.25
    assert bench.parse_fraction("0.25") == 0.25
    assert bench.parse_fraction("25") == 0.25
    for value in ("-1%", "101%", "bad"):
        try:
            bench.parse_fraction(value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)


def test_geometry_patterns_and_compaction():
    target = layout()
    geometry = bench.tile_geometry(target)
    assert (geometry.q_tiles, geometry.kv_tiles) == (5, 10)
    for pattern in ("uniform", "local", "shared"):
        first, metadata = bench.build_controlled_mask(
            target, 0.5, pattern=pattern, heads=2, seed=9
        )
        second, _ = bench.build_controlled_mask(
            target, 0.5, pattern=pattern, heads=2, seed=9
        )
        assert torch.equal(first, second)
        assert metadata["retained_video_kv_tiles"] == 4
        assert first[:, :, :geometry.pure_video_q_start].all()
        video = first[
            :, :, geometry.pure_video_q_start:, geometry.pure_video_kv_start:
        ]
        assert torch.equal(video.sum(dim=-1), torch.full_like(video.sum(dim=-1), 4))
        assert first[
            :, :, geometry.pure_video_q_start:, :geometry.pure_video_kv_start
        ].all()
    shared, _ = bench.build_controlled_mask(target, 0.5, pattern="shared", heads=1, seed=4)
    assert torch.equal(shared[:, :, 2], shared[:, :, 3])

    sample = torch.tensor([[[[False, True, False, True]]]])
    counts, indices = bench.compact_kv_blocks(sample)
    assert counts.item() == 2
    assert indices[0, 0, 0, :2].tolist() == [1, 3]
    block_mask = bench.make_flex_block_mask(
        sample, q_length=64, kv_length=256, q_tile=64, kv_tile=64
    )
    assert block_mask.seq_lengths == (64, 256)
    assert block_mask.kv_indices[0, 0, 0, :2].tolist() == [1, 3]


def test_hybrid_gather_plan():
    target = layout()
    geometry = bench.tile_geometry(target)
    empty = bench.build_hybrid_plan(
        target, 0.5, hard_q_fraction=0, hard_head_fraction=0, heads=8
    )
    assert empty["hard_q_tiles"] == [] and empty["hard_heads"] == []

    plan = bench.build_hybrid_plan(
        target,
        0.5,
        hard_q_fraction="50%",
        hard_head_fraction="25%",
        flex_q_tile=64,
        heads=8,
        seed=2,
    )
    assert len(plan["hard_q_tiles"]) == 2
    assert len(plan["hard_heads"]) == 2
    assert len(plan["hard_tokens"]) == 256
    assert len(plan["flex_q_rows"]) == 4
    for head in plan["hard_heads"]:
        for q_row in plan["hard_q_tiles"]:
            row = plan["placeholder_mask"][0, head, q_row]
            assert row[:geometry.pure_video_kv_start].all()
            assert not row[geometry.pure_video_kv_start:].any()
    selected_flex = plan["flex_mask"][:, :, plan["flex_q_rows"]]
    assert selected_flex.shape[2] == len(plan["flex_q_rows"])


def test_sweep_break_even_and_reports():
    cases = bench.sweep_cases()
    assert len(cases) == 6 * 3 * 10
    assert {case["flex_q_tile"] for case in cases} == {128, 64, 32}
    rows = [
        {"mode": "dense_sage", "budget": 0.5, "flex_q_tile": 64, "latency_ms": 10.0},
        {"mode": "hybrid_sage_flex", "budget": 0.5, "flex_q_tile": 64,
         "hard_q_fraction": 0.2, "latency_ms": 9.0},
        {"mode": "hybrid_sage_flex", "budget": 0.5, "flex_q_tile": 64,
         "hard_q_fraction": 0.4, "latency_ms": 11.0},
        {"mode": "hybrid_sage_flex", "budget": 0.5, "flex_q_tile": 32,
         "hard_q_fraction": 0.1, "latency_ms": 12.0},
    ]
    frontier = bench.derive_break_even(rows)
    assert frontier == {"0.5:32": None, "0.5:64": 0.2}

    temp_root = os.path.join(PACK, ".agent", "tmp")
    os.makedirs(temp_root, exist_ok=True)
    json_path = os.path.join(temp_root, "hybrid-benchmark-test.json")
    csv_path = os.path.join(temp_root, "hybrid-benchmark-test.csv")
    try:
        bench.write_reports({"sequence": 1}, rows, None, frontier, json_path, csv_path)
        with open(json_path, encoding="utf-8") as handle:
            assert json.load(handle)["break_even"] == frontier
        with open(csv_path, newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == len(rows)
    finally:
        for path in (json_path, csv_path):
            if os.path.exists(path):
                os.unlink(path)


def main():
    test_fraction_parsing()
    test_geometry_patterns_and_compaction()
    test_hybrid_gather_plan()
    test_sweep_break_even_and_reports()
    print("all hybrid sparse benchmark tests passed")


if __name__ == "__main__":
    main()
