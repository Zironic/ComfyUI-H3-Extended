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
    lut, valid = bench.block_mask_to_lut(sample)
    assert valid.item() == 2 and lut[0, 0, 0, :2].tolist() == [1, 2]


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


def test_sparse_prepare_uses_direct_router():
    target = layout()
    calls = []

    class Router:
        def build_lut(self, q, k, received_layout, budget):
            calls.append((q, k, received_layout, budget))
            q_tiles = bench.tile_geometry(received_layout).q_tiles
            kv_tiles = bench.tile_geometry(received_layout).kv_tiles
            lut = torch.ones((1, bench.HEADS, q_tiles, kv_tiles), dtype=torch.int32)
            valid = torch.full((1, bench.HEADS, q_tiles), kv_tiles, dtype=torch.int32)
            return lut, valid, {"requested_video_budget": float(budget)}

    prepared_args = {}

    class Sparse:
        def prepare(self, q, k, v, lut, valid, **kwargs):
            prepared_args.update({"q": q, "k": k, "v": v, "lut": lut,
                                  "valid": valid, **kwargs})
            return object()

        @staticmethod
        def execute(_prepared):
            return torch.empty(0)

    context = {
        "torch": torch,
        "layout": target,
        "device": torch.device("cpu"),
        "router": Router(),
        "sparse": Sparse(),
    }
    call = bench._prepare_call(
        context, "sparse_sage_128x64", 0.5, 0.0, 0.0, 64, "uniform", 7
    )
    call.execute()
    assert len(calls) == 1
    q, k, received_layout, budget = calls[0]
    assert q.dtype == torch.bfloat16 and q.shape == k.shape
    assert received_layout is target and budget == 0.5
    assert prepared_args["lut"].dtype == torch.int32
    assert prepared_args["valid"].dtype == torch.int32
    assert prepared_args["metadata"] == {"requested_video_budget": 0.5}


def test_sparse_timing_field_aggregation():
    summaries = [
        {"stages": {
            stage: {"mean_ms": value}
            for stage, value in zip(bench.SPARSE_TIMING_STAGES, (1, 5, 9, 13, 17))
        }},
        {"stages": {
            stage: {"mean_ms": value}
            for stage, value in zip(bench.SPARSE_TIMING_STAGES, (3, 7, 11, 15, 19))
        }},
        {"stages": {
            stage: {"mean_ms": value}
            for stage, value in zip(bench.SPARSE_TIMING_STAGES, (2, 6, 10, 14, 18))
        }},
    ]
    fields = bench._aggregate_timing_fields(summaries)
    assert fields == {
        "%s_ms" % stage: float(index * 4 + 2)
        for index, stage in enumerate(bench.SPARSE_TIMING_STAGES)
    }


def test_flex_compile_boundary():
    calls = []
    compiled = object()

    class FakeTorch:
        @staticmethod
        def compile(function, **kwargs):
            calls.append((function, kwargs))
            return compiled

    def flex_attention():
        raise AssertionError("stand-in must not execute")

    assert bench.compile_flex_attention(FakeTorch, flex_attention) is compiled
    assert calls == [(flex_attention, {"fullgraph": True})]
    assert bench.flex_kernel_options(64) == {"BLOCK_M": 64, "BLOCK_N": 64}
    assert bench.flex_kernel_options(128) == {"BLOCK_M": 128, "BLOCK_N": 64}


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
    test_sparse_prepare_uses_direct_router()
    test_sparse_timing_field_aggregation()
    test_flex_compile_boundary()
    test_sweep_break_even_and_reports()
    print("all hybrid sparse benchmark tests passed")


if __name__ == "__main__":
    main()
