"""CPU-only contracts for the static per-layer dense-teacher optimizer."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "h3_probe" / "static_layer_teacher.py"
SPEC = spec_from_file_location("h3_static_layer_teacher", MODULE_PATH)
static_layer_teacher = module_from_spec(SPEC)
SPEC.loader.exec_module(static_layer_teacher)


def record(layer, region, errors):
    rows = []
    for budget, retained in ((0.1, 1), (0.2, 2), (0.3, 3)):
        rows.append({
            "budget": budget,
            "direct_tile_keep_video_kv_tiles": retained,
            "direct_tile_video_kv_tiles": 10,
            "adaptive_teacher": {
                "sampled_teacher_rows": {
                    "fixed": {
                        "squared_error": errors[retained] * 0.5,
                        "teacher_energy": 50.0,
                    },
                },
            },
        })
    return {
        "layer": layer,
        "step": 2,
        "cond_or_uncond": 0,
        "kind": "video",
        "start": region * 100,
        "stop": region * 100 + 8,
        "moba3d": {"budgets": rows},
    }


def test_exact_static_layer_schedule():
    curves = {
        0: {1: 100.0, 2: 50.0, 3: 0.0},
        1: {1: 10.0, 2: 5.0, 3: 4.0},
        2: {1: 3.0, 2: 2.0, 3: 1.5},
    }
    records = [
        record(layer, region, curves[layer])
        for layer in range(3)
        for region in (0, 1)
    ]
    result = static_layer_teacher.summarize(records, expected_layers=range(3))
    assert result["status"] == "complete"
    assert result["uniform"]["selected_video_kv_tiles"] == 6
    schedule = result["optimized"]["micro_squared_error"]
    assert schedule["same_budget_as_uniform"]
    assert schedule["proven_optimal_for_local_teacher_surrogate"]
    assert schedule["retained_k_by_layer"] == {"0": 3, "1": 2, "2": 1}
    assert abs(schedule["squared_error"] - 8.0) < 1.0e-9
    assert schedule["squared_error_reduction_vs_uniform"] > 0.8
    assert len(result["two_region_holdout"]["micro_squared_error"]) == 2
    assert all(
        row["evaluation_squared_error_reduction_vs_uniform"] > 0.8
        for row in result["two_region_holdout"]["micro_squared_error"]
    )


def test_missing_baseline_is_unavailable():
    rec = record(0, 0, {1: 3.0, 2: 2.0, 3: 1.0})
    rec["moba3d"]["budgets"] = [rec["moba3d"]["budgets"][0]]
    result = static_layer_teacher.summarize([rec], expected_layers=(0,))
    assert result["status"] == "unavailable"
    assert result["missing_baseline_layers"] == [0]


if __name__ == "__main__":
    test_exact_static_layer_schedule()
    test_missing_baseline_is_unavailable()
    print("static layer teacher tests passed")
