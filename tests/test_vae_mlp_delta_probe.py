import importlib.util
from pathlib import Path
import sys

import torch


_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "probe_vae_mlp_delta.py"
_SPEC = importlib.util.spec_from_file_location("probe_vae_mlp_delta", _SCRIPT)
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


def test_tile_starts_match_h3_geometry():
    assert probe.split_tile_starts(54) == (0, 9, 18, 28, 38)
    assert probe.split_tile_starts(30) == (0, 7, 14)


def test_global_overlap_indices_match_coordinates():
    pair = probe.Pair(
        "horizontal",
        probe.Window(5, 18, 7),
        probe.Window(5, 18, 14),
    )
    indices_a, indices_b, coords, shared = probe.pair_indices(pair)
    assert shared == 7 * 16 * 9
    assert coords[0].tolist() == [5, 18, 14]
    assert coords[-1].tolist() == [11, 33, 22]
    assert indices_a[0].item() == probe.flat_index(pair.a, 5, 18, 14)
    assert indices_b[0].item() == probe.flat_index(pair.b, 5, 18, 14)


def test_full_delta_reproduces_linear_output():
    torch.manual_seed(1)
    a_a = torch.randn(5, 8)
    a_b = torch.randn(5, 8)
    weight = torch.randn(3, 8)
    bias = torch.randn(3)
    y_a = torch.nn.functional.linear(a_a, weight, bias)
    y_b = torch.nn.functional.linear(a_b, weight, bias)
    prediction = torch.nn.functional.linear(a_b - a_a, weight)
    metrics = probe.metric_values(y_b - y_a, prediction)
    assert metrics["r2"] > 0.99999
    assert metrics["relative_delta_error"] < 1e-6


def test_voxel_groups_are_local_and_bounded():
    coords = torch.cartesian_prod(torch.arange(2), torch.arange(6), torch.arange(9))
    group_ids, voxel_shape, counts = probe.voxel_groups(coords, 32)
    assert group_ids.shape == (108,)
    assert math_prod(voxel_shape) == 32
    assert counts.max().item() <= 32
    assert counts.sum().item() == 108


def math_prod(values):
    result = 1
    for value in values:
        result *= value
    return result


def test_pooled_gate_applies_kill_threshold():
    row = {
        "selector": "oracle_token",
        "overlap_type": "adjacent_horizontal",
        "active_fraction_requested": 0.25,
        "sse": 70.0,
        "sst": 100.0,
    }
    gate = probe.pooled_gate([row])
    assert abs(gate["pooled_r2"] - 0.3) < 1e-12
    assert gate["verdict"] == "stop"


def test_proxy_selector_names_parse():
    value = "proxy_gate,proxy_value,proxy_gate_value,proxy_swiglu,proxy_voxel"
    assert probe.parse_selectors(value) == tuple(value.split(","))


if __name__ == "__main__":
    test_tile_starts_match_h3_geometry()
    test_global_overlap_indices_match_coordinates()
    test_full_delta_reproduces_linear_output()
    test_voxel_groups_are_local_and_bounded()
    test_pooled_gate_applies_kill_threshold()
    test_proxy_selector_names_parse()
    print("H3 VAE MLP delta probe tests passed")
