"""CPU contracts for H3 Chipmunk selector/config/cache/report state."""

import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_chipmunk.config import H3ChipmunkConfig
from h3_chipmunk.report import _shadow_bucket, _group_shadow
from h3_chipmunk.selector import (
    logical_swiglu,
    group_scores,
    token_group_scores,
    select_top_groups,
    energy_capture_by_fraction,
    selected_mask,
    expand_selection,
)
from h3_chipmunk.state import H3ChipmunkSession


def test_config_contract():
    cfg = H3ChipmunkConfig()
    assert cfg.mode == "measure"
    assert cfg.feature_group == 256
    assert cfg.scope == "target_video"
    assert cfg.cache_location == "cpu"
    assert cfg.cache_budget_gb == 24.0
    assert cfg.chunk_rows == 2048
    assert cfg.effective_chunk_rows == 2048
    assert cfg.measure_layer_stride == 5
    assert cfg.shadow_layer_stride == 5
    assert cfg.shadow_sample_rows == 128
    assert H3ChipmunkConfig(chunk_rows=128).effective_chunk_rows == 2048
    assert H3ChipmunkConfig(mode="shadow_validate", chunk_rows=128).effective_chunk_rows == 2048
    assert H3ChipmunkConfig(mode="reference_delta", chunk_rows=128).effective_chunk_rows == 128
    for kwargs in (
        {"top_fraction": 0.0},
        {"cache_location": "disk"},
        {"cache_budget_gb": 0.0},
        {"layer_start": 10, "layer_stop": 10},
        {"measure_layer_stride": 0},
        {"shadow_layer_stride": 0},
        {"shadow_sample_rows": 31},
    ):
        try:
            H3ChipmunkConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config accepted: {kwargs}")


def test_shadow_depth_profile_and_sampling():
    cfg = H3ChipmunkConfig(mode="shadow_validate")
    assert cfg.shadow_fraction_for_layer(0) == 0.30
    assert cfg.shadow_fraction_for_layer(14) == 0.30
    assert cfg.shadow_fraction_for_layer(15) == 0.40
    assert cfg.shadow_fraction_for_layer(24) == 0.40
    assert cfg.shadow_fraction_for_layer(25) == 0.50
    assert cfg.shadow_fraction_for_layer(29) == 0.50
    assert cfg.shadow_fraction_for_layer(30) == 1.0
    assert cfg.shadow_fraction_for_layer(49) == 1.0
    enabled = [layer for layer in range(50) if cfg.shadow_layer_enabled(layer)]
    assert enabled == [0, 5, 10, 15, 20, 25, 29, 30]


def test_signature_tracks_cache_measure_and_shadow_policy():
    cpu = H3ChipmunkConfig(cache_location="cpu", cache_budget_gb=24.0)
    gpu = H3ChipmunkConfig(cache_location="gpu", cache_budget_gb=24.0)
    larger = H3ChipmunkConfig(cache_location="cpu", cache_budget_gb=48.0)
    all_layers = H3ChipmunkConfig(measure_layer_stride=1)
    shadow_dense = H3ChipmunkConfig(shadow_layer_stride=1)
    shadow_rows = H3ChipmunkConfig(shadow_sample_rows=256)
    assert cpu.signature != gpu.signature
    assert cpu.signature != larger.signature
    assert cpu.signature != all_layers.signature
    assert cpu.signature != shadow_dense.signature
    assert cpu.signature != shadow_rows.signature


def test_swiglu_pairing():
    gate = torch.tensor([[0.0, 1.0]])
    up = torch.tensor([[2.0, 3.0]])
    out = logical_swiglu(torch.cat((gate, up), dim=-1))
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(out, expected)


def test_group_selector_shapes():
    delta = torch.zeros((5, 1024))
    delta[:, 256:512] = 10.0
    scores = group_scores(delta, 256)
    assert scores.shape == (5, 4)
    grouped = token_group_scores(scores, 2)
    assert grouped.shape == (3, 4)
    indices, counts = select_top_groups(grouped, 0.25)
    assert indices.shape == (3, 1)
    assert torch.all(indices == 1)
    assert torch.all(counts == 1)
    mask = selected_mask(indices, counts, 4)
    assert mask.shape == (3, 4)
    expanded = expand_selection(mask, rows=5, token_group_rows=2, feature_group=256)
    assert expanded.shape == (5, 1024)
    assert expanded[:, 256:512].all()
    assert not expanded[:, :256].any()


def test_energy_capture_sweep():
    scores = torch.zeros((2, 10), dtype=torch.float32)
    scores[:, 3] = 10.0
    captures = energy_capture_by_fraction(scores)
    values = [float(captures[key]) for key in ("0.10", "0.20", "0.25", "0.30", "0.40", "0.50")]
    assert values[0] == 1.0
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_shadow_report_aggregation():
    rows = [
        {
            "path": "shadow_delta", "layer": 5, "step": 2,
            "raw_relative_l2": 0.10, "gated_relative_l2": 0.05,
            "block_relative_l2": 0.01, "raw_cosine": 0.99,
            "error_rms": 0.2, "dense_rms": 1.0, "max_abs_error": 0.8,
        },
        {
            "path": "shadow_delta", "layer": 5, "step": 3,
            "raw_relative_l2": 0.20, "gated_relative_l2": 0.10,
            "block_relative_l2": 0.02, "raw_cosine": 0.98,
            "error_rms": 0.3, "dense_rms": 1.0, "max_abs_error": 1.0,
        },
    ]
    bucket = _shadow_bucket(rows)
    assert abs(bucket["raw_relative_l2"]["mean"] - 0.15) < 1e-9
    assert bucket["block_relative_l2"]["max"] == 0.02
    grouped = _group_shadow(rows, "layer")
    assert set(grouped) == {"5"}
    assert grouped["5"]["raw_cosine"]["count"] == 2


def test_session_isolates_branch_layer_chunk():
    session = H3ChipmunkSession()

    class Snapshot:
        request_id = 4
        layout_signature = (123,)

    session.ensure_request(Snapshot())
    a = session.cache((0,), 3, 2)
    b = session.cache((1,), 3, 2)
    c = session.cache((0,), 4, 2)
    d = session.cache((0,), 3, 3)
    assert len({id(a), id(b), id(c), id(d)}) == 4
    a.output = torch.ones(1)
    session.invalidate_branch((0,))
    assert a.output is None
    assert b.output is None


def test_deferred_record_materialization():
    session = H3ChipmunkSession()
    session.record(
        path="shadow_delta",
        raw_relative_l2=torch.tensor(0.125),
        nested={"metric": torch.tensor(0.75)},
    )
    rows = session.materialize_records()
    assert rows == [{
        "path": "shadow_delta",
        "raw_relative_l2": 0.125,
        "nested": {"metric": 0.75},
    }]


def test_request_change_resets_cache_and_records():
    session = H3ChipmunkSession()

    class A:
        request_id = 1
        layout_signature = (1,)

    class B:
        request_id = 2
        layout_signature = (1,)

    session.ensure_request(A())
    session.cache((0,), 0, 0).output = torch.ones(1)
    session.record(path="x")
    session.ensure_request(B())
    assert not session.caches
    assert not session.records


if __name__ == "__main__":
    test_config_contract()
    test_shadow_depth_profile_and_sampling()
    test_signature_tracks_cache_measure_and_shadow_policy()
    test_swiglu_pairing()
    test_group_selector_shapes()
    test_energy_capture_sweep()
    test_shadow_report_aggregation()
    test_session_isolates_branch_layer_chunk()
    test_deferred_record_materialization()
    test_request_change_resets_cache_and_records()
    print("H3 Chipmunk CPU tests passed")
