"""CPU contracts for the production H3 Chipmunk path."""

import inspect
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_chipmunk.config import H3ChipmunkConfig  # noqa: E402
from h3_chipmunk import executor as chip_executor  # noqa: E402
from h3_chipmunk import report as chip_report  # noqa: E402
from h3_chipmunk import state as chip_state  # noqa: E402
from h3_chipmunk.selector import (  # noqa: E402
    logical_swiglu,
    group_scores,
    token_group_scores,
    select_top_groups,
    selected_mask,
    expand_selection,
)
from h3_chipmunk.state import H3ChipmunkSession  # noqa: E402


def test_config_contract():
    cfg = H3ChipmunkConfig()
    assert cfg.mode == "measure"
    assert cfg.feature_group == 256
    assert cfg.scope == "target_video"
    assert cfg.cache_location == "gpu"
    assert cfg.cache_budget_gb == 24.0
    assert cfg.chunk_rows == 2048
    assert cfg.effective_chunk_rows == 2048
    assert cfg.save_report is False

    assert H3ChipmunkConfig(chunk_rows=128).effective_chunk_rows == 2048
    assert (
        H3ChipmunkConfig(
            mode="reference_delta",
            cache_location="gpu",
            chunk_rows=128,
        ).effective_chunk_rows
        == 128
    )

    invalid = (
        {"mode": "shadow_validate"},
        {"top_fraction": 0.0},
        {"cache_location": "disk"},
        {"cache_budget_gb": 0.0},
        {"layer_start": 10, "layer_stop": 10},
        {"measure_layer_stride": 0},
        {"mode": "reference_delta", "cache_location": "cpu"},
    )
    for kwargs in invalid:
        try:
            H3ChipmunkConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config accepted: {kwargs}")


def test_signature_tracks_gpu_cache_policy():
    base = H3ChipmunkConfig(cache_budget_gb=24.0)
    larger = H3ChipmunkConfig(cache_budget_gb=48.0)
    all_layers = H3ChipmunkConfig(measure_layer_stride=1)
    assert base.signature != larger.signature
    assert base.signature != all_layers.signature


def test_swiglu_pairing():
    gate = torch.tensor([[0.0, 1.0]])
    up = torch.tensor([[2.0, 3.0]])
    out = logical_swiglu(torch.cat((gate, up), dim=-1))
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(out, expected)


def test_group_selector_shapes_without_scalar_reads():
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
    expanded = expand_selection(
        mask,
        rows=5,
        token_group_rows=2,
        feature_group=256,
    )
    assert expanded.shape == (5, 1024)
    assert expanded[:, 256:512].all()
    assert not expanded[:, :256].any()


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


def test_report_records_reject_tensors():
    session = H3ChipmunkSession()
    session.record(path="sparse_delta", step=2, active_fraction=0.30)
    assert session.records[0]["step"] == 2
    try:
        session.record(path="bad", metric=torch.tensor(0.5))
    except RuntimeError as exc:
        assert "may not contain tensors" in str(exc)
    else:
        raise AssertionError("tensor-valued production report record was accepted")


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


def test_production_modules_have_no_host_materialization_calls():
    # Hard contract for actual Comfy-node execution/reporting. Keep standalone
    # offline analysis separate if host materialization is ever needed again.
    sources = {
        "executor": inspect.getsource(chip_executor),
        "state": inspect.getsource(chip_state),
        "report": inspect.getsource(chip_report),
    }
    forbidden = (".item(", ".cpu(", 'to("cpu"', "to('cpu'", ".tolist(")
    for name, source in sources.items():
        for token in forbidden:
            assert token not in source, f"{name} reintroduced host materialization via {token}"


if __name__ == "__main__":
    test_config_contract()
    test_signature_tracks_gpu_cache_policy()
    test_swiglu_pairing()
    test_group_selector_shapes_without_scalar_reads()
    test_session_isolates_branch_layer_chunk()
    test_report_records_reject_tensors()
    test_request_change_resets_cache_and_records()
    test_production_modules_have_no_host_materialization_calls()
    print("H3 Chipmunk production no-sync tests passed")
