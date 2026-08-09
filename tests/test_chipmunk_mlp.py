"""CPU contracts for H3 Chipmunk selector/config/cache state."""

import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_chipmunk.config import H3ChipmunkConfig
from h3_chipmunk.selector import (
    logical_swiglu,
    group_scores,
    token_group_scores,
    select_top_groups,
    selected_mask,
    expand_selection,
)
from h3_chipmunk.state import H3ChipmunkSession


def test_config_contract():
    cfg = H3ChipmunkConfig()
    assert cfg.mode == "measure"
    assert cfg.feature_group == 256
    assert cfg.scope == "target_video"
    try:
        H3ChipmunkConfig(top_fraction=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero top_fraction accepted")


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
    assert b.output is None  # never initialized


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
    test_swiglu_pairing()
    test_group_selector_shapes()
    test_session_isolates_branch_layer_chunk()
    test_request_change_resets_cache_and_records()
    print("H3 Chipmunk CPU tests passed")
