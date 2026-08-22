"""CPU/static contracts for the production H3 Chipmunk path."""

import ast
import inspect
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_chipmunk.config import H3ChipmunkConfig  # noqa: E402
from h3_chipmunk import executor as chip_executor  # noqa: E402
from h3_chipmunk import offload as chip_offload  # noqa: E402
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
    assert cfg.cache_location == "async_pinned"
    assert cfg.cache_budget_gb == 1.0
    assert cfg.chunk_rows == 2048
    assert cfg.effective_chunk_rows == 2048
    assert cfg.save_report is False
    assert cfg.density_profile == "depth_safe_v1"
    assert cfg.staging_slots == 2

    assert H3ChipmunkConfig(chunk_rows=128).effective_chunk_rows == 2048
    assert (
        H3ChipmunkConfig(
            mode="reference_delta",
            cache_location="async_pinned",
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
        {"density_profile": "guess"},
        {"staging_slots": 1},
        {"mode": "reference_delta", "cache_location": "cpu"},
        {"mode": "reference_delta", "cache_location": "gpu"},
    )
    for kwargs in invalid:
        try:
            H3ChipmunkConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config accepted: {kwargs}")


def test_depth_profile_protects_destructive_front():
    cfg = H3ChipmunkConfig(mode="reference_delta")
    assert all(cfg.fraction_for_layer(layer) == 1.0 for layer in range(0, 11))
    assert not any(cfg.layer_eligible(layer) for layer in range(0, 11))
    assert all(cfg.fraction_for_layer(layer) == 0.40 for layer in range(11, 20))
    assert all(cfg.fraction_for_layer(layer) == 0.50 for layer in range(20, 30))
    assert all(cfg.fraction_for_layer(layer) == 0.60 for layer in range(30, 50))

    # Balanced whole-group rounding for H3's two 28-group halves.
    assert cfg.selected_features_for_layer(11, 14336) == 6144   # 24 / 56
    assert cfg.selected_features_for_layer(20, 14336) == 7168   # 28 / 56
    assert cfg.selected_features_for_layer(30, 14336) == 8704   # 34 / 56
    assert cfg.max_selected_features(14336) == 8704


def test_default_staging_fits_one_gib_budget():
    cfg = H3ChipmunkConfig(mode="reference_delta")
    rows = cfg.chunk_rows
    selected = cfg.max_selected_features(14336)
    selector_rows = (rows + cfg.token_group_rows - 1) // cfg.token_group_rows
    groups = selected // cfg.feature_group
    bytes_per_slot = (
        rows * selected * 2
        + rows * 5376 * 2
        + selector_rows * 14336 * 2
        + selector_rows * groups * 4
    )
    total = bytes_per_slot * cfg.staging_slots
    assert total < 1 * 1024**3
    # Current geometry is ~0.108 GiB, leaving substantial headroom for the
    # selected GEMM temporaries and normal H3 execution.
    assert total < 128 * 1024**2


def test_signature_tracks_offload_and_profile_policy():
    base = H3ChipmunkConfig(cache_budget_gb=1.0)
    larger = H3ChipmunkConfig(cache_budget_gb=2.0)
    uniform = H3ChipmunkConfig(density_profile="uniform")
    assert base.signature != larger.signature
    assert base.signature != uniform.signature


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
    cfg = H3ChipmunkConfig()
    session = H3ChipmunkSession(cfg)

    class Snapshot:
        request_id = 4
        layout_signature = (123,)

    session.ensure_request(Snapshot())
    a = session.cache((0,), 3, 2)
    b = session.cache((1,), 3, 2)
    c = session.cache((0,), 4, 2)
    d = session.cache((0,), 3, 3)
    assert len({id(a), id(b), id(c), id(d)}) == 4
    a.valid = b.valid = c.valid = d.valid = True
    session.invalidate_branch((0,))
    assert a.valid is False
    assert c.valid is False
    assert d.valid is False
    assert b.valid is True


def test_report_records_reject_tensors():
    session = H3ChipmunkSession(H3ChipmunkConfig())
    session.record(path="sparse_delta_async", step=2, active_fraction=0.50)
    assert session.records[0]["step"] == 2
    try:
        session.record(path="bad", metric=torch.tensor(0.5))
    except RuntimeError as exc:
        assert "may not contain tensors" in str(exc)
    else:
        raise AssertionError("tensor-valued production report record was accepted")


def test_request_change_resets_logical_cache_but_keeps_offloader():
    session = H3ChipmunkSession(H3ChipmunkConfig())
    offload_id = id(session.offload)

    class A:
        request_id = 1
        layout_signature = (1,)

    class B:
        request_id = 2
        layout_signature = (1,)

    session.ensure_request(A())
    session.cache((0,), 11, 0).valid = True
    session.record(path="x")
    session.ensure_request(B())
    assert not session.caches
    assert not session.records
    assert id(session.offload) == offload_id


def _forbidden_device_sync_calls(module):
    tree = ast.parse(inspect.getsource(module))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {
            "item",
            "cpu",
            "tolist",
            "synchronize",
        }:
            bad.append((func.attr, getattr(node, "lineno", None)))
        if isinstance(func, ast.Attribute) and func.attr == "to" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value == "cpu":
                bad.append(("to(cpu)", getattr(node, "lineno", None)))
    return bad


def test_production_modules_have_no_device_to_host_sync_calls():
    # Pinned CPU tensors are DMA backing storage and are allowed. What is banned
    # is materializing CUDA values or synchronizing the model thread on device
    # completion.
    for module in (chip_executor, chip_state, chip_report, chip_offload):
        assert not _forbidden_device_sync_calls(module), module.__name__

    source = inspect.getsource(chip_offload)
    assert "pin_memory=True" in source
    assert "non_blocking=True" in source
    assert "wait_event" in source
    assert "wait_stream" in source


if __name__ == "__main__":
    test_config_contract()
    test_depth_profile_protects_destructive_front()
    test_default_staging_fits_one_gib_budget()
    test_signature_tracks_offload_and_profile_policy()
    test_swiglu_pairing()
    test_group_selector_shapes_without_scalar_reads()
    test_session_isolates_branch_layer_chunk()
    test_report_records_reject_tensors()
    test_request_change_resets_logical_cache_but_keeps_offloader()
    test_production_modules_have_no_device_to_host_sync_calls()
    print("H3 Chipmunk async-pinned production tests passed")
