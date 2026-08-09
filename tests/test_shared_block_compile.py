"""CPU-only contracts for the shared H3 tensor block."""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

from h3_attention.hybrid.router import SparseTileGeometry  # noqa: E402
from h3_runtime.block_compile import (  # noqa: E402
    BlockCarriers,
    BlockTopology,
    H3BlockError,
    build_block_signature,
    make_compiled_block,
    validate_block_signature,
)


def _topology():
    geometry = SparseTileGeometry(
        signature=(256, (128, 256), ((0, 128, "audio"), (128, 256, "video")), (1, 1, 128), 64),
        sequence=256, q_tiles=2, kv_tiles=4, pure_video_q_start=1, pure_video_kv_start=2,
    )
    return BlockTopology(
        hidden_size=512, ffn_size=512, timestep_dim=64, heads=2, head_dim=128,
        norm_eps=1e-5, qk_norm_eps=1e-6, adaln_apply_silu=True,
        adaln_out_features=9216, rope_strides=(128, 1, 2, 3), has_rope=True,
        router_geometry=geometry, video_budget=0.5,
        mod_segments=((0, 128, 0), (128, 256, 1)),
        mlp_chunks=((0, 128, 0), (128, 256, 1)),
    )


def _tensor(shape, dtype):
    return torch.empty(shape, device="meta", dtype=dtype)


def _carriers():
    return BlockCarriers(
        _tensor((9216, 64), torch.bfloat16), _tensor((9216,), torch.bfloat16),
        _tensor((512,), torch.bfloat16), _tensor((512,), torch.bfloat16),
        _tensor((768, 512), torch.int8), _tensor((768,), torch.float32),
        _tensor((128,), torch.bfloat16), _tensor((128,), torch.bfloat16),
        _tensor((512, 256), torch.int8), _tensor((512,), torch.float32),
        _tensor((1024, 512), torch.int8), _tensor((1024,), torch.float32),
        _tensor((512, 512), torch.int8), _tensor((512,), torch.float32),
    )


def test_shared_callable_reuses_one_graph_for_distinct_bindings():
    topology = _topology()
    carriers = {3: _carriers(), 17: _carriers(), 42: _carriers()}
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(str(graph_module.graph))
        return graph_module.forward

    compiled = make_compiled_block(topology, carriers[3], backend=backend)
    x = _tensor((256, 512), torch.bfloat16)
    t_emb = _tensor((1, 64), torch.bfloat16)
    rope = _tensor((1, 256, 1, 48, 2, 2), torch.bfloat16)
    for layer in (3, 17, 42, 3, 42, 17):
        assert compiled(x, t_emb, rope, carriers[layer]).shape == (256, 512)

    assert len(graphs) == 1
    graph = graphs[0]
    for name in (
        "minimax_h3.fused_qkv", "minimax_h3.prepare_sparse_sage_v",
        "minimax_h3.sparse_sage_attention", "minimax_h3.convrot_fc1_tiles",
        "minimax_h3.convrot_fc2_tiles", "minimax_h3.convrot_linear",
        "matmul", "topk", "minimax_h3.sort_selected_indices",
    ):
        assert name in graph
    assert "module" not in graph.lower()
    assert "layer_index" not in graph
    assert "registry" not in graph.lower()

    bad = list(carriers[17])
    bad[8] = _tensor((512, 256), torch.bfloat16)
    try:
        compiled(x, t_emb, rope, tuple(bad))
    except H3BlockError:
        pass
    else:
        raise AssertionError("changed carrier metadata reached compiled graph")
    assert len(graphs) == 1


def test_signature_rejects_metadata_and_accepts_structural_fake_blocks():
    topology = _topology()
    original = _carriers()
    signature = build_block_signature(original, topology)
    assert validate_block_signature(signature, _carriers(), topology) == signature

    changed = list(_carriers())
    changed[8] = _tensor((512, 256), torch.bfloat16)
    try:
        validate_block_signature(signature, tuple(changed), topology)
    except H3BlockError as exc:
        assert "dtype mismatch" in str(exc)
    else:
        raise AssertionError("changed carrier dtype was accepted")

    changed = list(_carriers())
    changed[10] = _tensor((1024, 128), torch.int8)
    try:
        validate_block_signature(signature, tuple(changed), topology)
    except H3BlockError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("changed carrier shape was accepted")


if __name__ == "__main__":
    test_shared_callable_reuses_one_graph_for_distinct_bindings()
    test_signature_rejects_metadata_and_accepts_structural_fake_blocks()
    print("shared block CPU tests passed")
