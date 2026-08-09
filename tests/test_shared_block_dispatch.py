"""CPU-only contracts for the eager shared-block dispatcher."""

import os
import sys
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

import h3_runtime.block_dispatch as dispatch  # noqa: E402
from h3_activation_memory.config import (  # noqa: E402
    ActivationMemoryConfig,
    MODE_CONVROT_2SLICE,
)
from h3_attention.hybrid.config import (  # noqa: E402
    HybridSparseConfig,
    MODE_SAGE128_FUSED_QKV,
)


class FakeQuantized:
    def __init__(self, qdata, scale):
        self.qdata = qdata
        self.scale = scale
        self.shape = qdata.shape
        self._layout_cls = "TensorWiseINT8Layout"
        self._params = SimpleNamespace(
            transposed=False,
            convrot=True,
            convrot_groupsize=256,
        )


class FakeLayout:
    @staticmethod
    def get_plain_tensors(weight):
        return weight.qdata, weight.scale


def linear(weight, bias=None, in_features=None, out_features=None):
    return SimpleNamespace(
        weight=weight,
        bias=bias,
        in_features=int(in_features if in_features is not None else weight.shape[1]),
        out_features=int(out_features if out_features is not None else weight.shape[0]),
    )


def fake_block():
    hidden = 512
    inner = 256
    ffn = 512
    adaln_width = 6 * 3 * hidden
    quant = lambda shape: FakeQuantized(
        torch.empty(shape, dtype=torch.int8),
        torch.empty((shape[0], 1), dtype=torch.float32),
    )
    norm1 = SimpleNamespace(weight=torch.empty((hidden,), dtype=torch.bfloat16), eps=1e-5)
    norm2 = SimpleNamespace(weight=torch.empty((hidden,), dtype=torch.bfloat16), eps=1e-5)
    q_norm = SimpleNamespace(weight=torch.empty((128,), dtype=torch.bfloat16), eps=1e-6)
    k_norm = SimpleNamespace(weight=torch.empty((128,), dtype=torch.bfloat16), eps=1e-6)
    attn = SimpleNamespace(
        heads=2,
        head_dim=128,
        qkv_proj=linear(quant((3 * inner, hidden))),
        q_norm=q_norm,
        k_norm=k_norm,
        out_proj=linear(quant((hidden, inner))),
    )
    mlp = SimpleNamespace(
        fc1=linear(quant((2 * ffn, hidden))),
        fc2=linear(quant((hidden, ffn))),
    )
    adaln = SimpleNamespace(
        expand=6,
        modalities=3,
        hidden=hidden,
        apply_silu=True,
        linear=linear(
            torch.empty((adaln_width, 64), dtype=torch.bfloat16),
            torch.empty((adaln_width,), dtype=torch.bfloat16),
        ),
    )
    return SimpleNamespace(
        adaln_proj=adaln,
        norm1=norm1,
        norm2=norm2,
        attn=attn,
        mlp=mlp,
    )


def patched_function(**attributes):
    def forward(*args, **kwargs):
        raise AssertionError("not executed")
    for name, value in attributes.items():
        setattr(forward, name, value)
    return forward


def test_fifty_blocks_share_one_execution_signature():
    blocks = [fake_block() for _ in range(50)]
    signature = dispatch.validate_identical_blocks(blocks)
    assert signature == dispatch.block_execution_signature(blocks[42])

    blocks[17].attn.qkv_proj.weight.scale = torch.empty(
        (blocks[17].attn.qkv_proj.weight.scale.shape[0] - 1, 1),
        dtype=torch.float32,
    )
    try:
        dispatch.validate_identical_blocks(blocks)
    except dispatch.H3BlockError as exc:
        assert "block 17" in str(exc)
    else:
        raise AssertionError("different block 17 signature was accepted")


def test_install_replaces_fifty_block_closures_with_one_dispatcher():
    blocks = [fake_block() for _ in range(50)]
    activation = ActivationMemoryConfig(mode=MODE_CONVROT_2SLICE)
    object_patches = {}
    for index in range(50):
        object_patches["diffusion_model.blocks.%d.forward" % index] = patched_function(
            _h3_activation_memory=True,
            _h3_activation_config=activation.signature,
        )
        object_patches["diffusion_model.blocks.%d.attn.forward" % index] = patched_function(
            _h3_attention=True,
            _h3_backend="hybrid_sparse",
            _h3_projector="h3_fused_qkv",
        )

    class Patcher:
        def __init__(self):
            self.object_patches = object_patches
            self.model_options = {}

        def get_model_object(self, name):
            assert name == "diffusion_model.blocks"
            return blocks

        def add_object_patch(self, name, value):
            self.object_patches[name] = value

    backend = SimpleNamespace(
        name="hybrid_sparse",
        config=HybridSparseConfig(
            mode=MODE_SAGE128_FUSED_QKV,
            video_budget=0.1,
            strict=True,
        ),
        projector=object(),
        executor=SimpleNamespace(_use_sparse_sage_op=True),
        router=object(),
        collector=None,
    )
    patcher = Patcher()
    shared = dispatch.install_shared_block_dispatch(
        patcher,
        backend,
        activation,
        lambda graph, inputs: graph.forward,
    )
    assert patcher.model_options[dispatch.DISPATCHER_KEY] is shared
    assert all(
        getattr(
            patcher.object_patches["diffusion_model.blocks.%d.forward" % index],
            "_h3_shared_block_compile",
            False,
        )
        for index in range(50)
    )
    assert len({
        id(patcher.object_patches["diffusion_model.blocks.%d.forward" % index]._h3_shared_dispatcher)
        for index in range(50)
    }) == 1


if __name__ == "__main__":
    original_quantized = dispatch.QuantizedTensor
    original_layout = dispatch.TensorWiseINT8Layout
    dispatch.QuantizedTensor = FakeQuantized
    dispatch.TensorWiseINT8Layout = FakeLayout
    try:
        test_fifty_blocks_share_one_execution_signature()
        test_install_replaces_fifty_block_closures_with_one_dispatcher()
    finally:
        dispatch.QuantizedTensor = original_quantized
        dispatch.TensorWiseINT8Layout = original_layout
    print("shared block dispatcher CPU tests passed")
