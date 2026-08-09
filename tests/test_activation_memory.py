"""CPU self-tests for H3 activation-memory execution.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_activation_memory.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import comfy.ops  # noqa: E402
from comfy.ldm.minimax.model import DiTBlock  # noqa: E402

from h3_activation_memory import chunks  # noqa: E402
from h3_activation_memory.config import (  # noqa: E402
    MODE_CONVROT_2SLICE,
    MODE_NATIVE,
    ActivationMemoryConfig,
)
from h3_activation_memory.forward import make_forward  # noqa: E402
from h3_activation_memory.linear import HeldMLP  # noqa: E402
import h3_activation_memory.forward as forward_module  # noqa: E402
import h3_activation_memory.linear as linear_module  # noqa: E402
from h3_activation_memory.observer import observing  # noqa: E402
from h3_activation_memory import patch  # noqa: E402
from h3_attention.hybrid.stats import DeferredCudaTiming  # noqa: E402
from h3_runtime.timing import publish_timing  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("  ok: %s" % message)


def test_defaults():
    print("defaults")
    config = ActivationMemoryConfig()
    check(config.mode == MODE_NATIVE, "native SwiGLU is the default")
    check(config.chunk_rows == 2048, "2048 rows is the default slab size")
    check(config.prefer_held_weights, "held weights are enabled by default")
    check(
        ActivationMemoryConfig(mode=MODE_CONVROT_2SLICE).convrot_2slice,
        "two-slice ConvRot mode is selectable",
    )


def test_convrot_two_slice_cpu_fake():
    print("two-slice ConvRot fake")

    class FakeQuantized:
        def __init__(self, qdata, scale, **params):
            self.qdata = qdata
            self.scale = scale
            self._layout_cls = "TensorWiseINT8Layout"
            self._params = type("Params", (), params)()

    class FakeLayout:
        @staticmethod
        def get_plain_tensors(weight):
            return weight.qdata, weight.scale

    class FakeLinear:
        def __init__(self, weight):
            self.weight = weight

    hidden, ffn = 256, 512
    torch.manual_seed(27)
    fc1_q = torch.randint(-1, 2, (ffn * 2, hidden), dtype=torch.int8)
    fc2_q = torch.randint(-1, 2, (hidden, ffn), dtype=torch.int8)
    fc1 = FakeLinear(
        FakeQuantized(
            fc1_q,
            torch.ones(ffn * 2),
            transposed=False,
            convrot=True,
            convrot_groupsize=256,
        )
    )
    fc2 = FakeLinear(
        FakeQuantized(
            fc2_q,
            torch.ones(hidden),
            transposed=False,
            convrot=True,
            convrot_groupsize=256,
        )
    )
    mlp = type("MLP", (), {"fc1": fc1, "fc2": fc2})()
    original_cast = comfy.ops.cast_bias_weight
    original_quantized = linear_module.QuantizedTensor
    original_layout = linear_module.TensorWiseINT8Layout
    linear_module.QuantizedTensor = FakeQuantized
    linear_module.TensorWiseINT8Layout = FakeLayout
    comfy.ops.cast_bias_weight = lambda module, sample, **kwargs: (
        module.weight,
        None,
        None,
    )
    def fake_convrot(x, qdata, scale, input_act=None):
        weight = qdata.to(x.dtype)
        if input_act == "swiglu":
            gate, up = x.chunk(2, dim=-1)
            x = torch.nn.functional.silu(gate) * up
        return x @ weight.t()
    try:
        x = torch.randn(3, hidden, dtype=torch.bfloat16) * 0.01
        with linear_module.ConvRotTwoSliceMLP(mlp, x[:1], fake_convrot) as session:
            got, path = session.fc1_fc2(x)
            gate, up = (x @ fc1_q[:ffn].to(torch.bfloat16).t()), (
                x @ fc1_q[ffn:].to(torch.bfloat16).t()
            )
            expected = (torch.nn.functional.silu(gate) * up) @ fc2_q.to(torch.bfloat16).t()
            check(got.shape == (3, hidden), "two-slice output shape is preserved")
            check(torch.allclose(got, expected, atol=0.25, rtol=0.0), "two-slice accumulation matches fake ConvRot")
            check(path == "held_convrot_2slice", "two-slice path is reported")
        check(session.tiles is None, "prepared tiles release at session exit")
    finally:
        comfy.ops.cast_bias_weight = original_cast
        linear_module.QuantizedTensor = original_quantized
        linear_module.TensorWiseINT8Layout = original_layout


def test_convrot_scale_tiles():
    print("two-slice ConvRot scale layouts")
    fc1_q = torch.empty((1024, 256), dtype=torch.int8)
    fc2_q = torch.empty((256, 512), dtype=torch.int8)
    scalar = torch.tensor(0.25)
    scalar_fc1 = linear_module._convrot_fc1_tiles(fc1_q, scalar)
    scalar_fc2 = linear_module._convrot_fc2_tiles(fc2_q, scalar)
    check(scalar_fc1[1].shape == scalar.shape and scalar_fc1[3].shape == scalar.shape,
          "both fc1 tiles preserve a scalar scale")
    check(scalar_fc2[2].shape == scalar.shape,
          "fc2 tiles preserve a scalar scale")
    column_fc1 = linear_module._convrot_fc1_tiles(
        fc1_q, torch.empty((1024, 1))
    )
    column_fc2 = linear_module._convrot_fc2_tiles(
        fc2_q, torch.empty((256, 1))
    )
    check(column_fc1[1].shape == (512, 1) and column_fc1[3].shape == (512, 1),
          "both fc1 tiles preserve per-output scale columns")
    check(column_fc2[2].shape == (256, 1),
          "fc2 tiles preserve per-output scale columns")


def test_convrot_two_slice_rejects_contract():
    print("two-slice ConvRot contract")
    class FakeLinear:
        def __init__(self, weight):
            self.weight = weight

    class BadWeight:
        _layout_cls = "OtherLayout"
        _params = type("Params", (), {"transposed": False, "convrot": True, "convrot_groupsize": 256})()

    original_cast = comfy.ops.cast_bias_weight
    original_quantized = linear_module.QuantizedTensor
    comfy.ops.cast_bias_weight = lambda module, sample, **kwargs: (module.weight, None, None)
    linear_module.QuantizedTensor = BadWeight
    try:
        mlp = type("MLP", (), {"fc1": FakeLinear(BadWeight()), "fc2": FakeLinear(BadWeight())})()
        try:
            with linear_module.ConvRotTwoSliceMLP(mlp, torch.zeros(1, 256, dtype=torch.float32)):
                pass
        except TypeError as exc:
            check("requires BF16" in str(exc), "non-BF16 input is rejected clearly")
        else:
            raise AssertionError("non-BF16 input must be rejected")
        try:
            with linear_module.ConvRotTwoSliceMLP(mlp, torch.zeros(1, 256, dtype=torch.bfloat16)):
                pass
        except TypeError as exc:
            check("TensorWiseINT8Layout" in str(exc), "unsupported layout is rejected clearly")
        else:
            raise AssertionError("unsupported layout must be rejected")
    finally:
        comfy.ops.cast_bias_weight = original_cast
        linear_module.QuantizedTensor = original_quantized


def build_block(seed=0):
    torch.manual_seed(seed)
    block = DiTBlock(
        hidden=32,
        heads=2,
        head_dim=16,
        ffn=48,
        t_dim=24,
        eps=1e-6,
        qk_eps=1e-6,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    for parameter in block.parameters():
        with torch.no_grad():
            parameter.copy_(torch.randn_like(parameter) * 0.03)
        parameter.requires_grad_(False)
    return block


def build_inputs(seed=1):
    torch.manual_seed(seed)
    x = torch.randn(19, 32) * 0.1
    t_emb = torch.randn(1, 24) * 0.1
    segments = [(0, 5, 0), (5, 13, 1), (13, 19, 2)]
    return x, t_emb, segments


def test_convrot_two_slice_forward_dispatch():
    print("two-slice ConvRot forward dispatch")
    calls = []

    class FakeSession:
        def __init__(self, mlp, sample):
            calls.append(("init", tuple(sample.shape)))

        def __enter__(self):
            calls.append(("enter", None))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", None))

        def fc1_fc2(self, h, stage_factory=None):
            calls.append(("chunk", tuple(h.shape)))
            return torch.zeros_like(h), "held_convrot_2slice"

    original = forward_module.ConvRotTwoSliceMLP
    forward_module.ConvRotTwoSliceMLP = FakeSession
    try:
        block = build_block(seed=28)
        x, t_emb, segments = build_inputs(seed=29)
        config = ActivationMemoryConfig(
            mode=MODE_CONVROT_2SLICE,
            chunk_rows=256,
            alignment=256,
        )
        with torch.no_grad():
            got = make_forward(block, 0, config)(
                x, t_emb, segments, rope_freqs=None, transformer_options={}
            )
    finally:
        forward_module.ConvRotTwoSliceMLP = original

    check(got.shape == x.shape, "two-slice forward preserves the block shape")
    check(
        len([item for item in calls if item[0] == "chunk"]) == 3,
        "two-slice session handles every modulation chunk",
    )
    check(calls[-1][0] == "exit", "two-slice session releases after the block")


def test_chunks():
    print("chunk planner")
    segments = [(0, 5, 0), (5, 19, 1)]
    got = list(chunks.iter_mod_chunks(segments, 19, max_rows=8, alignment=4))
    check(
        [(c.start, c.stop, c.mod_row) for c in got]
        == [(0, 5, 0), (5, 13, 1), (13, 19, 1)],
        "chunks stay inside modulation boundaries",
    )
    check(sum(c.rows for c in got) == 19, "chunks cover every row once")

    for bad, phrase in (
        ([(0, 4, 0), (5, 8, 1)], "gap"),
        ([(0, 5, 0), (4, 8, 1)], "overlap"),
    ):
        try:
            chunks.validate_mod_segments(bad, 8)
        except ValueError as exc:
            check(phrase in str(exc), "%s is named" % phrase)
        else:
            raise AssertionError("invalid segments must raise")


def test_block_parity(mode):
    print("block parity: %s" % mode)
    block = build_block(seed=2)
    x, t_emb, segments = build_inputs(seed=3)
    with torch.no_grad():
        want = block.forward(
            x.clone(),
            t_emb,
            segments,
            rope_freqs=None,
            transformer_options={},
        )
        config = ActivationMemoryConfig(
            mode=mode,
            chunk_rows=256,
            alignment=256,
            prefer_held_weights=True,
        )
        got = make_forward(block, 0, config)(
            x.clone(),
            t_emb,
            segments,
            rope_freqs=None,
            transformer_options={},
        )

    check(got.shape == want.shape, "output shape preserved")
    check(
        torch.allclose(got, want, rtol=1e-5, atol=2e-6),
        "chunked block matches core within FP32 tolerance",
    )
    check(torch.isfinite(got).all(), "output remains finite")


def test_multiple_chunks():
    print("multiple MLP slabs")
    block = build_block(seed=4)
    x = torch.randn(700, 32) * 0.1
    t_emb = torch.randn(1, 24) * 0.1
    segments = [(0, 300, 0), (300, 600, 1), (600, 700, 2)]
    seen = []

    config = ActivationMemoryConfig(
        mode="mlp_chunked_bf16",
        chunk_rows=256,
        alignment=256,
    )
    options = {}
    with torch.no_grad(), observing(
        options,
        lambda event, layer, payload: seen.append(
            (event, payload.get("start"), payload.get("stop"))
        ),
    ):
        out = make_forward(block, 7, config)(
            x, t_emb, segments, None, transformer_options=options
        )

    enters = [row for row in seen if row[0] == "mlp_chunk_enter"]
    check(len(enters) == 5, "700 rows over three segments produce five slabs")
    check(out.shape == (700, 32), "large synthetic block completes")


class FakeTimingEvent:
    def __init__(self, clock, events):
        self.clock = clock
        self.events = events
        self.value = None

    def record(self):
        self.clock[0] += 1.0
        self.value = self.clock[0]
        self.events.append("record")

    def synchronize(self):
        self.events.append("synchronize")

    def elapsed_time(self, other):
        return other.value - self.value


def test_deferred_stage_counts():
    print("deferred activation stage timing")
    block = build_block(seed=15)
    x = torch.randn(700, 32) * 0.1
    t_emb = torch.randn(1, 24) * 0.1
    segments = [(0, 300, 0), (300, 600, 1), (600, 700, 2)]
    clock = [0.0]
    events = []
    timing = DeferredCudaTiming(
        True, event_factory=lambda **kwargs: FakeTimingEvent(clock, events)
    )
    timing.begin_request(4, cuda=True)
    options = {}
    publish_timing(options, timing)
    config = ActivationMemoryConfig(
        mode="mlp_chunked_bf16", chunk_rows=256, alignment=256
    )
    with torch.no_grad():
        make_forward(block, 15, config)(
            x, t_emb, segments, None, transformer_options=options
        )
    summary = timing.resolve(1.0)
    check(summary["stages"]["total_dit_block"]["count"] == 1,
          "one total DiT block timing event")
    for stage in ("adaln_proj", "norm1_modulation", "attention_residual_gate"):
        check(summary["stages"][stage]["count"] == 1,
              "%s runs once per block" % stage)
    for stage in ("norm2_modulation", "mlp_fc1", "mlp_swiglu_fc2", "final_mlp_gate"):
        check(summary["stages"][stage]["count"] == 5,
              "%s runs once per MLP chunk" % stage)
    check(events.count("synchronize") == 1,
          "activation timing synchronizes once at request end")


def test_held_mlp():
    print("held MLP")
    block = build_block(seed=5)
    x = torch.randn(8, 32)
    with torch.no_grad(), HeldMLP(block.mlp, x[:1]) as held:
        expanded = held.fc1(x)
        out, path = held.fc2_swiglu(expanded, native=False)
    check(out.shape == (8, 32), "held fc1/fc2 produce the expected shape")
    check(path == "held_bf16_swiglu", "BF16 held path is reported")


class FakePatcher:
    def __init__(self, blocks):
        self.blocks = torch.nn.ModuleList(blocks)
        self.object_patches = {}

    def get_model_object(self, name):
        if name == patch.BLOCKS_ATTR:
            return self.blocks
        raise AttributeError(name)

    def add_object_patch(self, name, value):
        self.object_patches[name] = value


def test_patch_install():
    print("patch install")
    model = FakePatcher([build_block(10), build_block(11)])
    config = ActivationMemoryConfig(
        mode="mlp_chunked_bf16", chunk_rows=256
    )
    check(patch.install(model, config) == 2, "every main block is patched")
    check(
        set(model.object_patches) == {patch.key_for(0), patch.key_for(1)},
        "patch keys own blocks.<i>.forward",
    )
    check(patch.install(model, config) == 0, "same configuration is idempotent")

    other = ActivationMemoryConfig(
        mode="mlp_chunked_native", chunk_rows=256
    )
    try:
        patch.install(model, other)
    except patch.H3ActivationPatchError as exc:
        check("already patched" in str(exc), "reconfiguration is explicit")
    else:
        raise AssertionError("stacked activation-memory configurations must raise")

    partial = FakePatcher([build_block(12), build_block(13)])
    partial.object_patches[patch.key_for(0)] = make_forward(
        partial.blocks[0], 0, config
    )
    try:
        patch.install(partial, config)
    except patch.H3ActivationPatchError as exc:
        check("mixed state" in str(exc), "partial patch state is rejected")
    else:
        raise AssertionError("partial activation-memory patch must raise")

    model = FakePatcher([build_block(14)])
    model.object_patches[patch.key_for(0)] = lambda *args, **kwargs: None
    try:
        patch.install(model, config)
    except patch.H3ActivationPatchError as exc:
        check("another patch" in str(exc), "foreign block patch is rejected")
    else:
        raise AssertionError("foreign block patch must raise")


def main():
    test_defaults()
    test_convrot_two_slice_cpu_fake()
    test_convrot_scale_tiles()
    test_convrot_two_slice_rejects_contract()
    test_convrot_two_slice_forward_dispatch()
    test_chunks()
    test_block_parity("mlp_chunked_bf16")
    test_block_parity("mlp_chunked_native")
    test_multiple_chunks()
    test_deferred_stage_counts()
    test_held_mlp()
    test_patch_install()
    print("\nall H3 activation-memory self-tests passed")


if __name__ == "__main__":
    main()
