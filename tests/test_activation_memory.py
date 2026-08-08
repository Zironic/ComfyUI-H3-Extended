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
from h3_activation_memory.config import MODE_NATIVE, ActivationMemoryConfig  # noqa: E402
from h3_activation_memory.forward import make_forward  # noqa: E402
from h3_activation_memory.linear import HeldMLP  # noqa: E402
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
