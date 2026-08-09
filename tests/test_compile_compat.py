"""CPU-only contracts for H3's selective Inductor backend."""

import os
import sys
from types import SimpleNamespace

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

from h3_runtime import compile_compat  # noqa: E402
from h3_runtime.context import RUNTIME_KEY, RuntimeSnapshot  # noqa: E402
from h3_runtime.timing import publish_timing, timed_stage  # noqa: E402
from h3_attention.hybrid.stats import DeferredCudaTiming  # noqa: E402
from h3_activation_memory.linear import (  # noqa: E402
    _convrot_fc1_tiles_op,
    _convrot_fc2_tiles_op,
    _convrot_linear_op,
)
import h3_activation_memory.linear as linear_module  # noqa: E402
from h3_activation_memory.config import (  # noqa: E402
    MODE_CONVROT_2SLICE,
    ActivationMemoryConfig,
)
from h3_activation_memory.forward import make_forward as make_activation_forward  # noqa: E402
from h3_attention.hybrid.fused_qkv import fused_qkv_op  # noqa: E402
import h3_attention.hybrid.fused_qkv as fused_qkv_module  # noqa: E402
from h3_attention.forward import make_forward as make_attention_forward  # noqa: E402
from h3_attention.hybrid.backend import HybridSparseBackend  # noqa: E402
from h3_attention.hybrid.config import (  # noqa: E402
    MODE_SAGE128_FUSED_QKV,
    HybridSparseConfig,
)
from h3_attention.hybrid.fused_qkv import FusedQKVProjector  # noqa: E402
from h3_attention.hybrid.router import SparseTileRouter  # noqa: E402
from h3_attention.hybrid.sparse_sage import (  # noqa: E402
    SparseSageKernelSpec,
    prepare_sparse_sage_v_op,
    sparse_sage_attention_op,
)


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("  ok: %s" % message)


def test_cpu_graph_stays_eager():
    print("CPU graph stays eager")
    graph_module = torch.fx.symbolic_trace(lambda x: x + 1)
    x = torch.tensor([2.0])
    original = torch._inductor.compile
    torch._inductor.compile = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("CPU graph must not reach Inductor")
    )
    try:
        callable_graph = compile_compat.cuda_only_inductor(graph_module, [x])
        check(torch.equal(callable_graph(x), torch.tensor([3.0])), "CPU graph executes eagerly")
    finally:
        torch._inductor.compile = original


def test_device_scan_ignores_unused_offloaded_weights():
    print("graph device scan")
    graph_module = torch.fx.symbolic_trace(lambda x: x + 1)
    graph_module.register_parameter(
        "unused_cpu_weight",
        torch.nn.Parameter(torch.ones(1), requires_grad=False),
    )
    mode = FakeTensorMode()
    cuda_input = FakeTensor(
        mode,
        torch.empty((2,), device="meta"),
        "cuda",
    )
    check(
        compile_compat.graph_tensor_devices(graph_module, [cuda_input])
        == frozenset(("cuda",)),
        "unused offloaded CPU parameters do not taint a CUDA graph",
    )

    class UsesCpuWeight(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)

        def forward(self, x):
            return x + self.weight

    referenced = torch.fx.symbolic_trace(UsesCpuWeight())
    check(
        compile_compat.graph_tensor_devices(referenced, [cuda_input])
        == frozenset(("cpu", "cuda")),
        "referenced CPU tensors still classify the graph as mixed",
    )


def test_cuda_graph_reaches_inductor_without_cuda():
    print("CUDA graph reaches Inductor")
    graph_module = torch.fx.symbolic_trace(lambda x: x + 1)
    calls = []
    original_devices = compile_compat.graph_tensor_devices
    original_compile = torch._inductor.compile
    compile_compat.graph_tensor_devices = lambda graph, inputs: frozenset(("cuda",))
    def compile_graph(graph, inputs):
        calls.append((
            graph,
            inputs,
            torch._inductor.config.triton.cudagraphs,
            torch._inductor.config.triton.cudagraph_trees,
        ))
        return graph.forward

    torch._inductor.compile = compile_graph
    try:
        callable_graph = compile_compat.cuda_only_inductor(graph_module, [torch.tensor([2.0])])
        check(torch.equal(callable_graph(torch.tensor([2.0])), torch.tensor([3.0])), "mock CUDA graph executes")
        check(len(calls) == 1, "CUDA-only graph is sent to Inductor once")
        check(calls[0][2:] == (False, False), "H3 Inductor graphs disable CUDA graph capture")
    finally:
        compile_compat.graph_tensor_devices = original_devices
        torch._inductor.compile = original_compile


def test_shared_block_compile_request_and_configuration():
    print("shared block compile configuration")

    class Patcher:
        def __init__(self):
            self.model_options = {"disable_dynamic_vbar_prefetch": True}
            self.removed = []
            self.wrappers = {}

        def remove_wrappers_with_key(self, wrapper_type, key):
            self.removed.append((wrapper_type, key))

        def get_wrappers(self, wrapper_type, key):
            return self.wrappers.get((wrapper_type, key), [])

    model = Patcher()
    compile_compat.request_shared_block_compile(model)
    kwargs = model.model_options[compile_compat.REQUEST_MARKER]
    check(kwargs["backend"] == "inductor", "H3 requests Inductor directly")
    check(kwargs["fullgraph"] and kwargs["dynamic"] is False,
          "shared block requests one static full graph")

    calls = []
    original = compile_compat.install_shared_block_dispatch
    compile_compat.install_shared_block_dispatch = lambda *args: calls.append(args)
    try:
        check(
            compile_compat.configure_shared_block_inductor(
                model,
                backend="hybrid",
                activation_config="convrot",
            ),
            "requested shared block compiler is installed",
        )
        check(len(calls) == 1, "shared dispatcher is installed once")
        check(calls[0][3] is compile_compat.cuda_only_inductor,
              "shared dispatcher uses the CUDA-only Inductor backend")
        check("disable_dynamic_vbar_prefetch" not in model.model_options,
              "eager H3 loop restores dynamic VBAR prefetch")
        check(
            compile_compat.configure_shared_block_inductor(
                model,
                backend="hybrid",
                activation_config="convrot",
            ),
            "configured shared block compiler remains recognized",
        )
        check(len(calls) == 1, "shared dispatcher is not installed twice")
        try:
            compile_compat.configure_shared_block_inductor(
                model,
                backend="changed-hybrid",
                activation_config="convrot",
            )
        except RuntimeError as exc:
            check("configured differently" in str(exc),
                  "changed compile configuration is rejected")
        else:
            raise AssertionError("changed compile configuration retained stale dispatcher")
    finally:
        compile_compat.install_shared_block_dispatch = original

    stock = Patcher()
    stock.model_options[compile_compat.TORCH_COMPILE_KWARGS] = {"backend": "inductor"}
    try:
        compile_compat.request_shared_block_compile(stock)
    except RuntimeError as exc:
        check("Remove TorchCompileModel" in str(exc),
              "stock whole-model compile is rejected explicitly")
    else:
        raise AssertionError("stock TorchCompileModel was accepted")


def test_convrot_ops_export_in_one_graph():
    print("ConvRot operator graph")

    def forward(x, fc1_qdata, fc1_scale, fc2_qdata, fc2_scale):
        w10, s10, _w11, _s11 = _convrot_fc1_tiles_op(fc1_qdata, fc1_scale)
        w20, _w21, s2 = _convrot_fc2_tiles_op(fc2_qdata, fc2_scale)
        expanded = _convrot_linear_op(x, w10, s10, None)
        return _convrot_linear_op(expanded, w20, s2, "swiglu")

    args = (
        torch.empty((4, 256), device="meta", dtype=torch.bfloat16),
        torch.empty((1024, 256), device="meta", dtype=torch.int8),
        torch.empty((1024,), device="meta", dtype=torch.float32),
        torch.empty((256, 512), device="meta", dtype=torch.int8),
        torch.empty((256,), device="meta", dtype=torch.float32),
    )
    graph_module, _guards = torch._dynamo.export(forward)(*args)
    graph = str(graph_module.graph)
    check("minimax_h3.convrot_fc1_tiles" in graph, "fc1 tile preparation stays in the graph")
    check("minimax_h3.convrot_fc2_tiles" in graph, "fc2 tile preparation stays in the graph")
    check(graph.count("minimax_h3.convrot_linear") == 2, "both ConvRot kernels stay in the graph")


def test_sparse_attention_ops_export_in_one_graph():
    print("fused sparse-attention operator graph")
    heads = 2
    sequence = 256
    hidden = 256
    router = SparseTileRouter()
    layout = SimpleNamespace(
        seq_len=sequence,
        video_range=(128, sequence),
        segments=((0, 64, "text"), (64, 128, "audio"), (128, sequence, "video")),
        video_shape=(1, 1, 128),
        audio_t=64,
    )

    def forward(x, qdata, weight_scale, q_norm, k_norm, rope, threshold):
        q, q_scale, k, k_scale, v, q_summary, k_summary = fused_qkv_op(
            x, qdata, weight_scale, q_norm, k_norm, rope,
            heads, 1e-6, True, [heads * 128 * 4, 4, 2, 1],
        )
        lut, valid, _metadata = router.build_lut_from_summaries(
            q_summary, k_summary, layout, 0.1
        )
        v_fp8, v_scale = prepare_sparse_sage_v_op(v)
        return sparse_sage_attention_op(
            q, k, v_fp8, lut, valid, threshold,
            q_scale, k_scale, v_scale, torch.bfloat16,
        )

    args = (
        torch.empty((sequence, hidden), device="meta", dtype=torch.bfloat16),
        torch.empty((heads * 128 * 3, hidden), device="meta", dtype=torch.int8),
        torch.empty((heads * 128 * 3,), device="meta", dtype=torch.float32),
        torch.empty((128,), device="meta", dtype=torch.bfloat16),
        torch.empty((128,), device="meta", dtype=torch.bfloat16),
        torch.empty((1, sequence, 1, 48, 2, 2), device="meta", dtype=torch.bfloat16),
        torch.empty((heads,), device="meta", dtype=torch.float32),
    )
    graph_module, _guards = torch._dynamo.export(forward)(*args)
    graph = str(graph_module.graph)
    check("minimax_h3.fused_qkv" in graph, "fused QKV stays in the graph")
    check(
        "matmul" in graph
        and "topk" in graph
        and "minimax_h3.sort_selected_indices" in graph,
        "sparse routing stays in the graph",
    )
    check("aten.sort" not in graph, "router sort avoids pathological Inductor lowering")
    check("minimax_h3.prepare_sparse_sage_v" in graph, "V preparation stays in the graph")
    check("minimax_h3.sparse_sage_attention" in graph, "Sparse Sage stays in the graph")


def test_stage_timing_does_not_fragment_compilation():
    print("compiled timing boundary")
    timing = DeferredCudaTiming(
        True,
        event_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("compiled stages must not allocate events")
        ),
    )
    timing.begin_request(0, cuda=True)
    options = {}
    publish_timing(options, timing)

    def forward(x):
        with timed_stage(options, "adaln_proj"):
            return x.sin() + 1

    result = torch._dynamo.explain(forward)(torch.randn(8))
    check(result.graph_count == 1 and result.graph_break_count == 0, "stage timing leaves one tensor graph")
    check(not timing._samples["adaln_proj"], "compiled stage timing is measured at the outer model boundary")


def test_active_convrot_block_reaches_operator_graphs():
    print("active ConvRot block graph")

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

    class FakeLinear:
        def __init__(self, weight):
            self.weight = weight
            self.bias = None

    class FakeMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = FakeLinear(FakeQuantized(
                torch.empty((1024, 256), device="meta", dtype=torch.int8),
                torch.empty((1024, 1), device="meta", dtype=torch.float32),
            ))
            self.fc2 = FakeLinear(FakeQuantized(
                torch.empty((256, 512), device="meta", dtype=torch.int8),
                torch.empty((256, 1), device="meta", dtype=torch.float32),
            ))

    class FakeBlock:
        def __init__(self):
            self.mlp = FakeMLP()
            self.adaln_proj = lambda t: tuple(t * 0 for _ in range(6))
            self.norm1 = lambda x: x * 1
            self.norm2 = lambda x: x * 1
            self.attn = lambda x, **kwargs: x * 0

        def forward(self, *args, **kwargs):
            raise AssertionError("patched forward must be used")

    block = FakeBlock()
    config = ActivationMemoryConfig(
        mode=MODE_CONVROT_2SLICE,
        chunk_rows=4096,
        alignment=256,
    )
    forward = make_activation_forward(block, 0, config)
    graphs = []

    def backend(graph_module, example_inputs):
        graphs.append(str(graph_module.graph))
        return graph_module.forward

    original_cast = comfy.ops.cast_bias_weight
    original_quantized = linear_module.QuantizedTensor
    original_layout = linear_module.TensorWiseINT8Layout
    comfy.ops.cast_bias_weight = lambda module, sample, **kwargs: (
        module.weight,
        None,
        None,
    )
    linear_module.QuantizedTensor = FakeQuantized
    linear_module.TensorWiseINT8Layout = FakeLayout
    try:
        compiled = torch.compile(forward, backend=backend, fullgraph=False)
        result = compiled(
            torch.empty((3, 256), device="meta", dtype=torch.bfloat16),
            torch.empty((1, 256), device="meta", dtype=torch.bfloat16),
            [(0, 3, 0)],
            None,
            transformer_options={},
        )
    finally:
        comfy.ops.cast_bias_weight = original_cast
        linear_module.QuantizedTensor = original_quantized
        linear_module.TensorWiseINT8Layout = original_layout
        torch._dynamo.reset()

    graph = "\n".join(graphs)
    check(result.shape == (3, 256), "active block preserves its output shape")
    check(len(graphs) == 1, "active block tensor work stays in one graph (got %d)" % len(graphs))
    check("minimax_h3.convrot_fc1_module" in graph, "active block captures fc1 acquisition and tiling")
    check("minimax_h3.convrot_fc2_module" in graph, "active block captures fc2 acquisition and tiling")
    check(graph.count("minimax_h3.convrot_linear") == 4, "active block captures both linears for both slices")


def test_active_hybrid_attention_reaches_operator_graphs():
    print("active Hybrid Sparse attention graph")
    heads = 2
    sequence = 256
    hidden = heads * 128
    mode = FakeTensorMode(allow_non_fake_inputs=True)

    def fake(shape, dtype):
        return FakeTensor(mode, torch.empty(shape, device="meta", dtype=dtype), "cuda:0")

    qdata = fake((hidden * 3, hidden), torch.int8)
    weight_scale = fake((hidden * 3,), torch.float32)
    norm = fake((128,), torch.bfloat16)
    x = fake((sequence, hidden), torch.bfloat16)
    rope = fake((1, sequence, 1, 48, 2, 2), torch.bfloat16)
    layout = SimpleNamespace(
        seq_len=sequence,
        video_range=(128, sequence),
        segments=((0, 64, "text"), (64, 128, "audio"), (128, sequence, "video")),
        video_shape=(1, 1, 128),
        audio_t=64,
    )
    options = {
        RUNTIME_KEY: RuntimeSnapshot(
            request_id=0,
            step_index=0,
            total_steps=20,
            sigma=1.0,
            branch=(0,),
            layout=layout,
            layout_signature=(sequence,),
            compute_dtype=torch.bfloat16,
            device=torch.device("cuda:0"),
        )
    }
    class FakeAttention:
        pass

    module = FakeAttention()
    module.heads = heads
    module.head_dim = 128
    module.qkv_proj = object()
    module.q_norm = SimpleNamespace(weight=norm, eps=1e-6)
    module.k_norm = SimpleNamespace(weight=norm, eps=1e-6)
    module.out_proj = lambda value: value
    api = SparseSageKernelSpec(
        version="test",
        architecture="sm89",
        capability=(8, 9),
        q_tile=128,
        kv_tile=64,
        v_format="fp8",
        kernel=lambda *args: None,
        accumulator="f32",
        fused_v_ops=object(),
        kernel_name="fake_sparse_kernel",
    )
    backend = HybridSparseBackend(
        HybridSparseConfig(
            mode=MODE_SAGE128_FUSED_QKV,
            video_budget=0.1,
            strict=True,
            timing=False,
        ),
        kernel_spec=api,
        projector=FusedQKVProjector(),
    )
    forward = make_attention_forward(
        module,
        0,
        backend=backend,
        projector=backend.projector,
    )
    graphs = []

    def graph_backend(graph_module, example_inputs):
        graphs.append(str(graph_module.graph))
        return graph_module.forward

    original_capability = torch.cuda.get_device_capability
    torch.cuda.get_device_capability = lambda device=None: (8, 9)
    torch._dynamo.utils.counters.clear()
    try:
        with mode:
            compiled = torch.compile(forward, backend=graph_backend, fullgraph=False)
            result = compiled(x, rope, transformer_options=options)
        graph_breaks = sum(
            torch._dynamo.utils.counters.get("graph_break", {}).values()
        )
    finally:
        torch.cuda.get_device_capability = original_capability
        torch._dynamo.reset()

    graph = "\n".join(graphs)
    check(result.shape == (sequence, hidden), "active attention preserves its output shape")
    check(len(graphs) == 1, "active attention tensor path is one graph (got %d)" % len(graphs))
    check(graph_breaks == 0, "active attention introduces no graph breaks")
    check("minimax_h3.fused_qkv_module" in graph, "active attention captures fused QKV")
    check(
        "matmul" in graph
        and "topk" in graph
        and "minimax_h3.sort_selected_indices" in graph,
        "active attention captures sparse routing",
    )
    check("aten.sort" not in graph, "active router avoids Inductor sort lowering")
    check("minimax_h3.prepare_sparse_sage_v" in graph, "active attention captures V preparation")
    check("minimax_h3.sparse_sage_attention" in graph, "active attention captures Sparse Sage")


def main():
    test_cpu_graph_stays_eager()
    test_device_scan_ignores_unused_offloaded_weights()
    test_cuda_graph_reaches_inductor_without_cuda()
    test_shared_block_compile_request_and_configuration()
    test_convrot_ops_export_in_one_graph()
    test_sparse_attention_ops_export_in_one_graph()
    test_stage_timing_does_not_fragment_compilation()
    test_active_convrot_block_reaches_operator_graphs()
    test_active_hybrid_attention_reaches_operator_graphs()
    print("\nall H3 compile-compat tests passed")


if __name__ == "__main__":
    main()
