"""CPU-only contracts for the fused-QKV benchmark's geometry mode."""

import os
import sys
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "benchmarks"))

sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

import bench_fused_qkv as benchmark  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("  ok: %s" % message)


def test_legacy_sequence_resolution():
    sequence, layout = benchmark.resolve_sequence(None, None, 1344, 768, 256)
    check(sequence == 54006, "sequence-only mode keeps the legacy 54006 default")
    check(layout is None, "sequence-only mode does not synthesize geometry")


def test_geometry_resolution_and_layout_contract():
    sequence, layout = benchmark.resolve_sequence(None, 22, 1344, 768, 256)
    check(layout is not None, "--frames creates a TokenLayout")
    check(sequence == layout.seq_len, "geometry sequence comes from the packed layout")
    check(layout.video_range[1] == sequence, "target video remains the final packed segment")
    check(layout.video_shape == (7, 24, 42), "layout uses the MiniMax 1x2x2 patch geometry")
    check(layout.audio_t == (layout.audio_range[1] - layout.audio_range[0]) // 2,
          "audio_t follows the channel-major packed rows")
    sequence_209, _layout_209 = benchmark.resolve_sequence(None, 209, 1344, 768, 256)
    check(sequence_209 == 63448, "209-frame 1 MP geometry resolves to 63448 tokens")


def test_conflicting_sequence_rejected():
    try:
        benchmark.resolve_sequence(54006, 22, 1344, 768, 256)
    except ValueError as exc:
        check("conflicts" in str(exc), "explicit conflicting sequence fails clearly")
    else:
        raise AssertionError("expected a conflicting --sequence to be rejected")


def test_compile_factory_uses_static_full_graph():
    calls = []
    compiled = object()

    class FakeTorch:
        @staticmethod
        def compile(core, **kwargs):
            calls.append((core, kwargs))
            return compiled

    core = object()
    check(benchmark.compile_fused_qkv_core(FakeTorch, core) is compiled,
          "compile flag selects the injected tensor core")
    check(calls == [(core, {"fullgraph": True, "dynamic": False})],
          "fused compile factory requests a static full graph")


def test_benchmark_launch_config_is_explicit():
    calls = []

    def core(*args, **kwargs):
        calls.append((args, kwargs))
        return "configured"

    configured = benchmark.bind_fused_qkv_launch_config(
        core, {"v_block_m": 64, "v_num_warps": 4},
    )
    check(configured("x", heads=2) == "configured",
          "benchmark launch binding preserves the core return")
    check(calls == [(("x",), {
        "heads": 2,
        "v_block_m": 64,
        "v_num_warps": 4,
    })], "benchmark launch binding forwards only the selected overrides")
    check(benchmark.LAUNCH_CONFIGS["production"] == {},
          "production benchmark case uses the core defaults")


def test_sm89_sweep_covers_requested_per_kind_space():
    q = benchmark.sm89_launch_sweep_candidates("q")
    k = benchmark.sm89_launch_sweep_candidates("k")
    v = benchmark.sm89_launch_sweep_candidates("v")
    check(len(q) == 24 and len(k) == 24,
          "Q/K sweep every requested K, warp, and stage combination")
    check(len(v) == 216,
          "V sweep covers M, wider N, K, warp, and stage combinations")
    check({config["q_block_k"] for _, config in q} == {32, 64, 128}
          and {config["q_num_warps"] for _, config in q} == {4, 8}
          and {config["q_num_stages"] for _, config in q} == {2, 3, 4, 5},
          "Q sweep spans the full requested dimensions")
    check({config["v_block_m"] for _, config in v} == {32, 64, 128}
          and {config["v_block_n"] for _, config in v} == {128, 256, 512}
          and {config["v_block_k"] for _, config in v} == {32, 64, 128},
          "V sweep spans the requested M/K and wider N dimensions")
    try:
        benchmark.sm89_launch_sweep_candidates("bad")
    except ValueError as exc:
        check("q, k, or v" in str(exc), "invalid sweep kind fails clearly")
    else:
        raise AssertionError("expected invalid launch sweep kind to fail")


def test_launch_candidates_require_exact_carriers():
    carriers = tuple(
        benchmark.torch.arange(4, dtype=benchmark.torch.float32)
        for _ in range(7)
    )
    exact = benchmark.require_exact_carriers(carriers, carriers)
    check(all(exact.values()), "identical launch carriers pass the parity gate")
    changed = list(carriers)
    changed[4] = changed[4] + 1
    try:
        benchmark.require_exact_carriers(tuple(changed), carriers)
    except RuntimeError as exc:
        check("v" in str(exc), "changed V carrier fails with its field name")
    else:
        raise AssertionError("expected changed launch carrier to fail")


def test_cuda_trace_counts_only_kernel_activities():
    class Interval:
        def __init__(self, duration_us):
            self.duration_us = duration_us

        def elapsed_us(self):
            return self.duration_us

    events = [
        SimpleNamespace(
            device_type=benchmark.torch.autograd.DeviceType.CUDA,
            activity_type="kernel",
            name="qk",
            time_range=Interval(2500),
            event_metadata=None,
        ),
        SimpleNamespace(
            device_type=benchmark.torch.autograd.DeviceType.CUDA,
            activity_type="ActivityType.CONCURRENT_KERNEL",
            name="qk",
            time_range=Interval(1500),
            event_metadata=None,
        ),
        SimpleNamespace(
            device_type=benchmark.torch.autograd.DeviceType.CUDA,
            activity_type="ActivityType.MEMCPY",
            name="copy",
            time_range=Interval(100),
            event_metadata=None,
        ),
        SimpleNamespace(
            device_type=benchmark.torch.autograd.DeviceType.CPU,
            activity_type="ActivityType.CONCURRENT_KERNEL",
            name="cpu",
            time_range=Interval(100),
            event_metadata=None,
        ),
    ]
    summary = benchmark.summarize_cuda_trace(events)
    check(summary["kernel_launches"] == 2,
          "CUDA trace counts only concurrent kernel activities")
    check(summary["kernels"] == [{
        "name": "qk", "count": 2, "device_time_ms": 4.0,
    }],
          "CUDA trace aggregates launch names deterministically")
    check(summary["cuda_activities"]["ActivityType.MEMCPY"] == 1,
          "CUDA trace reports non-kernel device activities separately")


def test_sparse_compile_factory_uses_static_full_graph():
    calls = []
    compiled = object()

    class FakeTorch:
        @staticmethod
        def compile(core, **kwargs):
            calls.append((core, kwargs))
            return compiled

    core = object()
    check(benchmark.compile_sparse_sage_kernel_core(FakeTorch, core) is compiled,
          "Sparse Sage compile factory selects the injected adapter")
    check(calls == [(core, {"fullgraph": True, "dynamic": False})],
          "Sparse Sage compile factory requests a static full graph")


def test_sparse_adapter_owns_output_and_preserves_exact_abi():
    calls = []

    def fake_kernel(*args):
        calls.append(args)
        output = args[3]
        output.fill_(7)
        return "ignored return value"

    tensors = tuple(
        benchmark.torch.empty((1,), dtype=benchmark.torch.float32)
        for _ in range(9)
    )
    adapter = benchmark.make_sparse_sage_kernel_adapter(
        fake_kernel, (1, 2, 3, 4), benchmark.torch.float32,
    )
    output = adapter(*tensors)
    check(output.shape == (1, 2, 3, 4), "adapter allocates the executor output shape")
    check(output.dtype == benchmark.torch.float32 and float(output.min()) == 7,
          "adapter returns the kernel-mutated output")
    expected_carrier = tensors[:3] + (calls[0][3],) + tensors[3:]
    check(len(calls) == 1 and all(actual is expected for actual, expected in zip(calls[0][:10], expected_carrier)),
          "adapter forwards the nine prepared carrier tensors in order")
    check(calls[0][10:] == (1, 0, 1, 128 ** -0.5, 0),
          "adapter forwards the exact Sparse Sage ABI flags")


def test_compile_sage_requires_geometry():
    try:
        benchmark.validate_compile_sage_request(True, None)
    except ValueError as exc:
        check("--frames" in str(exc),
              "Sparse Sage compilation outside geometry fails clearly")
    else:
        raise AssertionError("expected --compile-sage without geometry to fail")
    benchmark.validate_compile_sage_request(False, None)
    check(True, "disabled Sparse Sage compilation preserves sequence-only mode")


def test_compilation_warmup_is_not_measured():
    calls = []
    original_compile_warmup = benchmark._compile_warmup
    original_benchmark_case = benchmark.benchmark_case
    try:
        benchmark._compile_warmup = lambda fn, device: calls.append(
            ("compile", fn, device)
        ) or 123.0
        benchmark.benchmark_case = lambda fn, warmup, iterations: calls.append(
            ("measure", fn, warmup, iterations)
        ) or {"median_ms": 4.0, "min_ms": 3.0, "peak_bytes": 5}
        fn = object()
        result = benchmark.benchmark_compiled_case(fn, 0, 3, "cuda")
    finally:
        benchmark._compile_warmup = original_compile_warmup
        benchmark.benchmark_case = original_benchmark_case
    check(calls == [
        ("compile", fn, "cuda"),
        ("measure", fn, 0, 3),
    ], "first compilation is separate even when benchmark warmup is zero")
    check(result["compile_warmup_ms"] == 123.0,
          "first-call compile cost is reported separately")


def test_invalid_geometry_rejected():
    for values, text in (
        ((0, 1344, 768, 256), "frames"),
        ((23, 1344, 768, 256), "frames % 17"),
        ((22, 1333, 768, 256), "multiples of 32"),
        ((22, 1344, 768, -1), "text-len"),
    ):
        try:
            benchmark.resolve_sequence(None, *values)
        except ValueError as exc:
            check(text in str(exc), "invalid geometry reports %s" % text)
        else:
            raise AssertionError("expected invalid geometry to fail")


def test_projection_case_matrix_preserves_exact_boundaries():
    calls = []
    x = SimpleNamespace(shape=(17, 256), dtype="bf16")
    weight_scale = benchmark.torch.ones((3, 1))
    x_scale = benchmark.torch.ones((17, 1))

    def standard():
        calls.append(("standard",))
        return "standard"

    def kitchen_gemm(*args):
        calls.append(("kitchen", args))
        return "kitchen"

    def quantizer(value):
        calls.append(("quantizer", value))
        return "quantized"

    def fused_op(*args):
        calls.append(("fused_op", args))
        return "fused"

    def tensor_core(*args, **kwargs):
        calls.append(("tensor_core", args, kwargs))
        return "prequantized"

    cases = benchmark.make_projection_case_functions(
        standard_fn=standard,
        x=x,
        qdata="qdata",
        weight_scale=weight_scale,
        q_norm="q_norm",
        k_norm="k_norm",
        rope="rope",
        rope_strides=(11, 12, 13, 14),
        heads=2,
        epsilon=1e-6,
        x_int8="x_int8",
        x_scale=x_scale,
        kitchen_gemm=kitchen_gemm,
        quantizer=quantizer,
        fused_op=fused_op,
        tensor_core=tensor_core,
    )
    outputs = {name: fn() for name, fn in cases.items()}
    check(outputs == {
        "A_standard_preparation": "standard",
        "B_raw_kitchen_qkv_gemm": "kitchen",
        "C_fused_with_input_quantization": "fused",
        "D_fused_prequantized": "prequantized",
        "input_quantization": "quantized",
    }, "projection matrix exposes the requested A-D cases and quantizer")

    kitchen = next(call for call in calls if call[0] == "kitchen")
    check(kitchen[1][0] == "x_int8"
          and kitchen[1][1] == "qdata"
          and benchmark.torch.equal(kitchen[1][2], x_scale.reshape(-1))
          and kitchen[1][3] is weight_scale
          and kitchen[1][4] == "bf16",
          "raw Kitchen case receives only prequantized input and held QKV tensors")

    fused = next(call for call in calls if call[0] == "fused_op")
    check(fused[1][0] is x and fused[1][1] == "qdata",
          "fused total case starts from BF16 input")
    core = next(call for call in calls if call[0] == "tensor_core")
    check(core[1][0] == "x_int8"
          and benchmark.torch.equal(core[1][2], x_scale.reshape(-1)),
          "prequantized case bypasses the input quantizer")
    check(core[2] == {
        "heads": 2,
        "sequence": 17,
        "hidden": 256,
        "epsilon": 1e-6,
        "has_rope": True,
        "rope_strides": (11, 12, 13, 14),
        "output_dtype": "bf16",
    }, "prequantized case preserves the tensor-core ABI")
    check(sum(call[0] == "quantizer" for call in calls) == 1,
          "standalone quantizer case is isolated from the prebuilt carrier")
    calls.clear()
    original_benchmark_case = benchmark.benchmark_case
    try:
        benchmark.benchmark_case = lambda fn, warmup, iterations: fn()
        matrix = benchmark.benchmark_projection_cases(
            cases, 0, 1,
            precomputed={"A_standard_preparation": "already measured"},
        )
    finally:
        benchmark.benchmark_case = original_benchmark_case
    check(matrix["A_standard_preparation"]["measurement"] == "already measured"
          and not any(call[0] == "standard" for call in calls),
          "case A reuses the legacy baseline sample instead of running twice")


def test_projection_report_does_not_invent_profiler_metrics():
    capabilities = benchmark.projection_measurement_capabilities()
    check("cuda_event_elapsed_ms" in capabilities["measured"],
          "projection report identifies its direct CUDA timing evidence")
    for metric in (
        "tensor_core_utilization",
        "achieved_memory_bandwidth",
        "achieved_occupancy",
    ):
        check(metric in capabilities["requires_external_profiler"],
              "%s remains explicitly profiler-only" % metric)
    check("registers_per_thread" in capabilities["available_in_profile_case_d"],
          "one-launch mode reports compiler register usage")
    check("kernel_launches" in capabilities["available_with_profile_kernel_launches"],
          "launch trace mode reports measured kernel counts")


def test_routed_dispatch_uses_production_backend_contracts():
    calls = []

    class Timing:
        def begin(self, stage):
            calls.append(("timing_begin", stage))
            return stage

        def end(self, token):
            calls.append(("timing_end", token))

    class Projector:
        def project(self, module, x, rope, *, layer_index, transformer_options):
            calls.append(("projector", module, x, rope, layer_index))
            return "projected"

    class Backend:
        def __init__(self):
            self.timing = Timing()
            self.projector = Projector()

        def prepare(self, q, k, v, *, layer_index, transformer_options):
            calls.append(("prepare", q, k, v, layer_index))
            return SimpleNamespace(sparse=SimpleNamespace(metadata={"path": "established"}))

        def prepare_projected(self, projected, *, layer_index, transformer_options):
            calls.append(("prepare_projected", projected, layer_index))
            return SimpleNamespace(sparse=SimpleNamespace(metadata={"path": "fused"}))

        def execute(self, prepared):
            calls.append(("execute", prepared.sparse.metadata["path"]))
            return prepared.sparse.metadata["path"]

    original_ensure = benchmark._ensure_forward_imports
    original_project = benchmark.project_qkv
    original_to_hnd = benchmark.to_hnd
    try:
        benchmark._ensure_forward_imports = lambda: None
        benchmark.project_qkv = lambda module, x, rope, options: ("q", "k", "v")
        benchmark.to_hnd = lambda q, k, v: ("qh", "kh", "vh")
        backend = Backend()
        runtime = SimpleNamespace()
        output, metadata = benchmark._routed_call(
            backend, "module", "x", "rope", runtime, 7, False,
        )
        check(output == "established" and metadata["path"] == "established",
              "established geometry mode uses prepare and execute")
        check(("prepare", "qh", "kh", "vh", 7) in calls,
              "established geometry mode supplies projected HND Q/K/V")

        calls.clear()
        output, metadata = benchmark._routed_call(
            backend, "module", "x", "rope", runtime, 8, True,
        )
        check(output == "fused" and metadata["path"] == "fused",
              "fused geometry mode uses prepare_projected and execute")
        check(("projector", "module", "x", "rope", 8) in calls,
              "fused geometry mode invokes the backend-owned projector")
        check(("timing_begin", "fused_qkv_projection") in calls,
              "fused projection uses the production timing stage")
    finally:
        benchmark._ensure_forward_imports = original_ensure
        benchmark.project_qkv = original_project
        benchmark.to_hnd = original_to_hnd


def main():
    test_legacy_sequence_resolution()
    test_geometry_resolution_and_layout_contract()
    test_conflicting_sequence_rejected()
    test_compile_factory_uses_static_full_graph()
    test_benchmark_launch_config_is_explicit()
    test_sm89_sweep_covers_requested_per_kind_space()
    test_launch_candidates_require_exact_carriers()
    test_cuda_trace_counts_only_kernel_activities()
    test_sparse_compile_factory_uses_static_full_graph()
    test_sparse_adapter_owns_output_and_preserves_exact_abi()
    test_compile_sage_requires_geometry()
    test_compilation_warmup_is_not_measured()
    test_invalid_geometry_rejected()
    test_projection_case_matrix_preserves_exact_boundaries()
    test_projection_report_does_not_invent_profiler_metrics()
    test_routed_dispatch_uses_production_backend_contracts()
    print("\nall fused QKV benchmark tests passed")


if __name__ == "__main__":
    main()
