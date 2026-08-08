"""CPU-only contracts for the fused-QKV benchmark's geometry mode."""

import os
import sys
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "benchmarks"))

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
    test_compilation_warmup_is_not_measured()
    test_invalid_geometry_rejected()
    test_routed_dispatch_uses_production_backend_contracts()
    print("\nall fused QKV benchmark tests passed")


if __name__ == "__main__":
    main()
