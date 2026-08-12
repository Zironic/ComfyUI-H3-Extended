"""Contracts for the runnable combined Hybrid Sparse experiment."""

import os
import sys
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _PACK)
sys.path.insert(0, _ROOT)

sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import h3_sparse_attention.nodes as experimental  # noqa: E402


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


class FakeModel:
    def __init__(self, options=None):
        self.model_options = dict(options or {})
        self.clone_calls = 0

    def clone(self):
        self.clone_calls += 1
        return FakeModel(self.model_options)


def main():
    print("runnable Hybrid Sparse experiment")
    schema = experimental.MiniMaxH3HybridSparseAttention.define_schema()
    ids = [item.id for item in schema.inputs]
    check(
        not bool(getattr(schema, "is_deprecated", False)),
        "combined node is an active experiment rather than a deprecated shim",
    )
    check(
        ids == [
            "model",
            "enabled",
            "mode",
            "video_budget",
            "strict",
            "activation",
            "chunk_rows",
            "run_tag",
            "timing",
            "compile_backend",
            "density_mode",
            "min_video_density",
            "max_video_density",
            "adaptive_temperature",
            "adaptive_target_mass",
        ],
        "historical widget order is preserved",
    )
    check(
        "H3 adaptive sparse" in schema.search_aliases,
        "experimental routing aliases are published",
    )

    source = FakeModel()
    captured = {}

    class FakeHybridConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            captured["hybrid_config"] = self

    class FakeOptimizerConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            captured["optimizer_config"] = self

    class FakeBackend:
        runtime_listeners = ()

        def __init__(self, config, **kwargs):
            captured["backend_config"] = config
            captured["backend_kwargs"] = kwargs
            self.projector = object()

    class FakeDecision:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            captured["decision"] = self

    def fake_apply(model, config, decision):
        captured["applied_model"] = model
        captured["applied_config"] = config
        captured["applied_decision"] = decision

    with mock.patch.object(
        experimental, "HybridSparseConfig", FakeHybridConfig
    ), mock.patch.object(
        experimental, "MemoryOptimizerConfig", FakeOptimizerConfig
    ), mock.patch.object(
        experimental, "HybridStatsCollector", lambda *args: ("collector", args)
    ), mock.patch.object(
        experimental.RuntimeEnvironment,
        "detect",
        return_value=SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
        ),
    ), mock.patch.object(
        experimental,
        "preflight_sparse_sage",
        return_value=SimpleNamespace(),
    ), mock.patch.object(
        experimental, "HybridSparseBackend", FakeBackend
    ), mock.patch.object(
        experimental, "AttentionDecision", FakeDecision
    ), mock.patch.object(
        experimental, "apply", side_effect=fake_apply
    ), mock.patch.object(
        experimental, "_output_root", return_value="reports"
    ):
        result = experimental.MiniMaxH3HybridSparseAttention.execute(
            source,
            mode="sage128_fused_qkv",
            video_budget=0.37,
            strict=False,
            activation="mlp_chunked_native",
            chunk_rows=4096,
            run_tag="adaptive-test",
            timing=True,
            density_mode="adaptive_budget",
            min_video_density=0.08,
            max_video_density=0.75,
            adaptive_temperature=1.5,
            adaptive_target_mass=0.9,
        )

    hybrid = captured["hybrid_config"]
    optimizer = captured["optimizer_config"]
    check(
        hybrid.density_mode == "adaptive_budget"
        and hybrid.video_budget == 0.37
        and hybrid.min_video_density == 0.08
        and hybrid.max_video_density == 0.75
        and hybrid.adaptive_temperature == 1.5
        and hybrid.adaptive_target_mass == 0.9,
        "adaptive routing controls reach HybridSparseConfig",
    )
    check(
        hybrid.run_tag == "adaptive-test" and hybrid.timing is True,
        "timing and run-tag reporting remain active",
    )
    check(
        optimizer.activation == "mlp_chunked_native"
        and optimizer.chunk_rows == 4096,
        "legacy MLP controls reach the monolithic optimizer",
    )
    check(
        source.clone_calls == 1
        and captured["applied_model"] is result.args[0],
        "the experiment clones once and applies the coordinated transaction",
    )

    compile_source = FakeModel()
    with mock.patch.object(
        experimental, "HybridSparseConfig", FakeHybridConfig
    ), mock.patch.object(
        experimental, "MemoryOptimizerConfig", FakeOptimizerConfig
    ), mock.patch.object(
        experimental, "HybridStatsCollector", lambda *args: object()
    ), mock.patch.object(
        experimental.RuntimeEnvironment,
        "detect",
        return_value=SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
        ),
    ), mock.patch.object(
        experimental,
        "preflight_sparse_sage",
        return_value=SimpleNamespace(),
    ), mock.patch.object(
        experimental, "HybridSparseBackend", FakeBackend
    ), mock.patch.object(
        experimental, "AttentionDecision", FakeDecision
    ), mock.patch.object(
        experimental, "apply"
    ), mock.patch.object(
        experimental, "request_shared_block_compile"
    ) as compile_request, mock.patch.object(
        experimental, "_output_root", return_value="reports"
    ):
        experimental.MiniMaxH3HybridSparseAttention.execute(
            compile_source,
            mode="sage128_fused_qkv",
            activation="mlp_chunked_convrot_2slice",
            compile_backend="inductor",
            density_mode="fixed",
            timing=False,
        )
    check(
        compile_request.call_count == 1,
        "valid shared Inductor configuration requests compilation",
    )

    conflict = FakeModel(
        {experimental.PRODUCTION_PLAN_KEY: object()}
    )
    try:
        experimental.MiniMaxH3HybridSparseAttention.execute(conflict)
    except ValueError as exc:
        check(
            "cannot be combined" in str(exc),
            "production-node composition fails before cloning or GPU probing",
        )
    else:
        raise AssertionError("production plan was accepted by experiment")
    check(conflict.clone_calls == 0, "conflict rejection is mutation-free")

    print("\nall runnable Hybrid Sparse experiment tests passed")


if __name__ == "__main__":
    main()
