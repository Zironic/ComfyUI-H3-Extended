"""Contracts for the deprecated combined Hybrid Sparse adapter."""

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

import h3_sparse_attention.nodes as legacy  # noqa: E402
from h3_sage_optimizations.plan import (  # noqa: E402
    FUSED_QKV_AUTO,
    FUSED_QKV_REQUIRED,
    MLP_MEMORY_LEGACY_NATIVE,
    H3SageOptimizationPlan,
)


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def main():
    print("deprecated Hybrid Sparse compatibility")
    schema = legacy.MiniMaxH3HybridSparseAttention.define_schema()
    ids = [item.id for item in schema.inputs]
    check(schema.is_deprecated, "legacy combined node is marked deprecated")
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
        "legacy widget order is preserved for saved workflows",
    )
    check(
        "legacy H3 sparse" in schema.search_aliases,
        "legacy node publishes migration search aliases",
    )
    mode_input = next(item for item in schema.inputs if item.id == "mode")
    check(
        mode_input.options == ["auto", "sage128", "sage128_fused_qkv"]
        and mode_input.default == "auto",
        "new legacy nodes default to capability-safe fused QKV",
    )
    check(
        legacy._memory_request(
            "auto", "mlp_chunked_native", True, 2048,
        ).fused_qkv == FUSED_QKV_AUTO,
        "fixed-density auto delegates QKV selection to the production resolver",
    )

    environment = SimpleNamespace(capability=(8, 9))
    kernel_spec = SimpleNamespace(capability=(8, 9), q_tile=128, kv_tile=64)
    with mock.patch.object(
        legacy, "get_h3_blocks", return_value=("block",),
    ), mock.patch.object(
        legacy, "inspect_h3_linears", return_value="inventory",
    ), mock.patch.object(
        legacy, "resolve_qkv_provider", return_value=SimpleNamespace(fused=True),
    ) as resolve:
        resolved = legacy._resolve_adaptive_mode(
            object(), "auto", environment, kernel_spec,
        )
    check(
        resolved == "sage128_fused_qkv",
        "adaptive auto selects fused QKV after format and capability resolution",
    )
    kwargs = resolve.call_args.kwargs
    check(
        kwargs["request"] == FUSED_QKV_AUTO
        and kwargs["backend_kind"] == "sparse_sage"
        and kwargs["capability"] == (8, 9)
        and kwargs["sparse_spec"] is kernel_spec,
        "adaptive auto passes the actual Sparse Sage environment to the resolver",
    )
    with mock.patch.object(
        legacy, "get_h3_blocks", return_value=("block",),
    ), mock.patch.object(
        legacy, "inspect_h3_linears", return_value="inventory",
    ), mock.patch.object(
        legacy, "resolve_qkv_provider", return_value=SimpleNamespace(fused=False),
    ):
        fallback = legacy._resolve_adaptive_mode(
            object(), "auto", environment, kernel_spec,
        )
    check(
        fallback == "sage128",
        "adaptive auto preserves the standard-QKV fallback",
    )

    marker = object()
    captured = {}
    patched = object()

    def apply_plan(model, plan):
        captured["model"] = model
        captured["plan"] = plan
        return patched

    with mock.patch.object(
        legacy,
        "read_plan",
        return_value=H3SageOptimizationPlan(),
    ), mock.patch.object(
        legacy,
        "apply_plan",
        side_effect=apply_plan,
    ), mock.patch.object(
        legacy,
        "format_legacy_status",
        return_value="compatibility status",
    ):
        result = legacy.MiniMaxH3HybridSparseAttention.execute(
            marker,
            mode="sage128_fused_qkv",
            video_budget=0.4,
            strict=True,
            activation="mlp_chunked_native",
            timing=True,
        )

    plan = captured["plan"]
    check(
        captured["model"] is marker and result.args[0] is patched,
        "legacy node delegates execution to the production apply path",
    )
    check(
        plan.memory.fused_qkv == FUSED_QKV_REQUIRED,
        "strict legacy fused mode becomes an immediate required request",
    )
    check(
        plan.memory.mlp_memory == MLP_MEMORY_LEGACY_NATIVE,
        "legacy native MLP mode is preserved internally",
    )
    check(
        plan.sparse.video_budget == 0.4,
        "legacy video budget becomes the production Sparse request",
    )
    check(
        result.ui.value == "compatibility status",
        "legacy adapter returns visible migration status",
    )

    cases = (
        ({"compile_backend": "bogus"}, "compile backend"),
        ({"compile_backend": "inductor"}, "sage128_fused_qkv"),
        (
            {
                "compile_backend": "inductor",
                "mode": "sage128_fused_qkv",
            },
            "convrot_2slice",
        ),
        ({"chunk_rows": 128}, "chunk_rows"),
    )
    for kwargs, expected in cases:
        try:
            legacy.MiniMaxH3HybridSparseAttention.execute(
                marker, **kwargs
            )
        except ValueError as exc:
            check(
                expected in str(exc),
                "legacy %s validation remains preflight-safe" % expected,
            )
        else:
            raise AssertionError(
                "legacy validation accepted %r" % (kwargs,)
            )

    adaptive_model = SimpleNamespace()
    adaptive_model.clone = lambda: object()
    adaptive_captured = {}

    def capture_backend(config, **kwargs):
        adaptive_captured["backend_config"] = config
        adaptive_captured["backend_kwargs"] = kwargs
        return SimpleNamespace(projector="projector")

    def capture_apply(model, *, config, decision):
        adaptive_captured["patched_model"] = model
        adaptive_captured["optimizer_config"] = config
        adaptive_captured["decision"] = decision

    environment = SimpleNamespace(
        cuda_available=False,
        capability=None,
        architecture="none",
        device_name="cpu",
    )
    with mock.patch.object(
        legacy,
        "RuntimeEnvironment",
        SimpleNamespace(detect=mock.Mock(return_value=environment)),
    ), mock.patch.object(
        legacy,
        "preflight_sparse_sage",
        return_value="kernel-spec",
    ) as preflight, mock.patch.object(
        legacy,
        "HybridStatsCollector",
        side_effect=lambda root, tag: (root, tag),
    ), mock.patch.object(
        legacy,
        "HybridSparseBackend",
        side_effect=capture_backend,
    ), mock.patch.object(
        legacy,
        "apply_legacy",
        side_effect=capture_apply,
    ):
        result = legacy.MiniMaxH3HybridSparseAttention.execute(
            adaptive_model,
            mode="sage128_fused_qkv",
            video_budget=0.4,
            strict=False,
            activation="mlp_chunked_native",
            chunk_rows=4096,
            run_tag="adaptive-check",
            timing=True,
            density_mode="adaptive_budget",
            min_video_density=0.10,
            max_video_density=0.70,
            adaptive_temperature=2.5,
            adaptive_target_mass=0.65,
        )

    adaptive_config = adaptive_captured["backend_config"]
    check(
        result.args[0] is adaptive_captured["patched_model"],
        "adaptive legacy mode clones before applying the compatibility patch",
    )
    check(
        adaptive_config.mode == "sage128_fused_qkv"
        and adaptive_config.video_budget == 0.4
        and adaptive_config.density_mode == "adaptive_budget"
        and adaptive_config.min_video_density == 0.10
        and adaptive_config.max_video_density == 0.70
        and adaptive_config.adaptive_temperature == 2.5
        and adaptive_config.adaptive_target_mass == 0.65
        and adaptive_config.strict is False
        and adaptive_config.run_tag == "adaptive-check"
        and adaptive_config.timing is True,
        "adaptive controls are preserved in HybridSparseConfig",
    )
    optimizer_config = adaptive_captured["optimizer_config"]
    check(
        optimizer_config.activation == "mlp_chunked_native"
        and optimizer_config.chunk_rows == 4096
        and optimizer_config.activation_strict is False,
        "legacy MLP controls are preserved in MemoryOptimizerConfig",
    )
    check(
        adaptive_captured["decision"].requested == "hybrid_sparse"
        and adaptive_captured["decision"].backend.projector == "projector",
        "adaptive mode builds the historical hybrid AttentionDecision",
    )
    check(
        preflight.call_count == 1
        and adaptive_captured["backend_kwargs"]["kernel_spec"] == "kernel-spec",
        "adaptive mode preflights Sparse Sage before backend construction",
    )

    try:
        legacy.MiniMaxH3HybridSparseAttention.execute(
            marker,
            compile_backend="inductor",
            mode="sage128_fused_qkv",
            activation="mlp_chunked_convrot_2slice",
        )
    except ValueError as exc:
        check(
            "does not support shared Inductor" in str(exc),
            "fully valid legacy compile request reaches migration guidance",
        )
    else:
        raise AssertionError("compiled compatibility request was accepted")

    print("\nall deprecated Hybrid Sparse compatibility tests passed")


if __name__ == "__main__":
    main()
