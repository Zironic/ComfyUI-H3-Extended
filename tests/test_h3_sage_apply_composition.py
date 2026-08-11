"""Exercise apply_plan through Memory→Sparse and Sparse→Memory."""

import os
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import h3_sage_optimizations.apply as apply_module  # noqa: E402
from h3_sage_optimizations.plan import (  # noqa: E402
    H3SageOptimizationPlan,
    MemoryRequest,
    SparseRequest,
    STATUS_KEY,
    read_plan,
)
from h3_sage_optimizations.qkv.providers import (  # noqa: E402
    MLPProviderResolution,
    QKVProviderResolution,
)


class FakeModel:
    def __init__(self, options=None):
        self.model_options = deepcopy(options or {})
        self.object_patches = {}

    def clone(self):
        clone = FakeModel(self.model_options)
        clone.object_patches = dict(self.object_patches)
        return clone


def attention(plan):
    selected = "sparse_sage" if plan.sparse else "dense_sage_sm89"
    return apply_module.ResolvedAttention(
        requested=selected,
        selected=selected,
        backend=SimpleNamespace(
            name=selected,
            runtime_listeners=(),
        ),
        reason="synthetic",
        backend_kind=selected,
        projector=None,
    )


def run(base, first_request, second_request):
    first_plan = H3SageOptimizationPlan()
    if isinstance(first_request, MemoryRequest):
        first_plan = first_plan.with_memory(first_request)
    else:
        first_plan = first_plan.with_sparse(first_request)
    first = apply_module.apply_plan(base, first_plan)

    second_plan = read_plan(first)
    if isinstance(second_request, MemoryRequest):
        second_plan = second_plan.with_memory(second_request)
    else:
        second_plan = second_plan.with_sparse(second_request)
    return apply_module.apply_plan(first, second_plan)


def main():
    memory = MemoryRequest()
    sparse = SparseRequest(video_budget=0.5)
    inventory = SimpleNamespace(
        labels=lambda name: ("TensorWiseINT8Layout+convrot256",),
    )
    qkv = QKVProviderResolution(
        "standard_h3_qkv", False, "synthetic"
    )
    mlp = MLPProviderResolution(
        "generic_chunked_quantized",
        "mlp_chunked_native",
        "synthetic",
    )

    def resolve(plan, environment, actual_inventory):
        return attention(plan), qkv

    with mock.patch.object(
        apply_module, "is_minimax_h3", return_value=True
    ), mock.patch.object(
        apply_module, "get_h3_blocks", return_value=(object(),)
    ), mock.patch.object(
        apply_module,
        "inspect_h3_linears",
        return_value=inventory,
    ), mock.patch.object(
        apply_module.RuntimeEnvironment,
        "detect",
        return_value=SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
            device_name="fake 4090",
            architecture="sm89",
        ),
    ), mock.patch.object(
        apply_module, "_resolve_dense", side_effect=resolve
    ), mock.patch.object(
        apply_module, "_resolve_sparse", side_effect=resolve
    ), mock.patch.object(
        apply_module,
        "configure_backend",
        return_value=(object(), 50),
    ), mock.patch.object(
        apply_module,
        "_install_mlp",
        return_value=(mlp, 50),
    ), mock.patch.object(
        apply_module,
        "_ensure_sparse_runtime",
        return_value=(object(), True),
    ):
        left = run(FakeModel(), memory, sparse)
        right = run(FakeModel(), sparse, memory)

    left_status = left.model_options["transformer_options"][STATUS_KEY]
    right_status = right.model_options["transformer_options"][STATUS_KEY]
    if read_plan(left).signature != read_plan(right).signature:
        raise AssertionError("node order changed the final plan")
    if left_status["plan_signature"] != right_status["plan_signature"]:
        raise AssertionError("node order changed the resolved status")
    print("  ok: apply_plan composes both public node orders identically")


if __name__ == "__main__":
    main()
