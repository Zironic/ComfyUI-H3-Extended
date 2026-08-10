"""CPU contracts for the dense Sage fused-QKV carrier and consumer."""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torch  # noqa: E402

from h3_attention.sage_mem_eff import (  # noqa: E402
    PreparedSM89,
    SageSM89API,
    SM89SageMemoryEfficientBackend,
)
from h3_sage_optimizations.dense_backend import (  # noqa: E402
    ProjectedSM89SageBackend,
)
from h3_sage_optimizations.dense_fused_qkv import (  # noqa: E402
    DENSE_QK_FORMAT,
    PreparedDenseFusedQKV,
    validate_prepared_dense_fused_qkv,
)


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def projected(sequence=350, heads=2):
    shape = (1, heads, sequence, 128)
    q_scales = ((sequence + 127) // 128) * 32
    k_scales = ((sequence + 63) // 64) * 4
    return PreparedDenseFusedQKV(
        q_int8=torch.zeros(shape, dtype=torch.int8),
        q_scale=torch.ones((1, heads, q_scales), dtype=torch.float32),
        k_int8=torch.zeros(shape, dtype=torch.int8),
        k_scale=torch.ones((1, heads, k_scales), dtype=torch.float32),
        v=torch.zeros(shape, dtype=torch.bfloat16),
        output_dtype=torch.bfloat16,
        sequence=sequence,
        heads=heads,
        head_dim=128,
        layer_index=7,
    )


def main():
    print("dense fused QKV carrier")
    carrier = projected()
    check(
        validate_prepared_dense_fused_qkv(carrier) is carrier,
        "per-thread Q/K scale layouts validate",
    )
    check(
        carrier.qk_format == DENSE_QK_FORMAT,
        "carrier explicitly identifies the dense per-thread ABI",
    )

    kernel_calls = []

    def kernel(*args):
        kernel_calls.append(args)
        args[3].fill_(2)

    def per_channel_fp8(v, **kwargs):
        return (
            torch.zeros(v.shape, dtype=torch.int8),
            torch.ones((v.shape[0], v.shape[1], v.shape[3]), dtype=torch.float32),
            None,
        )

    api = SageSM89API(
        version="2.2.test",
        per_channel_fp8=per_channel_fp8,
        kernel=kernel,
        kernel_name="fake_dense_sage",
    )
    delegate = SM89SageMemoryEfficientBackend(
        api=api,
        allow_cpu_for_tests=True,
    )
    backend = ProjectedSM89SageBackend(delegate)
    prepared = backend.prepare_projected(
        carrier,
        layer_index=7,
        transformer_options={},
    )
    check(
        isinstance(prepared, PreparedSM89),
        "dense backend accepts a fused projected carrier",
    )
    check(
        prepared.q_int8 is carrier.q_int8
        and prepared.k_int8 is carrier.k_int8,
        "dense backend reuses fused Q/K without requantization",
    )
    output = backend.execute(prepared)
    check(
        kernel_calls and torch.all(output == 2),
        "dense Sage kernel executes the projected carrier",
    )
    print("\nall dense fused QKV tests passed")


if __name__ == "__main__":
    main()
