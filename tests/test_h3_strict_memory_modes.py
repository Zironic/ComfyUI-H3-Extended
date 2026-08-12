"""Pure contracts for strict fused-QKV and explicit MLP execution."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_activation_memory.config import (  # noqa: E402
    ActivationMemoryConfig,
    MODE_BF16,
    MODE_BF16_STRICT,
    MODE_NATIVE,
    MODE_NATIVE_STRICT,
)
from h3_sage_optimizations.plan import (  # noqa: E402
    FUSED_QKV_REQUIRED,
    MLP_MEMORY_LEGACY_BF16,
    MLP_MEMORY_LEGACY_NATIVE,
    MLP_MEMORY_STRICT_AUTO,
    MLP_MEMORY_STRICT_BF16,
    MLP_MEMORY_STRICT_NATIVE,
    MemoryRequest,
)
from h3_sage_optimizations.qkv.providers import (  # noqa: E402
    MLP_GENERIC_CHUNKED,
    resolve_mlp_provider,
)


class Inventory:
    fc1 = (object(),)
    fc2 = (object(),)
    mlp_convrot_int8_256 = False

    @staticmethod
    def homogeneous(name):
        return True

    @staticmethod
    def labels(name):
        return ("plain_bfloat16",)


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def main():
    strict_auto = MemoryRequest(strict=True)
    strict_bf16 = MemoryRequest(
        mlp_memory=MLP_MEMORY_LEGACY_BF16,
        strict=True,
    )
    strict_native = MemoryRequest(
        mlp_memory=MLP_MEMORY_LEGACY_NATIVE,
        strict=True,
    )
    check(
        strict_auto.fused_qkv == FUSED_QKV_REQUIRED
        and strict_auto.mlp_memory == MLP_MEMORY_STRICT_AUTO,
        "strict automatic selection canonicalizes QKV and MLP fallback policy",
    )
    check(
        strict_bf16.mlp_memory == MLP_MEMORY_STRICT_BF16,
        "strict BF16 execution has a distinct immutable request",
    )
    check(
        strict_native.mlp_memory == MLP_MEMORY_STRICT_NATIVE,
        "strict native execution has a distinct immutable request",
    )

    automatic = resolve_mlp_provider(
        Inventory(), request=strict_auto.mlp_memory
    )
    bf16 = resolve_mlp_provider(
        Inventory(), request=strict_bf16.mlp_memory
    )
    native = resolve_mlp_provider(
        Inventory(), request=strict_native.mlp_memory
    )
    check(
        automatic.provider_id == MLP_GENERIC_CHUNKED
        and automatic.activation_mode == MODE_NATIVE_STRICT,
        "strict auto preserves non-ConvRot formats through fail-closed native chunking",
    )
    check(
        bf16.provider_id == MLP_GENERIC_CHUNKED
        and bf16.activation_mode == MODE_BF16_STRICT,
        "strict BF16 resolves to generic chunking with fail-closed acquisition",
    )
    check(
        native.provider_id == MLP_GENERIC_CHUNKED
        and native.activation_mode == MODE_NATIVE_STRICT,
        "strict native resolves to the native fail-closed activation mode",
    )

    bf16_config = ActivationMemoryConfig(
        mode=MODE_BF16_STRICT,
        strict=False,
    )
    native_config = ActivationMemoryConfig(
        mode=MODE_NATIVE_STRICT,
        strict=False,
    )
    check(
        bf16_config.mode == MODE_BF16 and bf16_config.strict,
        "strict BF16 alias normalizes to the canonical mode and forces strictness",
    )
    check(
        native_config.mode == MODE_NATIVE
        and native_config.strict
        and native_config.native_swiglu,
        "strict native alias normalizes without breaking native SwiGLU",
    )
    print("\nall strict Memory-mode tests passed")


if __name__ == "__main__":
    main()
