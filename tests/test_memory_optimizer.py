"""CPU tests for unified H3 memory-optimizer selection and orchestration."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_memory_optimizer.attention import (  # noqa: E402
    ATTENTION_AUTO,
    ATTENTION_EXISTING,
    FALLBACK_ALLOW,
    FALLBACK_ERROR,
    AttentionResolutionError,
    RuntimeEnvironment,
    resolve_attention,
)
from h3_memory_optimizer.config import (  # noqa: E402
    ACTIVATION_OFF,
    MemoryOptimizerConfig,
)
from h3_memory_optimizer.patch import apply  # noqa: E402


def check(cond, message):
    if not cond:
        raise AssertionError(message)
    print("  ok: %s" % message)


class FakeAdapter:
    name = "fake_sm89"
    builds = 0
    supported = True
    fail = False

    @classmethod
    def probe(cls, environment):
        return cls.supported, "synthetic probe"

    @classmethod
    def build(cls):
        cls.builds += 1
        if cls.fail:
            raise RuntimeError("synthetic missing kernel")
        return object()


class FakePatcher:
    def __init__(self):
        self.model_options = {"transformer_options": {"existing": True}}
        self.calls = []


def env(capability=(8, 9)):
    return RuntimeEnvironment(
        cuda_available=capability is not None,
        device_index=0 if capability is not None else None,
        capability=capability,
        device_name="fake GPU" if capability is not None else "no CUDA device",
    )


def test_resolution():
    print("resolution")
    FakeAdapter.builds = 0
    FakeAdapter.supported = True
    FakeAdapter.fail = False
    decision = resolve_attention(
        ATTENTION_AUTO,
        FALLBACK_ALLOW,
        environment=env(),
        adapters=(FakeAdapter,),
    )
    check(decision.selected == FakeAdapter.name and decision.optimized,
          "auto selects a supported adapter")
    check(FakeAdapter.builds == 1, "selected adapter is preflight-built exactly once")

    FakeAdapter.supported = False
    decision = resolve_attention(
        ATTENTION_AUTO,
        FALLBACK_ALLOW,
        environment=env((8, 6)),
        adapters=(FakeAdapter,),
    )
    check(decision.selected == ATTENTION_EXISTING and not decision.optimized,
          "unsupported architecture preserves existing attention")

    try:
        resolve_attention(
            ATTENTION_AUTO,
            FALLBACK_ERROR,
            environment=env((8, 6)),
            adapters=(FakeAdapter,),
        )
    except AttentionResolutionError as exc:
        check("cannot select" in str(exc), "strict fallback raises during preflight")
    else:
        raise AssertionError("strict fallback must raise")

    FakeAdapter.supported = True
    FakeAdapter.fail = True
    decision = resolve_attention(
        ATTENTION_AUTO,
        FALLBACK_ALLOW,
        environment=env(),
        adapters=(FakeAdapter,),
    )
    check(decision.selected == ATTENTION_EXISTING,
          "missing low-level kernel falls back before model mutation")
    check("synthetic missing kernel" in decision.reason,
          "fallback reason retains the failed capability detail")


def test_config():
    print("config")
    config = MemoryOptimizerConfig(activation=ACTIVATION_OFF)
    check(config.activation_config() is None, "activation off produces no patch config")
    config = MemoryOptimizerConfig(
        activation="mlp_chunked_bf16",
        chunk_rows=4096,
        activation_strict=False,
    )
    activation = config.activation_config()
    check(activation.chunk_rows == 4096 and activation.strict is False,
          "portable BF16 activation config is preserved")


def test_apply_order_and_fallback():
    print("apply")
    patcher = FakePatcher()
    calls = []

    class Decision:
        requested = ATTENTION_AUTO
        selected = "fake_sm89"
        backend = object()
        reason = "supported"
        environment = env()

    def configure_attention(model, backend):
        calls.append("attention")
        model.calls.append(("attention", backend))
        return backend, 50

    def install_activation(model, config):
        calls.append("activation")
        model.calls.append(("activation", config.mode))
        return 50

    config = MemoryOptimizerConfig()
    result = apply(
        patcher,
        config=config,
        decision=Decision(),
        attention_configurer=configure_attention,
        activation_installer=install_activation,
    )
    check(calls == ["attention", "activation"],
          "attention forward is installed before the enclosing block forward")
    check(result.attention_blocks == 50 and result.activation_blocks == 50,
          "unified result reports both patch counts")
    status = patcher.model_options["transformer_options"]["minimax_h3_memory_optimizer"]
    check(status["attention_selected"] == "fake_sm89",
          "transformer options record the selected adapter")
    check(patcher.model_options["transformer_options"]["existing"] is True,
          "status recording preserves existing transformer options")

    patcher = FakePatcher()
    calls = []

    class FallbackDecision:
        requested = ATTENTION_AUTO
        selected = ATTENTION_EXISTING
        backend = None
        reason = "unsupported GPU"
        environment = env((7, 5))

    result = apply(
        patcher,
        config=config,
        decision=FallbackDecision(),
        attention_configurer=configure_attention,
        activation_installer=install_activation,
    )
    check(calls == ["activation"],
          "attention fallback leaves the incoming attention path untouched")
    check(result.attention_blocks == 0 and result.activation_blocks == 50,
          "activation optimization remains active after attention fallback")


def main():
    test_resolution()
    test_config()
    test_apply_order_and_fallback()
    print("\nall unified memory-optimizer tests passed")


if __name__ == "__main__":
    main()
