"""CPU tests for unified H3 selection and multi-feature orchestration."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_memory_optimizer.attention import (  # noqa: E402
    ATTENTION_AUTO,
    ATTENTION_EXISTING,
    ATTENTION_SOL,
    AUTO_ADAPTERS,
    FALLBACK_ALLOW,
    FALLBACK_ERROR,
    AttentionResolutionError,
    RuntimeEnvironment,
    SM80Adapter,
    SM86Adapter,
    SM89Adapter,
    SM90Adapter,
    SM12xAdapter,
    SolAdapter,
    resolve_attention,
)
from h3_memory_optimizer.config import ACTIVATION_OFF, MemoryOptimizerConfig  # noqa: E402
from h3_memory_optimizer.patch import apply  # noqa: E402


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


class FakeAdapter:
    name = "fake_sm89"
    builds = 0
    supported = True
    fail = False
    options = None

    @classmethod
    def probe(cls, environment):
        return cls.supported, "synthetic probe"

    @classmethod
    def build(cls, environment=None, options=None):
        cls.builds += 1
        cls.options = options
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


def test_architecture_probes():
    print("architecture probes")
    for adapter, capability in (
        (SM80Adapter, (8, 0)),
        (SM86Adapter, (8, 6)),
        (SM89Adapter, (8, 9)),
        (SM90Adapter, (9, 0)),
        (SM12xAdapter, (12, 0)),
        (SM12xAdapter, (12, 1)),
    ):
        supported, _ = adapter.probe(env(capability))
        check(supported, "%s accepts SM%d%d" % (adapter.name, *capability))
    supported, reason = SM89Adapter.probe(env((9, 0)))
    check(not supported and "requires SM89" in reason, "dense adapters require exact architecture families")
    check(SolAdapter not in AUTO_ADAPTERS, "auto never selects approximate Sol-Attn")


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
        adapter_options={"tau": 1.25},
    )
    check(decision.selected == FakeAdapter.name and decision.optimized, "supported adapter is selected")
    check(FakeAdapter.builds == 1 and FakeAdapter.options == {"tau": 1.25}, "adapter is built exactly once with options")

    FakeAdapter.supported = False
    decision = resolve_attention(
        ATTENTION_AUTO,
        FALLBACK_ALLOW,
        environment=env((8, 7)),
        adapters=(FakeAdapter,),
    )
    check(decision.selected == ATTENTION_EXISTING, "unsupported architecture preserves existing attention")
    try:
        resolve_attention(
            ATTENTION_AUTO,
            FALLBACK_ERROR,
            environment=env((8, 7)),
            adapters=(FakeAdapter,),
        )
    except AttentionResolutionError:
        check(True, "strict fallback raises during preflight")
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
    check(decision.selected == ATTENTION_EXISTING, "adapter build failure falls back before mutation")
    check("synthetic missing kernel" in decision.reason, "fallback preserves detailed reason")


def test_config():
    print("config")
    config = MemoryOptimizerConfig(activation=ACTIVATION_OFF)
    check(config.activation_config() is None, "activation off produces no block patch")
    sol = MemoryOptimizerConfig(
        attention=ATTENTION_SOL,
        sol_gate_heads=3,
        sol_density_heads=2,
        sol_sink_mode="prefix",
    )
    options = sol.attention_options()
    check(options["gate_heads"] == 3 and options["density_heads"] == 2, "Sol settings reach adapter options")
    check(sol.adaln_config().mode == "off" and sol.block_cache_config().mode == "off", "approximate/cache features remain off by default")


def test_apply_order_and_fallback():
    print("apply")
    patcher = FakePatcher()
    calls = []

    class Backend:
        approximate = True
        requires_runtime_context = True
        strict_runtime_layout = True

    class Decision:
        requested = ATTENTION_SOL
        selected = ATTENTION_SOL
        backend = Backend()
        reason = "supported"
        environment = env()

    class Provider:
        def as_status(self):
            return {"ready": True}

    class Cache:
        def as_status(self):
            return {"skipped_tails": 2}

    provider = Provider()
    cache = Cache()

    def attention(model, backend):
        calls.append("attention")
        return backend, 50

    def activation(model, config):
        calls.append("activation")
        return 50

    def adaln(model, config):
        calls.append("adaln")
        return provider, 50

    def block_cache(model, config):
        calls.append("cache")
        return cache, 50

    def runtime(model, session):
        calls.append("runtime")
        return session

    config = MemoryOptimizerConfig(
        attention=ATTENTION_SOL,
        adaln_precompute="on",
        block_cache="first_block",
    )
    result = apply(
        patcher,
        config=config,
        decision=Decision(),
        attention_configurer=attention,
        activation_installer=activation,
        adaln_installer=adaln,
        block_cache_installer=block_cache,
        runtime_installer=runtime,
    )
    check(calls == ["attention", "activation", "adaln", "cache", "runtime"], "components install in non-conflicting order")
    check(result.runtime_installed and result.block_cache_blocks == 50, "runtime and cache are reported")
    status = patcher.model_options["transformer_options"]["minimax_h3_memory_optimizer"]
    check(status["attention_approximate"] and status["block_cache_approximate"], "status labels approximate features")
    check(patcher.model_options["transformer_options"]["existing"] is True, "existing transformer options survive")

    patcher = FakePatcher()
    calls = []

    class Fallback:
        requested = ATTENTION_AUTO
        selected = ATTENTION_EXISTING
        backend = None
        reason = "unsupported GPU"
        environment = env((7, 5))

    def disabled_adaln(model, config):
        calls.append("adaln")
        return None, 0

    def disabled_cache(model, config):
        calls.append("cache")
        return None, 0

    result = apply(
        patcher,
        config=MemoryOptimizerConfig(),
        decision=Fallback(),
        attention_configurer=attention,
        activation_installer=activation,
        adaln_installer=disabled_adaln,
        block_cache_installer=disabled_cache,
        runtime_installer=runtime,
    )
    check(calls == ["activation", "adaln", "cache"], "dense fallback leaves attention untouched and avoids runtime overhead")
    check(result.activation_blocks == 50 and result.attention_blocks == 0, "portable MLP optimization remains active")


def test_backend_runtime_capabilities():
    print("backend runtime capabilities")
    listener = object()

    class Backend:
        approximate = True
        requires_runtime_context = True
        strict_runtime_layout = True
        runtime_listeners = (listener,)

        def as_status(self):
            return {"phase": "test"}

    class Decision:
        requested = "hybrid_sparse"
        selected = "hybrid_sparse"
        backend = Backend()
        reason = "explicit test backend"
        environment = env()

    patcher = FakePatcher()
    seen = {}

    def attention(model, backend):
        return backend, 50

    def disabled(model, config):
        return None, 0

    def runtime(model, session):
        seen["session"] = session
        return session

    result = apply(
        patcher,
        config=MemoryOptimizerConfig(
            attention=ATTENTION_EXISTING,
            activation=ACTIVATION_OFF,
        ),
        decision=Decision(),
        attention_configurer=attention,
        adaln_installer=disabled,
        block_cache_installer=disabled,
        runtime_installer=runtime,
    )
    session = seen["session"]
    check(result.attention_requested == "hybrid_sparse" and result.runtime_installed,
          "custom backend identity and runtime requirement are preserved")
    check(session.strict_layout and listener in session.listeners,
          "backend strict-layout flag and listener configure the shared runtime")
    status = patcher.model_options["transformer_options"]["minimax_h3_memory_optimizer"]
    check(status["attention_approximate"] and status["attention_backend"]["phase"] == "test",
          "backend capability and status replace Sol-specific checks")


def main():
    test_architecture_probes()
    test_resolution()
    test_config()
    test_apply_order_and_fallback()
    test_backend_runtime_capabilities()
    print("\nall unified memory-optimizer tests passed")


if __name__ == "__main__":
    main()
