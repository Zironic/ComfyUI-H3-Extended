"""Regression contract for composing H3 model-patch nodes around one runtime."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_memory_optimizer.attention import RuntimeEnvironment  # noqa: E402
from h3_memory_optimizer.config import ACTIVATION_OFF, MemoryOptimizerConfig  # noqa: E402
from h3_memory_optimizer.patch import apply  # noqa: E402
from h3_runtime.context import H3RuntimeSession, RUNTIME_SESSION_KEY  # noqa: E402


class FakePatcher:
    def __init__(self, runtime):
        self.model_options = {
            "transformer_options": {
                RUNTIME_SESSION_KEY: runtime,
                "existing": True,
            }
        }


class Backend:
    approximate = True
    requires_runtime_context = True
    strict_runtime_layout = True

    def __init__(self, listener):
        self.runtime_listeners = (listener,)


class Decision:
    def __init__(self, backend):
        self.requested = "hybrid_sparse"
        self.selected = "hybrid_sparse"
        self.backend = backend
        self.projector = None
        self.reason = "runtime reuse regression"
        self.environment = RuntimeEnvironment(
            cuda_available=True,
            device_index=0,
            capability=(8, 9),
            device_name="fake GPU",
        )


def test_existing_runtime_is_reused():
    chipmunk_listener = object()
    hybrid_listener = object()
    existing = H3RuntimeSession(
        strict_layout=False,
        listeners=[chipmunk_listener],
    )
    patcher = FakePatcher(existing)
    installer_calls = []

    def attention(model, backend):
        return backend, 50

    def disabled(model, config):
        return None, 0

    def runtime_installer(model, session):
        installer_calls.append(session)
        raise AssertionError("a second runtime wrapper must not be installed")

    result = apply(
        patcher,
        config=MemoryOptimizerConfig(
            attention="existing",
            activation=ACTIVATION_OFF,
        ),
        decision=Decision(Backend(hybrid_listener)),
        attention_configurer=attention,
        activation_installer=lambda model, config: 0,
        adaln_installer=disabled,
        block_cache_installer=disabled,
        runtime_installer=runtime_installer,
    )

    options = patcher.model_options["transformer_options"]
    assert result.runtime_installed
    assert not installer_calls
    assert options[RUNTIME_SESSION_KEY] is existing
    assert existing.strict_layout
    assert chipmunk_listener in existing.listeners
    assert hybrid_listener in existing.listeners
    assert options["existing"] is True


if __name__ == "__main__":
    test_existing_runtime_is_reused()
    print("H3 runtime session reuse test passed")
