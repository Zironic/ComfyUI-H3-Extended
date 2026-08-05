"""CPU tests for the cudaMallocAsync soft release-threshold policy."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_memory_optimizer.config import MemoryOptimizerConfig  # noqa: E402
from h3_memory_optimizer.cuda_pool import (  # noqa: E402
    GIB,
    configure_cuda_async_soft_gc,
)


def check(cond, message):
    if not cond:
        raise AssertionError(message)
    print("  ok: %s" % message)


class DeviceContext:
    def __init__(self, events, index):
        self.events = events
        self.index = index

    def __enter__(self):
        self.events.append(("device_enter", self.index))

    def __exit__(self, exc_type, exc, tb):
        self.events.append(("device_exit", self.index))


class FakeCuda:
    def __init__(self, backend, events, available=True):
        self.backend = backend
        self.events = events
        self.available = available

    def is_available(self):
        self.events.append(("available", self.available))
        return self.available

    def get_allocator_backend(self):
        self.events.append(("backend", self.backend))
        return self.backend

    def current_device(self):
        self.events.append(("current_device", 2))
        return 2

    def device(self, index):
        return DeviceContext(self.events, index)


class FakeTorch:
    uint8 = object()

    def __init__(self, backend="cudaMallocAsync", available=True):
        self.events = []
        self.cuda = FakeCuda(backend, self.events, available=available)

    def empty(self, *args, **kwargs):
        self.events.append(("empty", args, kwargs))
        return object()


class FakeDriver:
    def __init__(self, events, actual_delta=0, fail=None):
        self.events = events
        self.actual_delta = actual_delta
        self.fail = fail

    def set_release_threshold(self, device_index, threshold_bytes):
        self.events.append(("set", device_index, threshold_bytes))
        if self.fail is not None:
            raise self.fail
        return threshold_bytes + self.actual_delta


def test_disabled_is_inert():
    print("disabled")
    fake = FakeTorch()
    result = configure_cuda_async_soft_gc(
        False,
        11.0,
        torch_module=fake,
    )
    check(not result.requested and not result.applied,
          "disabled policy reports an inert result")
    check(fake.events == [],
          "disabled policy does not initialize or query CUDA")


def test_non_async_backend_is_a_noop():
    print("native backend")
    fake = FakeTorch(backend="native")
    driver_events = []
    result = configure_cuda_async_soft_gc(
        True,
        11.0,
        torch_module=fake,
        driver=FakeDriver(driver_events),
        initializer=lambda _torch, _device: driver_events.append(("init", _device)),
    )
    check(result.requested and not result.applied,
          "native allocator does not receive an async-pool policy")
    check(result.backend == "native",
          "no-op result records the active allocator")
    check(driver_events == [],
          "native allocator path never touches the CUDA pool")


def test_async_policy_orders_initialization_before_override():
    print("async policy")
    fake = FakeTorch()
    events = []
    requested = int(10.5 * GIB)
    result = configure_cuda_async_soft_gc(
        True,
        10.5,
        device_index=1,
        torch_module=fake,
        driver=FakeDriver(events),
        initializer=lambda _torch, device: events.append(("init", device)),
    )
    check(result.applied and result.applied_bytes == requested,
          "async policy applies and verifies the requested threshold")
    check(events == [("init", 1), ("set", 1, requested)],
          "PyTorch allocator initialization precedes the pool override")
    check(result.as_status()["requested_bytes"] == requested,
          "status serialization preserves byte-exact policy values")


def test_driver_failure_is_fail_open():
    print("failure")
    fake = FakeTorch()
    events = []
    result = configure_cuda_async_soft_gc(
        True,
        11.0,
        torch_module=fake,
        driver=FakeDriver(events, fail=RuntimeError("synthetic driver failure")),
        initializer=lambda _torch, device: events.append(("init", device)),
    )
    check(result.requested and not result.applied,
          "driver failures do not abort the model optimization")
    check("synthetic driver failure" in result.reason,
          "failure reason remains visible for diagnostics")


def test_config_validation():
    print("config")
    config = MemoryOptimizerConfig(
        cuda_async_soft_gc=True,
        cuda_async_release_threshold_gib=11.0,
    )
    check(config.cuda_async_soft_gc,
          "unified config retains the soft-GC toggle")
    check(config.cuda_async_release_threshold_gib == 11.0,
          "unified config retains the release threshold")

    try:
        MemoryOptimizerConfig(
            cuda_async_release_threshold_gib=0.0,
        )
    except ValueError as exc:
        check("greater than zero" in str(exc),
              "non-positive release thresholds are rejected")
    else:
        raise AssertionError("zero release threshold must be rejected")


def main():
    test_disabled_is_inert()
    test_non_async_backend_is_a_noop()
    test_async_policy_orders_initialization_before_override()
    test_driver_failure_is_fail_open()
    test_config_validation()
    print("\nall CUDA async-pool policy tests passed")


if __name__ == "__main__":
    main()
