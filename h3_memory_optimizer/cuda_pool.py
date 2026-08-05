"""Soft release policy for PyTorch's cudaMallocAsync default pool.

PyTorch deliberately sets the CUDA default memory-pool release threshold to
UINT64_MAX when its cudaMallocAsync allocator initializes. That maximizes reuse
but also lets the pool retain a high-water footprint indefinitely. This module
optionally replaces that value with a finite soft retention target.

The policy is not an allocation limit, does not call empty_cache/trim, and does
not synchronize the device. CUDA may grow above the threshold when live demand
requires it and may release excess backing on a later stream/event/device sync.
"""

from dataclasses import dataclass
import ctypes
import logging
import math
import sys
import threading

GIB = 1024 ** 3
CUDA_MALLOC_ASYNC = "cudaMallocAsync"
CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4
LOG_PREFIX = "[H3 memory optimizer]"

_POLICY_LOCK = threading.Lock()


@dataclass(frozen=True)
class AsyncPoolPolicyResult:
    requested: bool
    applied: bool
    backend: str
    device_index: int | None
    requested_bytes: int
    applied_bytes: int | None
    reason: str

    def as_status(self):
        return {
            "requested": bool(self.requested),
            "applied": bool(self.applied),
            "backend": self.backend,
            "device_index": self.device_index,
            "requested_bytes": int(self.requested_bytes),
            "applied_bytes": (
                None if self.applied_bytes is None else int(self.applied_bytes)
            ),
            "reason": self.reason,
        }


class CudaDriverError(RuntimeError):
    pass


class CudaDefaultPoolDriver:
    """Minimal CUDA Driver API wrapper; no CUDA toolkit package is required."""

    def __init__(self, library=None):
        self.library = library or self._load_library()
        self._bind()

    @staticmethod
    def _load_library():
        names = (
            ("nvcuda.dll",)
            if sys.platform == "win32"
            else ("libcuda.so.1", "libcuda.so")
        )
        errors = []
        loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
        for name in names:
            try:
                return loader(name)
            except OSError as exc:
                errors.append("%s: %s" % (name, exc))
        raise CudaDriverError(
            "could not load the NVIDIA CUDA driver library (%s)"
            % "; ".join(errors)
        )

    def _bind(self):
        lib = self.library
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype = ctypes.c_int
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuDeviceGet.restype = ctypes.c_int
        lib.cuDeviceGetDefaultMemPool.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        lib.cuDeviceGetDefaultMemPool.restype = ctypes.c_int
        lib.cuMemPoolSetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.cuMemPoolSetAttribute.restype = ctypes.c_int
        lib.cuMemPoolGetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.cuMemPoolGetAttribute.restype = ctypes.c_int

        if hasattr(lib, "cuGetErrorName"):
            lib.cuGetErrorName.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_char_p),
            ]
            lib.cuGetErrorName.restype = ctypes.c_int
        if hasattr(lib, "cuGetErrorString"):
            lib.cuGetErrorString.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_char_p),
            ]
            lib.cuGetErrorString.restype = ctypes.c_int

    def _error_text(self, result):
        parts = ["CUDA driver error %d" % int(result)]
        for function_name in ("cuGetErrorName", "cuGetErrorString"):
            function = getattr(self.library, function_name, None)
            if function is None:
                continue
            value = ctypes.c_char_p()
            try:
                if function(int(result), ctypes.byref(value)) == 0 and value.value:
                    text = value.value.decode("utf-8", errors="replace")
                    if text not in parts:
                        parts.append(text)
            except Exception:
                pass
        return ": ".join(parts)

    def _check(self, result, operation):
        if int(result) != 0:
            raise CudaDriverError(
                "%s failed: %s" % (operation, self._error_text(result))
            )

    def set_release_threshold(self, device_index, threshold_bytes):
        with _POLICY_LOCK:
            self._check(self.library.cuInit(0), "cuInit")

            device = ctypes.c_int()
            self._check(
                self.library.cuDeviceGet(
                    ctypes.byref(device), int(device_index)
                ),
                "cuDeviceGet",
            )

            pool = ctypes.c_void_p()
            self._check(
                self.library.cuDeviceGetDefaultMemPool(
                    ctypes.byref(pool), device.value
                ),
                "cuDeviceGetDefaultMemPool",
            )
            if not pool.value:
                raise CudaDriverError(
                    "cuDeviceGetDefaultMemPool returned a null pool"
                )

            requested = ctypes.c_uint64(int(threshold_bytes))
            self._check(
                self.library.cuMemPoolSetAttribute(
                    pool,
                    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    ctypes.byref(requested),
                ),
                "cuMemPoolSetAttribute(release threshold)",
            )

            actual = ctypes.c_uint64()
            self._check(
                self.library.cuMemPoolGetAttribute(
                    pool,
                    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    ctypes.byref(actual),
                ),
                "cuMemPoolGetAttribute(release threshold)",
            )
            return int(actual.value)


def _allocator_backend(torch_module):
    cuda = torch_module.cuda
    getter = getattr(cuda, "get_allocator_backend", None)
    if getter is None:
        getter = getattr(
            getattr(cuda, "memory", None),
            "get_allocator_backend",
            None,
        )
    if getter is None:
        return "unknown"
    return str(getter())


def _initialize_async_allocator(torch_module, device_index):
    """Force allocator init before overriding PyTorch's UINT64_MAX setting."""
    cuda = torch_module.cuda
    with cuda.device(int(device_index)):
        probe = torch_module.empty(
            1,
            dtype=torch_module.uint8,
            device="cuda:%d" % int(device_index),
        )
        del probe


def configure_cuda_async_soft_gc(
    enabled,
    release_threshold_gib,
    *,
    device_index=None,
    torch_module=None,
    driver=None,
    initializer=None,
):
    """Apply a finite release threshold to PyTorch's async default pool.

    This function is deliberately fail-open: the model optimization remains
    usable if CUDA, cudaMallocAsync, or the driver pool API is unavailable.
    """
    requested_bytes = 0
    if enabled:
        try:
            threshold = float(release_threshold_gib)
            if not math.isfinite(threshold) or threshold <= 0:
                raise ValueError(
                    "release_threshold_gib must be finite and greater than zero"
                )
            requested_bytes = int(threshold * GIB)
        except Exception as exc:
            result = AsyncPoolPolicyResult(
                requested=True,
                applied=False,
                backend="unknown",
                device_index=device_index,
                requested_bytes=0,
                applied_bytes=None,
                reason="%s: %s" % (type(exc).__name__, exc),
            )
            logging.warning(
                "%s CUDA async soft GC not applied: %s",
                LOG_PREFIX,
                result.reason,
            )
            return result
    else:
        return AsyncPoolPolicyResult(
            requested=False,
            applied=False,
            backend="not queried",
            device_index=device_index,
            requested_bytes=0,
            applied_bytes=None,
            reason="disabled",
        )

    try:
        if torch_module is None:
            import torch as torch_module

        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        backend = _allocator_backend(torch_module)
        if backend != CUDA_MALLOC_ASYNC:
            result = AsyncPoolPolicyResult(
                requested=True,
                applied=False,
                backend=backend,
                device_index=device_index,
                requested_bytes=requested_bytes,
                applied_bytes=None,
                reason=(
                    "allocator backend is %s, not %s"
                    % (backend, CUDA_MALLOC_ASYNC)
                ),
            )
            logging.info(
                "%s CUDA async soft GC skipped: %s",
                LOG_PREFIX,
                result.reason,
            )
            return result

        if device_index is None:
            device_index = int(torch_module.cuda.current_device())
        else:
            device_index = int(device_index)

        (initializer or _initialize_async_allocator)(
            torch_module, device_index
        )
        driver = driver or CudaDefaultPoolDriver()
        actual_bytes = driver.set_release_threshold(
            device_index, requested_bytes
        )

        result = AsyncPoolPolicyResult(
            requested=True,
            applied=True,
            backend=backend,
            device_index=device_index,
            requested_bytes=requested_bytes,
            applied_bytes=actual_bytes,
            reason="applied",
        )
        logging.info(
            "%s CUDA async soft GC armed on cuda:%d: release threshold "
            "%.3f GiB (readback %.3f GiB); no hard limit or explicit trim",
            LOG_PREFIX,
            device_index,
            requested_bytes / GIB,
            actual_bytes / GIB,
        )
        return result
    except Exception as exc:
        backend = "unknown"
        try:
            backend = _allocator_backend(torch_module)
        except Exception:
            pass
        result = AsyncPoolPolicyResult(
            requested=True,
            applied=False,
            backend=backend,
            device_index=device_index,
            requested_bytes=requested_bytes,
            applied_bytes=None,
            reason="%s: %s" % (type(exc).__name__, exc),
        )
        logging.warning(
            "%s CUDA async soft GC not applied: %s",
            LOG_PREFIX,
            result.reason,
        )
        return result
