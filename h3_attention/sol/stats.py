"""Process-local diagnostics for the H3 Sol-Attn adapter."""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_DEFAULTS = {
    "configured": 0,
    "sparse_calls": 0,
    "dense_calls": 0,
    "kernel_errors": 0,
    "gate_passes": 0,
    "gate_failures": 0,
    "prepared_bf16_bytes": 0,
}
_STATS = dict(_DEFAULTS)
_DECLINES = {}
_LAST = {}


def reset():
    with _LOCK:
        _STATS.clear()
        _STATS.update(_DEFAULTS)
        _DECLINES.clear()
        _LAST.clear()


def increment(name, value=1):
    with _LOCK:
        _STATS[name] = _STATS.get(name, 0) + value


def decline(reason):
    with _LOCK:
        _DECLINES[reason] = _DECLINES.get(reason, 0) + 1


def set_last(name, value):
    with _LOCK:
        _LAST[name] = value


def get():
    with _LOCK:
        return {**_STATS, "declines": dict(_DECLINES), "last": dict(_LAST)}
