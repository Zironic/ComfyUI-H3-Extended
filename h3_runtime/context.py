"""Shared per-request state for MiniMax H3 acceleration features.

The outer-sample wrapper supplies an explicit request boundary. The diffusion
wrapper then publishes the packed layout, sampler step, and CFG branch before
the block stack starts. Sigma/layout inference remains as a fallback for direct
model calls and tests that do not pass through Comfy's sampler wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
from typing import Any, Iterable

import torch

RUNTIME_KEY = "minimax_h3_runtime"
RUNTIME_SESSION_KEY = "minimax_h3_runtime_session"
WRAPPER_KEY = "h3_runtime_context"
OUTER_WRAPPER_KEY = "h3_runtime_request_boundary"
LOG_PREFIX = "[H3 runtime]"


@dataclass(frozen=True)
class RuntimeSnapshot:
    request_id: int
    step_index: int
    total_steps: int
    sigma: float
    branch: tuple[int, ...]
    layout: Any | None
    layout_signature: tuple | None
    compute_dtype: torch.dtype | None
    device: torch.device | None
    error: str | None = None

    @property
    def valid_layout(self):
        return self.layout is not None and self.error is None

    @property
    def is_first_step(self):
        return self.step_index == 0


class H3RuntimeSession:
    """Tracks one cloned model across repeated sampler requests."""

    def __init__(self, *, strict_layout=False, listeners: Iterable[Any] = ()):
        self.strict_layout = bool(strict_layout)
        self.listeners = list(listeners)
        self.request_id = -1
        self.layout_signature = None
        self._last_step = {}
        self._last_sigma = {}
        self._direction = {}
        self._lock = threading.RLock()
        self._outer_local = threading.local()
        self._outer_serial = 0
        self._active_outer_token = None
        self._explicit_layout_pending = False
        self.last_snapshot = None

    def add_listener(self, listener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    @staticmethod
    def _branch(transformer_options):
        value = transformer_options.get("cond_or_uncond")
        if value is None:
            return (0,)
        if torch.is_tensor(value):
            value = value.detach().flatten().tolist()
        if isinstance(value, (list, tuple)):
            return tuple(int(x) for x in value) or (0,)
        return (int(value),)

    @staticmethod
    def _scalar_sigma(timestep, transformer_options):
        sigma = transformer_options.get("sigmas")
        if sigma is not None:
            try:
                return float(sigma.detach().flatten()[0].float().item())
            except Exception:
                pass
        if timestep is not None:
            try:
                return float(timestep.detach().flatten()[0].float().item()) / 1000.0
            except Exception:
                pass
        return float("nan")

    @staticmethod
    def _step_index(sigma, transformer_options):
        schedule = transformer_options.get("sample_sigmas")
        if schedule is None or not torch.is_tensor(schedule) or schedule.numel() == 0:
            return -1, 0
        flat = schedule.detach().flatten().to("cpu", torch.float32)
        total = max(0, int(flat.numel()) - 1)
        if not math.isfinite(sigma):
            return -1, total
        index = int(torch.argmin((flat - float(sigma)).abs()).item())
        return (min(index, max(0, total - 1)) if total else index), total

    @staticmethod
    def _layout_signature_of(layout):
        return (
            int(layout.seq_len),
            tuple((int(a), int(b), str(k)) for a, b, k in layout.segments),
            tuple(int(x) for x in layout.video_shape),
            int(layout.audio_t),
        )

    @staticmethod
    def _resolve_layout(x, context, payload):
        try:
            from ..h3_probe.layout import resolve_layout
        except ImportError:
            from h3_probe.layout import resolve_layout
        return resolve_layout(x, context, payload)

    def _direction_reversal(self, branch, sigma):
        if not math.isfinite(sigma):
            return False
        previous = self._last_sigma.get(branch)
        self._last_sigma[branch] = sigma
        if previous is None:
            return False
        delta = sigma - previous
        if abs(delta) <= 1e-8:
            return False
        direction = self._direction.get(branch, 0)
        sign = 1 if delta > 0 else -1
        if direction == 0:
            self._direction[branch] = sign
            return False
        return sign != direction

    def _reset_request(self, layout_signature):
        self.request_id += 1
        self.layout_signature = layout_signature
        self._last_step.clear()
        self._last_sigma.clear()
        self._direction.clear()
        for listener in tuple(self.listeners):
            callback = getattr(listener, "on_request_reset", None)
            if callback is not None:
                callback(self.request_id)

    def begin_outer_request(self):
        """Start one sampler request, even when shape and schedule are identical."""
        depth = int(getattr(self._outer_local, "depth", 0))
        if depth:
            self._outer_local.depth = depth + 1
            return self._outer_local.token
        with self._lock:
            if self._active_outer_token is not None:
                raise RuntimeError("concurrent sampler requests share one H3 runtime session")
            self._outer_serial += 1
            token = self._outer_serial
            self._active_outer_token = token
            self._explicit_layout_pending = True
            self._reset_request(None)
        self._outer_local.depth = 1
        self._outer_local.token = token
        return token

    def end_outer_request(self, token):
        depth = int(getattr(self._outer_local, "depth", 0))
        if depth > 1:
            self._outer_local.depth = depth - 1
            return
        self._outer_local.depth = 0
        self._outer_local.token = None
        with self._lock:
            if self._active_outer_token == token:
                self._active_outer_token = None
                self._explicit_layout_pending = False

    def observe(self, x, timestep, context, transformer_options, payload=None):
        payload = payload or {}
        branch = self._branch(transformer_options)
        sigma = self._scalar_sigma(timestep, transformer_options)
        step_index, total_steps = self._step_index(sigma, transformer_options)

        layout = None
        layout_signature = None
        error = None
        try:
            layout = self._resolve_layout(x, context, payload)
            layout_signature = self._layout_signature_of(layout)
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
            if self.strict_layout:
                raise RuntimeError(
                    "%s could not resolve packed layout: %s" % (LOG_PREFIX, error)
                ) from exc

        device = None
        compute_dtype = getattr(context, "dtype", None)
        try:
            video = x[0] if isinstance(x, (list, tuple)) else x
            device = video.device
        except Exception:
            pass

        with self._lock:
            explicit = self._active_outer_token is not None
            last = self._last_step.get(branch)
            new_request = self.request_id < 0
            if explicit and self._explicit_layout_pending:
                self.layout_signature = layout_signature
                self._explicit_layout_pending = False
                new_request = False
            elif explicit:
                # A layout change inside one sampler request is still a hard
                # state boundary, but ordinary CFG branches keep one request.
                new_request = layout_signature != self.layout_signature
            elif not new_request:
                new_request = layout_signature != self.layout_signature
                if not new_request:
                    if step_index >= 0 and last is not None and step_index < last:
                        new_request = True
                    elif step_index < 0 and self._direction_reversal(branch, sigma):
                        new_request = True
            if new_request:
                self._reset_request(layout_signature)
                if explicit:
                    self._explicit_layout_pending = False

            if step_index >= 0:
                self._last_step[branch] = step_index
                if math.isfinite(sigma):
                    self._last_sigma[branch] = sigma

            snapshot = RuntimeSnapshot(
                request_id=self.request_id,
                step_index=step_index,
                total_steps=total_steps,
                sigma=sigma,
                branch=branch,
                layout=layout,
                layout_signature=layout_signature,
                compute_dtype=compute_dtype,
                device=device,
                error=error,
            )
            self.last_snapshot = snapshot
            transformer_options[RUNTIME_KEY] = snapshot
            for listener in tuple(self.listeners):
                callback = getattr(listener, "before_forward", None)
                if callback is not None:
                    callback(snapshot, transformer_options, payload)
            return snapshot

    def after_forward(self, snapshot, result, transformer_options):
        for listener in tuple(self.listeners):
            callback = getattr(listener, "after_forward", None)
            if callback is not None:
                callback(snapshot, result, transformer_options)


def get_runtime_snapshot(transformer_options):
    if not transformer_options:
        return None
    value = transformer_options.get(RUNTIME_KEY)
    return value if isinstance(value, RuntimeSnapshot) else None


def make_outer_wrapper(session):
    def wrapper(executor, *args, **kwargs):
        token = session.begin_outer_request()
        try:
            return executor(*args, **kwargs)
        finally:
            session.end_outer_request(token)
    return wrapper


def make_diffusion_wrapper(session):
    def wrapper(executor, *args, **kwargs):
        transformer_options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
        if transformer_options is None:
            transformer_options = {}
            if len(args) > 3:
                mutable = list(args)
                mutable[3] = transformer_options
                args = tuple(mutable)
            else:
                kwargs["transformer_options"] = transformer_options
        x = args[0] if args else kwargs.get("x")
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
        context = args[2] if len(args) > 2 else kwargs.get("context")
        payload = kwargs.get("minimax_payload") or {}
        snapshot = session.observe(x, timestep, context, transformer_options, payload)
        result = executor(*args, **kwargs)
        session.after_forward(snapshot, result, transformer_options)
        return result
    return wrapper


def install_runtime_wrapper(model_patcher, session=None):
    """Install explicit sampler and diffusion wrappers on one ModelPatcher."""
    import comfy.patcher_extension

    session = session or H3RuntimeSession()
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        OUTER_WRAPPER_KEY,
        make_outer_wrapper(session),
        model_patcher.model_options,
        is_model_options=True,
    )
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        WRAPPER_KEY,
        make_diffusion_wrapper(session),
        model_patcher.model_options,
        is_model_options=True,
    )
    options = model_patcher.model_options["transformer_options"] = (
        model_patcher.model_options.get("transformer_options", {}).copy()
    )
    options[RUNTIME_SESSION_KEY] = session
    logging.info("%s installed explicit sampler and diffusion request context", LOG_PREFIX)
    return session
