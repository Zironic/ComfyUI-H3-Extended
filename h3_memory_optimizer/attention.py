"""Capability-driven attention selection for the unified H3 memory patch.

The first optimized adapter is deliberately SM89-only. The registry and decision
objects are architecture-neutral so SM80/86/90/120 adapters can be added without
changing the node or activation-memory implementation.
"""

from dataclasses import dataclass
import logging

import torch

ATTENTION_AUTO = "auto"
ATTENTION_SM89 = "efficient_sage_sm89"
ATTENTION_EXISTING = "existing"
ATTENTION_MODES = (ATTENTION_AUTO, ATTENTION_SM89, ATTENTION_EXISTING)

FALLBACK_ALLOW = "allow"
FALLBACK_ERROR = "error"
FALLBACK_MODES = (FALLBACK_ALLOW, FALLBACK_ERROR)

LOG_PREFIX = "[H3 memory optimizer]"


class AttentionResolutionError(RuntimeError):
    """No requested optimized adapter could be selected safely."""


@dataclass(frozen=True)
class RuntimeEnvironment:
    cuda_available: bool
    device_index: int | None
    capability: tuple[int, int] | None
    device_name: str

    @classmethod
    def detect(cls):
        if not torch.cuda.is_available():
            return cls(False, None, None, "no CUDA device")
        try:
            try:
                import comfy.model_management as model_management
                device = torch.device(model_management.get_torch_device())
            except Exception:
                device = torch.device("cuda", torch.cuda.current_device())
            if device.type != "cuda":
                return cls(False, None, None, "ComfyUI model device is %s" % device)
            index = int(device.index if device.index is not None else torch.cuda.current_device())
            capability = tuple(int(v) for v in torch.cuda.get_device_capability(index))
            name = str(torch.cuda.get_device_name(index))
            return cls(True, index, capability, name)
        except Exception as exc:
            return cls(False, None, None, "CUDA probe failed: %s: %s" % (type(exc).__name__, exc))

    @property
    def architecture(self):
        if self.capability is None:
            return "none"
        return "sm%d%d" % self.capability


@dataclass(frozen=True)
class AttentionDecision:
    requested: str
    selected: str
    backend: object | None
    adapter: str | None
    reason: str
    environment: RuntimeEnvironment

    @property
    def optimized(self):
        return self.backend is not None


class SM89Adapter:
    """Current tested RTX 40-series prepared-QKV Sage adapter."""

    name = ATTENTION_SM89

    @classmethod
    def probe(cls, environment):
        if not environment.cuda_available:
            return False, environment.device_name
        if environment.capability != (8, 9):
            return False, "requires SM89, detected %s (%s)" % (
                environment.architecture,
                environment.device_name,
            )
        return True, "SM89 detected"

    @classmethod
    def build(cls):
        try:
            from ..h3_attention import SM89SageMemoryEfficientBackend
            from ..h3_attention.triton_i64 import TRITON_AVAILABLE
        except ImportError:
            from h3_attention import SM89SageMemoryEfficientBackend
            from h3_attention.triton_i64 import TRITON_AVAILABLE

        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is unavailable")

        from comfy.ldm.modules.attention import get_attention_function

        if get_attention_function("sage", default=None) is None:
            raise RuntimeError("ComfyUI has no registered 'sage' attention backend")
        return SM89SageMemoryEfficientBackend()


ADAPTERS = (SM89Adapter,)
ADAPTER_BY_NAME = {adapter.name: adapter for adapter in ADAPTERS}


def _fallback(requested, environment, fallback, reasons):
    reason = "; ".join(reasons) if reasons else "existing attention requested"
    if fallback == FALLBACK_ERROR and requested != ATTENTION_EXISTING:
        raise AttentionResolutionError(
            "cannot select %s on %s: %s" % (requested, environment.device_name, reason)
        )
    return AttentionDecision(
        requested=requested,
        selected=ATTENTION_EXISTING,
        backend=None,
        adapter=None,
        reason=reason,
        environment=environment,
    )


def resolve_attention(requested=ATTENTION_AUTO, fallback=FALLBACK_ALLOW,
                      environment=None, adapters=None):
    """Resolve an attention strategy without mutating the model.

    Expected compatibility failures are converted to an ``existing`` decision
    when fallback is allowed. Once an optimized adapter has been installed,
    runtime kernel failures remain hard errors because a CUDA fault may poison
    the context and a silent backend switch would invalidate results.
    """
    if requested not in ATTENTION_MODES:
        raise ValueError("unknown attention mode %r" % requested)
    if fallback not in FALLBACK_MODES:
        raise ValueError("unknown attention fallback %r" % fallback)

    environment = environment or RuntimeEnvironment.detect()
    if requested == ATTENTION_EXISTING:
        return _fallback(requested, environment, FALLBACK_ALLOW, [])

    registry = tuple(adapters or ADAPTERS)
    if requested == ATTENTION_AUTO:
        candidates = registry
    else:
        adapter = next((item for item in registry if item.name == requested), None)
        if adapter is None:
            return _fallback(
                requested, environment, fallback,
                ["no adapter named %s is registered" % requested],
            )
        candidates = (adapter,)

    reasons = []
    for adapter in candidates:
        supported, probe_reason = adapter.probe(environment)
        if not supported:
            reasons.append("%s: %s" % (adapter.name, probe_reason))
            continue
        try:
            backend = adapter.build()
        except Exception as exc:
            reasons.append(
                "%s preflight failed: %s: %s"
                % (adapter.name, type(exc).__name__, exc)
            )
            continue
        return AttentionDecision(
            requested=requested,
            selected=adapter.name,
            backend=backend,
            adapter=adapter.name,
            reason=probe_reason,
            environment=environment,
        )

    decision = _fallback(requested, environment, fallback, reasons)
    logging.warning(
        "%s attention fallback: requested=%s selected=%s device=%s arch=%s reason=%s",
        LOG_PREFIX,
        requested,
        decision.selected,
        environment.device_name,
        environment.architecture,
        decision.reason,
    )
    return decision
