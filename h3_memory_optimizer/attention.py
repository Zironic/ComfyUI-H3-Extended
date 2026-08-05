"""Capability-driven attention selection for the unified H3 memory patch.

The registry mirrors SageAttention 2.2's public architecture dispatch:

- SM80: CUDA FP16-V path
- SM86: Triton per-block FP16-V path
- SM89: CUDA FP8-V path
- SM90: Hopper FP8-V path
- SM120/121: SM89-family FP8 kernel with per-warp Q/K

Unknown architectures or incomplete wheels fall back before model mutation.
"""

from dataclasses import dataclass
import logging

import torch

ATTENTION_AUTO = "auto"
ATTENTION_SM80 = "efficient_sage_sm80"
ATTENTION_SM86 = "efficient_sage_sm86"
ATTENTION_SM89 = "efficient_sage_sm89"
ATTENTION_SM90 = "efficient_sage_sm90"
ATTENTION_SM12X = "efficient_sage_sm12x"
ATTENTION_EXISTING = "existing"
ATTENTION_MODES = (
    ATTENTION_AUTO,
    ATTENTION_SM80,
    ATTENTION_SM86,
    ATTENTION_SM89,
    ATTENTION_SM90,
    ATTENTION_SM12X,
    ATTENTION_EXISTING,
)

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
                return cls(
                    False,
                    None,
                    None,
                    "ComfyUI model device is %s" % device,
                )
            index = int(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            capability = tuple(
                int(v) for v in torch.cuda.get_device_capability(index)
            )
            name = str(torch.cuda.get_device_name(index))
            return cls(True, index, capability, name)
        except Exception as exc:
            return cls(
                False,
                None,
                None,
                "CUDA probe failed: %s: %s"
                % (type(exc).__name__, exc),
            )

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


def _require_registered_sage():
    from comfy.ldm.modules.attention import get_attention_function

    if get_attention_function("sage", default=None) is None:
        raise RuntimeError(
            "ComfyUI has no registered 'sage' attention backend"
        )


def _require_local_triton():
    try:
        from ..h3_attention.triton_i64 import TRITON_AVAILABLE
    except ImportError:
        from h3_attention.triton_i64 import TRITON_AVAILABLE

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is unavailable")


class _ExactCapabilityAdapter:
    name = None
    capabilities = frozenset()
    backend_name = None
    requires_local_triton = False

    @classmethod
    def probe(cls, environment):
        if not environment.cuda_available:
            return False, environment.device_name
        if environment.capability not in cls.capabilities:
            supported = ", ".join(
                "SM%d%d" % capability
                for capability in sorted(cls.capabilities)
            )
            return (
                False,
                "requires %s, detected %s (%s)"
                % (
                    supported,
                    environment.architecture,
                    environment.device_name,
                ),
            )
        return True, "%s detected" % environment.architecture.upper()

    @classmethod
    def _backend_class(cls):
        try:
            from .. import h3_attention
        except ImportError:
            import h3_attention
        return getattr(h3_attention, cls.backend_name)

    @classmethod
    def build(cls):
        _require_registered_sage()
        if cls.requires_local_triton:
            _require_local_triton()
        return cls._backend_class()()


class SM80Adapter(_ExactCapabilityAdapter):
    name = ATTENTION_SM80
    capabilities = frozenset({(8, 0)})
    backend_name = "SageSM80MemoryEfficientBackend"
    requires_local_triton = True


class SM86Adapter(_ExactCapabilityAdapter):
    name = ATTENTION_SM86
    capabilities = frozenset({(8, 6)})
    backend_name = "SageSM86MemoryEfficientBackend"
    requires_local_triton = True


class SM89Adapter(_ExactCapabilityAdapter):
    name = ATTENTION_SM89
    capabilities = frozenset({(8, 9)})
    backend_name = "SM89SageMemoryEfficientBackend"
    requires_local_triton = True


class SM90Adapter(_ExactCapabilityAdapter):
    name = ATTENTION_SM90
    capabilities = frozenset({(9, 0)})
    backend_name = "SageSM90MemoryEfficientBackend"
    requires_local_triton = True


class SM12xAdapter(_ExactCapabilityAdapter):
    name = ATTENTION_SM12X
    capabilities = frozenset({(12, 0), (12, 1)})
    backend_name = "SageSM12xMemoryEfficientBackend"
    # Upstream intentionally uses CUDA per-warp quantization on Blackwell.
    requires_local_triton = False


ADAPTERS = (
    SM80Adapter,
    SM86Adapter,
    SM89Adapter,
    SM90Adapter,
    SM12xAdapter,
)
ADAPTER_BY_NAME = {
    adapter.name: adapter
    for adapter in ADAPTERS
}


def _fallback(requested, environment, fallback, reasons):
    reason = (
        "; ".join(reasons)
        if reasons
        else "existing attention requested"
    )
    if fallback == FALLBACK_ERROR and requested != ATTENTION_EXISTING:
        raise AttentionResolutionError(
            "cannot select %s on %s: %s"
            % (requested, environment.device_name, reason)
        )
    return AttentionDecision(
        requested=requested,
        selected=ATTENTION_EXISTING,
        backend=None,
        adapter=None,
        reason=reason,
        environment=environment,
    )


def resolve_attention(
    requested=ATTENTION_AUTO,
    fallback=FALLBACK_ALLOW,
    environment=None,
    adapters=None,
):
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
        return _fallback(
            requested,
            environment,
            FALLBACK_ALLOW,
            [],
        )

    registry = tuple(adapters or ADAPTERS)
    if requested == ATTENTION_AUTO:
        candidates = registry
    else:
        adapter = next(
            (item for item in registry if item.name == requested),
            None,
        )
        if adapter is None:
            return _fallback(
                requested,
                environment,
                fallback,
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
                % (
                    adapter.name,
                    type(exc).__name__,
                    exc,
                )
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

    decision = _fallback(
        requested,
        environment,
        fallback,
        reasons,
    )
    logging.warning(
        "%s attention fallback: requested=%s selected=%s "
        "device=%s arch=%s reason=%s",
        LOG_PREFIX,
        requested,
        decision.selected,
        environment.device_name,
        environment.architecture,
        decision.reason,
    )
    return decision
