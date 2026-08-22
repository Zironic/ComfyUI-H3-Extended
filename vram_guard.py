"""Reject H3 forwards whose irreducible physical-VRAM peak cannot fit.

The model-installed guard proves ``floor + working set + mandatory AIMDO pages
+ margin <= physical VRAM`` once per distinct forward signature. A small
physical-free check remains as an emergency monitor between proofs and around
preview decoding.
"""

import contextlib
from dataclasses import dataclass
import logging
import os

import torch

import comfy.model_management

try:
    import comfy_aimdo.control as aimdo_control
except ImportError:
    aimdo_control = None

try:
    from . import run_context
    from .h3_activation_memory.observer import observing
    from .h3_runtime.timing import observing_stages
except ImportError:  # the self-tests import this file as a top-level module
    import run_context
    from h3_activation_memory.observer import observing
    from h3_runtime.timing import observing_stages

MB = 1024 * 1024

try:
    from .weight_footprint import PAGE_SIZE, footprint
    from .working_set import (
        UnknownWorkingSet,
        make_signature,
        record_observed,
        resolve_layout,
        upper_bound,
    )
except ImportError:
    from weight_footprint import PAGE_SIZE, footprint
    from working_set import UnknownWorkingSet, make_signature, record_observed, resolve_layout, upper_bound

QKV_STAGES = frozenset(("qkv_proj", "fused_qkv_projection"))


@dataclass(frozen=True)
class PhaseMemory:
    qkv: int | None
    mlp: int | None
    forward: int | None


class PhaseMemoryProfiler:
    """Sample H3 live allocations at request-local QKV and MLP seams."""

    def __init__(self, device):
        self.device = device
        self.qkv = 0
        self.mlp = 0
        self.qkv_seen = False
        self.mlp_seen = False
        self.mlp_baselines = {}

    def _allocated(self):
        return int(torch.cuda.memory_allocated(self.device))

    def begin(self, stage):
        if stage in QKV_STAGES:
            return "qkv", self._allocated()
        if stage.startswith("mlp_"):
            return "mlp", self._allocated()
        return None

    def end(self, token):
        if token is None:
            return
        phase, before = token
        delta = max(0, self._allocated() - before)
        if phase == "qkv":
            self.qkv_seen = True
            self.qkv = max(self.qkv, delta)
        else:
            self.mlp_seen = True
            self.mlp = max(self.mlp, delta)

    def __call__(self, event, layer_index, payload):
        chunk_index = payload.get("chunk_index")
        key = int(layer_index), chunk_index
        if event == "mlp_chunk_enter":
            self.mlp_baselines[key] = self._allocated()
            self.mlp_seen = True
            return
        if event not in ("mlp_fc2_ready", "mlp_epilogue_residual_ready", "mlp_chunk_gated"):
            return
        before = self.mlp_baselines.get(key)
        if before is not None:
            self.mlp = max(self.mlp, max(0, self._allocated() - before))
        if event == "mlp_chunk_gated":
            self.mlp_baselines.pop(key, None)

    def finish(self, forward):
        return PhaseMemory(
            self.qkv if self.qkv_seen else None,
            self.mlp if self.mlp_seen else None,
            forward,
        )


@dataclass
class AllocatorPolicy:
    backend: str
    previous_fraction: float | None = None
    fraction: float | None = None
    limit: int | None = None
    gc_threshold: float | None = None
    gc_source: str | None = None
    gc_target: int | None = None
    desired_gc_target: int | None = None
    changed: bool = False
    note: str | None = None
    stats_before: dict | None = None
    stats_after: dict | None = None


def _is_cuda(device):
    return getattr(device, "type", None) == "cuda"


def _free_physical(device):
    """Bytes the CUDA driver can still hand out on `device`."""
    free, total = torch.cuda.mem_get_info(device)
    return free, total


def _allocator_backend():
    try:
        return torch.cuda.memory.get_allocator_backend()
    except Exception:
        return "unavailable"


def _gc_threshold():
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        value = os.environ.get(name)
        if not value:
            continue
        for setting in value.split(","):
            key, separator, raw = setting.partition(":")
            if separator and key.strip() == "garbage_collection_threshold":
                try:
                    threshold = float(raw)
                except ValueError:
                    return None, name
                return threshold, name
    return 1.0, "PyTorch default"


def _allocator_stats(device):
    try:
        stats = torch.cuda.memory_stats(device)
    except Exception:
        return None
    keys = (
        "num_device_free",
        "segment.all.freed",
        "reserved_bytes.all.freed",
        "num_alloc_retries",
        "num_ooms",
        "num_oom_rejections",
    )
    return {key: int(stats.get(key, 0)) for key in keys}


def _stats_delta(before, after, key):
    if before is None or after is None:
        return None
    return max(0, int(after.get(key, 0)) - int(before.get(key, 0)))


def _aimdo_device_usage(device):
    if aimdo_control is None or aimdo_control.lib is None:
        return None
    index = getattr(device, "index", None)
    if index is None:
        index = torch.cuda.current_device()
    try:
        devctx = aimdo_control.get_devctx(int(index))
        return int(aimdo_control.lib.get_total_vram_usage(devctx))
    except Exception:
        return None


@contextlib.contextmanager
def _native_memory_fraction(device, guard_bytes):
    """Put native allocator GC one AIMDO page ahead of the physical guard."""
    backend = _allocator_backend()
    policy = AllocatorPolicy(backend=backend)
    if not _is_cuda(device) or backend != "native":
        policy.note = "per-process fraction is only adjusted for the native allocator"
        yield policy
        return

    threshold, source = _gc_threshold()
    policy.gc_threshold = threshold
    policy.gc_source = source
    if threshold is None or not 0.0 < threshold < 1.0:
        policy.note = "native threshold GC is not enabled with a value between 0 and 1"
        yield policy
        return

    try:
        free, total = _free_physical(device)
        reserved = int(torch.cuda.memory_reserved(device))
        allocated = int(torch.cuda.memory_allocated(device))
        previous = float(torch.cuda.get_per_process_memory_fraction(device))
        physical_used = max(0, int(total) - int(free))
        non_torch = max(0, physical_used - reserved)
        desired_target = max(0, int(total) - int(guard_bytes) - PAGE_SIZE - non_torch)
        requested_limit = int(desired_target / threshold)
        minimum_limit = min(int(total), allocated + PAGE_SIZE)
        computed_limit = min(int(total), max(requested_limit, minimum_limit))
        fraction = min(previous, computed_limit / int(total))
        limit = int(fraction * int(total))

        policy.previous_fraction = previous
        policy.fraction = fraction
        policy.limit = limit
        policy.desired_gc_target = desired_target
        policy.gc_target = int(limit * threshold)

        if minimum_limit > requested_limit:
            policy.note = "live PyTorch allocations forced the GC target inside the requested boundary"
        elif previous < computed_limit / int(total):
            policy.note = "kept the existing stricter PyTorch memory limit"
        elif requested_limit > int(total):
            policy.note = "the full-card limit already triggers GC earlier than the requested boundary"
        else:
            policy.note = "GC target aligned one 32 MiB AIMDO page ahead of the guard"

        if fraction < previous:
            torch.cuda.set_per_process_memory_fraction(fraction, device)
            policy.changed = True
        policy.stats_before = _allocator_stats(device)
    except Exception as exc:
        policy.note = "memory-fraction control unavailable: %s" % exc
        logging.warning("[H3 Extended] PyTorch memory-fraction control unavailable: %s", exc)

    try:
        yield policy
    finally:
        policy.stats_after = _allocator_stats(device)
        if policy.changed:
            try:
                torch.cuda.set_per_process_memory_fraction(policy.previous_fraction, device)
            except Exception as exc:
                logging.error("[H3 Extended] Could not restore PyTorch memory fraction: %s", exc)


def _available(device, model_patcher=None):
    """Physical free plus verified current-model unpinned AIMDO pages."""
    free, total = _free_physical(device)
    if model_patcher is not None:
        try:
            reclaimable = footprint(getattr(model_patcher, "model", None), device=device).resident_unpinned_bytes
        except Exception:
            reclaimable = 0
    else:
        reclaimable = 0
    return free + reclaimable, free, total


def _model_device(model_patcher, fallback=None):
    if fallback is not None:
        return fallback
    device = getattr(model_patcher, "load_device", None)
    if device is not None:
        return device
    return comfy.model_management.get_torch_device()


def _forward_device(args, model_patcher=None):
    tensors = getattr(args.get("input"), "tensors", None)
    if tensors:
        device = getattr(tensors[0], "device", None)
        if device is not None:
            return device
    return _model_device(model_patcher)


def _memory_diagnostic_lines(device, model_patcher=None, guard_bytes=None,
                             policy=None, phases=None, weights=None):
    try:
        free, total = _free_physical(device)
        reserved = int(torch.cuda.memory_reserved(device))
        allocated = int(torch.cuda.memory_allocated(device))
    except Exception as exc:
        return ["  Full memory diagnostic unavailable: %s" % exc]

    physical_used = max(0, int(total) - int(free))
    cache = max(0, reserved - allocated)
    if weights is None and model_patcher is not None:
        try:
            weights = footprint(getattr(model_patcher, "model", None), device=device)
        except Exception:
            weights = None
    aimdo_usage = _aimdo_device_usage(device)
    model_resident = weights.resident_bytes if weights is not None else None
    accounted_aimdo = aimdo_usage if aimdo_usage is not None else (model_resident or 0)
    other = max(0, physical_used - reserved - accounted_aimdo)

    lines = [
        "  Physical card: total %.0f MB; used %.0f MB; free %.0f MB"
        % (total / MB, physical_used / MB, free / MB),
    ]
    if aimdo_usage is not None:
        aimdo = "  AIMDO on this device: native-reported %.0f MB" % (aimdo_usage / MB)
    else:
        aimdo = "  AIMDO on this device: native total unavailable"
    if weights is not None:
        aimdo += "; current H3 model resident %.0f MB (pinned %.0f MB, reclaimable %.0f MB)" % (
            weights.resident_bytes / MB,
            weights.pinned_bytes / MB,
            weights.resident_unpinned_bytes / MB,
        )
    lines.append(aimdo)
    lines.append(
        "  Other/unattributed card use: %.0f MB residual; includes other processes and CUDA/driver/library allocations"
        % (other / MB)
    )
    lines.append(
        "  PyTorch current: reserved %.0f MB; allocated %.0f MB; cached %.0f MB"
        % (reserved / MB, allocated / MB, cache / MB)
    )

    if phases is None:
        lines.append("  PyTorch H3 QKV: pending or unavailable")
        lines.append("  PyTorch H3 MLP: pending or unavailable")
        lines.append("  PyTorch whole-forward peak increment: pending or unavailable")
    else:
        qkv = "unavailable" if phases.qkv is None else "%.0f MB" % (phases.qkv / MB)
        mlp = "unavailable" if phases.mlp is None else "%.0f MB" % (phases.mlp / MB)
        forward = "unavailable" if phases.forward is None else "%.0f MB" % (phases.forward / MB)
        lines.append("  PyTorch H3 QKV observed live delta: %s" % qkv)
        lines.append("  PyTorch H3 MLP observed live delta: %s" % mlp)
        lines.append("  PyTorch whole-forward allocator peak increment: %s" % forward)
    lines.append(
        "  PyTorch other/current allocations: the current %.0f MB active total is untagged; transient phase values above are not additive"
        % (allocated / MB)
    )

    backend = policy.backend if policy is not None else _allocator_backend()
    fraction = policy.fraction if policy is not None else None
    limit = policy.limit if policy is not None else None
    if fraction is None:
        try:
            fraction = float(torch.cuda.get_per_process_memory_fraction(device))
            limit = int(fraction * int(total))
        except Exception:
            pass
    if fraction is None or limit is None:
        lines.append("  PyTorch allocator: backend %s; process memory limit unavailable" % backend)
    else:
        label = "temporary guarded-forward limit" if policy is not None else "process memory limit"
        lines.append(
            "  PyTorch allocator: backend %s; %s %.0f MB (%.2f%% of %.0f MB physical)"
            % (backend, label, limit / MB, fraction * 100.0, total / MB)
        )

    gc_threshold = policy.gc_threshold if policy is not None else None
    gc_source = policy.gc_source if policy is not None else None
    gc_target = policy.gc_target if policy is not None else None
    if gc_threshold is None:
        gc_threshold, gc_source = _gc_threshold()
        if gc_threshold is not None and limit is not None:
            gc_target = int(gc_threshold * limit)
    if backend == "native" and gc_threshold is not None and 0.0 < gc_threshold < 1.0 and gc_target:
        pressure = reserved * 100.0 / gc_target
        lines.append(
            "  PyTorch native GC: threshold %.3f (%s); trigger %.0f MB; current reserved %.1f%% of trigger"
            % (gc_threshold, gc_source, gc_target / MB, pressure)
        )
        if policy is not None and policy.note:
            lines.append("  PyTorch native GC placement: %s" % policy.note)
    elif backend == "native":
        lines.append(
            "  PyTorch native GC: threshold collection inactive or unavailable (%s)"
            % (gc_source or "unknown configuration")
        )
    else:
        lines.append("  PyTorch GC placement: not adjusted for backend %s" % backend)

    stats = policy.stats_after if policy is not None and policy.stats_after is not None else _allocator_stats(device)
    if policy is not None and policy.stats_before is not None and stats is not None:
        device_frees = _stats_delta(policy.stats_before, stats, "num_device_free")
        segments = _stats_delta(policy.stats_before, stats, "segment.all.freed")
        freed_bytes = _stats_delta(policy.stats_before, stats, "reserved_bytes.all.freed")
        retries = _stats_delta(policy.stats_before, stats, "num_alloc_retries")
        lines.append(
            "  Last guarded forward reclamation evidence: device frees +%d; segments freed +%d; reserved bytes freed +%.0f MB; allocation retries +%d"
            % (device_frees, segments, freed_bytes / MB, retries)
        )
    elif stats is not None:
        lines.append(
            "  PyTorch allocator lifetime counters: device frees %d; segments freed %d; reserved bytes freed %.0f MB; allocation retries %d; OOMs %d; OOM rejections %d"
            % (
                stats["num_device_free"],
                stats["segment.all.freed"],
                stats["reserved_bytes.all.freed"] / MB,
                stats["num_alloc_retries"],
                stats["num_ooms"],
                stats["num_oom_rejections"],
            )
        )
    lines.append(
        "  GC attribution: PyTorch exposes no dedicated threshold-GC trigger flag; free counters are generic reclamation evidence"
    )
    if guard_bytes is not None:
        lines.append("  VRAM guard physical-free floor: %.0f MB" % (guard_bytes / MB))
    return lines


def _log_memory_diagnostic(where, device, model_patcher=None, guard_bytes=None,
                           policy=None, phases=None, weights=None, level=logging.INFO):
    lines = ["[H3 Extended] VRAM full diagnostic at %s" % where]
    lines.extend(_memory_diagnostic_lines(
        device,
        model_patcher=model_patcher,
        guard_bytes=guard_bytes,
        policy=policy,
        phases=phases,
        weights=weights,
    ))
    logging.log(level, "\n".join(lines))


@dataclass(frozen=True)
class CapacityProof:
    total: int
    floor: int
    working: int
    working_source: str
    mandatory: int
    mandatory_group: str | None
    margin: int

    @property
    def predicted(self):
        return self.floor + self.working + self.mandatory + self.margin

    @property
    def headroom(self):
        return self.total - self.predicted


def _capacity_proof(model_patcher, args, margin_mb, device=None, signature=None):
    """Check the conservative H3 capacity equation before apply_model."""
    if not margin_mb or margin_mb <= 0 or model_patcher is None:
        return None
    device = _model_device(model_patcher, device)
    if not _is_cuda(device):
        return None
    torch.cuda.synchronize(device)
    comfy.model_management.soft_empty_cache(force=True)
    free, total = torch.cuda.mem_get_info(device)
    physical_used = max(0, int(total) - int(free))
    model = getattr(model_patcher, "model", None)
    try:
        weights = footprint(model, device=device)
    except Exception as exc:
        logging.error("[H3 Extended] VRAM capacity unavailable: AIMDO page accounting failed: %s", exc)
        raise comfy.model_management.InterruptProcessingException() from exc
    # The current model's resident, unpinned pages are reclaimable.  Keep the
    # pinned portion in the non-reclaimable floor and do not add it twice.
    floor = max(0, physical_used - weights.resident_unpinned_bytes)
    signature = signature or make_signature(args, model_patcher)
    working, source = upper_bound(signature)
    margin = int(margin_mb) * MB
    mandatory = int(weights.mandatory_bytes)
    proof = CapacityProof(int(total), floor, working, source, mandatory,
                          weights.mandatory_group, margin)
    logging.info(
        "[H3 Extended] VRAM capacity proof: total=%d MB floor=%d MB "
        "working=%d MB (%s) mandatory AIMDO pages=%d MB (%s) margin=%d MB "
        "predicted=%d MB %s=%d MB",
        proof.total // MB, proof.floor // MB, proof.working // MB, proof.working_source,
        proof.mandatory // MB, proof.mandatory_group or "none", proof.margin // MB,
        proof.predicted // MB, "headroom" if proof.headroom >= 0 else "deficit",
        abs(proof.headroom) // MB,
    )
    if proof.headroom >= 0:
        _log_memory_diagnostic(
            "capacity proof",
            device,
            model_patcher=model_patcher,
            guard_bytes=margin,
            weights=weights,
        )
    if proof.headroom < 0:
        lines = [
            "[H3 Extended] VRAM capacity cancelling run before apply_model",
            "  Physical VRAM: %d MB" % (proof.total // MB),
            "  Non-reclaimable starting floor: %d MB" % (proof.floor // MB),
            "  H3 working-set upper bound: %d MB (%s)" % (proof.working // MB, proof.working_source),
            "  Mandatory AIMDO pages: %d MB (%s)" % (proof.mandatory // MB, proof.mandatory_group or "none"),
            "  Safety margin: %d MB" % (proof.margin // MB),
            "  Predicted physical peak: %d MB" % (proof.predicted // MB),
            "  Deficit: %d MB" % (-proof.headroom // MB),
        ]
        lines.extend(_memory_diagnostic_lines(
            device,
            model_patcher=model_patcher,
            guard_bytes=margin,
            weights=weights,
        ))
        try:
            lines.extend(_run_details(args))
        except Exception:
            logging.exception("[H3 Extended] VRAM guard could not describe the run")
        logging.error(
            "\n".join(lines)
        )
        raise comfy.model_management.InterruptProcessingException()
    return signature, proof


def _memory_report(device, free, total, reclaimable=None):
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    text = (
        "free physical %.0f MB / %.0f MB total; "
        "torch reserved %.0f MB, allocated %.0f MB"
        % (free / MB, total / MB, reserved / MB, allocated / MB)
    )
    if reclaimable:
        text += ("; AIMDO reclaimable %.0f MB -> %.0f MB available"
                 % (reclaimable / MB, (free + reclaimable) / MB))
    return text


def check_vram(threshold_mb, where, device=None, detail_lines=None, model_patcher=None):
    """Secondary emergency check for a low physical-free watermark.

    `detail_lines` is an optional zero-arg callable returning extra log lines. It
    is only called on the cancel path, so describing the run costs nothing on the
    thousands of checks that pass.

    Returns the free byte count (or None when the check does not apply), so
    callers can log it. Raises InterruptProcessingException on a breach that
    survives a cache release.
    """
    if not threshold_mb or threshold_mb <= 0:
        return None
    if device is None:
        device = comfy.model_management.get_torch_device()
    if not _is_cuda(device):
        return None

    threshold = threshold_mb * MB
    available, free, total = _available(device, model_patcher=model_patcher)
    if available >= threshold:
        return available

    trip_lines = [
        "[H3 Extended] VRAM guard tripped at %s: %s (threshold %d MB) - releasing cached blocks and re-checking"
        % (where, _memory_report(device, free, total, available - free), threshold_mb)
    ]
    trip_lines.extend(_memory_diagnostic_lines(
        device,
        model_patcher=model_patcher,
        guard_bytes=threshold,
    ))
    logging.warning("\n".join(trip_lines))

    comfy.model_management.soft_empty_cache(force=True)
    available, free, total = _available(device, model_patcher=model_patcher)
    if available >= threshold:
        recovered_lines = [
            "[H3 Extended] VRAM guard recovered at %s: %s - continuing"
            % (where, _memory_report(device, free, total, available - free))
        ]
        recovered_lines.extend(_memory_diagnostic_lines(
            device,
            model_patcher=model_patcher,
            guard_bytes=threshold,
        ))
        logging.warning("\n".join(recovered_lines))
        return available

    lines = [
        "[H3 Extended] VRAM guard cancelling run at %s" % where,
        "  memory: %s (threshold %d MB, still under it after releasing cached blocks)"
        % (_memory_report(device, free, total, available - free), threshold_mb),
    ]
    lines.extend(_memory_diagnostic_lines(
        device,
        model_patcher=model_patcher,
        guard_bytes=threshold,
    ))
    if detail_lines is not None:
        try:
            lines.extend(detail_lines())
        except Exception:  # a broken description must not mask the cancellation
            logging.exception("[H3 Extended] VRAM guard could not describe the run")
    lines.append(
        "  Cancelling now rather than letting the next allocation raise a CUDA OOM, "
        "which can kill the prompt worker. Reduce length/resolution, lower the "
        "reference image size, or free VRAM used by other processes."
    )
    logging.error("\n".join(lines))
    raise comfy.model_management.InterruptProcessingException()


def _latent_shapes(x):
    """(video, audio) latent shapes from the H3 NestedTensor being denoised."""
    tensors = getattr(x, "tensors", None)
    if tensors and len(tensors) >= 2:
        return tuple(tensors[0].shape), tuple(tensors[1].shape)
    shape = getattr(x, "shape", None)
    return (tuple(shape) if shape is not None else None), None


def _packed_seq_len(x, args):
    """Describe the packed token sequence attention will run over, if resolvable.

    This is the number that actually drives attention memory, and unlike the node
    inputs it is measured from the live forward pass rather than remembered.
    """
    try:
        return resolve_layout(args).describe()
    except UnknownWorkingSet:
        return None


def _run_details(args):
    """Everything worth knowing about the run being cancelled.

    `args` is the unet wrapper's argument dict. A caller outside the forward pass
    (the sampler node, between steps) has no conditioning to hand over, so it
    passes just `{"input": latent}` and the packed-token line is skipped.
    """
    lines = []
    x = args.get("input")
    video_shape, audio_shape = _latent_shapes(x)

    sampling = []
    if video_shape is not None:
        sampling.append("video latent %s" % (list(video_shape),))
    if audio_shape is not None:
        sampling.append("audio latent %s" % (list(audio_shape),))
    cond_or_uncond = args.get("cond_or_uncond")
    if cond_or_uncond is not None:
        sampling.append("cond_or_uncond %s" % (cond_or_uncond,))
    if sampling:
        lines.append("  sampling: %s" % ", ".join(sampling))

    try:
        packed = _packed_seq_len(x, args)
    except Exception as e:
        packed = "unavailable (%s)" % e
    if packed:
        lines.append("  packed tokens: %s" % packed)

    latent_shape = video_shape if video_shape and len(video_shape) == 5 else None
    node_lines = run_context.describe(latent_shape)
    if node_lines:
        lines.append("  node inputs for this run:")
        lines.extend("  " + line for line in node_lines)
    else:
        lines.append("  node inputs: not recorded (no (Zi) H3 conditioning node ran)")
    return lines


def check_latent(threshold_mb, where, latent, device=None):
    """Check from outside the forward pass, describing the latent being sampled.

    Used by the sampler node between steps, where the preview decoder — a
    resident 2.26 GiB fp8 model — has just allocated and the pre-forward check
    has not run yet.
    """
    return check_vram(threshold_mb, where, device=device,
                      detail_lines=lambda: _run_details({"input": latent}))


def _make_guard(threshold_mb, previous, model_patcher=None):
    """The unet wrapper itself: check, then call through."""

    checked_signatures = set()
    emergency_profiled = False

    def guard(apply_model, args):
        nonlocal emergency_profiled
        sigma = args.get("timestep")
        where = "DiT forward"
        if sigma is not None:
            try:
                where = "DiT forward (sigma %.4f)" % float(sigma.flatten()[0])
            except (RuntimeError, ValueError, IndexError):
                pass
        signature = None
        first_signature = False
        device = _forward_device(args, model_patcher)
        if model_patcher is not None:
            try:
                signature = make_signature(args, model_patcher)
            except Exception as exc:
                logging.error("[H3 Extended] VRAM capacity unavailable before apply_model: %s", exc)
                raise comfy.model_management.InterruptProcessingException() from exc
            if signature not in checked_signatures:
                _capacity_proof(model_patcher, args, threshold_mb, device=device, signature=signature)
                first_signature = True
        check_vram(threshold_mb, where, detail_lines=lambda: _run_details(args),
                   model_patcher=model_patcher, device=device)
        profile_this = first_signature or (model_patcher is None and not emergency_profiled)
        phase_memory = None
        with _native_memory_fraction(device, threshold_mb * MB) as policy:
            before = 0
            profiling = False
            profiler = None
            if profile_this and _is_cuda(device):
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                    before = int(torch.cuda.memory_allocated(device))
                    profiler = PhaseMemoryProfiler(device)
                    profiling = True
                except Exception as exc:
                    logging.warning("[H3 Extended] VRAM working-set observation unavailable: %s", exc)

            transformer_options = args.get("c", {}).get("transformer_options")
            with contextlib.ExitStack() as stack:
                if profiling and isinstance(transformer_options, dict):
                    stack.enter_context(observing_stages(transformer_options, profiler))
                    stack.enter_context(observing(transformer_options, profiler))
                if previous is not None:
                    result = previous(apply_model, args)
                else:
                    result = apply_model(args["input"], args["timestep"], **args["c"])

            if profiling:
                try:
                    peak = int(torch.cuda.max_memory_allocated(device))
                    forward_peak = max(0, peak - before)
                    phase_memory = profiler.finish(forward_peak)
                    if signature is not None:
                        record_observed(signature, forward_peak)
                except Exception as exc:
                    logging.warning("[H3 Extended] VRAM working-set observation failed: %s", exc)

        reclaimed = (
            _stats_delta(policy.stats_before, policy.stats_after, "num_device_free") or 0
        ) or (
            _stats_delta(policy.stats_before, policy.stats_after, "segment.all.freed") or 0
        ) or (
            _stats_delta(policy.stats_before, policy.stats_after, "num_alloc_retries") or 0
        )
        if profile_this or reclaimed:
            _log_memory_diagnostic(
                "first successful %s" % where if profile_this else "%s allocator reclamation" % where,
                device,
                model_patcher=model_patcher,
                guard_bytes=threshold_mb * MB,
                policy=policy,
                phases=phase_memory,
            )
        if first_signature:
            checked_signatures.add(signature)
        if model_patcher is None and profile_this:
            emergency_profiled = True
        return result

    guard._h3_vram_guard = True
    guard._h3_vram_capacity = model_patcher is not None
    guard._h3_checked_signatures = checked_signatures
    return guard


def install_in_model_options(model_options, threshold_mb, label="", model_patcher=None):
    """Install capacity plus emergency checks, or an emergency-only wrapper.

    Chains onto any wrapper already installed rather than replacing it, so this
    composes with other model patches — but skips if an H3 guard is already
    present, so arming from both the model patch node and the sampler node does
    not stack two identical checks.

    Returns a zero-arg callable that undoes the install, or None if nothing was
    installed. Callers that write into a *shared* dict — `CFGGuider.model_options`
    is the model patcher's own dict, not a copy — must call it in a `finally`,
    or the wrapper survives onto the cached patcher and re-arms every later run.
    """
    if not threshold_mb or threshold_mb <= 0:
        return None

    previous = model_options.get("model_function_wrapper")
    if getattr(previous, "_h3_vram_guard", False):
        return None

    guard = _make_guard(threshold_mb, previous, model_patcher=model_patcher)
    model_options["model_function_wrapper"] = guard
    if model_patcher is None:
        logging.info(
            "[H3 Extended] VRAM emergency guard armed%s: %d MB physical-free floor",
            label, threshold_mb,
        )
    else:
        logging.info(
            "[H3 Extended] VRAM capacity guard armed%s: %d MB safety margin and emergency floor",
            label, threshold_mb,
        )

    def restore():
        if model_options.get("model_function_wrapper") is guard:
            if previous is None:
                model_options.pop("model_function_wrapper", None)
            else:
                model_options["model_function_wrapper"] = previous

    return restore


@contextlib.contextmanager
def guarded(model_options, threshold_mb, label="", model_patcher=None):
    """Arm the guard for the duration of one sampling run, then take it back off."""
    restore = install_in_model_options(
        model_options, threshold_mb, label=label, model_patcher=model_patcher,
    )
    try:
        yield
    finally:
        if restore is not None:
            restore()


def install_unet_guard(model_patcher, threshold_mb):
    """Arm the guard permanently on a cloned model patcher (the model patch node)."""
    return install_in_model_options(model_patcher.model_options, threshold_mb,
                                    label=" on the model", model_patcher=model_patcher)
