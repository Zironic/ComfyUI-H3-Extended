"""Reject H3 forwards whose irreducible physical-VRAM peak cannot fit.

The model-installed guard proves ``floor + working set + mandatory AIMDO pages
+ margin <= physical VRAM`` once per distinct forward signature. A small
physical-free check remains as an emergency monitor between proofs and around
preview decoding.
"""

import contextlib
from dataclasses import dataclass
import logging

import torch

import comfy.model_management

try:
    from . import run_context
except ImportError:  # the self-tests import this file as a top-level module
    import run_context

MB = 1024 * 1024

try:
    from .weight_footprint import footprint
    from .working_set import (
        UnknownWorkingSet,
        make_signature,
        record_observed,
        resolve_layout,
        upper_bound,
    )
except ImportError:
    from weight_footprint import footprint
    from working_set import UnknownWorkingSet, make_signature, record_observed, resolve_layout, upper_bound


def _is_cuda(device):
    return getattr(device, "type", None) == "cuda"


def _free_physical(device):
    """Bytes the CUDA driver can still hand out on `device`."""
    free, total = torch.cuda.mem_get_info(device)
    return free, total


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

    logging.warning(
        "[H3 Extended] VRAM guard tripped at %s: %s (threshold %d MB) - "
        "releasing cached blocks and re-checking",
        where, _memory_report(device, free, total, available - free), threshold_mb,
    )

    comfy.model_management.soft_empty_cache(force=True)
    available, free, total = _available(device, model_patcher=model_patcher)
    if available >= threshold:
        logging.warning(
            "[H3 Extended] VRAM guard recovered at %s: %s - continuing",
            where, _memory_report(device, free, total, available - free),
        )
        return available

    lines = [
        "[H3 Extended] VRAM guard cancelling run at %s" % where,
        "  memory: %s (threshold %d MB, still under it after releasing cached blocks)"
        % (_memory_report(device, free, total, available - free), threshold_mb),
    ]
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

    def guard(apply_model, args):
        sigma = args.get("timestep")
        where = "DiT forward"
        if sigma is not None:
            try:
                where = "DiT forward (sigma %.4f)" % float(sigma.flatten()[0])
            except (RuntimeError, ValueError, IndexError):
                pass
        signature = None
        first_signature = False
        device = _forward_device(args, model_patcher) if model_patcher is not None else None
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
        before = 0
        profiling = False
        if first_signature and _is_cuda(device):
            try:
                torch.cuda.reset_peak_memory_stats(device)
                before = int(torch.cuda.memory_allocated(device))
                profiling = True
            except Exception as exc:
                logging.warning("[H3 Extended] VRAM working-set observation unavailable: %s", exc)
        if previous is not None:
            result = previous(apply_model, args)
        else:
            result = apply_model(args["input"], args["timestep"], **args["c"])
        if profiling:
            try:
                peak = int(torch.cuda.max_memory_allocated(device))
                record_observed(signature, max(0, peak - before))
            except Exception as exc:
                logging.warning("[H3 Extended] VRAM working-set observation failed: %s", exc)
        if first_signature:
            checked_signatures.add(signature)
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
def guarded(model_options, threshold_mb, label=""):
    """Arm the guard for the duration of one sampling run, then take it back off."""
    restore = install_in_model_options(model_options, threshold_mb, label=label)
    try:
        yield
    finally:
        if restore is not None:
            restore()


def install_unet_guard(model_patcher, threshold_mb):
    """Arm the guard permanently on a cloned model patcher (the model patch node)."""
    return install_in_model_options(model_patcher.model_options, threshold_mb,
                                    label=" on the model", model_patcher=model_patcher)
