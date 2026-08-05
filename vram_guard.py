"""Abort an H3 run before the driver refuses an allocation.

12 GB is tight for H3's ~20 GB of dynamically-loaded weights, and an OOM raised
from inside the DiT forward tends to cascade through model_management's recovery
path and take the prompt_worker thread with it - which needs a full server
restart, not just a re-queue.

So instead of waiting for `CUDA error: out of memory`, this checks free *physical*
VRAM (driver-level, via `torch.cuda.mem_get_info`) before each DiT forward. Below
the threshold it dumps the memory picture and raises
`InterruptProcessingException`, which is the same exception the Cancel button
raises: the executor unwinds cleanly, marks the prompt cancelled, and the worker
survives.

Note this deliberately does *not* use `comfy.model_management.get_free_memory`,
which adds torch's reserved-but-unused bytes back in. The question here is what
the driver can still hand out, and under cudaMallocAsync a full pool is exactly
the condition worth trimming - so a breach first tries to release cached blocks
and only cancels if the memory does not come back.
"""

import contextlib
import logging

import torch

import comfy.model_management

try:
    from . import run_context
except ImportError:  # the self-tests import this file as a top-level module
    import run_context

MB = 1024 * 1024


def _is_cuda(device):
    return getattr(device, "type", None) == "cuda"


def _free_physical(device):
    """Bytes the CUDA driver can still hand out on `device`."""
    free, total = torch.cuda.mem_get_info(device)
    return free, total


def _memory_report(device, free, total):
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    return (
        "free physical %.0f MB / %.0f MB total; "
        "torch reserved %.0f MB, allocated %.0f MB"
        % (free / MB, total / MB, reserved / MB, allocated / MB)
    )


def check_vram(threshold_mb, where, device=None, detail_lines=None):
    """Log and cancel the run if free physical VRAM is under `threshold_mb`.

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
    free, total = _free_physical(device)
    if free >= threshold:
        return free

    logging.warning(
        "[H3 Extended] VRAM guard tripped at %s: %s (threshold %d MB) - "
        "releasing cached blocks and re-checking",
        where, _memory_report(device, free, total), threshold_mb,
    )

    comfy.model_management.soft_empty_cache(force=True)
    free, total = _free_physical(device)
    if free >= threshold:
        logging.warning(
            "[H3 Extended] VRAM guard recovered at %s: %s - continuing",
            where, _memory_report(device, free, total),
        )
        return free

    lines = [
        "[H3 Extended] VRAM guard cancelling run at %s" % where,
        "  memory: %s (threshold %d MB, still under it after releasing cached blocks)"
        % (_memory_report(device, free, total), threshold_mb),
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
        from .h3_probe import layout as h3_layout
    except ImportError:  # the self-tests import this file as a top-level module
        from h3_probe import layout as h3_layout

    c = args.get("c") or {}
    context = c.get("c_crossattn")
    payload = c.get("minimax_payload") or {}
    if context is None:
        return None
    tensors = getattr(x, "tensors", None)
    if not tensors or len(tensors) < 2:
        return None
    return h3_layout.resolve_layout(tensors, context, payload).describe()


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


def _make_guard(threshold_mb, previous):
    """The unet wrapper itself: check, then call through."""

    def guard(apply_model, args):
        sigma = args.get("timestep")
        where = "DiT forward"
        if sigma is not None:
            try:
                where = "DiT forward (sigma %.4f)" % float(sigma.flatten()[0])
            except (RuntimeError, ValueError, IndexError):
                pass
        check_vram(threshold_mb, where, detail_lines=lambda: _run_details(args))
        if previous is not None:
            return previous(apply_model, args)
        return apply_model(args["input"], args["timestep"], **args["c"])

    guard._h3_vram_guard = True
    return guard


def install_in_model_options(model_options, threshold_mb, label=""):
    """Check free physical VRAM before every DiT forward.

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

    guard = _make_guard(threshold_mb, previous)
    model_options["model_function_wrapper"] = guard
    logging.info(
        "[H3 Extended] VRAM guard armed%s: cancelling the run if free physical "
        "VRAM drops below %d MB during sampling", label, threshold_mb,
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
                                    label=" on the model")
