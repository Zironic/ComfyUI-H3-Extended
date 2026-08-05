"""Sampling wrappers for masked Ref2V measurement.

Two wrappers, both installed on a cloned `MODEL`:

* `OUTER_SAMPLE` - one run, one output directory, one source-latent device copy.
* `DIFFUSION_MODEL` - per model call: validate, run the *unmodified* dense
  forward, then score its output against the source.

Stage 0 never touches what the model returns. The wrapper calls the executor and
hands back exactly the object it got, so `measure` mode is a pure observer: if
it ever changes a pixel, that is a bug, not a tuning parameter.

Validation happens *before* the executor, so `strict=True` can raise while the
forward has not yet run - a measurement run that silently degrades to "no
source found, sampled anyway" is worse than one that stops, because the output
looks like evidence and is not.
"""

import logging
import time

import torch

from . import mask as mask_ops
from .source import resolve_source

try:                                                    # loaded as a custom node
    from ..h3_probe import layout as h3_layout
except ImportError:                                     # tests put the package dir on sys.path
    from h3_probe import layout as h3_layout

LOG_PREFIX = "[H3 Extended] masked cache"

# Below this sigma the forced-velocity clamp of later stages divides by an
# unstable value; measurement stops there too, so the reported score
# distributions describe the same sigma range the optimization would run in.
MIN_SIGMA = 1e-3


class MeasurementUnavailable(Exception):
    """A validation failure that must fall back to dense, or raise under strict."""


def _step_index(transformer_options):
    """Locate the current sigma in the run's schedule."""
    sched = transformer_options.get("sample_sigmas")
    sigma = transformer_options.get("sigmas")
    if sigma is None:
        return -1, float("nan")
    s = float(sigma.flatten()[0])
    if sched is None:
        return -1, s
    sched = sched.flatten().to("cpu", torch.float32)
    return int(torch.argmin((sched - s).abs())), s


def _video_input(x):
    """H3's packed model input is `[video, audio]`."""
    if not isinstance(x, (list, tuple)) or len(x) < 2 or not torch.is_tensor(x[0]):
        raise MeasurementUnavailable(
            "model input is not H3's [video, audio] pair (is this a MiniMax H3 model?)")
    return x[0]


def _video_output(out):
    if not isinstance(out, (list, tuple)) or not out or not torch.is_tensor(out[0]):
        raise MeasurementUnavailable("model output is not H3's [video, audio] pair")
    return out[0]


def make_outer_wrapper(session):
    """OUTER_SAMPLE: one trace and one source copy per sampling run."""

    def wrapper(executor, *args, **kwargs):
        session.begin()
        try:
            return executor(*args, **kwargs)
        finally:
            try:
                path = session.end()
                if path:
                    logging.info("%s report -> %s", LOG_PREFIX, path)
            except Exception:
                logging.exception("%s final report write failed", LOG_PREFIX)

    return wrapper


def make_diffusion_wrapper(session):
    """DIFFUSION_MODEL: validate, run dense, score the result against the source."""

    def wrapper(executor, *args, **kwargs):
        run = session.run
        if run is None or run.disabled_reason is not None:
            return executor(*args, **kwargs)

        transformer_options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
        payload = kwargs.get("minimax_payload") or {}
        cfg = session.config

        try:
            video_x = _video_input(args[0])
            source_res, sigma_t, step, sigma, cond_or_uncond = _prepare(
                session, run, args, transformer_options, payload, video_x)
        except MeasurementUnavailable as exc:
            reason = str(exc)
            if cfg.strict:
                raise RuntimeError(
                    "%s refusing to sample: %s. Set strict=False to sample dense "
                    "and report the fallback instead." % (LOG_PREFIX, reason)) from exc
            logging.warning("%s disabled for this run: %s", LOG_PREFIX, reason)
            run.disable(reason)
            return executor(*args, **kwargs)

        started = time.perf_counter()
        out = executor(*args, **kwargs)
        dense_wall_s = time.perf_counter() - started

        # only the conditional branch is observed; the mask has to be one mask,
        # and the conditional pass is the one that describes the requested edit
        if cond_or_uncond != 0:
            return out

        try:
            _score(session, run, out, video_x, source_res, sigma_t, step, sigma,
                   cond_or_uncond, dense_wall_s)
        except MeasurementUnavailable as exc:
            logging.warning("%s scoring disabled for this run: %s", LOG_PREFIX, exc)
            run.disable(str(exc))
        except Exception:
            # the forward already happened and its output is correct; a broken
            # observer must not take the generation down with it
            logging.exception("%s scoring failed", LOG_PREFIX)
            run.disable("scoring raised (see traceback)")
        return out

    return wrapper


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def _prepare(session, run, args, transformer_options, payload, video_x):
    """Everything that can be checked before the model runs."""
    cfg = session.config
    context = args[2] if len(args) > 2 else None

    if session.model_sampling is None:
        raise MeasurementUnavailable(
            "no model_sampling captured; the node must be applied to a MODEL")

    sigma_t = transformer_options.get("sigmas")
    if sigma_t is None:
        raise MeasurementUnavailable("no sigma in transformer_options")
    step, sigma = _step_index(transformer_options)
    if not (sigma > MIN_SIGMA):
        raise MeasurementUnavailable(
            "sigma %.3g at or below the %.3g floor" % (sigma, MIN_SIGMA))

    source_res = resolve_source(payload, video_x, cfg.source_video_ref)
    if not source_res.valid:
        raise MeasurementUnavailable(source_res.reason)

    if run.layout is None:
        try:
            layout = h3_layout.resolve_layout(args[0], context, payload)
        except Exception as exc:
            raise MeasurementUnavailable("could not resolve the packed layout: %s" % exc) from exc
        run.layout = layout
        run.source = source_res
        run.notes["total_steps"] = max(1, transformer_options["sample_sigmas"].numel() - 1) \
            if transformer_options.get("sample_sigmas") is not None else 0
        run.notes["attention_backend"] = (
            "override" if "optimized_attention_override" in transformer_options else "comfy default")
        run.notes["easycache"] = "easycache" in transformer_options
        if run.notes["easycache"]:
            # EasyCache skips whole forwards, so observed sigmas are a subset of
            # the schedule. Harmless while measuring, recorded so nobody reads
            # the resulting step count as the sampler's.
            logging.warning("%s EasyCache is active; some sigmas will not be observed", LOG_PREFIX)
        _log_run_header(run, video_x, source_res, cfg)

    cu = transformer_options.get("cond_or_uncond") or [0]
    run.observe_sigma(sigma)
    return source_res, sigma_t, step, sigma, int(cu[0])


def _score(session, run, out, video_x, source_res, sigma_t, step, sigma,
           cond_or_uncond, dense_wall_s):
    """Predicted clean latent -> token scores -> one trace row."""
    cfg = session.config
    video_out = _video_output(out)
    source = session.sources.get(source_res, video_x.device, torch.float32)

    with torch.no_grad():
        # the configured sampling object, not a re-derived flow formula: the
        # node is meant to sit downstream of the sigma-shift node, and the shift
        # is exactly the kind of thing a duplicated formula would miss
        x0 = session.model_sampling.calculate_denoised(
            sigma_t, video_out.float(), video_x.float())
        scores = mask_ops.latent_score(x0, source, cfg.score_absolute_floor)
        tokens = mask_ops.token_score(scores)

    expected = run.layout.video_shape if run.layout is not None else None
    if expected is not None and tuple(tokens.shape) != tuple(expected):
        raise MeasurementUnavailable(
            "token grid %s does not match the packed layout's video shape %s"
            % (list(tokens.shape), list(expected)))

    row = session.record_step(run, tokens.to("cpu"), step, sigma, cond_or_uncond, dense_wall_s)
    logging.info("%s step %d sigma %.4f: active %.1f%% core -> %.1f%% expanded"
                 "%s (threshold %.3g)", LOG_PREFIX, step, sigma,
                 100.0 * row["active_core"], 100.0 * row["active_expanded"],
                 "" if row["jaccard_prev"] is None else ", J=%.3f vs previous" % row["jaccard_prev"],
                 cfg.score_threshold)

    try:
        from .report import write_run
        write_run(run)
    except Exception:
        logging.exception("%s incremental report write failed", LOG_PREFIX)


def _log_run_header(run, video_x, source_res, cfg):
    layout = run.layout
    video_rows = layout.video_range[1] - layout.video_range[0]
    logging.info(
        "%s\n"
        "  mode:              %s (strict=%s)\n"
        "  source video ref:  %d (payload index %d, %s)\n"
        "  source latent:     %s\n"
        "  target latent:     %s\n"
        "  dense sequence:    %d rows (%d target video)\n"
        "  token grid:        t=%d %dx%d\n"
        "  tile / halo:       %dx%d tiles, spatial %d, temporal %d\n"
        "  threshold:         %.3g (floor %.3g)\n"
        "  attention:         %s\n"
        "  output:            %s",
        LOG_PREFIX, cfg.mode, cfg.strict,
        source_res.ref_ordinal, source_res.payload_index, source_res.kind,
        list(source_res.latent.shape), list(video_x.shape),
        layout.seq_len, video_rows,
        layout.video_shape[0], layout.video_shape[1], layout.video_shape[2],
        cfg.tile_h, cfg.tile_w, cfg.spatial_halo, cfg.temporal_halo,
        cfg.score_threshold, cfg.score_absolute_floor,
        run.notes.get("attention_backend"), run.out_dir)
