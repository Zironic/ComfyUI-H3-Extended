"""Output-neutral sampling observers for masked Ref2V measurement.

The diffusion wrapper validates H3/source/layout state.  The post-CFG observer
scores the denoised prediction actually returned to the sampler.  The outer
wrapper also scores the final sampled latent, which is the Stage-0 ground truth.
"""

import logging

import torch
import comfy.utils

from . import mask as mask_ops
from .source import resolve_source

try:
    from ..h3_probe import layout as h3_layout
except ImportError:
    from h3_probe import layout as h3_layout

LOG_PREFIX = "[H3 Extended] masked cache"
MIN_SIGMA = 1e-3


class MeasurementUnavailable(Exception):
    pass


def _step_index(transformer_options):
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
    if not isinstance(x, (list, tuple)) or len(x) < 2 or not torch.is_tensor(x[0]):
        raise MeasurementUnavailable(
            "model input is not H3's [video, audio] pair (is this a MiniMax H3 model?)")
    return x[0]


def _unpack_video(packed, latent_shapes):
    if not torch.is_tensor(packed):
        raise MeasurementUnavailable("expected a packed latent tensor")
    if not latent_shapes or len(latent_shapes) < 2:
        raise MeasurementUnavailable("sampling run did not expose H3 latent_shapes")
    try:
        streams = comfy.utils.unpack_latents(packed, latent_shapes)
    except Exception as exc:
        raise MeasurementUnavailable("could not unpack H3 AV latent: %s" % exc) from exc
    if not isinstance(streams, (list, tuple)) or not streams or not torch.is_tensor(streams[0]):
        raise MeasurementUnavailable("unpacked sample has no video stream")
    return streams[0]


def _score_maps(video_x0, source, cfg):
    error, scale = mask_ops.score_components(video_x0.float(), source.float())
    relative = mask_ops.relative_score(error, scale, cfg.score_absolute_floor)
    return mask_ops.token_score(relative), error, scale


def make_outer_wrapper(session):
    """One trace per run, plus final-latent ground truth."""
    def wrapper(executor, *args, **kwargs):
        latent_shapes = kwargs.get("latent_shapes")
        run = session.begin(latent_shapes=latent_shapes)
        result = None
        try:
            result = executor(*args, **kwargs)
            if run.disabled_reason is None and run.source is not None:
                try:
                    final_video = _unpack_video(result, run.latent_shapes)
                    source = session.sources.get(run.source, final_video.device, torch.float32)
                    tokens, error, scale = _score_maps(final_video, source, session.config)
                    session.record_final(run, tokens.cpu(), error.cpu(), scale.cpu())
                except MeasurementUnavailable as exc:
                    logging.warning("%s final sample was not scored: %s", LOG_PREFIX, exc)
                    run.note_fallback("final sample not scored: %s" % exc)
                except Exception:
                    logging.exception("%s final sample scoring failed", LOG_PREFIX)
                    run.note_fallback("final sample scoring raised")
            return result
        finally:
            try:
                path = session.end()
                if path:
                    logging.info("%s report -> %s", LOG_PREFIX, path)
            except Exception:
                logging.exception("%s final report write failed", LOG_PREFIX)
    return wrapper


def make_diffusion_wrapper(session):
    """Validate each dense H3 call and publish state for the post-CFG observer."""
    def wrapper(executor, *args, **kwargs):
        run = session.run
        if run is None or run.disabled_reason is not None:
            return executor(*args, **kwargs)
        transformer_options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
        payload = kwargs.get("minimax_payload") or {}
        cfg = session.config
        try:
            video_x = _video_input(args[0])
            _prepare(session, run, args, transformer_options, payload, video_x)
        except MeasurementUnavailable as exc:
            reason = str(exc)
            if cfg.strict:
                raise RuntimeError(
                    "%s refusing to sample: %s. Set strict=False to sample dense and "
                    "report the fallback instead." % (LOG_PREFIX, reason)) from exc
            logging.warning("%s disabled for this run: %s", LOG_PREFIX, reason)
            run.disable(reason)
        return executor(*args, **kwargs)
    return wrapper


def make_post_cfg_observer(session):
    """Observe Comfy's final guided denoised prediction without modifying it."""
    def observer(args):
        denoised = args["denoised"]
        run = session.run
        if run is None or run.disabled_reason is not None or run.source is None:
            return denoised
        try:
            sigma_t = args.get("sigma")
            if sigma_t is None:
                raise MeasurementUnavailable("post-CFG observer received no sigma")
            sigma = float(sigma_t.flatten()[0])
            if not (sigma > MIN_SIGMA):
                return denoised
            schedule = run.notes.get("sample_sigmas_tensor")
            if schedule is None:
                step = -1
            else:
                step = int(torch.argmin((schedule - sigma).abs()))
            run.observe_sigma(sigma)
            video_x0 = _unpack_video(denoised, run.latent_shapes)
            source = session.sources.get(run.source, video_x0.device, torch.float32)
            tokens, error, scale = _score_maps(video_x0, source, session.config)
            row = session.record_step(
                run, tokens.cpu(), error.cpu(), scale.cpu(), step, sigma,
                source_kind="guided")
            logging.info(
                "%s step %d sigma %.4f: active %.1f%% core -> %.1f%% expanded%s "
                "(threshold %.3g)",
                LOG_PREFIX, step, sigma, 100.0 * row["active_core"],
                100.0 * row["active_expanded"],
                "" if row["jaccard_prev"] is None else
                ", J=%.3f vs previous" % row["jaccard_prev"],
                session.config.score_threshold)
            try:
                from .report import write_run
                write_run(run)
            except Exception:
                logging.exception("%s incremental report write failed", LOG_PREFIX)
        except MeasurementUnavailable as exc:
            logging.warning("%s guided scoring disabled: %s", LOG_PREFIX, exc)
            run.disable(str(exc))
        except Exception:
            logging.exception("%s guided scoring failed", LOG_PREFIX)
            run.disable("guided scoring raised (see traceback)")
        return denoised
    return observer


def _prepare(session, run, args, transformer_options, payload, video_x):
    cfg = session.config
    context = args[2] if len(args) > 2 else None
    sigma_t = transformer_options.get("sigmas")
    if sigma_t is None:
        raise MeasurementUnavailable("no sigma in transformer_options")
    _, sigma = _step_index(transformer_options)
    if not (sigma > MIN_SIGMA):
        return

    if "easycache" in transformer_options:
        raise MeasurementUnavailable(
            "EasyCache is active; whole-step reuse contaminates the Stage-0 trajectory")

    source_res = resolve_source(payload, video_x, cfg.source_video_ref)
    if not source_res.valid:
        raise MeasurementUnavailable(source_res.reason)

    if run.layout is None:
        try:
            run.layout = h3_layout.resolve_layout(args[0], context, payload)
        except Exception as exc:
            raise MeasurementUnavailable("could not resolve the packed layout: %s" % exc) from exc
        run.source = source_res
        sched = transformer_options.get("sample_sigmas")
        run.notes["total_steps"] = max(1, sched.numel() - 1) if sched is not None else 0
        run.notes["sample_sigmas"] = sched.detach().cpu().float().tolist() if sched is not None else None
        run.notes["sample_sigmas_tensor"] = sched.detach().cpu().float() if sched is not None else None
        override = transformer_options.get("optimized_attention_override")
        run.notes["attention_backend"] = getattr(override, "_h3_backend", None) or (
            "override" if override is not None else "comfy default")
        _log_run_header(run, video_x, source_res, cfg)


def _log_run_header(run, video_x, source_res, cfg):
    layout = run.layout
    video_rows = layout.video_range[1] - layout.video_range[0]
    logging.info(
        "%s\n  mode:              %s (strict=%s)\n"
        "  source video ref:  %d (payload index %d, %s)\n"
        "  source latent:     %s\n  target latent:     %s\n"
        "  dense sequence:    %d rows (%d target video)\n"
        "  token grid:        t=%d %dx%d\n"
        "  tile / halo:       %dx%d tiles, spatial %d, temporal %d\n"
        "  warmup / refresh:  %d / %d\n"
        "  threshold:         %.3g (floor %.3g)\n"
        "  attention:         %s\n  output:            %s",
        LOG_PREFIX, cfg.mode, cfg.strict,
        source_res.ref_ordinal, source_res.payload_index, source_res.kind,
        list(source_res.latent.shape), list(video_x.shape), layout.seq_len, video_rows,
        layout.video_shape[0], layout.video_shape[1], layout.video_shape[2],
        cfg.tile_h, cfg.tile_w, cfg.spatial_halo, cfg.temporal_halo,
        cfg.warmup_steps, cfg.refresh_interval,
        cfg.score_threshold, cfg.score_absolute_floor,
        run.notes.get("attention_backend"), run.out_dir)
