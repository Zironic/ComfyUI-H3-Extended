"""Selective capture for the probe-only 3D MoBA-style H3 router."""

from __future__ import annotations

import logging
import os

from . import capture, latent_dynamics, layout as h3_layout, moba3d

try:
    from ..h3_attention.observer import observing
except ImportError:
    from h3_attention.observer import observing

try:
    from ..h3_vector_accel.diagnostics import current_callback_metadata
except ImportError:
    from h3_vector_accel.diagnostics import current_callback_metadata


class MobaProbeRun:
    def __init__(self, session, tag, out_dir):
        self.tag = tag
        self.out_dir = out_dir
        self.layers_spec = session.layers_spec
        self.steps_spec = session.steps_spec
        self.n_time = session.n_time
        self.n_spatial = session.n_spatial
        self.query_block = session.query_block
        self.include_audio = session.include_audio
        self.include_text = session.include_text
        self.capture_uncond = session.capture_uncond
        self.capture_latent_dynamics = session.capture_latent_dynamics
        self.capture_attention = session.capture_attention
        self.block_t = session.block_t
        self.block_h = session.block_h
        self.block_w = session.block_w
        self.budgets = session.budgets
        self.execution_geometry = session.execution_geometry
        self.sage_q_tile = session.sage_q_tile
        self.sage_kv_tile = session.sage_kv_tile
        self.layers = set()
        self.steps = set()
        self.records = []
        self.layout = None
        self.notes = {}
        self.anchor_frames = []
        self.dynamics_queries = []
        self.latent_dynamics = []
        self.latent_activity_maps = []
        self.latent_energy_maps = []
        self.dynamics_tracker = latent_dynamics.LatentDynamicsTracker()


class MobaProbeSession:
    def __init__(
        self,
        tag,
        layers_spec,
        steps_spec,
        n_time,
        n_spatial,
        query_block,
        include_audio,
        include_text,
        capture_uncond,
        capture_latent_dynamics,
        block_t,
        block_h,
        block_w,
        budgets,
        base_dir,
        capture_attention=True,
        execution_geometry="logical",
        sage_q_tile=128,
        sage_kv_tile=64,
    ):
        self.tag = tag
        self.layers_spec = layers_spec
        self.steps_spec = steps_spec
        self.n_time = int(n_time)
        self.n_spatial = int(n_spatial)
        self.query_block = int(query_block)
        self.include_audio = bool(include_audio)
        self.include_text = bool(include_text)
        self.capture_uncond = bool(capture_uncond)
        self.capture_latent_dynamics = bool(capture_latent_dynamics)
        self.capture_attention = bool(capture_attention)
        self.block_t = int(block_t)
        self.block_h = int(block_h)
        self.block_w = int(block_w)
        self.budgets = moba3d.parse_budgets(budgets)
        self.base_dir = base_dir
        self.execution_geometry = str(execution_geometry or "logical").strip().lower()
        if self.execution_geometry not in ("logical", "sage_sparse"):
            raise ValueError("execution_geometry must be logical or sage_sparse")
        self.sage_q_tile = max(1, int(sage_q_tile))
        self.sage_kv_tile = max(1, int(sage_kv_tile))
        self.run = None

    def begin(self):
        import time

        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = "%s_%s" % (self.tag, stamp)
        self.run = MobaProbeRun(self, name, os.path.join(self.base_dir, name))
        return self.run

    def end(self):
        run = self.run
        self.run = None
        if run is None:
            return None
        run.dynamics_tracker.close()
        if not run.records and not run.latent_dynamics:
            logging.warning("[H3 MoBA3D probe] run finished with no captures")
            return None
        from .moba_report import write_run

        path = write_run(run)
        logging.info(
            "[H3 MoBA3D probe] %d attention records, %d dynamics records -> %s",
            len(run.records),
            len(run.latent_dynamics),
            path,
        )
        return path


class ForwardMobaProbe:
    """Live state for one H3 forward pass."""

    def __init__(self, run, layout, step, sigma, cond_or_uncond):
        self.run = run
        self.layout = layout
        self.step = step
        self.sigma = sigma
        self.cond_or_uncond = cond_or_uncond
        self.layer = -1
        self.queries = None
        self.explicit = False

    def observe(self, q, k, v, layer_index=None):
        if q.shape[2] != self.layout.seq_len:
            return
        if v is None:
            raise RuntimeError(
                "MoBA3D output-error probe requires V, but the attention "
                "observation seam did not provide it"
            )

        if layer_index is None:
            if self.explicit:
                return
            self.layer += 1
            layer = self.layer
        else:
            self.explicit = True
            layer = int(layer_index)
        if layer not in self.run.layers:
            return

        if self.queries is None:
            self.queries = capture.select_query_blocks(
                self.layout,
                self.run.n_time,
                self.run.n_spatial,
                self.run.query_block,
                include_audio=self.run.include_audio,
                include_text=self.run.include_text,
            )

        # Mean-pooled keys are query-independent, so build the 3D index only
        # once for this attention layer and reuse it for every sampled query.
        prepared = moba3d.prepare_video_router(
            k,
            self.layout,
            block_t=self.run.block_t,
            block_h=self.run.block_h,
            block_w=self.run.block_w,
        )
        for spec in self.queries:
            result = moba3d.analyze_routing(
                q,
                k,
                v,
                self.layout,
                spec["start"],
                spec["stop"],
                block_t=self.run.block_t,
                block_h=self.run.block_h,
                block_w=self.run.block_w,
                budgets=self.run.budgets,
                prepared=prepared,
                execution_geometry=self.run.execution_geometry,
                sage_q_tile=self.run.sage_q_tile,
                sage_kv_tile=self.run.sage_kv_tile,
            )
            rec = dict(spec)
            rec.update(
                {
                    "layer": layer,
                    "step": self.step,
                    "sigma": self.sigma,
                    "cond_or_uncond": self.cond_or_uncond,
                    "moba3d": result,
                }
            )
            self.run.records.append(rec)


def _replace_outer_callback(args, kwargs, callback):
    """Return args/kwargs with CFGGuider.outer_sample's callback replaced."""
    if len(args) > 5:
        args = list(args)
        args[5] = callback
        return tuple(args), kwargs
    kwargs = dict(kwargs)
    kwargs["callback"] = callback
    return args, kwargs


def make_outer_wrapper(session):
    def wrapper(executor, *args, **kwargs):
        run = session.begin()
        original_callback = args[5] if len(args) > 5 else kwargs.get("callback")
        latent_shapes = kwargs.get("latent_shapes")

        if run.capture_latent_dynamics:
            def probe_callback(step, x0, x, total_steps):
                checkpoint_arrays = False
                try:
                    rec = run.dynamics_tracker.capture(
                        run,
                        step,
                        x0,
                        x,
                        total_steps,
                        latent_shapes,
                        run.dynamics_queries,
                        callback_metadata=current_callback_metadata(),
                    )
                    if rec is not None:
                        run.latent_dynamics.append(rec)
                        checkpoint_arrays = (
                            bool(run.latent_activity_maps)
                            and len(run.latent_activity_maps) % 10 == 0
                        )
                except Exception:
                    # Diagnostics must never break sampling or the user's preview
                    # callback. The attention probe follows the same policy.
                    logging.exception(
                        "[H3 MoBA3D probe] latent dynamics capture failed at step %s",
                        step,
                    )
                if checkpoint_arrays:
                    try:
                        from .moba_report import write_run

                        write_run(run, arrays=True)
                    except Exception:
                        logging.exception(
                            "[H3 MoBA3D probe] latent dynamics checkpoint failed at step %s",
                            step,
                        )
                if original_callback is not None:
                    return original_callback(step, x0, x, total_steps)
                return None

            args, kwargs = _replace_outer_callback(args, kwargs, probe_callback)

        try:
            return executor(*args, **kwargs)
        finally:
            try:
                session.end()
            except Exception:
                logging.exception("[H3 MoBA3D probe] final report write failed")

    return wrapper


def make_wrapper(session):
    def wrapper(executor, *args, **kwargs):
        run = session.run
        transformer_options = (
            args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
        )
        x = args[0]
        context = args[2]
        payload = kwargs.get("minimax_payload") or {}
        try:
            layout = h3_layout.resolve_layout(x, context, payload)
        except Exception:
            logging.exception("[H3 MoBA3D probe] could not resolve token layout")
            return executor(*args, **kwargs)

        transformer_options["minimax_h3_token_layout"] = layout
        transformer_options["minimax_h3_token_ranges"] = layout.as_dict()
        if run is None:
            return executor(*args, **kwargs)

        if run.layout is None:
            run.layout = layout
            sched = transformer_options.get("sample_sigmas")
            total_steps = max(1, sched.numel() - 1) if sched is not None else 1
            model = getattr(executor, "class_obj", None)
            num_layers = len(getattr(model, "blocks", [])) or 50
            run.layers = set(capture.resolve_indices(run.layers_spec, num_layers))
            run.steps = set(capture.resolve_indices(run.steps_spec, total_steps))
            run.anchor_frames = latent_dynamics.resolve_anchor_frames(
                payload, layout.latent_t
            )
            if run.capture_latent_dynamics:
                # Match the video query regions used by the attention probe, but
                # omit audio/text because sampler dynamics are target-video only.
                run.dynamics_queries = capture.select_query_blocks(
                    layout,
                    run.n_time,
                    run.n_spatial,
                    run.query_block,
                    include_audio=False,
                    include_text=False,
                )
            run.notes.update(
                {
                    "total_steps": int(total_steps),
                    "num_layers": int(num_layers),
                    "mode": (
                        "latent-dynamics-only"
                        if not run.capture_attention
                        else "probe-only per-query-token 3D mean-pooled routing "
                             "with exact sparse-output comparison"
                    ),
                    "latent_dynamics": bool(run.capture_latent_dynamics),
                    "capture_attention": bool(run.capture_attention),
                    "execution_geometry": run.execution_geometry,
                    "sage_q_tile": int(run.sage_q_tile),
                    "sage_kv_tile": int(run.sage_kv_tile),
                    "latent_dynamics_source": "sampler callback x/x0",
                    "latent_dynamics_patch": [1, 2, 2],
                    "anchor_frames": list(run.anchor_frames),
                }
            )
            logging.info(
                "[H3 MoBA3D probe] layout=%s layers=%s steps=%s "
                "block=%dx%dx%d budgets=%s execution=%s q_tile=%d kv_tile=%d per-query-token routing "
                "latent_dynamics=%s attention=%s anchors=%s",
                layout.describe(),
                sorted(run.layers),
                sorted(run.steps),
                run.block_t,
                run.block_h,
                run.block_w,
                run.budgets,
                run.execution_geometry,
                run.sage_q_tile,
                run.sage_kv_tile,
                run.capture_latent_dynamics,
                run.capture_attention,
                run.anchor_frames,
            )

        if not run.capture_attention:
            return executor(*args, **kwargs)

        step, sigma = capture._step_index(transformer_options)
        cu = transformer_options.get("cond_or_uncond") or [0]
        cond_or_uncond = int(cu[0])
        if step not in run.steps or (
            not run.capture_uncond and cond_or_uncond != 0
        ):
            return executor(*args, **kwargs)

        probe = ForwardMobaProbe(run, layout, step, sigma, cond_or_uncond)
        try:
            with observing(transformer_options, probe.observe):
                return executor(*args, **kwargs)
        finally:
            try:
                from .moba_report import write_run

                write_run(run, arrays=False)
            except Exception:
                logging.exception("[H3 MoBA3D probe] report write failed")

    return wrapper
