"""Lightweight H3 sparse-router characterization for HASTE/static-topology work.

This experiment augments the existing MoBA3D probe without changing inference.
It has two jobs:

1. Observe the direct production-like Q=128/KV=64 tile router on every sampled
   denoising step and layer, measuring temporal Q/K drift and route stability.
2. Add direct-tile sparse-output error curves to the existing expensive MoBA3D
   snapshots, so head/layer density calibration is based on the route we could
   actually execute rather than on fine logical masks coalesced afterward.

The lightweight path never forms dense attention probabilities.  It stores only
small per-head drift/reuse metrics plus sampled topology counts.  The existing
MoBA3D exact-error path remains selective according to the node's ``layers`` and
``steps`` controls.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import capture, latent_dynamics, layout as h3_layout, moba3d, moba_capture, moba_report

try:
    from ..h3_attention.observer import observing
except ImportError:
    from h3_attention.observer import observing


DEFAULT_DYNAMICS_BUDGET = 0.50
DEFAULT_TOPOLOGY_Q_SAMPLES = 8
_INSTALLED = False
_ORIGINAL_ANALYZE_ROUTING = None
_ORIGINAL_WRITE_RUN = None
_ORIGINAL_MAKE_WRAPPER = None

# One attention layer is queried several times by the expensive probe.  Keep the
# latest direct-router summary/score state so all query regions reuse it.
_DIRECT_STATE_CACHE = None


def _mean_pool(x, block):
    """Mean-pool the packed sequence at the requested execution granularity."""
    block = max(1, int(block))
    sequence = int(x.shape[-2])
    full = sequence // block
    remainder = sequence % block
    pieces = []
    if full:
        pieces.append(
            x[..., : full * block, :]
            .reshape(*x.shape[:-2], full, block, x.shape[-1])
            .float()
            .mean(dim=-2)
        )
    if remainder:
        pieces.append(x[..., full * block :, :].float().mean(dim=-2, keepdim=True))
    if not pieces:
        raise ValueError("cannot pool an empty sequence")
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)


def _geometry(layout, q_tile, kv_tile):
    q_tile = max(1, int(q_tile))
    kv_tile = max(1, int(kv_tile))
    sequence = int(layout.seq_len)
    video_start, video_stop = (int(x) for x in layout.video_range)
    if not (0 <= video_start < video_stop == sequence):
        raise ValueError(
            "direct tile characterization requires target video to be the final packed segment"
        )
    q_tiles = (sequence + q_tile - 1) // q_tile
    kv_tiles = (sequence + kv_tile - 1) // kv_tile
    pure_q_start = (video_start + q_tile - 1) // q_tile
    pure_kv_start = (video_start + kv_tile - 1) // kv_tile
    return {
        "sequence": sequence,
        "q_tile": q_tile,
        "kv_tile": kv_tile,
        "q_tiles": q_tiles,
        "kv_tiles": kv_tiles,
        "pure_q_start": pure_q_start,
        "pure_kv_start": pure_kv_start,
        "pure_q": q_tiles - pure_q_start,
        "pure_kv": kv_tiles - pure_kv_start,
    }


@dataclass
class _DirectRouterState:
    q_identity: tuple
    k_identity: tuple
    geometry: dict
    q_summary: torch.Tensor
    k_summary: torch.Tensor
    scores: torch.Tensor
    indices_by_budget: dict


def _tensor_identity(value):
    try:
        ptr = int(value.untyped_storage().data_ptr())
    except Exception:
        ptr = id(value)
    return (ptr, tuple(int(x) for x in value.shape), str(value.device), str(value.dtype))


def _prepare_direct_state(q, k, layout, q_tile, kv_tile):
    global _DIRECT_STATE_CACHE
    geometry = _geometry(layout, q_tile, kv_tile)
    q_identity = _tensor_identity(q)
    k_identity = _tensor_identity(k)
    cached = _DIRECT_STATE_CACHE
    if (
        cached is not None
        and cached.q_identity == q_identity
        and cached.k_identity == k_identity
        and cached.geometry == geometry
    ):
        return cached

    q_summary = _mean_pool(q, geometry["q_tile"])
    k_summary = _mean_pool(k, geometry["kv_tile"])
    if geometry["pure_q"] <= 0 or geometry["pure_kv"] <= 0:
        raise ValueError("packed layout has no pure-video execution tiles")
    scores = torch.matmul(
        q_summary[..., geometry["pure_q_start"] :, :],
        k_summary[..., geometry["pure_kv_start"] :, :].transpose(-1, -2),
    )
    cached = _DirectRouterState(
        q_identity=q_identity,
        k_identity=k_identity,
        geometry=geometry,
        q_summary=q_summary,
        k_summary=k_summary,
        scores=scores,
        indices_by_budget={},
    )
    _DIRECT_STATE_CACHE = cached
    return cached


def _budget_target(budget, pure_kv):
    budget = float(budget)
    if budget > 1.0:
        budget /= 100.0
    if not 0.0 < budget <= 1.0:
        raise ValueError("router budget must lie in (0, 1]")
    return min(int(pure_kv), max(1, int(math.ceil(budget * pure_kv))))


def _selected_indices(state, budget):
    target = _budget_target(budget, state.geometry["pure_kv"])
    cached = state.indices_by_budget.get(target)
    if cached is None:
        cached = torch.topk(state.scores, target, dim=-1, sorted=False).indices
        cached = cached.sort(dim=-1).values
        state.indices_by_budget[target] = cached
    return cached, target


def _direct_keep_for_query_range(state, selected, qs, qe):
    """Expand direct per-tile routes to a packed-token mask for sampled Q rows."""
    geometry = state.geometry
    qs, qe = int(qs), int(qe)
    if not 0 <= qs < qe <= geometry["sequence"]:
        raise ValueError("query range lies outside the packed sequence")

    device = selected.device
    heads = int(selected.shape[1])
    queries = qe - qs
    q_global = torch.arange(qs, qe, device=device)
    q_tile_ids = torch.div(q_global, geometry["q_tile"], rounding_mode="floor")
    kv_global = torch.arange(geometry["sequence"], device=device)
    kv_tile_ids = torch.div(kv_global, geometry["kv_tile"], rounding_mode="floor")

    tile_keep = torch.ones(
        heads,
        queries,
        geometry["kv_tiles"],
        dtype=torch.bool,
        device=device,
    )
    sparse_query = q_tile_ids >= geometry["pure_q_start"]
    if bool(sparse_query.any()):
        sparse_positions = torch.nonzero(sparse_query, as_tuple=False).flatten()
        tile_keep[:, sparse_positions, geometry["pure_kv_start"] :] = False
        route_rows = q_tile_ids[sparse_positions] - geometry["pure_q_start"]
        routed = selected[0].index_select(1, route_rows)
        absolute = routed + geometry["pure_kv_start"]
        head_index = torch.arange(heads, device=device)[:, None, None].expand_as(absolute)
        query_index = sparse_positions[None, :, None].expand_as(absolute)
        tile_keep[head_index, query_index, absolute] = True

    return tile_keep.index_select(2, kv_tile_ids)


def _direct_tile_calibration(
    q,
    k,
    v,
    layout,
    qs,
    qe,
    budgets,
    *,
    head_chunk,
    q_tile,
    kv_tile,
):
    """Exact sparse-output error for the direct production-like tile router."""
    state = _prepare_direct_state(q, k, layout, q_tile, kv_tile)
    budgets = tuple(float(x) for x in moba3d.parse_budgets(budgets))
    scale = float(q.shape[-1]) ** -0.5
    heads = int(q.shape[1])
    head_chunk = max(1, int(head_chunk))
    accum = {
        frac: {"density": [], "rel_l2": [], "mean_abs": [], "max_abs": []}
        for frac in budgets
    }

    for h0 in range(0, heads, head_chunk):
        h1 = min(heads, h0 + head_chunk)
        qh = q[0, h0:h1, qs:qe, :].float()
        kh = k[0, h0:h1].float()
        vh = v[0, h0:h1].float()
        probs = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)) * scale, dim=-1)
        dense_out = torch.matmul(probs, vh)

        for frac in budgets:
            selected, _target = _selected_indices(state, frac)
            keep = _direct_keep_for_query_range(
                state,
                selected[:, h0:h1],
                qs,
                qe,
            )
            _mass, sparse_out = moba3d._renormalized_masked_output(probs, vh, keep)
            rel_l2, mean_abs, max_abs = moba3d._per_head_error(sparse_out, dense_out)
            bucket = accum[frac]
            bucket["density"].append(keep.float().mean(-1).detach().cpu())
            bucket["rel_l2"].append(rel_l2.detach().cpu())
            bucket["mean_abs"].append(mean_abs.detach().cpu())
            bucket["max_abs"].append(max_abs.detach().cpu())

    result = {}
    for frac in budgets:
        bucket = accum[frac]
        density = torch.cat(bucket["density"], dim=0)
        rel_l2 = torch.cat(bucket["rel_l2"], dim=0)
        mean_abs = torch.cat(bucket["mean_abs"], dim=0)
        max_abs = torch.cat(bucket["max_abs"], dim=0)
        selected, target = _selected_indices(state, frac)
        result[frac] = {
            "direct_tile_keep_video_kv_tiles": int(target),
            "direct_tile_video_kv_tiles": int(state.geometry["pure_kv"]),
            "direct_tile_video_density": float(target / state.geometry["pure_kv"]),
            "direct_tile_effective_token_density_mean": float(density.mean()),
            "direct_tile_effective_token_density_max": float(density.max()),
            "direct_tile_sparse_output_rel_l2_mean_head": float(rel_l2.mean()),
            "direct_tile_sparse_output_rel_l2_median_head": float(rel_l2.median()),
            "direct_tile_sparse_output_rel_l2_max_head": float(rel_l2.max()),
            "direct_tile_sparse_output_mean_abs_mean_head": float(mean_abs.mean()),
            "direct_tile_sparse_output_max_abs": float(max_abs.max()),
            "direct_tile_head_rel_l2": [float(x) for x in rel_l2.tolist()],
            "direct_tile_q_tile": int(state.geometry["q_tile"]),
            "direct_tile_kv_tile": int(state.geometry["kv_tile"]),
        }
        del selected
    return result


def _analyze_routing_with_direct_calibration(*args, **kwargs):
    result = _ORIGINAL_ANALYZE_ROUTING(*args, **kwargs)
    if result.get("execution_geometry") != "sage_sparse":
        return result
    if len(args) < 6:
        return result

    q, k, v, layout, qs, qe = args[:6]
    if v is None:
        return result
    budgets = kwargs.get("budgets", moba3d.DEFAULT_BUDGETS)
    head_chunk = kwargs.get("head_chunk", 4)
    q_tile = int(kwargs.get("sage_q_tile", result.get("sage_q_tile") or 128))
    kv_tile = int(kwargs.get("sage_kv_tile", result.get("sage_kv_tile") or 64))

    calibration = _direct_tile_calibration(
        q,
        k,
        v,
        layout,
        int(qs),
        int(qe),
        budgets,
        head_chunk=head_chunk,
        q_tile=q_tile,
        kv_tile=kv_tile,
    )
    for row in result.get("budgets", ()):
        row.update(calibration[float(row["budget"])])
    result["direct_tile_calibration"] = True
    result["direct_tile_q_tile"] = q_tile
    result["direct_tile_kv_tile"] = kv_tile
    return result


def _route_hash(indices):
    """Two inexpensive int64 moments used only for exact-row reuse detection."""
    values = indices.to(torch.int64) + 1
    rank = torch.arange(
        1,
        values.shape[-1] + 1,
        device=values.device,
        dtype=torch.int64,
    )
    first = torch.sum(values * rank, dim=-1)
    second = torch.sum(values * values * (rank * 2 + 1), dim=-1)
    return first, second


def _sample_q_rows(pure_q, count, device):
    count = min(max(1, int(count)), max(1, int(pure_q)))
    if count == 1:
        return torch.tensor([max(0, pure_q // 2)], dtype=torch.long, device=device)
    return torch.linspace(0, pure_q - 1, steps=count, device=device).round().long().unique()


def _sample_route_mask(indices, pure_kv, sample_rows):
    sampled = indices.index_select(2, sample_rows)
    mask = torch.zeros(
        sampled.shape[0],
        sampled.shape[1],
        sampled.shape[2],
        int(pure_kv),
        dtype=torch.bool,
        device=indices.device,
    )
    mask.scatter_(3, sampled, True)
    return mask


def _signature(summary, start):
    value = summary[..., int(start) :, :].mean(dim=-2)
    return F.normalize(value.float(), dim=-1, eps=1e-12)


@dataclass
class _PreviousRoute:
    q_signature: torch.Tensor
    k_signature: torch.Tensor
    hash_a: torch.Tensor
    hash_b: torch.Tensor
    sampled_mask: torch.Tensor


class RouterDynamicsTracker:
    """Request-scoped direct-router state for temporal and topology measurements."""

    def __init__(
        self,
        *,
        q_tile=128,
        kv_tile=64,
        budget=DEFAULT_DYNAMICS_BUDGET,
        topology_q_samples=DEFAULT_TOPOLOGY_Q_SAMPLES,
    ):
        self.q_tile = max(1, int(q_tile))
        self.kv_tile = max(1, int(kv_tile))
        self.budget = float(budget)
        self.topology_q_samples = max(1, int(topology_q_samples))
        self.previous = {}
        self.topology = {}

    def capture(self, q, k, layout, *, step, sigma, branch, layer):
        state = _prepare_direct_state(q, k, layout, self.q_tile, self.kv_tile)
        selected, target = _selected_indices(state, self.budget)
        geometry = state.geometry
        q_signature = _signature(state.q_summary, geometry["pure_q_start"])
        k_signature = _signature(state.k_summary, geometry["pure_kv_start"])
        hash_a, hash_b = _route_hash(selected)
        sample_rows = _sample_q_rows(
            geometry["pure_q"], self.topology_q_samples, selected.device
        )
        sampled_mask = _sample_route_mask(selected, geometry["pure_kv"], sample_rows)

        key = (int(branch), int(layer))
        previous = self.previous.get(key)
        if previous is None:
            q_cos = k_cos = exact = jaccard = None
        else:
            q_cos = (q_signature * previous.q_signature).sum(dim=-1)[0]
            k_cos = (k_signature * previous.k_signature).sum(dim=-1)[0]
            exact_rows = (hash_a == previous.hash_a) & (hash_b == previous.hash_b)
            exact = exact_rows.float().mean(dim=-1)[0]
            intersection = (sampled_mask & previous.sampled_mask).sum(dim=(-1, -2)).float()[0]
            union = (sampled_mask | previous.sampled_mask).sum(dim=(-1, -2)).float()[0]
            jaccard = intersection / union.clamp_min(1.0)

        self.previous[key] = _PreviousRoute(
            q_signature=q_signature.detach(),
            k_signature=k_signature.detach(),
            hash_a=hash_a.detach(),
            hash_b=hash_b.detach(),
            sampled_mask=sampled_mask.detach(),
        )

        # Aggregate sampled topology on CPU.  This is intentionally tiny relative
        # to Q/K/V and gives cross-run static-topology analysis without archiving
        # full masks for every step.
        topo = self.topology.get(int(layer))
        sampled_cpu = sampled_mask[0].to(device="cpu", dtype=torch.int16)
        sample_global = (sample_rows + geometry["pure_q_start"]).to(device="cpu")
        if topo is None:
            topo = {
                "counts": torch.zeros_like(sampled_cpu, dtype=torch.int16),
                "q_tiles": sample_global.clone(),
                "observations": 0,
                "pure_kv": int(geometry["pure_kv"]),
                "pure_kv_start": int(geometry["pure_kv_start"]),
            }
            self.topology[int(layer)] = topo
        topo["counts"].add_(sampled_cpu)
        topo["observations"] += 1

        def _head_list(value):
            return None if value is None else [float(x) for x in value.detach().cpu().tolist()]

        record = {
            "step": int(step),
            "sigma": float(sigma),
            "branch": int(branch),
            "layer": int(layer),
            "q_tile": int(self.q_tile),
            "kv_tile": int(self.kv_tile),
            "budget": float(self.budget),
            "retained_video_kv_tiles": int(target),
            "pure_video_q_tiles": int(geometry["pure_q"]),
            "pure_video_kv_tiles": int(geometry["pure_kv"]),
            "q_cosine_by_head": _head_list(q_cos),
            "k_cosine_by_head": _head_list(k_cos),
            "exact_route_reuse_fraction_by_head": _head_list(exact),
            "sampled_route_jaccard_by_head": _head_list(jaccard),
        }
        for key_name in (
            "q_cosine_by_head",
            "k_cosine_by_head",
            "exact_route_reuse_fraction_by_head",
            "sampled_route_jaccard_by_head",
        ):
            values = record[key_name]
            record[key_name.replace("_by_head", "_mean")] = (
                None if values is None else float(sum(values) / max(1, len(values)))
            )
        return record

    def write_topology(self, path):
        raw = {
            "q_tile": np.asarray([self.q_tile], dtype=np.int32),
            "kv_tile": np.asarray([self.kv_tile], dtype=np.int32),
            "budget": np.asarray([self.budget], dtype=np.float32),
        }
        for layer, topo in sorted(self.topology.items()):
            prefix = "layer_%02d_" % int(layer)
            raw[prefix + "counts"] = topo["counts"].numpy().astype(np.uint16)
            raw[prefix + "q_tiles"] = topo["q_tiles"].numpy().astype(np.int32)
            raw[prefix + "observations"] = np.asarray(
                [topo["observations"]], dtype=np.int32
            )
            raw[prefix + "pure_kv"] = np.asarray([topo["pure_kv"]], dtype=np.int32)
            raw[prefix + "pure_kv_start"] = np.asarray(
                [topo["pure_kv_start"]], dtype=np.int32
            )
        if len(raw) > 3:
            np.savez_compressed(path, **raw)


class _CombinedForwardProbe:
    """Observe every layer cheaply; run the old exact probe only on selected snapshots."""

    def __init__(self, run, layout, step, sigma, branch, snapshot_enabled):
        self.run = run
        self.layout = layout
        self.step = int(step)
        self.sigma = sigma
        self.branch = int(branch)
        self.snapshot_enabled = bool(snapshot_enabled)
        self.snapshot = (
            moba_capture.ForwardMobaProbe(run, layout, step, sigma, branch)
            if snapshot_enabled
            else None
        )
        self.layer = -1
        self.explicit = False

    def observe(self, q, k, v, layer_index=None):
        if q.shape[2] != self.layout.seq_len:
            return
        if layer_index is None:
            if self.explicit:
                return
            self.layer += 1
            layer = self.layer
        else:
            self.explicit = True
            layer = int(layer_index)

        tracker = getattr(self.run, "router_dynamics_tracker", None)
        if tracker is not None:
            try:
                rec = tracker.capture(
                    q,
                    k,
                    self.layout,
                    step=self.step,
                    sigma=self.sigma,
                    branch=self.branch,
                    layer=layer,
                )
                self.run.router_dynamics.append(rec)
            except Exception:
                logging.exception(
                    "[H3 MoBA3D probe] router dynamics failed at step=%d layer=%d",
                    self.step,
                    layer,
                )

        if self.snapshot is not None:
            self.snapshot.observe(q, k, v, layer_index=layer)


def _make_wrapper_with_router_dynamics(session):
    """Variant of the existing wrapper that observes consecutive denoising steps."""

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
            run.anchor_frames = latent_dynamics.resolve_anchor_frames(payload, layout.latent_t)
            if run.capture_latent_dynamics:
                run.dynamics_queries = capture.select_query_blocks(
                    layout,
                    run.n_time,
                    run.n_spatial,
                    run.query_block,
                    include_audio=False,
                    include_text=False,
                )
            run.router_dynamics = []
            run.router_dynamics_tracker = (
                RouterDynamicsTracker(
                    q_tile=run.sage_q_tile,
                    kv_tile=run.sage_kv_tile,
                    budget=DEFAULT_DYNAMICS_BUDGET,
                    topology_q_samples=DEFAULT_TOPOLOGY_Q_SAMPLES,
                )
                if run.capture_attention
                else None
            )
            run.notes.update(
                {
                    "total_steps": int(total_steps),
                    "num_layers": int(num_layers),
                    "mode": (
                        "latent-dynamics-only"
                        if not run.capture_attention
                        else "MoBA exact snapshots + direct-tile router dynamics"
                    ),
                    "latent_dynamics": bool(run.capture_latent_dynamics),
                    "capture_attention": bool(run.capture_attention),
                    "execution_geometry": run.execution_geometry,
                    "sage_q_tile": int(run.sage_q_tile),
                    "sage_kv_tile": int(run.sage_kv_tile),
                    "router_dynamics": bool(run.capture_attention),
                    "router_dynamics_budget": float(DEFAULT_DYNAMICS_BUDGET),
                    "router_dynamics_steps": "all conditional denoising evaluations",
                    "router_dynamics_layers": "all observed H3 attention layers",
                    "router_topology_q_samples": int(DEFAULT_TOPOLOGY_Q_SAMPLES),
                    "latent_dynamics_source": "sampler callback x/x0",
                    "latent_dynamics_patch": [1, 2, 2],
                    "anchor_frames": list(run.anchor_frames),
                }
            )
            logging.info(
                "[H3 MoBA3D probe] router dynamics armed: all steps/layers budget=%.0f%% q=%d kv=%d; exact snapshots layers=%s steps=%s",
                100.0 * DEFAULT_DYNAMICS_BUDGET,
                run.sage_q_tile,
                run.sage_kv_tile,
                sorted(run.layers),
                sorted(run.steps),
            )

        if not run.capture_attention:
            return executor(*args, **kwargs)

        step, sigma = capture._step_index(transformer_options)
        cu = transformer_options.get("cond_or_uncond") or [0]
        branch = int(cu[0])
        if not run.capture_uncond and branch != 0:
            return executor(*args, **kwargs)

        snapshot_enabled = step in run.steps
        probe = _CombinedForwardProbe(
            run,
            layout,
            step,
            sigma,
            branch,
            snapshot_enabled=snapshot_enabled,
        )
        try:
            with observing(transformer_options, probe.observe):
                return executor(*args, **kwargs)
        finally:
            if snapshot_enabled:
                try:
                    # arrays=False deliberately avoids rewriting topology archives
                    # at every snapshot; final session.end() writes them once.
                    moba_report.write_run(run, arrays=False)
                except Exception:
                    logging.exception("[H3 MoBA3D probe] report checkpoint failed")

    return wrapper


def _router_summary(records):
    by_layer = {}
    for rec in records:
        if rec.get("exact_route_reuse_fraction_mean") is None:
            continue
        bucket = by_layer.setdefault(
            str(int(rec["layer"])),
            {"q": [], "k": [], "reuse": [], "jaccard": []},
        )
        bucket["q"].append(float(rec["q_cosine_mean"]))
        bucket["k"].append(float(rec["k_cosine_mean"]))
        bucket["reuse"].append(float(rec["exact_route_reuse_fraction_mean"]))
        bucket["jaccard"].append(float(rec["sampled_route_jaccard_mean"]))

    out = {}
    for layer, bucket in by_layer.items():
        out[layer] = {
            "transitions": len(bucket["reuse"]),
            "q_cosine_mean": float(sum(bucket["q"]) / len(bucket["q"])),
            "k_cosine_mean": float(sum(bucket["k"]) / len(bucket["k"])),
            "exact_route_reuse_fraction_mean": float(
                sum(bucket["reuse"]) / len(bucket["reuse"])
            ),
            "sampled_route_jaccard_mean": float(
                sum(bucket["jaccard"]) / len(bucket["jaccard"])
            ),
        }
    return {"by_layer": out}


def _write_router_artifacts(run):
    records = list(getattr(run, "router_dynamics", ()) or ())
    tracker = getattr(run, "router_dynamics_tracker", None)
    if not records and tracker is None:
        return
    os.makedirs(run.out_dir, exist_ok=True)
    payload = {
        "tag": run.tag,
        "layout": run.layout.as_dict() if run.layout else None,
        "budget": float(DEFAULT_DYNAMICS_BUDGET),
        "q_tile": int(getattr(run, "sage_q_tile", 128)),
        "kv_tile": int(getattr(run, "sage_kv_tile", 64)),
        "summary": _router_summary(records),
        "records": records,
    }
    with open(
        os.path.join(run.out_dir, "router_dynamics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2)
    if tracker is not None:
        tracker.write_topology(os.path.join(run.out_dir, "router_topology.npz"))


def _write_run_with_router_artifacts(run, arrays=True):
    path = _ORIGINAL_WRITE_RUN(run, arrays=arrays)
    if arrays:
        try:
            _write_router_artifacts(run)
        except Exception:
            logging.exception("[H3 MoBA3D probe] router artifact write failed")
    return path


def install():
    global _INSTALLED, _ORIGINAL_ANALYZE_ROUTING, _ORIGINAL_WRITE_RUN, _ORIGINAL_MAKE_WRAPPER
    if _INSTALLED:
        return

    current_analyze = moba3d.analyze_routing
    if not getattr(current_analyze, "_h3_direct_tile_calibration", False):
        _ORIGINAL_ANALYZE_ROUTING = current_analyze
        _analyze_routing_with_direct_calibration._h3_direct_tile_calibration = True
        _analyze_routing_with_direct_calibration._h3_original = current_analyze
        moba3d.analyze_routing = _analyze_routing_with_direct_calibration
    else:
        _ORIGINAL_ANALYZE_ROUTING = getattr(current_analyze, "_h3_original", current_analyze)

    _ORIGINAL_WRITE_RUN = moba_report.write_run
    if not getattr(_ORIGINAL_WRITE_RUN, "_h3_router_artifacts", False):
        _write_run_with_router_artifacts._h3_router_artifacts = True
        moba_report.write_run = _write_run_with_router_artifacts

    _ORIGINAL_MAKE_WRAPPER = moba_capture.make_wrapper
    if not getattr(_ORIGINAL_MAKE_WRAPPER, "_h3_router_dynamics", False):
        _make_wrapper_with_router_dynamics._h3_router_dynamics = True
        moba_capture.make_wrapper = _make_wrapper_with_router_dynamics

    _INSTALLED = True


install()
