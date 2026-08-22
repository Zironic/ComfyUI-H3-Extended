"""Dense-teacher characterization for fixed and adaptive Sparse-Sage routes."""

from __future__ import annotations

import math

import torch

try:
    from ..h3_attention.hybrid.config import (
        DENSITY_ADAPTIVE_BUDGET,
        DENSITY_FIXED,
        HybridSparseConfig,
    )
    from ..h3_attention.hybrid.router import SparseTileRouter
except ImportError:
    from h3_attention.hybrid.config import (
        DENSITY_ADAPTIVE_BUDGET,
        DENSITY_FIXED,
        HybridSparseConfig,
    )
    from h3_attention.hybrid.router import SparseTileRouter


def _distribution(values):
    values = torch.as_tensor(values, dtype=torch.float32).flatten()
    if not values.numel():
        return None
    return {
        "mean": float(values.mean()),
        "p05": float(torch.quantile(values, 0.05)),
        "p50": float(values.median()),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _rankdata(values):
    values = [float(value) for value in values]
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) * 0.5
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return torch.tensor(ranks, dtype=torch.float64)


def _spearman(left, right):
    left = torch.as_tensor(left).flatten().tolist()
    right = torch.as_tensor(right).flatten().tolist()
    if len(left) < 2 or len(left) != len(right):
        return None
    left = _rankdata(left)
    right = _rankdata(right)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def _arm_bucket():
    return {
        "retained": [],
        "row_rel_l2": [],
        "row_mean_abs": [],
        "squared_error": 0.0,
        "teacher_energy": 0.0,
    }


def _arm_report(bucket):
    squared_error = float(bucket["squared_error"])
    teacher_energy = max(float(bucket["teacher_energy"]), 1.0e-24)
    return {
        "retained_dense_attention_mass": _distribution(torch.cat(bucket["retained"])),
        "sparse_output_rel_l2_by_token_row": _distribution(
            torch.cat(bucket["row_rel_l2"])
        ),
        "sparse_output_mean_abs_by_token_row": _distribution(
            torch.cat(bucket["row_mean_abs"])
        ),
        "micro_relative_l2": math.sqrt(squared_error / teacher_energy),
        "squared_error": squared_error,
        "teacher_energy": teacher_energy,
    }


def _lagrangian_allocation(costs, minimum, maximum, target):
    """Return an exact-budget feasible prefix allocation.

    Per-row prefix error is not assumed convex. The Lagrangian search is followed
    by unit repairs, so this is a diagnostic control rather than a proven optimum.
    """
    rows = int(costs.shape[0])
    choices = torch.arange(minimum, maximum + 1, dtype=torch.int64)
    table = costs[:, minimum - 1:maximum].to(torch.float64)
    span = max(1.0, float((table.max() - table.min()).abs()))
    low = -span * 4.0
    high = span * 4.0

    def choose(penalty):
        index = torch.argmin(table + choices.to(torch.float64) * penalty, dim=1)
        return choices[index].clone()

    for _ in range(64):
        midpoint = (low + high) * 0.5
        allocation = choose(midpoint)
        if int(allocation.sum()) > target:
            low = midpoint
        else:
            high = midpoint
    allocation = choose(high)

    while int(allocation.sum()) < target:
        eligible = allocation < maximum
        current = allocation - minimum
        delta = torch.full((rows,), float("inf"), dtype=torch.float64)
        row = torch.arange(rows)[eligible]
        delta[eligible] = table[row, current[eligible] + 1] - table[row, current[eligible]]
        allocation[int(torch.argmin(delta))] += 1
    while int(allocation.sum()) > target:
        eligible = allocation > minimum
        current = allocation - minimum
        delta = torch.full((rows,), float("inf"), dtype=torch.float64)
        row = torch.arange(rows)[eligible]
        delta[eligible] = table[row, current[eligible] - 1] - table[row, current[eligible]]
        allocation[int(torch.argmin(delta))] -= 1

    fixed = torch.full_like(allocation, target // rows)
    fixed[: target - int(fixed.sum())] += 1
    if int(fixed.min()) >= minimum and int(fixed.max()) <= maximum:
        row = torch.arange(rows)
        allocated_cost = table[row, allocation - minimum].sum()
        fixed_cost = table[row, fixed - minimum].sum()
        if fixed_cost < allocated_cost:
            allocation = fixed
    return allocation


class AdaptiveTeacherExperiment:
    """Accumulate paired ideal-mask metrics from an existing dense teacher."""

    def __init__(self, q, k, layout, qs, qe, budgets, *, q_tile, kv_tile):
        self.q = q
        self.k = k
        self.layout = layout
        self.qs = int(qs)
        self.qe = int(qe)
        self.budgets = tuple(float(value) for value in budgets)
        self.router = SparseTileRouter(q_tile=q_tile, kv_tile=kv_tile)
        self.geometry = self.router.geometry(layout)
        self.q_tile = int(q_tile)
        self.kv_tile = int(kv_tile)
        self.sequence = int(q.shape[-2])
        self.query_tile_ids = torch.div(
            torch.arange(self.qs, self.qe, device=q.device),
            self.q_tile,
            rounding_mode="floor",
        )
        self.pure_video_token_start = (
            self.geometry.pure_video_kv_start * self.kv_tile
        )
        self.routes = {}
        self.accum = {}
        self.row_curves = []

        for budget in self.budgets:
            fixed = SparseTileRouter(HybridSparseConfig(
                video_budget=budget,
                density_mode=DENSITY_FIXED,
            ), q_tile=q_tile, kv_tile=kv_tile).build_lut(q, k, layout, budget)
            adaptive = SparseTileRouter(HybridSparseConfig(
                video_budget=budget,
                density_mode=DENSITY_ADAPTIVE_BUDGET,
            ), q_tile=q_tile, kv_tile=kv_tile).build_lut(q, k, layout, budget)
            fixed_counts = self._video_counts(fixed[1])
            adaptive_counts = self._video_counts(adaptive[1])
            self.routes[budget] = {
                "fixed": fixed,
                "adaptive": adaptive,
                "fixed_counts": fixed_counts,
                "adaptive_counts": adaptive_counts,
            }
            self.accum[budget] = {
                "fixed": _arm_bucket(),
                "adaptive": _arm_bucket(),
                "mass_delta": [],
                "rel_l2_delta": [],
                "mean_abs_delta": [],
                "sampled_fixed_k": [],
                "sampled_adaptive_k": [],
            }

        q_means = self.router._mean_pool(q, self.q_tile)
        k_means = self.router._mean_pool(k, self.kv_tile)
        self.production_scores = torch.matmul(
            q_means[..., self.geometry.pure_video_q_start:, :],
            k_means[..., self.geometry.pure_video_kv_start:, :].transpose(-1, -2),
        )

    def _video_counts(self, valid):
        return (
            valid[..., self.geometry.pure_video_q_start:]
            - self.geometry.pure_video_kv_start
        ).to(torch.int64)

    def _sampled_counts(self, valid, h0, h1):
        selected = valid[0, h0:h1].index_select(1, self.query_tile_ids)
        return selected.to(torch.int64) - self.geometry.pure_video_kv_start

    def _keep(self, lut, valid, h0, h1):
        absolute = torch.cumsum(lut[0, h0:h1], dim=-1).to(torch.long)
        selected = absolute.index_select(1, self.query_tile_ids)
        selected_valid = valid[0, h0:h1].index_select(1, self.query_tile_ids)
        rank = torch.arange(lut.shape[-1], device=lut.device)
        active = rank.view(1, 1, -1) < selected_valid[..., None]
        tile_keep = torch.zeros(
            selected.shape[:-1] + (self.geometry.kv_tiles,),
            dtype=torch.bool,
            device=lut.device,
        )
        tile_keep.scatter_(2, selected.clamp(0, self.geometry.kv_tiles - 1), active)
        token_tiles = torch.div(
            torch.arange(self.sequence, device=lut.device),
            self.kv_tile,
            rounding_mode="floor",
        )
        return tile_keep.index_select(2, token_tiles)

    @staticmethod
    def _observe_arm(bucket, probs, values, dense_out, keep):
        masked = probs * keep.to(probs.dtype)
        retained = masked.sum(-1).clamp_min(1.0e-12)
        sparse_out = torch.matmul(masked, values) / retained.unsqueeze(-1)
        diff = sparse_out - dense_out
        row_rel_l2 = torch.linalg.vector_norm(diff, dim=-1) / torch.linalg.vector_norm(
            dense_out, dim=-1
        ).clamp_min(1.0e-12)
        row_mean_abs = diff.abs().mean(-1)
        bucket["retained"].append(retained.detach().cpu().flatten())
        bucket["row_rel_l2"].append(row_rel_l2.detach().cpu().flatten())
        bucket["row_mean_abs"].append(row_mean_abs.detach().cpu().flatten())
        bucket["squared_error"] += float(diff.square().sum())
        bucket["teacher_energy"] += float(dense_out.square().sum())
        return retained, row_rel_l2, row_mean_abs

    def _observe_row_curves(self, h0, h1, probs, values, dense_out):
        pure_start = self.pure_video_token_start
        pure_tokens = self.sequence - pure_start
        padded_tokens = self.geometry.pure_video_kv_tiles * self.kv_tile
        unique_q_tiles = torch.unique(self.query_tile_ids)
        for local_head, head in enumerate(range(h0, h1)):
            for q_tile in unique_q_tiles.tolist():
                if q_tile < self.geometry.pure_video_q_start:
                    continue
                positions = torch.nonzero(
                    self.query_tile_ids == q_tile, as_tuple=False
                ).flatten()
                block_probs = probs[local_head].index_select(0, positions)[..., pure_start:]
                block_values = values[local_head, pure_start:]
                if pure_tokens < padded_tokens:
                    block_probs = torch.nn.functional.pad(
                        block_probs, (0, padded_tokens - pure_tokens)
                    )
                    block_values = torch.nn.functional.pad(
                        block_values, (0, 0, 0, padded_tokens - pure_tokens)
                    )
                block_probs = block_probs.reshape(
                    positions.numel(), self.geometry.pure_video_kv_tiles, self.kv_tile
                )
                block_values = block_values.reshape(
                    self.geometry.pure_video_kv_tiles, self.kv_tile, values.shape[-1]
                )
                block_mass_by_token = block_probs.sum(-1)
                block_numerator = torch.einsum(
                    "qbt,btd->qbd", block_probs, block_values
                )
                context_probs = probs[local_head].index_select(0, positions)[..., :pure_start]
                context_mass = context_probs.sum(-1)
                context_numerator = torch.matmul(
                    context_probs, values[local_head, :pure_start]
                )
                want = dense_out[local_head].index_select(0, positions)
                exact_mass = block_mass_by_token.mean(0)
                q_row = q_tile - self.geometry.pure_video_q_start
                production_order = torch.argsort(
                    self.production_scores[0, head, q_row].float(),
                    descending=True,
                    stable=True,
                )
                teacher_order = torch.argsort(
                    exact_mass, descending=True, stable=True
                )

                def curves(order):
                    mass = block_mass_by_token.index_select(1, order).cumsum(1)
                    numerator = block_numerator.index_select(1, order).cumsum(1)
                    output = (
                        context_numerator[:, None, :] + numerator
                    ) / (
                        context_mass[:, None, None] + mass[:, :, None]
                    ).clamp_min(1.0e-12)
                    error = (output - want[:, None, :]).square().sum(dim=(0, 2))
                    retained = exact_mass.index_select(0, order).cumsum(0)
                    return error.detach().cpu(), retained.detach().cpu()

                production_error, production_mass = curves(production_order)
                teacher_error, teacher_mass = curves(teacher_order)
                total_mass = float(exact_mass.sum())
                oracle_k = {}
                for name, fraction in (("k80", 0.80), ("k90", 0.90), ("k95", 0.95), ("k99", 0.99)):
                    if total_mass <= 1.0e-12:
                        oracle_k[name] = 1
                    else:
                        oracle_k[name] = int(
                            (teacher_mass < total_mass * fraction).sum()
                        ) + 1
                self.row_curves.append({
                    "head": head,
                    "q_tile": q_tile,
                    "production_error": production_error,
                    "production_mass": production_mass,
                    "teacher_error": teacher_error,
                    "teacher_mass": teacher_mass,
                    "total_pure_video_mass": total_mass,
                    "oracle_k": oracle_k,
                })

    def observe(self, h0, h1, probs, dense_out, values):
        self._observe_row_curves(h0, h1, probs, values, dense_out)
        for budget in self.budgets:
            route = self.routes[budget]
            bucket = self.accum[budget]
            observed = {}
            for arm in ("fixed", "adaptive"):
                lut, valid, _metadata = route[arm]
                keep = self._keep(lut, valid, h0, h1)
                observed[arm] = self._observe_arm(
                    bucket[arm], probs, values, dense_out, keep
                )
                bucket["sampled_%s_k" % arm].append(
                    self._sampled_counts(valid, h0, h1).detach().cpu().flatten()
                )
            fixed = observed["fixed"]
            adaptive = observed["adaptive"]
            bucket["mass_delta"].append((adaptive[0] - fixed[0]).detach().cpu().flatten())
            bucket["rel_l2_delta"].append((adaptive[1] - fixed[1]).detach().cpu().flatten())
            bucket["mean_abs_delta"].append((adaptive[2] - fixed[2]).detach().cpu().flatten())

    def _oracle_report(self, budget, route):
        rows = len(self.row_curves)
        if not rows:
            return None
        metadata = route["adaptive"][2]
        minimum = int(metadata.configured_minimum_video_kv_tiles)
        maximum = int(metadata.configured_maximum_video_kv_tiles)
        target = int(metadata.retained_video_kv_tiles) * rows
        production_cost = torch.stack([
            row["production_error"] for row in self.row_curves
        ])
        allocation = _lagrangian_allocation(
            production_cost, minimum, maximum, target
        )
        row_index = torch.arange(rows)
        allocation_error = production_cost[
            row_index, allocation - 1
        ].sum()
        fixed = torch.full((rows,), int(metadata.retained_video_kv_tiles), dtype=torch.int64)
        fixed_error = production_cost[row_index, fixed - 1].sum()

        teacher_mass = torch.stack([
            row["teacher_mass"] for row in self.row_curves
        ])
        mass_allocation = torch.full((rows,), minimum, dtype=torch.int64)
        remaining = target - minimum * rows
        if remaining:
            candidates = (
                teacher_mass[:, minimum:maximum]
                - teacher_mass[:, minimum - 1:maximum - 1]
            ).reshape(-1)
            selected = torch.topk(candidates, remaining).indices
            selected_rows = torch.div(
                selected, maximum - minimum, rounding_mode="floor"
            )
            mass_allocation += torch.bincount(selected_rows, minlength=rows)
        teacher_error = torch.stack([
            row["teacher_error"] for row in self.row_curves
        ])
        mass_oracle_error = teacher_error[
            row_index, mass_allocation - 1
        ].sum()
        mass_oracle_retained = teacher_mass[
            row_index, mass_allocation - 1
        ].sum()
        return {
            "scope": "sampled head/Q-tile rows in this exact query record",
            "rows": rows,
            "target_selected_video_tiles": target,
            "output_allocation_control": {
                "ranking": "stable BF16 pooled-QK score order",
                "solver": "Lagrangian search plus exact-budget unit repair",
                "proven_optimal": False,
                "selected_video_tiles": int(allocation.sum()),
                "k": _distribution(allocation),
                "squared_error": float(allocation_error),
                "uniform_fixed_squared_error": float(fixed_error),
                "relative_error_reduction_vs_uniform_fixed": float(
                    (fixed_error - allocation_error) / fixed_error.clamp_min(1.0e-24)
                ),
            },
            "mass_oracle": {
                "ranking": "exact dense pure-video block mass",
                "selected_video_tiles": int(mass_allocation.sum()),
                "k": _distribution(mass_allocation),
                "retained_pure_video_mass": float(mass_oracle_retained),
                "squared_output_error_at_mass_allocation": float(mass_oracle_error),
            },
        }

    def finalize(self):
        result = {}
        oracle_k95 = torch.tensor([
            row["oracle_k"]["k95"] for row in self.row_curves
        ], dtype=torch.int64)
        total_video_mass = torch.tensor([
            row["total_pure_video_mass"] for row in self.row_curves
        ], dtype=torch.float32)
        for budget in self.budgets:
            route = self.routes[budget]
            bucket = self.accum[budget]
            fixed_counts = route["fixed_counts"].detach().cpu().flatten()
            adaptive_counts = route["adaptive_counts"].detach().cpu().flatten()
            sampled_fixed = torch.cat(bucket["sampled_fixed_k"])
            sampled_adaptive = torch.cat(bucket["sampled_adaptive_k"])
            mass_delta = torch.cat(bucket["mass_delta"])
            rel_l2_delta = torch.cat(bucket["rel_l2_delta"])
            mean_abs_delta = torch.cat(bucket["mean_abs_delta"])
            sampled_row_adaptive = torch.tensor([
                int(route["adaptive_counts"][0, row["head"], row["q_tile"] - self.geometry.pure_video_q_start])
                for row in self.row_curves
            ], dtype=torch.int64)
            metadata = route["adaptive"][2]
            exact_target = int(metadata.retained_video_kv_tiles) * adaptive_counts.numel()
            result[budget] = {
                "policy": {
                    "temperature": float(metadata.adaptive_temperature),
                    "target_mass": float(metadata.adaptive_target_mass),
                    "minimum_video_kv_tiles": int(metadata.configured_minimum_video_kv_tiles),
                    "maximum_video_kv_tiles": int(metadata.configured_maximum_video_kv_tiles),
                },
                "full_router_call": {
                    "route_rows": int(adaptive_counts.numel()),
                    "fixed_selected_video_tiles": int(fixed_counts.sum()),
                    "adaptive_selected_video_tiles": int(adaptive_counts.sum()),
                    "target_selected_video_tiles": exact_target,
                    "exact_budget_match": bool(
                        int(fixed_counts.sum()) == exact_target
                        and int(adaptive_counts.sum()) == exact_target
                    ),
                    "fixed_k": _distribution(fixed_counts),
                    "adaptive_k": _distribution(adaptive_counts),
                },
                "sampled_teacher_rows": {
                    "fixed_k": _distribution(sampled_fixed),
                    "adaptive_k": _distribution(sampled_adaptive),
                    "adaptive_minus_fixed_k": _distribution(
                        sampled_adaptive - sampled_fixed
                    ),
                    "fixed": _arm_report(bucket["fixed"]),
                    "adaptive": _arm_report(bucket["adaptive"]),
                    "adaptive_minus_fixed_retained_mass": _distribution(mass_delta),
                    "adaptive_minus_fixed_rel_l2": _distribution(rel_l2_delta),
                    "adaptive_minus_fixed_mean_abs": _distribution(mean_abs_delta),
                    "adaptive_retained_mass_win_fraction": float((mass_delta > 0).float().mean()),
                    "adaptive_rel_l2_win_fraction": float((rel_l2_delta < 0).float().mean()),
                    "adaptive_mean_abs_win_fraction": float((mean_abs_delta < 0).float().mean()),
                    "adaptive_retained_mass_outcomes": {
                        "win_fraction": float((mass_delta > 0).float().mean()),
                        "tie_fraction": float((mass_delta == 0).float().mean()),
                        "loss_fraction": float((mass_delta < 0).float().mean()),
                    },
                    "adaptive_rel_l2_outcomes": {
                        "win_fraction": float((rel_l2_delta < 0).float().mean()),
                        "tie_fraction": float((rel_l2_delta == 0).float().mean()),
                        "loss_fraction": float((rel_l2_delta > 0).float().mean()),
                    },
                    "adaptive_mean_abs_outcomes": {
                        "win_fraction": float((mean_abs_delta < 0).float().mean()),
                        "tie_fraction": float((mean_abs_delta == 0).float().mean()),
                        "loss_fraction": float((mean_abs_delta > 0).float().mean()),
                    },
                },
                "demand_teacher": {
                    "rows": len(self.row_curves),
                    "total_pure_video_attention_mass": _distribution(total_video_mass),
                    "oracle_k95": _distribution(oracle_k95),
                    "spearman_adaptive_k_vs_oracle_k95": _spearman(
                        sampled_row_adaptive, oracle_k95
                    ),
                    "spearman_adaptive_k_vs_total_pure_video_attention_mass": _spearman(
                        sampled_row_adaptive, total_video_mass
                    ),
                },
                "sampled_allocation_controls": self._oracle_report(budget, route),
            }
        return result
