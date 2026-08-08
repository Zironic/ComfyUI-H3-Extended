"""Benchmark H3 dense, sparse, Flex, Sol, and hybrid attention headlessly.

The benchmark synthesizes H3's fused BF16 QKV allocation and packed modality
layout, then calls the same prepared backends and kernels used by the H3 pack.
It never constructs a ComfyUI graph or loads a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable

Q_TILE = 128
KV_TILE = 64
HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM
FUSED_STRIDE = INNER * 3
GIB = 1024 ** 3

DEFAULT_BACKENDS = (
    "dense_sage",
    "sparse_sage_128x64",
    "flex_128x64",
    "flex_64x64",
    "sol",
    "hybrid_sage_flex",
)
ALL_BACKENDS = DEFAULT_BACKENDS + ("flex_32x64",)


def parse_fraction(value: Any, *, name: str = "fraction", allow_zero: bool = True) -> float:
    """Accept 0.5, 50, or ``50%`` and return a normalized fraction."""
    text = str(value).strip()
    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError("%s must be a number or percentage" % name) from exc
    if percent or 1.0 < number <= 100.0:
        number /= 100.0
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError("%s must be between 0 and 1, or between 0%% and 100%%" % name)
    if not allow_zero and number == 0.0:
        raise ValueError("%s must be greater than zero" % name)
    return number


@dataclass(frozen=True)
class TileGeometry:
    sequence: int
    video_start: int
    q_tile: int
    kv_tile: int
    q_tiles: int
    kv_tiles: int
    pure_video_q_start: int
    pure_video_kv_start: int

    @property
    def pure_video_q_tiles(self):
        return self.q_tiles - self.pure_video_q_start

    @property
    def pure_video_kv_tiles(self):
        return self.kv_tiles - self.pure_video_kv_start


@dataclass
class PreparedCall:
    execute: Any
    details: dict
    timing: Any = None


SPARSE_TIMING_STAGES = (
    "direct_lut_construction",
    "v_fp8_preparation",
    "q_k_int8_quantization",
    "sparse_sage_low_level_kernel",
    "total_hybrid_attention",
)


def _aggregate_timing_fields(summaries):
    """Return median production-stage timings from deferred timing summaries."""
    return {
        "%s_ms" % stage: statistics.median([
            float(summary["stages"][stage]["mean_ms"])
            for summary in summaries
        ])
        for stage in SPARSE_TIMING_STAGES
    }


def tile_geometry(layout, q_tile=Q_TILE, kv_tile=KV_TILE):
    sequence = int(layout.seq_len)
    video_start, video_stop = (int(x) for x in layout.video_range)
    if sequence <= 0 or not (0 <= video_start < video_stop == sequence):
        raise ValueError("layout must end with a non-empty target video segment")
    if q_tile <= 0 or kv_tile <= 0:
        raise ValueError("tile sizes must be positive")
    geometry = TileGeometry(
        sequence=sequence,
        video_start=video_start,
        q_tile=int(q_tile),
        kv_tile=int(kv_tile),
        q_tiles=math.ceil(sequence / q_tile),
        kv_tiles=math.ceil(sequence / kv_tile),
        pure_video_q_start=math.ceil(video_start / q_tile),
        pure_video_kv_start=math.ceil(video_start / kv_tile),
    )
    if geometry.pure_video_q_tiles <= 0 or geometry.pure_video_kv_tiles <= 0:
        raise ValueError("layout has no pure-video attention tiles")
    return geometry


def _generator(torch, device, seed):
    return torch.Generator(device=device).manual_seed(int(seed))


def build_controlled_mask(layout, budget, *, pattern="uniform", q_tile=Q_TILE,
                          kv_tile=KV_TILE, heads=HEADS, seed=0, device="cpu"):
    """Return a deterministic [1,H,Q,K] mask with exact video density."""
    import torch

    budget = parse_fraction(budget, name="budget")
    geometry = tile_geometry(layout, q_tile, kv_tile)
    retained = min(
        geometry.pure_video_kv_tiles,
        math.ceil(budget * geometry.pure_video_kv_tiles),
    )
    mask = torch.ones(
        (1, int(heads), geometry.q_tiles, geometry.kv_tiles),
        dtype=torch.bool,
        device=device,
    )
    video = torch.zeros(
        (int(heads), geometry.pure_video_q_tiles, geometry.pure_video_kv_tiles),
        dtype=torch.bool,
        device=device,
    )
    if retained:
        generator = _generator(torch, device, seed)
        if pattern == "uniform":
            scores = torch.rand(video.shape, generator=generator, device=device)
            indices = torch.topk(scores, retained, dim=-1).indices
            video.scatter_(-1, indices, True)
        elif pattern == "local":
            local_count = min(retained, max(1, retained // 2))
            q_centers = (
                torch.arange(geometry.pure_video_q_tiles, device=device, dtype=torch.float32)
                * q_tile + geometry.pure_video_q_start * q_tile + q_tile / 2
            )
            k_centers = (
                torch.arange(geometry.pure_video_kv_tiles, device=device, dtype=torch.float32)
                * kv_tile + geometry.pure_video_kv_start * kv_tile + kv_tile / 2
            )
            nearest = torch.topk(
                -(q_centers[:, None] - k_centers[None, :]).abs(),
                local_count,
                dim=-1,
            ).indices
            video.scatter_(-1, nearest.unsqueeze(0).expand(int(heads), -1, -1), True)
            distant_count = retained - local_count
            if distant_count:
                scores = torch.rand(video.shape, generator=generator, device=device)
                scores.masked_fill_(video, -1.0)
                video.scatter_(-1, torch.topk(scores, distant_count, dim=-1).indices, True)
        elif pattern == "shared":
            cluster_size = 4
            clusters = math.ceil(geometry.pure_video_q_tiles / cluster_size)
            scores = torch.rand(
                (int(heads), clusters, geometry.pure_video_kv_tiles),
                generator=generator,
                device=device,
            )
            shared = torch.zeros_like(scores, dtype=torch.bool)
            shared.scatter_(-1, torch.topk(scores, retained, dim=-1).indices, True)
            video.copy_(
                shared.repeat_interleave(cluster_size, dim=1)[
                    :, :geometry.pure_video_q_tiles
                ]
            )
        else:
            raise ValueError("pattern must be uniform, local, or shared")
    elif pattern not in ("uniform", "local", "shared"):
        raise ValueError("pattern must be uniform, local, or shared")

    mask[
        0,
        :,
        geometry.pure_video_q_start:,
        geometry.pure_video_kv_start:,
    ] = video
    non_video_kv = geometry.pure_video_kv_start
    true_blocks_per_head = (
        geometry.pure_video_q_start * geometry.kv_tiles
        + geometry.pure_video_q_tiles * (non_video_kv + retained)
    )
    metadata = {
        "requested_video_budget": budget,
        "actual_video_tile_density": retained / geometry.pure_video_kv_tiles,
        "full_mask_density": true_blocks_per_head / (geometry.q_tiles * geometry.kv_tiles),
        "retained_video_kv_tiles": retained,
        "q_tile": q_tile,
        "kv_tile": kv_tile,
        "q_tiles": geometry.q_tiles,
        "kv_tiles": geometry.kv_tiles,
        "pure_video_q_tiles": geometry.pure_video_q_tiles,
        "pure_video_kv_tiles": geometry.pure_video_kv_tiles,
        "pattern": pattern,
    }
    return mask.contiguous(), metadata


def block_mask_to_lut(mask):
    """Convert a synthetic benchmark mask to Sparge's delta LUT carrier."""
    import torch

    kv_tiles = mask.shape[-1]
    indices = torch.arange(kv_tiles, dtype=torch.int32, device=mask.device)
    selected = torch.where(mask, indices, kv_tiles).sort(dim=-1).values
    previous = torch.cat(
        (torch.zeros_like(selected[..., :1]), selected[..., :-1]), dim=-1
    )
    valid = mask.sum(dim=-1, dtype=torch.int32)
    return (selected - previous).contiguous(), valid.contiguous()


def compact_kv_blocks(mask):
    """Convert a dense tile mask to Flex's compact KV block rows."""
    import torch

    if mask.ndim != 4 or mask.dtype != torch.bool:
        raise ValueError("Flex tile mask must be rank-4 bool")
    counts = mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.argsort(
        mask.to(torch.int8), dim=-1, descending=True, stable=True
    ).to(torch.int32)
    return counts, indices


def make_flex_block_mask(mask, q_length, kv_length, q_tile, kv_tile=KV_TILE):
    from torch.nn.attention.flex_attention import BlockMask

    counts, indices = compact_kv_blocks(mask)
    return BlockMask.from_kv_blocks(
        counts,
        indices,
        BLOCK_SIZE=(int(q_tile), int(kv_tile)),
        seq_lengths=(int(q_length), int(kv_length)),
    )


def compile_flex_attention(torch, flex_attention):
    """Compile Flex once so it lowers to its fused block-sparse kernel."""
    return torch.compile(flex_attention, fullgraph=True)


def flex_kernel_options(q_tile, kv_tile=KV_TILE):
    return {"BLOCK_M": int(q_tile), "BLOCK_N": int(kv_tile)}


def _rank(seed, kind, index):
    value = "%d:%d:%d" % (int(seed), int(kind), int(index))
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "little")


def select_hard_tiles(pure_q_tiles, heads, q_fraction, head_fraction, seed=0):
    q_fraction = parse_fraction(q_fraction, name="hard Q fraction")
    head_fraction = parse_fraction(head_fraction, name="hard head fraction")
    q_count = min(pure_q_tiles, math.floor(pure_q_tiles * q_fraction + 0.5))
    head_count = min(heads, math.floor(heads * head_fraction + 0.5))
    q_rows = sorted(sorted(range(pure_q_tiles), key=lambda x: _rank(seed, 1, x))[:q_count])
    head_rows = sorted(sorted(range(heads), key=lambda x: _rank(seed, 2, x))[:head_count])
    return q_rows, head_rows


def build_hybrid_plan(layout, budget, *, hard_q_fraction, hard_head_fraction,
                      pattern="uniform", flex_q_tile=64, heads=HEADS, seed=0,
                      device="cpu"):
    """Build placeholder Sage rows and gathered Flex row metadata."""
    sparse_mask, sparse_metadata = build_controlled_mask(
        layout, budget, pattern=pattern, q_tile=Q_TILE, heads=heads,
        seed=seed, device=device,
    )
    geometry = tile_geometry(layout)
    q_offsets, hard_heads = select_hard_tiles(
        geometry.pure_video_q_tiles, heads, hard_q_fraction,
        hard_head_fraction, seed,
    )
    hard_q_tiles = [geometry.pure_video_q_start + value for value in q_offsets]
    placeholder = sparse_mask.clone()
    for head in hard_heads:
        for q_row in hard_q_tiles:
            placeholder[0, head, q_row, geometry.pure_video_kv_start:] = False

    hard_tokens = []
    flex_q_rows = []
    for q_row in hard_q_tiles:
        start = q_row * Q_TILE
        stop = min(layout.seq_len, start + Q_TILE)
        hard_tokens.extend(range(start, stop))
        flex_q_rows.extend(range(start // flex_q_tile, math.ceil(stop / flex_q_tile)))

    flex_mask, flex_metadata = build_controlled_mask(
        layout, budget, pattern=pattern, q_tile=flex_q_tile,
        heads=len(hard_heads), seed=seed + 17, device=device,
    )
    return {
        "placeholder_mask": placeholder,
        "sparse_metadata": sparse_metadata,
        "flex_mask": flex_mask,
        "flex_metadata": flex_metadata,
        "hard_q_tiles": hard_q_tiles,
        "hard_heads": hard_heads,
        "hard_tokens": hard_tokens,
        "flex_q_rows": flex_q_rows,
        "flex_q_tile": int(flex_q_tile),
    }


def sweep_cases():
    return [
        {"budget": budget, "hard_q_fraction": hard, "flex_q_tile": q_tile}
        for budget in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        for q_tile in (128, 64, 32)
        for hard in (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.0)
    ]


def derive_break_even(rows: Iterable[dict]):
    dense = next(
        (float(row["latency_ms"]) for row in rows if row.get("mode") == "dense_sage"),
        None,
    )
    keys = {
        (float(row["budget"]), int(row["flex_q_tile"]))
        for row in rows
        if row.get("mode") == "hybrid_sage_flex"
    }
    result = {}
    for budget, q_tile in sorted(keys):
        faster = [
            float(row["hard_q_fraction"])
            for row in rows
            if row.get("mode") == "hybrid_sage_flex"
            and float(row["budget"]) == budget
            and int(row["flex_q_tile"]) == q_tile
            and dense is not None
            and float(row["latency_ms"]) < dense
        ]
        result["%g:%d" % (budget, q_tile)] = max(faster) if faster else None
    return result


def write_reports(metadata, rows, router, break_even, json_path, csv_path):
    payload = {
        "metadata": metadata,
        "rows": rows,
        "router": router,
        "break_even": break_even,
    }
    if json_path:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    if csv_path:
        fields = sorted({key for row in rows for key in row})
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=141)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--text-len", type=int, default=256)
    parser.add_argument("--budget", default="50%")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--pattern", choices=("uniform", "local", "shared"), default="uniform")
    parser.add_argument("--flex-fraction", default="10%", help="hard 128Q tile fraction")
    parser.add_argument("--hard-head-fraction", default="100%")
    parser.add_argument("--hybrid-flex-q", type=int, choices=(128, 64, 32), default=64)
    parser.add_argument("--timing", choices=("kernel", "end_to_end"), default="end_to_end")
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--seed", type=int, default=6841)
    parser.add_argument("--router", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--json", metavar="PATH")
    parser.add_argument("--csv", metavar="PATH")
    return parser


def validate_args(args):
    if args.frames <= 0 or args.text_len < 0:
        raise ValueError("frames must be positive and text-len cannot be negative")
    if args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32:
        raise ValueError("width and height must be positive multiples of 32")
    if args.repeats <= 0 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups cannot be negative")
    parse_fraction(args.budget, name="budget")
    parse_fraction(args.flex_fraction, name="flex fraction")
    parse_fraction(args.hard_head_fraction, name="hard head fraction")
    modes = [value.strip() for value in args.backends.split(",") if value.strip()]
    unknown = [value for value in modes if value not in ALL_BACKENDS]
    if not modes:
        raise ValueError("at least one backend is required")
    if unknown:
        raise ValueError("unknown backend(s): %s" % ", ".join(unknown))


def _layout(probe, TokenLayout, args):
    raw = probe.build_layout(args.frames, args.width, args.height, args.text_len)
    text = audio = video = None
    references = []
    for start, stop, kind in raw["segments"]:
        if kind == "text":
            text = (start, stop)
        elif kind == "audio":
            audio = (start, stop)
        elif kind == "video":
            video = (start, stop)
        else:
            references.append((kind, start, stop))
    return TokenLayout(
        seq_len=raw["seq_len"],
        text_range=text,
        audio_range=audio,
        video_range=video,
        video_shape=(raw["latent_t"], args.height // 32, args.width // 32),
        audio_t=(audio[1] - audio[0]) // 2,
        reference_ranges=references,
        segments=list(raw["segments"]),
    )


def _fused_qkv(torch, sequence, device, seed):
    fused = torch.randn(
        sequence,
        FUSED_STRIDE,
        dtype=torch.bfloat16,
        device=device,
        generator=_generator(torch, device, seed),
    )
    q, k, v = fused.split(INNER, dim=-1)
    return (
        q.view(sequence, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0),
        k.view(sequence, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0),
        v.view(sequence, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0),
    )


def _prepare_call(context, mode, budget, hard_q_fraction, hard_head_fraction,
                  flex_q_tile, pattern, seed, timing=None):
    torch = context["torch"]
    layout = context["layout"]
    device = context["device"]
    q, k, v = _fused_qkv(torch, layout.seq_len, device, seed)

    if mode == "dense_sage":
        prepared = context["dense"].prepare(
            q, k, v, layer_index=10, transformer_options={}
        )
        return PreparedCall(lambda: context["dense"].execute(prepared), {})

    if mode == "sol":
        prepared = context["sol"].prepare(
            q, k, v, layer_index=10,
            transformer_options={context["runtime_key"]: context["runtime"]},
        )
        return PreparedCall(lambda: context["sol"].execute(prepared), {})

    if mode == "sparse_sage_128x64":
        total_timing = timing.begin("total_hybrid_attention") if timing is not None else None
        router_timing = timing.begin("direct_lut_construction") if timing is not None else None
        try:
            lut, valid, metadata = context["router"].build_lut(
                q, k, layout, budget
            )
        except Exception:
            if timing is not None:
                timing.end(total_timing)
            raise
        finally:
            if timing is not None:
                timing.end(router_timing)
        details = metadata.as_dict() if hasattr(metadata, "as_dict") else dict(metadata)
        try:
            prepare_kwargs = {
                "layer_index": 10,
                "metadata": details,
            }
            if timing is not None:
                prepare_kwargs["timing"] = timing
            prepared = context["sparse"].prepare(q, k, v, lut, valid, **prepare_kwargs)
        except Exception:
            if timing is not None:
                timing.end(total_timing)
            raise

        def execute_sparse():
            try:
                return context["sparse"].execute(prepared)
            finally:
                if timing is not None:
                    timing.end(total_timing)

        return PreparedCall(execute_sparse, details, timing)

    if mode.startswith("flex_"):
        q_tile = int(mode.split("_")[1].split("x")[0])
        mask, details = build_controlled_mask(
            layout, budget, pattern=pattern, q_tile=q_tile,
            heads=HEADS, seed=seed, device=device,
        )
        block_mask = make_flex_block_mask(
            mask, layout.seq_len, layout.seq_len, q_tile
        )
        return PreparedCall(
            lambda: context["flex"](
                q, k, v, block_mask=block_mask, scale=HEAD_DIM ** -0.5,
                kernel_options=flex_kernel_options(q_tile),
            ),
            details,
        )

    plan = build_hybrid_plan(
        layout, budget, hard_q_fraction=hard_q_fraction,
        hard_head_fraction=hard_head_fraction, pattern=pattern,
        flex_q_tile=flex_q_tile, heads=HEADS, seed=seed, device=device,
    )
    sparse_lut, sparse_valid = block_mask_to_lut(plan["placeholder_mask"])
    sparse_prepared = context["sparse"].prepare(
        q, k, v, sparse_lut, sparse_valid, layer_index=10,
        metadata=plan["sparse_metadata"],
    )
    details = dict(plan["sparse_metadata"])
    details.update({
        "hard_q_fraction": hard_q_fraction,
        "hard_head_fraction": hard_head_fraction,
        "hard_q_tiles": len(plan["hard_q_tiles"]),
        "hard_heads": len(plan["hard_heads"]),
        "flex_q_tile": flex_q_tile,
    })
    if not plan["hard_heads"] or not plan["hard_tokens"]:
        return PreparedCall(lambda: context["sparse"].execute(sparse_prepared), details)

    head_index = torch.tensor(plan["hard_heads"], dtype=torch.long, device=device)
    token_index = torch.tensor(plan["hard_tokens"], dtype=torch.long, device=device)
    flex_row_index = torch.tensor(plan["flex_q_rows"], dtype=torch.long, device=device)
    flex_q = q.index_select(1, head_index).index_select(2, token_index).contiguous()
    flex_k = k.index_select(1, head_index).contiguous()
    flex_v = v.index_select(1, head_index).contiguous()
    flex_mask = plan["flex_mask"].index_select(2, flex_row_index).contiguous()
    block_mask = make_flex_block_mask(
        flex_mask, flex_q.shape[2], layout.seq_len, flex_q_tile
    )

    def execute_hybrid():
        output = context["sparse"].execute(sparse_prepared)
        fallback = context["flex"](
            flex_q, flex_k, flex_v,
            block_mask=block_mask,
            scale=HEAD_DIM ** -0.5,
            kernel_options=flex_kernel_options(flex_q_tile),
        )
        output[0, head_index[:, None], token_index[None, :], :] = fallback[0]
        return output

    return PreparedCall(execute_hybrid, details)


def _measure_case(context, mode, budget, hard_q_fraction, hard_head_fraction,
                  flex_q_tile, args):
    torch = context["torch"]
    device = context["device"]
    times = []
    peaks = []
    details = {}
    timing_summaries = []
    for index in range(args.warmups + args.repeats):
        torch.cuda.empty_cache()
        timing = None
        if mode == "sparse_sage_128x64":
            timing = context["DeferredCudaTiming"](enabled=True)
            timing.begin_request(index, cuda=True)
        prepare_args = (
            context, mode, budget, hard_q_fraction, hard_head_fraction,
            flex_q_tile, args.pattern, args.seed + index,
        )
        prepare_kwargs = {} if timing is None else {"timing": timing}
        if args.timing == "kernel":
            call = _prepare_call(*prepare_args, **prepare_kwargs)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            base = torch.cuda.memory_allocated(device)
            started = time.perf_counter()
        else:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            base = torch.cuda.memory_allocated(device)
            started = time.perf_counter()
            call = _prepare_call(*prepare_args, **prepare_kwargs)
        output = call.execute()
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - started) * 1000.0
        peak = (torch.cuda.max_memory_allocated(device) - base) / GIB
        details = call.details
        del output, call
        if index >= args.warmups:
            times.append(elapsed)
            peaks.append(peak)
            if timing is not None:
                timing_summaries.append(timing.resolve())
    result = {
        "mode": mode,
        "budget": float(budget),
        "hard_q_fraction": float(hard_q_fraction),
        "hard_head_fraction": float(hard_head_fraction),
        "flex_q_tile": int(flex_q_tile),
        "latency_ms": statistics.median(times),
        "peak_allocated_gib": statistics.median(peaks),
        **details,
    }
    if timing_summaries:
        result.update(_aggregate_timing_fields(timing_summaries))
    return result


def _mean_pool(x, block):
    import torch

    full = x.shape[-2] // block
    pieces = []
    if full:
        pieces.append(
            x[..., :full * block, :]
            .reshape(*x.shape[:-2], full, block, x.shape[-1])
            .mean(dim=-2)
        )
    if x.shape[-2] % block:
        pieces.append(x[..., full * block:, :].mean(dim=-2, keepdim=True))
    if not pieces:
        return x.new_empty((*x.shape[:-2], 0, x.shape[-1]))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)


def _router_metrics(context, budget, args):
    torch = context["torch"]
    device = context["device"]
    layout = context["layout"]
    geometry = tile_geometry(layout)
    retained = max(1, math.ceil(budget * geometry.pure_video_kv_tiles))
    values = {key: [] for key in (
        "q128_mean_ms", "k64_mean_ms", "qmean_kmean_ms", "topk_ms",
        "compatibility_ms", "router_ms",
    )}
    compatibility = []
    for index in range(args.warmups + args.repeats):
        q, k, v = _fused_qkv(torch, layout.seq_len, device, args.seed + 10000 + index)
        k = q * 0.75 + k * 0.25
        del v
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        q128 = _mean_pool(q, 128)
        torch.cuda.synchronize(device)
        q_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        k64 = _mean_pool(k, 64)
        torch.cuda.synchronize(device)
        k_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        scores = torch.matmul(
            q128[..., geometry.pure_video_q_start:, :],
            k64[..., geometry.pure_video_kv_start:, :].transpose(-1, -2),
        )
        torch.cuda.synchronize(device)
        qk_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        route128 = torch.topk(scores, retained, dim=-1).indices
        torch.cuda.synchronize(device)
        topk_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        q64 = _mean_pool(q, 64)
        scores64 = torch.matmul(
            q64[..., geometry.pure_video_q_start * 2:, :],
            k64[..., geometry.pure_video_kv_start:, :].transpose(-1, -2),
        )
        route64 = torch.topk(scores64, retained, dim=-1).indices
        pairs = min(route128.shape[-2], route64.shape[-2] // 2)
        route128_sorted = route128[..., :pairs, :].sort(dim=-1).values
        child0 = route64[..., :pairs * 2:2, :].sort(dim=-1).values
        child1 = route64[..., 1:pairs * 2:2, :].sort(dim=-1).values
        compatible = (
            (route128_sorted == child0).all(dim=-1)
            & (route128_sorted == child1).all(dim=-1)
        ).float().mean()
        torch.cuda.synchronize(device)
        compatibility_ms = (time.perf_counter() - started) * 1000
        if index >= args.warmups:
            values["q128_mean_ms"].append(q_ms)
            values["k64_mean_ms"].append(k_ms)
            values["qmean_kmean_ms"].append(qk_ms)
            values["topk_ms"].append(topk_ms)
            values["compatibility_ms"].append(compatibility_ms)
            values["router_ms"].append(q_ms + k_ms + qk_ms + topk_ms + compatibility_ms)
            compatibility.append(float(compatible))
        del q, k, q128, q64, k64, scores, scores64, route128, route64
    return {
        key: statistics.median(samples) for key, samples in values.items()
    } | {"compatible_128_vs_two64_fraction": statistics.median(compatibility)}


def _build_context(args, modes):
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for benchmark execution")
    here = os.path.dirname(os.path.abspath(__file__))
    pack = os.path.dirname(here)
    sys.path.insert(0, pack)
    sys.path.insert(0, os.path.abspath(os.path.join(pack, "..", "..")))

    import _minimax_vram_probe_base as probe
    from h3_memory_optimizer.attention import (
        ATTENTION_AUTO,
        ATTENTION_SOL,
        FALLBACK_ERROR,
        RuntimeEnvironment,
        resolve_attention,
    )
    from h3_probe.layout import TokenLayout
    from h3_runtime.context import RUNTIME_KEY, RuntimeSnapshot

    device = torch.device("cuda", torch.cuda.current_device())
    layout = _layout(probe, TokenLayout, args)
    runtime = RuntimeSnapshot(
        request_id=0,
        step_index=10,
        total_steps=20,
        sigma=0.5,
        branch=(0,),
        layout=layout,
        layout_signature=(layout.seq_len, tuple(layout.segments)),
        compute_dtype=torch.bfloat16,
        device=device,
    )
    environment = RuntimeEnvironment.detect()
    context = {
        "torch": torch,
        "device": device,
        "layout": layout,
        "runtime": runtime,
        "runtime_key": RUNTIME_KEY,
    }
    if "dense_sage" in modes:
        context["dense"] = resolve_attention(
            ATTENTION_AUTO, FALLBACK_ERROR, environment=environment
        ).backend
    if "sol" in modes:
        context["sol"] = resolve_attention(
            ATTENTION_SOL,
            FALLBACK_ERROR,
            environment=environment,
            adapter_options={
                "tau": 1.0,
                "thresh_type": "diag",
                "dense_steps": 0,
                "dense_layers": 0,
                "sink_mode": "prefix",
                "correctness_gate": True,
                "strict": True,
                "kv_splits": 1,
                "gate_heads": 4,
                "density_heads": 4,
                "max_sink_fraction": 1.0,
            },
        ).backend
    if any(mode == "sparse_sage_128x64" or mode == "hybrid_sage_flex" for mode in modes):
        from h3_attention.hybrid.sparse_sage import (
            SparseSageExecutor,
            preflight_sparse_sage,
        )
        context["sparse"] = SparseSageExecutor(preflight_sparse_sage())
    if "sparse_sage_128x64" in modes:
        from h3_attention.hybrid.router import SparseTileRouter
        from h3_attention.hybrid.stats import DeferredCudaTiming
        context["router"] = SparseTileRouter()
        context["DeferredCudaTiming"] = DeferredCudaTiming
    if any(mode.startswith("flex_") or mode == "hybrid_sage_flex" for mode in modes):
        from torch.nn.attention.flex_attention import flex_attention
        context["flex"] = compile_flex_attention(torch, flex_attention)
    context["environment"] = environment
    return context


def run_benchmark(args):
    budget = parse_fraction(args.budget, name="budget")
    hard_q = parse_fraction(args.flex_fraction, name="flex fraction")
    hard_heads = parse_fraction(args.hard_head_fraction, name="hard head fraction")
    requested = [value.strip() for value in args.backends.split(",") if value.strip()]
    modes = tuple(requested)
    if args.sweep:
        modes = ("dense_sage", "sparse_sage_128x64", "hybrid_sage_flex")
    context = _build_context(args, modes)
    rows = []
    if args.sweep:
        rows.append(_measure_case(context, "dense_sage", budget, 0.0, 0.0, 64, args))
        for sweep_budget in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            rows.append(_measure_case(
                context, "sparse_sage_128x64", sweep_budget, 0.0, 0.0, 64, args
            ))
        for case in sweep_cases():
            rows.append(_measure_case(
                context,
                "hybrid_sage_flex",
                case["budget"],
                case["hard_q_fraction"],
                hard_heads,
                case["flex_q_tile"],
                args,
            ))
    else:
        for mode in requested:
            rows.append(_measure_case(
                context, mode, budget,
                hard_q if mode == "hybrid_sage_flex" else 0.0,
                hard_heads if mode == "hybrid_sage_flex" else 0.0,
                args.hybrid_flex_q if mode == "hybrid_sage_flex" else (
                    int(mode.split("_")[1].split("x")[0]) if mode.startswith("flex_") else 64
                ),
                args,
            ))

    dense_ms = next(
        (row["latency_ms"] for row in rows if row["mode"] == "dense_sage"), None
    )
    router = _router_metrics(context, budget, args) if args.router else None
    for row in rows:
        row["speedup_vs_dense"] = dense_ms / row["latency_ms"] if dense_ms else None
        row["attention_ms"] = row["latency_ms"]
        saved = dense_ms - row["latency_ms"] if dense_ms is not None else None
        row["saved_attention_ms"] = saved
        row["router_as_percent_of_saved_attention"] = (
            100 * router["router_ms"] / saved
            if router is not None and saved is not None and saved > 0 else None
        )
    metadata = {
        "frames": args.frames,
        "width": args.width,
        "height": args.height,
        "text_len": args.text_len,
        "sequence": context["layout"].seq_len,
        "video_tokens": context["layout"].video_range[1] - context["layout"].video_range[0],
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "budget": budget,
        "pattern": args.pattern,
        "timing": args.timing,
        "gpu": context["environment"].device_name,
        "architecture": context["environment"].architecture,
        "sweep": bool(args.sweep),
    }
    return metadata, rows, router


def _print_results(metadata, rows, break_even):
    print("H3 Hybrid Sparse Benchmark")
    print("GPU: %s / %s" % (metadata["gpu"], metadata["architecture"].upper()))
    print("sequence: %d" % metadata["sequence"])
    print("heads: %d  head_dim: %d  video tokens: %d" % (
        metadata["heads"], metadata["head_dim"], metadata["video_tokens"]
    ))
    print("pattern: %s  timing: %s" % (metadata["pattern"], metadata["timing"]))
    print()
    print("backend                    budget  hard Q  hard H   latency    peak GiB  vs dense")
    print("-------------------------------------------------------------------------------")
    for row in rows:
        speedup = row.get("speedup_vs_dense")
        print("%-27s %6.1f%% %6.1f%% %6.1f%% %8.2f ms %9.3f %8s" % (
            row["mode"], 100 * row["budget"], 100 * row["hard_q_fraction"],
            100 * row["hard_head_fraction"], row["latency_ms"],
            row["peak_allocated_gib"], "-" if speedup is None else "%.2fx" % speedup,
        ))
    sparse_rows = [row for row in rows if row.get("mode") == "sparse_sage_128x64"]
    if sparse_rows:
        print("\nSparse Sage stage medians (ms):")
        for row in sparse_rows:
            values = ", ".join(
                "%s=%.3f" % (stage, row["%s_ms" % stage])
                for stage in SPARSE_TIMING_STAGES
            )
            print("  budget %.1f%%: %s" % (100 * row["budget"], values))
    if break_even:
        print("\nMeasured break-even frontier (greatest faster sampled hard-Q fraction):")
        for key, value in break_even.items():
            print("  %s: %s" % (key, "none" if value is None else "%.1f%%" % (100 * value)))


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    metadata, rows, router = run_benchmark(args)
    break_even = derive_break_even(rows)
    write_reports(metadata, rows, router, break_even, args.json, args.csv)
    _print_results(metadata, rows, break_even)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
