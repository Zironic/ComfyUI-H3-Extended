"""Runtime patching and measurement for the MiniMax H3 arm matrix."""

from dataclasses import dataclass
import statistics
import threading
import time

import torch

import _minimax_vram_probe_base as base
from _minimax_vram_probe_ab_cli import MLP_MODES, QKV_MODES, selected_arms

MIB = 1024 ** 2


@dataclass(frozen=True)
class ForwardMeasurement:
    peak_bytes: int
    median_ms: float
    physical_free_start: int
    physical_free_inputs: int
    physical_free_min: int
    physical_free_end: int
    physical_free_recovered: int
    physical_samples: int

    @property
    def physical_drop(self):
        return max(0, self.physical_free_start - self.physical_free_min)


class PhysicalFreeMonitor:
    """Best-effort cudaMemGetInfo sampler for one untimed residency probe.

    The timing iterations run after this monitor stops, so polling overhead does
    not contaminate the reported median. With cudaMallocAsync the pool normally
    retains its high-water pages after the warm-up, but polling also catches a
    shorter physical-free minimum when it does not.
    """

    def __init__(self, device, poll_ms):
        self.device = device
        self.interval = max(float(poll_ms), 0.1) / 1000.0
        self.minimum = None
        self.samples = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None

    def _sample(self):
        free, _ = torch.cuda.mem_get_info(self.device)
        with self._lock:
            self.minimum = free if self.minimum is None else min(self.minimum, free)
            self.samples += 1

    def _run(self):
        torch.cuda.set_device(self.device)
        self._sample()
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self):
        self._sample()
        self._thread = threading.Thread(
            target=self._run,
            name="h3-physical-vram-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()


def _route_arm(block, attention_forward, mlp_forward, variant):
    """Call the shared block with explicit attention and MLP implementations."""

    def forward(
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options=None,
    ):
        previous_attention = block.attn.forward
        previous_mlp = block.mlp.forward
        block.attn.forward = attention_forward
        block.mlp.forward = mlp_forward
        try:
            return block.forward(
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=(
                    transformer_options if transformer_options is not None else {}
                ),
            )
        finally:
            block.mlp.forward = previous_mlp
            block.attn.forward = previous_attention

    forward._h3_probe_variant = variant
    return forward


def runtime_snapshot_for_layout(layout, device, dtype):
    """Build the real packed layout/runtime metadata required by hybrid Sage."""
    from h3_probe.layout import TokenLayout
    from h3_runtime.context import RuntimeSnapshot

    ranges = {kind: (int(start), int(stop))
              for start, stop, kind in layout["segments"]}
    references = [
        (str(kind), int(start), int(stop))
        for start, stop, kind in layout["segments"]
        if kind not in ("text", "audio", "video")
    ]
    named = TokenLayout(
        seq_len=int(layout["seq_len"]),
        text_range=ranges["text"],
        audio_range=ranges["audio"],
        video_range=ranges["video"],
        video_shape=(int(layout["latent_t"]), int(layout["height"] // 32),
                     int(layout["width"] // 32)),
        audio_t=int(layout["audio_t"]),
        reference_ranges=references,
        segments=[(int(a), int(b), str(kind)) for a, b, kind in layout["segments"]],
    )
    return RuntimeSnapshot(
        request_id=0, step_index=0, total_steps=1, sigma=0.5,
        branch=(0,), layout=named,
        layout_signature=(named.seq_len, tuple(named.segments)),
        compute_dtype=dtype, device=device,
    )


def load_block_weights(block, checkpoint, block_index=0):
    """Attach only one block's real QKV/norm and MLP checkpoint modules."""
    from benchmarks.bench_fused_qkv import build_attention
    from benchmarks.benchmark_h3_activation_memory import (
        build_checkpoint_mlp,
        load_block_mlp_tensors,
    )

    attention, attention_hidden, attention_prefix = build_attention(
        checkpoint,
        block_index,
        block.attn.q_norm.eps,
    )
    loaded_mlp = load_block_mlp_tensors(checkpoint, block_index)
    mlp, mlp_hidden, _ = build_checkpoint_mlp(
        loaded_mlp,
        torch.bfloat16,
        hidden=attention_hidden,
    )
    if attention_hidden != mlp_hidden:
        raise ValueError("checkpoint attention and MLP hidden dimensions differ")

    block.attn.qkv_proj = attention.qkv_proj
    block.attn.q_norm.weight = torch.nn.Parameter(
        attention.q_norm.weight, requires_grad=False
    )
    block.attn.k_norm.weight = torch.nn.Parameter(
        attention.k_norm.weight, requires_grad=False
    )
    block.mlp.fc1 = mlp.fc1
    block.mlp.fc2 = mlp.fc2
    block.requires_grad_(False)
    return attention_prefix, loaded_mlp["prefix"]


def build_attention_backend(qkv_mode, backend_cls=None, config_cls=None):
    """Construct the production hybrid backend selected by one probe arm."""
    if qkv_mode not in QKV_MODES:
        raise ValueError("unsupported QKV mode %r" % qkv_mode)
    if backend_cls is None or config_cls is None:
        from h3_attention.hybrid.backend import HybridSparseBackend
        from h3_attention.hybrid.config import HybridSparseConfig
        backend_cls = backend_cls or HybridSparseBackend
        config_cls = config_cls or HybridSparseConfig
    return backend_cls(config_cls(
        mode=qkv_mode,
        timing=False,
        run_tag="minimax_vram_probe_%s" % qkv_mode,
    ))


def _tile_ranges(width, tile_count):
    tile_count = int(tile_count)
    if tile_count not in (2, 4):
        raise ValueError("ConvRot probe tile count must be 2 or 4")
    if width % tile_count:
        raise ValueError("H3 FFN width is not divisible by %d tiles" % tile_count)
    tile_width = width // tile_count
    if tile_width % 256:
        raise ValueError("every ConvRot tile must preserve complete 256-wide groups")
    return tuple(
        (index * tile_width, (index + 1) * tile_width)
        for index in range(tile_count)
    )


class ProbeConvRotTiledMLP:
    """One-invocation ConvRot tile session; packed copies never cross arms."""

    def __init__(self, mlp, sample, tile_count, *, acquire_fn=None,
                 parts_fn=None, linear_fn=None):
        self.mlp = mlp
        self.sample = sample
        self.tile_count = int(tile_count)
        self.acquire_fn = acquire_fn
        self.parts_fn = parts_fn
        self.linear_fn = linear_fn
        self.tiles = None

    def __enter__(self):
        if self.sample.dtype != torch.bfloat16:
            raise TypeError("tiled ConvRot probe arms require BF16 input")
        if self.acquire_fn is None or self.parts_fn is None or self.linear_fn is None:
            from h3_activation_memory.linear import (
                _convrot_linear,
                _convrot_parts,
                acquire_linear,
            )
            self.acquire_fn = self.acquire_fn or acquire_linear
            self.parts_fn = self.parts_fn or _convrot_parts
            self.linear_fn = self.linear_fn or _convrot_linear

        fc1 = self.acquire_fn(self.mlp.fc1, self.sample)
        try:
            if fc1.bias is not None:
                raise ValueError("fc1 ConvRot weight must not have a bias")
            fc1_qdata, fc1_scale = self.parts_fn(fc1.weight, "fc1")
            if fc1_qdata.shape[0] % 2:
                raise ValueError("fc1 ConvRot output width must be divisible by two")
            ffn_width = int(fc1_qdata.shape[0]) // 2
            hidden_width = int(fc1_qdata.shape[1])
            ranges = _tile_ranges(ffn_width, self.tile_count)
            fc1_tiles = []
            for start, stop in ranges:
                scale = (
                    fc1_scale.contiguous().clone()
                    if fc1_scale.numel() == 1
                    else torch.cat((
                        fc1_scale[start:stop],
                        fc1_scale[ffn_width + start:ffn_width + stop],
                    ), dim=0).contiguous()
                )
                fc1_tiles.append((torch.cat((
                    fc1_qdata[start:stop],
                    fc1_qdata[ffn_width + start:ffn_width + stop],
                ), dim=0).contiguous(), scale))
        finally:
            fc1.release()

        fc2 = self.acquire_fn(self.mlp.fc2, self.sample)
        try:
            if fc2.bias is not None:
                raise ValueError("fc2 ConvRot weight must not have a bias")
            fc2_qdata, fc2_scale = self.parts_fn(fc2.weight, "fc2")
            if tuple(fc2_qdata.shape) != (hidden_width, ffn_width):
                raise ValueError("fc1/fc2 ConvRot dimensions are incompatible")
            self.tiles = tuple(
                {
                    "fc1_weight": fc1_weight,
                    "fc1_scale": fc1_tile_scale,
                    "fc2_weight": fc2_qdata[:, start:stop].contiguous(),
                    "fc2_scale": fc2_scale.contiguous().clone(),
                }
                for (start, stop), (fc1_weight, fc1_tile_scale)
                in zip(ranges, fc1_tiles)
            )
        finally:
            fc2.release()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.tiles = None
        self.sample = None
        self.mlp = None
        return False

    def forward(self, x, chunk_rows):
        if self.tiles is None:
            raise RuntimeError("ConvRot tiled probe session is not active")
        output = x.new_empty((x.shape[0], self.tiles[0]["fc2_weight"].shape[0]))
        for row_start in range(0, x.shape[0], int(chunk_rows)):
            row_stop = min(x.shape[0], row_start + int(chunk_rows))
            chunk = x[row_start:row_stop]
            accumulated = None
            for tile in self.tiles:
                expanded = self.linear_fn(
                    chunk,
                    tile["fc1_weight"],
                    tile["fc1_scale"],
                )
                partial = self.linear_fn(
                    expanded,
                    tile["fc2_weight"],
                    tile["fc2_scale"],
                    input_act="swiglu",
                )
                if accumulated is None:
                    accumulated = partial
                else:
                    accumulated.add_(partial)
                del expanded, partial
            output[row_start:row_stop] = accumulated
        return output


def make_tiled_mlp_forward(mlp, tile_count, chunk_rows):
    if int(chunk_rows) <= 0:
        raise ValueError("activation chunk rows must be positive")

    def forward(x):
        with ProbeConvRotTiledMLP(mlp, x[:1], tile_count) as tiled:
            return tiled.forward(x, chunk_rows)

    forward._h3_probe_tile_count = int(tile_count)
    return forward


def build_forwards(block, args):
    """Build one forward per selected QKV x MLP arm over one shared block."""
    from h3_attention.forward import make_forward as make_attention_forward

    arms = selected_arms(args)
    selected_qkv = {label.split("/", 1)[0] for label in arms}
    stock_mlp_forward = block.mlp.forward
    forwards = {}
    backends = {}
    configs = {}
    for qkv_mode in QKV_MODES:
        if qkv_mode not in selected_qkv:
            continue
        backend = build_attention_backend(qkv_mode)
        attention = make_attention_forward(
            block.attn, layer_index=0, backend=backend,
            projector=backend.projector,
        )
        backends[qkv_mode] = backend
        for mlp_mode in MLP_MODES:
            label = "%s/%s" % (qkv_mode, mlp_mode)
            if label not in arms:
                continue
            if mlp_mode == "untiled":
                mlp_forward = stock_mlp_forward
                tile_count = 0
            else:
                tile_count = 2 if mlp_mode == "convrot2" else 4
                mlp_forward = make_tiled_mlp_forward(
                    block.mlp,
                    tile_count,
                    args.activation_chunk_rows,
                )
            configs[label] = {
                "mode": mlp_mode,
                "tile_count": tile_count,
                "chunk_rows": (
                    None if tile_count == 0 else int(args.activation_chunk_rows)
                ),
            }
            forwards[label] = _route_arm(
                block, attention, mlp_forward, label,
            )
            forwards[label]._h3_probe_qkv_mode = qkv_mode
            forwards[label]._h3_probe_mlp_mode = mlp_mode
    return forwards, backends, configs


def measure_forward(
    forward_fn,
    layout,
    arch,
    dtype,
    device,
    *,
    warmup,
    iterations,
    seed,
    physical_poll_ms,
):
    """Measure Torch peak, physical-free floor, and unmonitored median time.

    ``physical_free_min`` is sampled during warm-up plus one extra untimed
    forward. The timed iterations run after the sampler stops. All physical
    values come from ``cudaMemGetInfo`` and are absolute bytes, not projected
    production totals.
    """
    from comfy.ldm.minimax.model import rope_rotation_table

    torch.manual_seed(seed)
    seq_len = layout["seq_len"]
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    physical_free_start, _ = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocation_base = torch.cuda.memory_allocated(device)

    x = torch.randn(seq_len, arch["hidden_size"], dtype=dtype, device=device)
    rope = rope_rotation_table(
        torch.randn(
            seq_len,
            arch["rope_inv_freq_len"] * 6,
            dtype=torch.float32,
            device=device,
        ),
        dtype,
    )
    t_emb = torch.randn(
        layout["n_t_classes"],
        arch["time_embed_dim"],
        dtype=torch.float16,
        device=device,
    )
    torch.cuda.synchronize(device)
    physical_free_inputs, _ = torch.cuda.mem_get_info(device)

    def invoke():
        return forward_fn(
            x,
            t_emb,
            layout["mod_segments"],
            rope,
            transformer_options={
                "minimax_h3_runtime": runtime_snapshot_for_layout(
                    layout, device, dtype,
                ),
            },
        )

    monitor = PhysicalFreeMonitor(device, physical_poll_ms)
    monitor.start()
    try:
        with torch.no_grad():
            for _ in range(max(0, warmup)):
                y = invoke()
                del y
        torch.cuda.synchronize(device)

        # One untimed probe keeps physical monitoring out of the timing numbers.
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            y = invoke()
        torch.cuda.synchronize(device)
        peaks = [torch.cuda.max_memory_allocated(device) - allocation_base]
        del y
    finally:
        monitor.stop()

    times = []
    for _ in range(max(1, iterations)):
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.no_grad():
            y = invoke()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - started) * 1000.0)
        peaks.append(torch.cuda.max_memory_allocated(device) - allocation_base)
        del y

    # cudaMallocAsync normally retains the high-water pool, so this boundary
    # sample also catches any timed-iteration growth missed by the untimed poller.
    physical_free_end, _ = torch.cuda.mem_get_info(device)
    physical_free_min = min(
        monitor.minimum if monitor.minimum is not None else physical_free_end,
        physical_free_inputs,
        physical_free_end,
    )

    del x, rope, t_emb
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    physical_free_recovered, _ = torch.cuda.mem_get_info(device)

    return ForwardMeasurement(
        peak_bytes=max(peaks),
        median_ms=statistics.median(times),
        physical_free_start=physical_free_start,
        physical_free_inputs=physical_free_inputs,
        physical_free_min=physical_free_min,
        physical_free_end=physical_free_end,
        physical_free_recovered=physical_free_recovered,
        physical_samples=monitor.samples,
    )


def safe_measure(*args, **kwargs):
    try:
        return measure_forward(*args, **kwargs), None
    except base.OOM_ERRORS as exc:
        if not base.is_oom(exc):
            raise
        torch.cuda.empty_cache()
        return None, str(exc)


def update_resident_fit(
    points,
    sequence,
    ms,
    spill_ratio,
    *,
    include_in_fit=True,
):
    """Compare with the current resident curve and optionally add the point.

    LOW-physical-free points are excluded from the resident fit even when they
    have not yet produced a timing cliff. That keeps the reference curve from
    learning WDDM pressure as normal behavior.
    """
    coefficient = base.fit_ms(points)
    predicted = (
        coefficient[0] * sequence + coefficient[1] * sequence * sequence
        if coefficient
        else None
    )
    spill = bool(predicted and ms > predicted * spill_ratio)
    if include_in_fit and not spill:
        points.append((sequence, ms))
    return spill, (ms / predicted if predicted else None)


def fmt_gib(value):
    return "   OOM" if value is None else f"{value / base.GB:>6.3f}G"


def fmt_ms(value):
    return "    OOM" if value is None else f"{value:>7.1f}"


def fmt_mib(value):
    return "   OOM" if value is None else f"{value / MIB:>6.0f}M"
