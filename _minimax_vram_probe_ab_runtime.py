"""Runtime patching and measurement for the activation-memory VRAM A/B probe."""

from dataclasses import dataclass
import statistics
import threading
import time

import torch

import _minimax_vram_probe_base as base

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


def build_forwards(block, args):
    # Imported only after base.select_attention(True) has selected Sage.
    from h3_attention.forward import make_forward as make_attention_forward
    from h3_attention.sage_mem_eff import SM89SageMemoryEfficientBackend
    from h3_activation_memory.config import ActivationMemoryConfig
    from h3_activation_memory.forward import make_forward as make_activation_forward

    stock_block_forward = block.forward
    backend = SM89SageMemoryEfficientBackend()
    block.attn.forward = make_attention_forward(
        block.attn, layer_index=0, backend=backend
    )
    config = ActivationMemoryConfig(
        mode=args.activation_mode,
        chunk_rows=args.activation_chunk_rows,
        alignment=args.activation_alignment,
        strict=not args.activation_nonstrict,
        prefer_held_weights=not args.no_held_weights,
    )
    activation_forward = make_activation_forward(
        block,
        layer_index=0,
        config=config,
        original_forward=stock_block_forward,
    )
    return stock_block_forward, activation_forward, backend, config


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
            transformer_options={},
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
