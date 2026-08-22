"""CPU-safe self-test for the H3 VRAM capacity and emergency guards.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_vram_guard.py

Driver, allocator, cache-release, and VBAR residency calls are stubbed. The test
covers the accounting and wrapper contracts, not live physical-VRAM behavior.
"""

import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

# model_management selects its device while importing; keep this self-test
# runnable on CI hosts without CUDA while the driver calls below remain mocked.
if "--cpu" not in sys.argv:
    sys.argv.append("--cpu")
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

import torch  # noqa: E402

import comfy.model_management  # noqa: E402
import run_context  # noqa: E402
import vram_guard  # noqa: E402
import weight_footprint  # noqa: E402
import working_set  # noqa: E402
from h3_runtime.timing import observing_stages, timed_stage  # noqa: E402

MB = vram_guard.MB
CUDA = torch.device("cuda:0")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok: %s" % msg)


class FakeGPU:
    """Stands in for the driver: `free` is what mem_get_info reports next."""

    def __init__(self, free_mb, recovers_to_mb=None, total_mb=12282,
                 allocated_mb=8, peak_mb=12, reserved_mb=9000,
                 backend="native", fraction=1.0):
        self.free = free_mb * MB
        self.total = total_mb * MB
        self.recovers_to = None if recovers_to_mb is None else recovers_to_mb * MB
        self.allocated = allocated_mb * MB
        self.peak = peak_mb * MB
        self.reserved = reserved_mb * MB
        self.backend = backend
        self.fraction = float(fraction)
        self.fraction_sets = []
        self.stats = {
            "num_device_free": 0,
            "segment.all.freed": 0,
            "reserved_bytes.all.freed": 0,
            "num_alloc_retries": 0,
            "num_ooms": 0,
            "num_oom_rejections": 0,
        }
        self.releases = 0
        self.synchronizes = 0
        self.peak_resets = 0

    def mem_get_info(self, device=None):
        return self.free, self.total

    def release(self, force=False):
        self.releases += 1
        if self.recovers_to is not None:
            self.free = self.recovers_to

    def synchronize(self, device=None):
        self.synchronizes += 1

    def reset_peak(self, device=None):
        self.peak_resets += 1

    def get_fraction(self, device=None):
        return self.fraction

    def set_fraction(self, fraction, device=None):
        self.fraction = float(fraction)
        self.fraction_sets.append(self.fraction)

    def memory_stats(self, device=None):
        return dict(self.stats)


def with_gpu(gpu, fn):
    real_info = torch.cuda.mem_get_info
    real_release = comfy.model_management.soft_empty_cache
    real_reserved = torch.cuda.memory_reserved
    real_alloc = torch.cuda.memory_allocated
    real_peak = torch.cuda.max_memory_allocated
    real_reset = torch.cuda.reset_peak_memory_stats
    real_sync = torch.cuda.synchronize
    real_capability = torch.cuda.get_device_capability
    real_backend = torch.cuda.memory.get_allocator_backend
    real_get_fraction = torch.cuda.get_per_process_memory_fraction
    real_set_fraction = torch.cuda.set_per_process_memory_fraction
    real_stats = torch.cuda.memory_stats
    real_device = comfy.model_management.get_torch_device
    real_alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF")
    real_cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    torch.cuda.mem_get_info = gpu.mem_get_info
    comfy.model_management.soft_empty_cache = gpu.release
    torch.cuda.memory_reserved = lambda device=None: gpu.reserved
    torch.cuda.memory_allocated = lambda device=None: gpu.allocated
    torch.cuda.max_memory_allocated = lambda device=None: gpu.peak
    torch.cuda.reset_peak_memory_stats = gpu.reset_peak
    torch.cuda.synchronize = gpu.synchronize
    torch.cuda.get_device_capability = lambda device=None: (8, 9)
    torch.cuda.memory.get_allocator_backend = lambda: gpu.backend
    torch.cuda.get_per_process_memory_fraction = gpu.get_fraction
    torch.cuda.set_per_process_memory_fraction = gpu.set_fraction
    torch.cuda.memory_stats = gpu.memory_stats
    comfy.model_management.get_torch_device = lambda: CUDA
    os.environ["PYTORCH_ALLOC_CONF"] = "backend:%s,garbage_collection_threshold:0.95" % gpu.backend
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    try:
        return fn()
    finally:
        torch.cuda.mem_get_info = real_info
        comfy.model_management.soft_empty_cache = real_release
        torch.cuda.memory_reserved = real_reserved
        torch.cuda.memory_allocated = real_alloc
        torch.cuda.max_memory_allocated = real_peak
        torch.cuda.reset_peak_memory_stats = real_reset
        torch.cuda.synchronize = real_sync
        torch.cuda.get_device_capability = real_capability
        torch.cuda.memory.get_allocator_backend = real_backend
        torch.cuda.get_per_process_memory_fraction = real_get_fraction
        torch.cuda.set_per_process_memory_fraction = real_set_fraction
        torch.cuda.memory_stats = real_stats
        comfy.model_management.get_torch_device = real_device
        if real_alloc_conf is None:
            os.environ.pop("PYTORCH_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_ALLOC_CONF"] = real_alloc_conf
        if real_cuda_alloc_conf is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = real_cuda_alloc_conf


class FakePatcher:
    def __init__(self, existing=None, model=None):
        self.model_options = {}
        self.model = model
        self.load_device = CUDA
        if existing is not None:
            self.model_options["model_function_wrapper"] = existing

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn


class FakeVBAR:
    def __init__(self, residency, base_addr=0x10000000, device=0):
        self.residency = list(residency)
        self.base_addr = base_addr
        self.device = device
        self.reads = 0

    def get_residency(self):
        self.reads += 1
        return list(self.residency)


class FakeWeight(torch.nn.Module):
    def __init__(self, allocation):
        super().__init__()
        self._v = allocation


class FakeNested:
    def __init__(self, tensors):
        self.tensors = tensors


class FakeTensorMeta:
    def __init__(self, shape, dtype=torch.bfloat16, device=CUDA):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device


def allocation(vbar, page, pages=1, offset=0):
    return (vbar, vbar.base_addr + page * weight_footprint.PAGE_SIZE + offset,
            pages * weight_footprint.PAGE_SIZE - offset)


def fake_h3_model():
    vbar = FakeVBAR([1, 3, 0, 0, 0])
    diffusion = torch.nn.Module()
    block0 = torch.nn.Module()
    block0.a = FakeWeight(allocation(vbar, 0))
    block0.overlap = FakeWeight(allocation(vbar, 0, offset=4096))
    block0.b = FakeWeight(allocation(vbar, 1))
    block0.c = FakeWeight(allocation(vbar, 2))
    block0.d = FakeWeight(allocation(vbar, 3))
    block0.attn = torch.nn.Module()
    block1 = torch.nn.Module()
    block1.a = FakeWeight(allocation(vbar, 2))
    block1.b = FakeWeight(allocation(vbar, 3))
    block1.c = FakeWeight(allocation(vbar, 4))
    block1.attn = torch.nn.Module()
    diffusion.blocks = torch.nn.ModuleList([block0, block1])
    diffusion.final_layer = torch.nn.Module()
    model = torch.nn.Module()
    model.diffusion_model = diffusion
    model.model_loaded_weight_memory = 9999 * MB
    return model, vbar


def forward_args(text_len=2):
    video = FakeTensorMeta((1, 24, 2, 4, 4))
    audio = FakeTensorMeta((1, 32, 2, 3))
    return {
        "input": FakeNested([video, audio]),
        "timestep": torch.tensor([0.7]),
        "c": {
            "c_crossattn": torch.zeros(1, text_len, 64),
            "minimax_payload": {},
            "transformer_options": {"minimax_h3_attention_backend": "sage"},
            "y": 1,
        },
        "cond_or_uncond": [0],
    }


def main():
    logging.basicConfig(level=logging.INFO, format="    %(levelname)s %(message)s")

    print("VBAR page accounting")
    model, vbar = fake_h3_model()
    pages = weight_footprint.pages_for_allocation(allocation(vbar, 0, offset=4096))
    check(len(pages) == 1, "an unaligned allocation maps to its containing 32 MiB page")
    crossing = weight_footprint.pages_for_allocation(
        (vbar, vbar.base_addr + weight_footprint.PAGE_SIZE - 512, 1024)
    )
    check(len(crossing) == 2, "an allocation crossing a page boundary includes both pages")
    fp = weight_footprint.footprint(model, device=CUDA)
    check(fp.resident_unpinned_pages == 1 and fp.pinned_pages == 1,
          "resident and pinned page bits are classified separately")
    check(fp.mandatory_group == "blocks.0" and fp.mandatory_pages == 4,
          "the largest complete block is selected without combining sequential blocks")
    check(fp.mandatory_pinned_pages == 1 and fp.mandatory_bytes == 3 * weight_footprint.PAGE_SIZE,
          "mandatory pages already pinned in the floor are not added twice")
    check(vbar.reads >= 1, "residency is read without faulting or pinning pages")

    print("working-set signature and profile")
    patcher = FakePatcher(model=model)
    sig = with_gpu(FakeGPU(900), lambda: working_set.make_signature(forward_args(), patcher))
    bound, source = working_set.upper_bound(sig)
    check(sig.seq_len == 16 and len(sig.segments) == 3,
          "signature uses the resolved packed layout and segment geometry")
    check(bound == 16 * working_set.CALIBRATED_BYTES_PER_ROW and "calibrated" in source,
          "unprofiled signatures use the conservative 128 KiB-per-row envelope")
    working_set.clear_observed()
    working_set.record_observed(sig, bound // 2)
    check(working_set.upper_bound(sig)[0] == bound,
          "a low observation never lowers the conservative bound")
    working_set.record_observed(sig, bound * 2)
    check(working_set.upper_bound(sig)[0] > bound,
          "a higher observation raises the bound with an allowance")
    working_set.clear_observed()

    print("request-scoped QKV and MLP attribution")
    gpu = FakeGPU(free_mb=9000, allocated_mb=100)

    def sample_phases():
        profiler = vram_guard.PhaseMemoryProfiler(CUDA)
        options = {}
        with observing_stages(options, profiler):
            with timed_stage(options, "qkv_proj"):
                gpu.allocated = 132 * MB
        gpu.allocated = 140 * MB
        profiler("mlp_chunk_enter", 0, {"chunk_index": 0})
        gpu.allocated = 220 * MB
        profiler("mlp_fc2_ready", 0, {"chunk_index": 0})
        gpu.allocated = 150 * MB
        profiler("mlp_chunk_gated", 0, {"chunk_index": 0})
        return profiler.finish(120 * MB)

    phases = with_gpu(gpu, sample_phases)
    check(phases.qkv == 32 * MB, "QKV allocation is sampled at the actual projection stage")
    check(phases.mlp == 80 * MB, "MLP allocation spans the complete live chunk")
    check(phases.forward == 120 * MB, "the whole-forward peak remains a separate observation")

    print("native allocator GC placement")
    saved_alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF")
    saved_cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native,garbage_collection_threshold:0.93"
    try:
        check(vram_guard._gc_threshold() == (0.93, "PYTORCH_CUDA_ALLOC_CONF"),
              "the legacy allocator-config alias used by this Comfy install is recognized")
    finally:
        if saved_alloc_conf is not None:
            os.environ["PYTORCH_ALLOC_CONF"] = saved_alloc_conf
        else:
            os.environ.pop("PYTORCH_ALLOC_CONF", None)
        if saved_cuda_alloc_conf is not None:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = saved_cuda_alloc_conf
        else:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    gpu = FakeGPU(free_mb=3000, total_mb=12282, reserved_mb=9000, allocated_mb=8000)

    def exercise_fraction():
        with vram_guard._native_memory_fraction(CUDA, 800 * MB) as policy:
            check(gpu.fraction == policy.fraction < 1.0,
                  "native allocator receives a temporary lower memory fraction")
            gpu.stats["num_device_free"] += 2
            gpu.stats["segment.all.freed"] += 2
            gpu.stats["reserved_bytes.all.freed"] += 256 * MB
        return policy

    policy = with_gpu(gpu, exercise_fraction)
    check(abs(policy.gc_target - policy.desired_gc_target) <= 1,
          "the 0.95 GC coefficient maps to one page ahead of the physical guard")
    check(gpu.fraction == 1.0 and gpu.fraction_sets[-1] == 1.0,
          "the previous process memory fraction is restored")
    check(vram_guard._stats_delta(policy.stats_before, policy.stats_after,
                                  "num_device_free") == 2,
          "generic allocator reclamation evidence is captured across the forward")

    async_gpu = FakeGPU(free_mb=3000, backend="cudaMallocAsync")

    def exercise_async():
        with vram_guard._native_memory_fraction(CUDA, 800 * MB) as async_policy:
            return async_policy

    async_policy = with_gpu(async_gpu, exercise_async)
    check(not async_gpu.fraction_sets and async_policy.backend == "cudaMallocAsync",
          "cudaMallocAsync is diagnosed but never receives native-only fraction control")

    diagnostic = with_gpu(gpu, lambda: "\n".join(vram_guard._memory_diagnostic_lines(
        CUDA,
        model_patcher=patcher,
        guard_bytes=800 * MB,
        policy=policy,
        phases=phases,
    )))
    check("Other/unattributed card use" in diagnostic and "AIMDO on this device" in diagnostic,
          "full diagnostic separates residual card use from AIMDO")
    check("PyTorch H3 QKV observed live delta" in diagnostic and
          "PyTorch H3 MLP observed live delta" in diagnostic,
          "full diagnostic includes QKV and MLP observations")
    check("temporary guarded-forward limit" in diagnostic and "current reserved" in diagnostic,
          "full diagnostic includes physical limit and native GC pressure")
    check("no dedicated threshold-GC trigger flag" in diagnostic,
          "full diagnostic states the limit of PyTorch GC attribution")

    print("capacity proof")
    gpu = FakeGPU(free_mb=176, total_mb=256)
    signature, proof = with_gpu(
        gpu, lambda: vram_guard._capacity_proof(patcher, forward_args(), 32, device=CUDA)
    )
    check(signature.seq_len == 16, "capacity proof returns the checked signature")
    check(proof.floor == 48 * MB,
          "floor subtracts only the current model's resident-unpinned page")
    check(proof.mandatory == 96 * MB,
          "capacity adds only incremental pages from the largest phase")
    check(proof.predicted <= proof.total,
          "capacity accepts when floor + working + weights + margin fits")
    check(gpu.synchronizes == 1 and gpu.releases == 1,
          "capacity settles CUDA and releases the Torch cache before measuring")
    check(proof.floor < model.model_loaded_weight_memory,
          "force-loaded weights already in physical usage are not double-counted")

    exact_gpu = FakeGPU(free_mb=98, total_mb=178)
    _, exact = with_gpu(
        exact_gpu, lambda: vram_guard._capacity_proof(
            patcher, forward_args(), 32, device=CUDA)
    )
    check(exact.headroom == 0, "predicted peak equal to physical VRAM is accepted")

    records = []
    real_error = logging.error
    logging.error = lambda msg, *a: records.append(msg % a if a else msg)
    try:
        gpu = FakeGPU(free_mb=80, total_mb=160)
        try:
            with_gpu(gpu, lambda: vram_guard._capacity_proof(
                patcher, forward_args(), 32, device=CUDA))
            raise AssertionError("expected the capacity proof to cancel")
        except comfy.model_management.InterruptProcessingException:
            check(True, "capacity rejects before apply_model when the equation exceeds VRAM")
    finally:
        logging.error = real_error
    capacity_log = "\n".join(records)
    check("Non-reclaimable starting floor" in capacity_log and "Deficit:" in capacity_log,
          "capacity cancellation logs every proof term and the deficit")

    print("above threshold")
    gpu = FakeGPU(free_mb=3000)
    free = with_gpu(gpu, lambda: vram_guard.check_vram(800, "test", device=CUDA))
    check(free == 3000 * MB, "returns the free byte count when there is headroom")
    check(gpu.releases == 0, "no cache release when above the threshold")

    print("disabled")
    gpu = FakeGPU(free_mb=10)
    check(with_gpu(gpu, lambda: vram_guard.check_vram(0, "test", device=CUDA)) is None,
          "threshold 0 disables the guard entirely")
    check(gpu.releases == 0, "disabled guard does not even query the driver")

    print("non-cuda device")
    check(vram_guard.check_vram(800, "test", device=torch.device("cpu")) is None,
          "cpu device is skipped rather than crashing")

    print("breach that recovers")
    gpu = FakeGPU(free_mb=500, recovers_to_mb=2500)
    free = with_gpu(gpu, lambda: vram_guard.check_vram(800, "test", device=CUDA))
    check(gpu.releases == 1, "a breach releases cached blocks once")
    check(free == 2500 * MB, "recovered memory lets the run continue")

    print("breach that does not recover")
    gpu = FakeGPU(free_mb=500, recovers_to_mb=600)
    try:
        with_gpu(gpu, lambda: vram_guard.check_vram(800, "test", device=CUDA))
        raise AssertionError("expected the guard to cancel")
    except comfy.model_management.InterruptProcessingException:
        check(True, "cancels with InterruptProcessingException, same as the Cancel button")
    check(gpu.releases == 1, "cancels after exactly one release attempt")

    print("boundary")
    gpu = FakeGPU(free_mb=800)
    with_gpu(gpu, lambda: vram_guard.check_vram(800, "test", device=CUDA))
    check(gpu.releases == 0, "free == threshold is not a breach")

    print("wrapper install")
    calls = []
    model, _ = fake_h3_model()
    patcher = FakePatcher(model=model)
    vram_guard.install_unet_guard(patcher, 800)
    wrapper = patcher.model_options["model_function_wrapper"]
    check(callable(wrapper), "wrapper installed on the patcher")

    def apply_model(x, timestep, **c):
        calls.append((x, timestep, c))
        return "denoised"

    args = forward_args()
    gpu = FakeGPU(free_mb=11000)
    out = with_gpu(gpu, lambda: wrapper(apply_model, args))
    check(out == "denoised", "wrapper calls through to apply_model with headroom")
    check(calls[-1][2]["y"] == 1 and "c_crossattn" in calls[-1][2],
          "conditioning kwargs are forwarded unchanged")
    check(gpu.releases == 1 and gpu.peak_resets == 1,
          "the first signature is proved and observed once")
    check(len(gpu.fraction_sets) == 2 and gpu.fraction_sets[-1] == 1.0,
          "the guarded forward restores its temporary PyTorch fraction")
    with_gpu(gpu, lambda: wrapper(apply_model, args))
    check(gpu.releases == 1 and gpu.peak_resets == 1,
          "the same successful signature is not proved again")
    changed = forward_args(text_len=3)
    with_gpu(gpu, lambda: wrapper(apply_model, changed))
    check(gpu.releases == 2 and gpu.peak_resets == 2,
          "a changed packed signature receives a new capacity proof")

    retry_patcher = FakePatcher(model=model)
    vram_guard.install_unet_guard(retry_patcher, 800)
    retry_wrapper = retry_patcher.model_options["model_function_wrapper"]
    retry_gpu = FakeGPU(free_mb=11000)

    def fail_forward(*_args, **_kwargs):
        raise ValueError("forward failed")

    try:
        with_gpu(retry_gpu, lambda: retry_wrapper(
            fail_forward,
            forward_args(text_len=4),
        ))
    except ValueError:
        pass
    check(retry_gpu.fraction == 1.0 and retry_gpu.fraction_sets[-1] == 1.0,
          "a failing forward also restores the previous PyTorch fraction")
    with_gpu(retry_gpu, lambda: retry_wrapper(apply_model, forward_args(text_len=4)))
    check(retry_gpu.releases == 2,
          "a failed first forward is proved again before the signature is trusted")

    broken_patcher = FakePatcher(model=model)
    vram_guard.install_unet_guard(broken_patcher, 800)
    broken_calls = []
    broken_args = forward_args()
    broken_args["c"].pop("c_crossattn")
    try:
        with_gpu(FakeGPU(free_mb=11000), lambda: broken_patcher.model_options[
            "model_function_wrapper"](lambda *_a, **_k: broken_calls.append(True), broken_args))
        raise AssertionError("expected an unknown signature to fail closed")
    except comfy.model_management.InterruptProcessingException:
        check(not broken_calls, "an unresolved working-set signature fails closed before apply_model")

    gpu = FakeGPU(free_mb=100, recovers_to_mb=100)
    n_before = len(calls)
    try:
        with_gpu(gpu, lambda: wrapper(apply_model, args))
        raise AssertionError("expected the wrapper to cancel")
    except comfy.model_management.InterruptProcessingException:
        check(len(calls) == n_before,
              "cancels BEFORE the forward runs, so the OOM allocation never happens")

    print("wrapper chaining")
    chained = []

    def previous(apply_model, args):
        chained.append(args)
        return "from previous wrapper"

    model, _ = fake_h3_model()
    patcher = FakePatcher(existing=previous, model=model)
    vram_guard.install_unet_guard(patcher, 800)
    gpu = FakeGPU(free_mb=11000)
    out = with_gpu(gpu, lambda: patcher.model_options["model_function_wrapper"](apply_model, args))
    check(out == "from previous wrapper" and len(chained) == 1,
          "an existing wrapper is chained, not replaced")

    print("disabled install")
    patcher = FakePatcher()
    vram_guard.install_unet_guard(patcher, 0)
    check("model_function_wrapper" not in patcher.model_options,
          "threshold 0 installs nothing at all")

    print("temporary arming")
    # CFGGuider.model_options IS the cached patcher's dict, so an install that is
    # not undone would re-arm -- and re-wrap -- on every later run
    shared = {}
    with vram_guard.guarded(shared, 800):
        armed = shared.get("model_function_wrapper")
        check(armed is not None, "armed inside the block")
        check(getattr(armed, "_h3_vram_guard", False), "the wrapper is marked as an H3 guard")
    check("model_function_wrapper" not in shared,
          "disarmed on exit, leaving the patcher exactly as it was")

    model, _ = fake_h3_model()
    sampler_patcher = FakePatcher(model=model)
    sampler_gpu = FakeGPU(free_mb=11000)
    with vram_guard.guarded(shared, 800, model_patcher=sampler_patcher):
        sampler_wrapper = shared["model_function_wrapper"]
        with_gpu(sampler_gpu, lambda: sampler_wrapper(apply_model, forward_args(text_len=5)))
    check(sampler_gpu.releases == 1,
          "a sampler-supplied patcher enables the capacity proof before its first forward")
    check("model_function_wrapper" not in shared,
          "the sampler capacity guard is also disarmed on exit")

    existing = lambda apply_model, args: "other patch"  # noqa: E731
    shared = {"model_function_wrapper": existing}
    with vram_guard.guarded(shared, 800):
        check(shared["model_function_wrapper"] is not existing, "an unrelated wrapper is wrapped")
    check(shared["model_function_wrapper"] is existing,
          "and handed back untouched afterwards")

    with vram_guard.guarded(shared, 800):
        pass
    try:
        with vram_guard.guarded(shared, 800):
            raise ValueError("sampling blew up")
    except ValueError:
        pass
    check(shared["model_function_wrapper"] is existing,
          "disarmed even when the run raises")

    print("no double arming")
    shared = {}
    outer = vram_guard.install_in_model_options(shared, 800)
    inner = vram_guard.install_in_model_options(shared, 800)
    check(inner is None, "a second guard on the same options is skipped, not stacked")
    outer()
    check("model_function_wrapper" not in shared, "the first guard still uninstalls cleanly")

    print("run context")
    run_context.clear()
    live = (2, 24, 12, 48, 84)          # batched cond+uncond
    created = (1, 24, 12, 48, 84)       # what the node made
    run_context.record("Ref to Video (Zi)", "14",
                       [("canvas", "1344x768"),
                        ("ref_image_1", "2048x2048 source -> 1024x1024 encoded")],
                       video_latent_shape=created)
    text = "\n".join(run_context.describe(live))
    check("ref_image_1" in text and "2048x2048" in text,
          "recorded reference resolutions reach the log")
    check("stale" not in text,
          "a batched cond+uncond latent still matches the record it came from")

    run_context.record("Ref to Video (Zi)", "14", [("canvas", "512x512")],
                       video_latent_shape=(1, 24, 4, 32, 32))
    text = "\n".join(run_context.describe(live))
    check(text.count("Ref to Video (Zi)") == 1,
          "re-running a node overwrites its record instead of accumulating")
    check("stale" in text, "a record for a different latent is flagged, not shown as current")

    print("node id is None-tolerant")

    class NoHidden:
        hidden = None

    class WithHidden:
        class hidden:  # noqa: N801  (mirrors ComfyNode.hidden)
            unique_id = "7"

    check(run_context.node_id(NoHidden) is None,
          "an unpopulated hidden holder yields no id instead of raising")
    check(run_context.node_id(object()) is None, "a class without `hidden` yields no id")
    check(run_context.node_id(WithHidden) == "7", "the graph id is read when present")
    check(run_context.image_res(None) == "none" and run_context.audio_desc({}) == "unreadable",
          "malformed inputs describe as unreadable rather than failing the node")

    print("cancel log contents")
    run_context.clear()
    run_context.record("Ref to Video (Zi)", "14",
                       [("length", "124 requested -> 124 frames"),
                        ("ref_video_1", "1920x1080 x240 frames source -> 1344x768 canvas")],
                       video_latent_shape=created)

    class FakeNested:
        def __init__(self, tensors):
            self.tensors = tensors

    x = FakeNested([torch.zeros(live), torch.zeros(2, 32, 2, 206)])
    args = {"input": x, "timestep": torch.tensor([0.7]), "c": {}, "cond_or_uncond": [1, 0]}
    records = []
    real_error = logging.error
    logging.error = lambda msg, *a: records.append(msg % a if a else msg)
    gpu = FakeGPU(free_mb=100, recovers_to_mb=100)
    try:
        with_gpu(gpu, lambda: vram_guard.check_vram(
            800, "DiT forward", device=CUDA, detail_lines=lambda: vram_guard._run_details(args)))
        raise AssertionError("expected the guard to cancel")
    except comfy.model_management.InterruptProcessingException:
        pass
    finally:
        logging.error = real_error

    log = records[0]
    print("    --- cancel log ---")
    for line in log.splitlines():
        print("    " + line)
    check("free physical 100 MB" in log, "cancel log states free physical VRAM")
    check("video latent [2, 24, 12, 48, 84]" in log, "cancel log states the live latent shape")
    check("audio latent [2, 32, 2, 206]" in log, "cancel log states the audio latent shape")
    check("124 requested -> 124 frames" in log, "cancel log states the requested frame length")
    check("1920x1080 x240 frames" in log, "cancel log states input video resolution")
    check("Reduce length/resolution" in log, "cancel log ends with what to do about it")

    print("packed token layout")
    # the seq_len is what actually drives attention memory, so it is resolved from
    # the live forward pass against core's real PackedLayout, not remembered
    args_with_ctx = dict(args, c={"c_crossattn": torch.zeros(2, 300, 64),
                                  "minimax_payload": {}})
    detail = "\n".join(vram_guard._run_details(args_with_ctx))
    check("packed tokens: seq_len=" in detail, "packed sequence length resolved from the live pass")
    check("video=" in detail and "t=12,24x42" in detail,
          "packed line reports the video token block (1x2x2 patching halves h/w)")

    detail_no_ctx = "\n".join(vram_guard._run_details(args))
    check("packed tokens" not in detail_no_ctx,
          "an unresolvable layout is skipped rather than failing the dump")

    print("description failure is non-fatal")
    gpu = FakeGPU(free_mb=100, recovers_to_mb=100)

    def boom():
        raise RuntimeError("describe exploded")

    try:
        with_gpu(gpu, lambda: vram_guard.check_vram(
            800, "test", device=CUDA, detail_lines=boom))
        raise AssertionError("expected the guard to cancel anyway")
    except comfy.model_management.InterruptProcessingException:
        check(True, "a broken description still cancels the run")

    run_context.clear()
    print("\nall VRAM-guard self-tests passed")


if __name__ == "__main__":
    main()
