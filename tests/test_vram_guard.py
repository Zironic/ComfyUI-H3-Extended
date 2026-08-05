"""Self-test for the H3 VRAM guard.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_vram_guard.py

The driver query and the cache release are the two things that cannot be
exercised honestly without pushing a real GPU to the edge, so both are stubbed
and the decision logic around them is what gets tested: threshold comparison,
the release-and-recheck second chance, that a breach raises the same exception
the Cancel button raises, and that the wrapper chains rather than replaces.
"""

import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import torch  # noqa: E402

import comfy.model_management  # noqa: E402
import run_context  # noqa: E402
import vram_guard  # noqa: E402

MB = vram_guard.MB
CUDA = torch.device("cuda:0")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok: %s" % msg)


class FakeGPU:
    """Stands in for the driver: `free` is what mem_get_info reports next."""

    def __init__(self, free_mb, recovers_to_mb=None):
        self.free = free_mb * MB
        self.recovers_to = None if recovers_to_mb is None else recovers_to_mb * MB
        self.releases = 0

    def mem_get_info(self, device=None):
        return self.free, 12282 * MB

    def release(self, force=False):
        self.releases += 1
        if self.recovers_to is not None:
            self.free = self.recovers_to


def with_gpu(gpu, fn):
    real_info = torch.cuda.mem_get_info
    real_release = comfy.model_management.soft_empty_cache
    real_reserved = torch.cuda.memory_reserved
    real_alloc = torch.cuda.memory_allocated
    torch.cuda.mem_get_info = gpu.mem_get_info
    comfy.model_management.soft_empty_cache = gpu.release
    torch.cuda.memory_reserved = lambda device=None: 9000 * MB
    torch.cuda.memory_allocated = lambda device=None: 8000 * MB
    try:
        return fn()
    finally:
        torch.cuda.mem_get_info = real_info
        comfy.model_management.soft_empty_cache = real_release
        torch.cuda.memory_reserved = real_reserved
        torch.cuda.memory_allocated = real_alloc


class FakePatcher:
    def __init__(self, existing=None):
        self.model_options = {}
        if existing is not None:
            self.model_options["model_function_wrapper"] = existing

    def set_model_unet_function_wrapper(self, fn):
        self.model_options["model_function_wrapper"] = fn


def main():
    logging.basicConfig(level=logging.INFO, format="    %(levelname)s %(message)s")

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
    patcher = FakePatcher()
    vram_guard.install_unet_guard(patcher, 800)
    wrapper = patcher.model_options["model_function_wrapper"]
    check(callable(wrapper), "wrapper installed on the patcher")

    def apply_model(x, timestep, **c):
        calls.append((x, timestep, c))
        return "denoised"

    args = {"input": torch.zeros(1), "timestep": torch.tensor([0.7]), "c": {"y": 1},
            "cond_or_uncond": [0]}
    gpu = FakeGPU(free_mb=3000)
    out = with_gpu(gpu, lambda: wrapper(apply_model, args))
    check(out == "denoised", "wrapper calls through to apply_model with headroom")
    check(calls[-1][2] == {"y": 1}, "conditioning kwargs are forwarded unchanged")

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

    patcher = FakePatcher(existing=previous)
    vram_guard.install_unet_guard(patcher, 800)
    gpu = FakeGPU(free_mb=3000)
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
