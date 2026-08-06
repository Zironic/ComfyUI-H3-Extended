"""Measure FirstBlockCache on a real H3 text-to-video sampler run.

Unlike ``bench_first_block_cache.py``, which only replays a residual sequence
through the decision logic, this runs the actual model and reports wall time,
memory and the per-step decision trace for one variant per arm.

The ``threshold=0`` arm is the control.  The coordinator skips on
``diff <= threshold`` and real diffs are strictly positive, so that arm pays the
entire cost of the cache - the block-0 input clone, the residual reduction, both
persistent buffers and the disabled dynamic prefetch - while never skipping a
step.  Comparing it against ``off`` separates wrapper overhead from algorithmic
saving.

Run from the ComfyUI root, for example:

    python custom_nodes/ComfyUI-H3-Extended/benchmarks/bench_first_block_cache_live.py \
        --frames 90 --steps 10 --prompt-file prompt.txt

Latents are saved rather than decoded.  A VAE decode competes for the same VRAM
the measurement is trying to characterize, and quality comparisons can run later
from the saved tensors without repeating the sampling.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_PACK))

DESKTOP_MODEL_PATHS = os.path.join(
    os.environ.get("APPDATA", ""), "Comfy Desktop", "shared_model_paths.yaml"
)
DEFAULT_UNET = r"hf_minimax_h3\minimax_h3_fl2va_pruned_int8_convrot.safetensors"
DEFAULT_CLIP = r"hf_minimax_h3\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"


def bootstrap():
    """Mirror the long-form CLI: path setup before any comfy import parses argv."""
    for path in (_PACK, _ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    sys.argv = [sys.argv[0]]
    import comfy.options

    comfy.options.enable_args_parsing()
    if os.name == "nt":
        os.environ["MIMALLOC_PURGE_DELAY"] = "0"
    import cuda_malloc  # noqa: F401  sets the async allocator before torch


def load_model_paths(explicit=None):
    import utils.extra_config

    for candidate in (explicit, DESKTOP_MODEL_PATHS):
        if candidate and os.path.isfile(candidate):
            utils.extra_config.load_extra_path_config(candidate)
            logging.info("loaded model paths from %s", candidate)
            return candidate
    logging.warning("no extra model paths config found")
    return None


def parse_arms(raw):
    """``off`` plus the thresholds, in the order they will run."""
    arms = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "off":
            arms.append(("off", None))
        else:
            arms.append(("first_block", float(token)))
    return arms


def arm_label(mode, threshold):
    return "off" if mode == "off" else "t%s" % ("%.4f" % threshold).rstrip("0").rstrip(".")


def free_vram(label):
    import torch
    import comfy.model_management

    gc.collect()
    comfy.model_management.soft_empty_cache(True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        logging.info(
            "%s: %.0f MiB physical free of %.0f MiB",
            label,
            free / (1024 ** 2),
            total / (1024 ** 2),
        )


def run_arm(
    *,
    mode,
    threshold,
    base_model,
    conditioning,
    latent,
    sampler,
    sigmas,
    seed,
    warmup_steps,
    report_directory,
):
    import torch
    import comfy.model_management

    from h3_memory_optimizer.attention import resolve_attention
    from h3_memory_optimizer.config import MemoryOptimizerConfig
    from h3_memory_optimizer.patch import apply as apply_optimizer
    from chunked_ref2v.harness import sample

    config = MemoryOptimizerConfig(
        attention="auto",
        activation="mlp_chunked_bf16",
        adaln_precompute="off",
        block_cache=mode,
        block_cache_threshold=0.0 if threshold is None else float(threshold),
        block_cache_warmup_steps=int(warmup_steps),
        block_cache_report_directory=report_directory if mode != "off" else "",
    )
    decision = resolve_attention(
        config.attention,
        config.attention_fallback,
        adapter_options=config.attention_options(),
    )
    patched = base_model.clone()
    apply_optimizer(patched, config=config, decision=decision)

    coordinator = patched.model_options["transformer_options"].get(
        "minimax_h3_first_block_cache"
    )

    free_vram("before %s" % arm_label(mode, threshold))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    samples = sample(
        model=patched,
        conditioning=conditioning,
        latent=latent,
        sampler=sampler,
        sigmas=sigmas,
        seed=seed,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    record = {
        "arm": arm_label(mode, threshold),
        "mode": mode,
        "threshold": threshold,
        "warmup_steps": int(warmup_steps),
        "sampler_seconds": elapsed,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available()
            else None
        ),
    }
    if coordinator is not None:
        report = coordinator.report(elapsed)
        record["skip_fraction"] = report["skip_fraction"]
        record["computed_tails"] = report["computed_tails"]
        record["skipped_tails"] = report["skipped_tails"]
        record["cache_bytes"] = report["cache_bytes"]
        record["minimum_physical_free_mib"] = report["minimum_physical_free_mib"]
        record["sequence_length"] = report["sequence_length"]
        record["steps"] = report["steps"]

    del patched, coordinator
    comfy.model_management.unload_all_models()
    free_vram("after %s" % arm_label(mode, threshold))
    return record, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sampler", default="res_multistep")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--arms",
        default="off,0,0.02,0.04,0.06,0.08",
        help="comma-separated: 'off' plus thresholds, run in this order",
    )
    parser.add_argument("--unet", default=DEFAULT_UNET)
    parser.add_argument("--clip", default=DEFAULT_CLIP)
    parser.add_argument("--out", default="")
    parser.add_argument("--model-paths", default=None)
    parser.add_argument("--save-latents", action="store_true")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )

    # Order is load-bearing: argparse first because bootstrap clears sys.argv,
    # and bootstrap before torch because the async allocator is only read when
    # torch first initializes CUDA.
    bootstrap()
    load_model_paths(args.model_paths)

    import torch

    with torch.inference_mode():
        run(args, out_root=args.out)


def run(args, out_root=""):
    import torch
    import comfy.samplers
    import comfy.sd
    import folder_paths

    from chunked_ref2v.geometry import FPS
    from chunked_ref2v.harness import empty_av_latent_frames

    out = out_root or os.path.join(
        _HERE, "fbc_live_%s" % time.strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out, exist_ok=True)

    with open(args.prompt_file, encoding="utf-8") as handle:
        prompt = handle.read()

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", args.unet)
    base_model = comfy.sd.load_diffusion_model(unet_path)

    clip_path = folder_paths.get_full_path_or_raise("text_encoders", args.clip)
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=getattr(comfy.sd.CLIPType, "MINIMAX"),
        model_options={},
    )
    # Text-only: no reference items, so the packed layout carries text, audio and
    # video segments and nothing else.
    tokens = clip.tokenize(prompt, minimax_ref_items={})
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    del clip
    free_vram("after text encode")

    sampler = comfy.samplers.sampler_object(args.sampler)
    sigmas = comfy.samplers.calculate_sigmas(
        base_model.get_model_object("model_sampling"), args.scheduler, args.steps
    )
    latent = empty_av_latent_frames((args.width, args.height), args.frames, FPS)

    logging.info(
        "H3 FirstBlockCache live benchmark: %d frames, %d steps, %dx%d, sampler=%s",
        args.frames,
        args.steps,
        args.width,
        args.height,
        args.sampler,
    )

    records = []
    for mode, threshold in parse_arms(args.arms):
        label = arm_label(mode, threshold)
        logging.info("=== arm %s ===", label)
        record, samples = run_arm(
            mode=mode,
            threshold=threshold,
            base_model=base_model,
            conditioning=conditioning,
            latent=latent,
            sampler=sampler,
            sigmas=sigmas,
            seed=args.seed,
            warmup_steps=args.warmup_steps,
            report_directory=out,
        )
        if args.save_latents:
            torch.save(
                [tensor.cpu() for tensor in samples],
                os.path.join(out, "latent_%s.pt" % label),
            )
        del samples
        records.append(record)
        logging.info(
            "%s: %.1f s, peak %.2f GiB, skip_fraction %s",
            label,
            record["sampler_seconds"],
            record["peak_allocated_gib"] or 0.0,
            record.get("skip_fraction"),
        )
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump({"arms": records}, handle, indent=2)

    baseline = next((r for r in records if r["mode"] == "off"), None)
    print("\n%-8s %10s %9s %9s %12s" % ("arm", "seconds", "speedup", "peak_gib", "skip_frac"))
    for record in records:
        speedup = (
            baseline["sampler_seconds"] / record["sampler_seconds"]
            if baseline and record["sampler_seconds"]
            else float("nan")
        )
        print(
            "%-8s %10.1f %9.3f %9.2f %12s"
            % (
                record["arm"],
                record["sampler_seconds"],
                speedup,
                record["peak_allocated_gib"] or 0.0,
                record.get("skip_fraction", "-"),
            )
        )
    print("\nreports written to %s" % out)


if __name__ == "__main__":
    main()
