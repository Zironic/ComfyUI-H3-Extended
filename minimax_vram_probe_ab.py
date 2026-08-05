"""A/B one real MiniMax-H3 DiT block at production packed-sequence shapes.

Dispatched by ``minimax_vram_probe.py --ab-activation-memory``. One block and one
random BF16 weight set remain resident while the probe compares:

    baseline:  efficient Sage + stock H3 DiTBlock forward
    candidate: efficient Sage + token-chunked MLP block forward

Typical target run:

    python custom_nodes/ComfyUI-H3-Extended/minimax_vram_probe.py \
        --ab-activation-memory --ab-frames 73,90 --budget 11 \
        --mode ref2v --width 1216 --height 672 --text-len 1500 \
        --anchor --ref-audio --calibrate-to 90 \
        --activation-chunk-rows 4096 --ckpt <dit>

Use ``--ab-frames grid`` for the complete 17k+5 ladder. Checkpoint headers supply
architecture and disk size only; the probe intentionally keeps random BF16 block
weights to isolate activation geometry, as the legacy probe does.
"""

import os

import torch

import _minimax_vram_probe_base as base
from _minimax_vram_probe_ab_cli import build_parser
from _minimax_vram_probe_ab_runtime import build_forwards
from _minimax_vram_probe_ab_sweep import run as run_sweep


def main():
    parser = build_parser(__doc__)
    args = parser.parse_args()
    if args.analytic:
        parser.error("--ab-activation-memory requires CUDA; --analytic is unsupported")
    if not torch.cuda.is_available():
        parser.error("--ab-activation-memory requires CUDA")
    if args.ab_warmup < 0 or args.ab_iterations < 1:
        parser.error("--ab-warmup must be >= 0 and --ab-iterations must be >= 1")
    if args.physical_warning_mb < 0:
        parser.error("--physical-warning-mb must be >= 0")
    if args.physical_poll_ms <= 0:
        parser.error("--physical-poll-ms must be > 0")

    arch = dict(base.DEFAULT_ARCH)
    ckpt_gb = None
    arch_source = "H3 defaults (pass --ckpt to confirm against your weights)"
    if args.ckpt:
        arch, total_bytes = base.arch_from_ckpt(args.ckpt)
        arch_source = os.path.basename(args.ckpt)
        ckpt_gb = total_bytes / base.GB
    for key in base.DEFAULT_ARCH:
        value = getattr(args, key)
        if value is not None:
            arch[key] = value

    streamed = False
    reserve_gb = args.reserve_gb
    if reserve_gb is None:
        if ckpt_gb is not None and ckpt_gb < args.budget:
            reserve_gb = ckpt_gb
        else:
            per_block = (
                arch["hidden_size"] * arch["num_attention_heads"]
                * arch["attention_head_dim"] * 4
                + arch["hidden_size"] * arch["ffn_hidden_size"] * 3
            )
            reserve_gb = per_block / base.GB
            streamed = ckpt_gb is not None

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Must run before importing the efficient-Sage implementation.
    selected = base.select_attention(True)

    print(
        f"canvas      {args.width}x{args.height}  ->  "
        f"latent {args.height // 16}x{args.width // 16}"
    )
    print(f"arch        {arch_source}")
    print(
        "            "
        + "  ".join(f"{key}={value}" for key, value in arch.items())
    )
    free, total = torch.cuda.mem_get_info(device)
    print(
        f"gpu         {torch.cuda.get_device_name(device)}  "
        f"{total / base.GB:.1f} GB total, {free / base.GB:.1f} GB free"
    )
    if ckpt_gb is not None:
        print(
            f"checkpoint  {ckpt_gb:.2f} GB on disk"
            + (
                f"  -- exceeds the {args.budget:.1f} GB budget; production streams it"
                if streamed
                else "  (fits, held resident)"
            )
        )
    print(
        f"budget      {args.budget:.1f} GB  "
        f"(reserve {reserve_gb:.2f} GB outside transient)"
    )
    print(f"method      measured A/B on GPU; Comfy selected {selected}")
    print(
        "physical    cudaMemGetInfo sampled every %.2f ms during an untimed "
        "residency pass; LOW below %d MiB"
        % (args.physical_poll_ms, args.physical_warning_mb)
    )
    print(
        "weights     random BF16 block weights; activation geometry, "
        "not checkpoint INT8 streaming"
    )

    if args.mode == "ref2v":
        bits = [f"ref video {args.ref_frames}"]
        if args.ref_width or args.ref_height:
            bits.append(
                f"ref canvas {args.ref_width or args.width}x"
                f"{args.ref_height or args.height}"
            )
        if args.ref_audio:
            bits.append("ref audio")
        if args.anchor:
            bits.append("anchor keyframe")
        if args.static_refs:
            pixels = args.static_ref_pixels or args.width * args.height
            bits.append(
                f"{args.static_refs} static ref(s) @ {pixels / 1e6:.2f} MP"
            )
        print(f"task        ref2v -- {', '.join(bits)}")
    else:
        print("task        t2va -- target stream only")

    block = base.build_block(arch, dtype, device)
    baseline, candidate, backend, config = build_forwards(block, args)
    print(
        "A/B         baseline=sage_mem_eff; candidate=+%s rows=%d "
        "alignment=%d held=%s"
        % (
            config.mode,
            config.chunk_rows,
            config.alignment,
            config.prefer_held_weights,
        )
    )
    print(
        "Sage        version=%s kernel=%s accumulation=%s"
        % (
            backend.api.version,
            backend.api.kernel_name,
            backend.api.accumulation,
        )
    )
    if config.native_swiglu:
        print(
            "note        BF16 probe weights cannot exercise native "
            "TensorWise-INT8 FC2"
        )
    print()

    run_sweep(
        args,
        parser,
        arch,
        dtype,
        device,
        reserve_gb,
        streamed,
        baseline,
        candidate,
    )


if __name__ == "__main__":
    main()
