"""One real MiniMax-H3 DiT block across a selectable QKV x MLP arm matrix.

Dispatched by ``minimax_vram_probe.py --ab-activation-memory``. One block and
one shared checkpoint weight set remain resident while the selected arms compare
production ``sage128``/``sage128_fused_qkv`` with untiled/2-tile/4-tile MLPs.

Use ``--ab-frames grid`` for the complete 17k+5 ladder. The probe loads only one
block's QKV/norm and MLP tensors from ``--ckpt``; the remaining block parameters
stay synthetic and shared by every arm.
"""

import os

import torch

import _minimax_vram_probe_base as base
from _minimax_vram_probe_ab_cli import build_parser, selected_arms
from _minimax_vram_probe_ab_runtime import build_forwards, load_block_weights
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
    if args.activation_chunk_rows <= 0:
        parser.error("--activation-chunk-rows must be > 0")
    if not args.ckpt:
        parser.error("--ckpt is required for the real-weight arm matrix")
    try:
        arms = selected_arms(args)
    except ValueError as exc:
        parser.error(str(exc))

    arch = dict(base.DEFAULT_ARCH)
    ckpt_gb = None
    arch_source = "H3 defaults (pass --ckpt to confirm against your weights)"
    if args.ckpt:
        from _minimax_vram_probe_ab_sweep import resolve_checkpoint
        args.ckpt = resolve_checkpoint(args.ckpt)
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

    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16

    print(f"canvas      {args.width}x{args.height}  ->  latent {args.height // 16}x{args.width // 16}")
    print(f"arch        {arch_source}")
    print("            " + "  ".join(f"{key}={value}" for key, value in arch.items()))
    free, total = torch.cuda.mem_get_info(device)
    print(f"gpu         {torch.cuda.get_device_name(device)}  {total / base.GB:.1f} GB total, {free / base.GB:.1f} GB free")
    if ckpt_gb is not None:
        print(
            f"checkpoint  {ckpt_gb:.2f} GB on disk"
            + (f"  -- exceeds the {args.budget:.1f} GB budget; production streams it"
               if streamed else "  (fits, held resident)")
        )
    print(f"budget      {args.budget:.1f} GB  (reserve {reserve_gb:.2f} GB outside transient)")
    print("method      measured selected production hybrid-Sage QKV x MLP arms on GPU")
    print("weights     one shared real checkpoint block; selected tensors only")
    print(
        f"physical    LOW below {args.physical_warning_mb} MiB; "
        f"cudaMemGetInfo sampled every {args.physical_poll_ms:g} ms during untimed probes"
    )

    if args.mode == "ref2v":
        bits = [f"ref video {args.ref_frames}"]
        if args.ref_width or args.ref_height:
            bits.append(f"ref canvas {args.ref_width or args.width}x{args.ref_height or args.height}")
        if args.ref_audio:
            bits.append("ref audio")
        if args.anchor:
            bits.append("anchor keyframe")
        if args.static_refs:
            pixels = args.static_ref_pixels or args.width * args.height
            bits.append(f"{args.static_refs} static ref(s) @ {pixels / 1e6:.2f} MP")
        print(f"task        ref2v -- {', '.join(bits)}")
    else:
        print("task        t2va -- target stream only")

    block = base.build_block(arch, dtype, device)
    load_block_weights(block, args.ckpt)
    forwards, backends, configs = build_forwards(block, args)
    print("arms        " + ", ".join(arms))
    for label in arms:
        config = configs[label]
        backend = backends[label.split("/", 1)[0]]
        mlp = config["mode"] if not config["tile_count"] else "%d-tile ConvRot rows=%d" % (
            config["tile_count"], config["chunk_rows"]
        )
        print("            %s: backend=%s fused_qkv=%s mlp=%s" %
              (label, backend.config.mode, backend.projector is not None, mlp))
    print()

    run_sweep(
        args,
        parser,
        arch,
        dtype,
        device,
        reserve_gb,
        streamed,
        forwards,
    )


if __name__ == "__main__":
    main()
