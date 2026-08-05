"""How many MiniMax-H3 frames fit in a VRAM budget?

Sweeps the model's 17k+5 frame grid and, for each candidate length, measures the
real peak allocation of one DiT block forward at that packed-sequence length
(bf16, random weights, on the actual GPU). Blocks run sequentially and their
activations are freed between layers, so the sampling transient is essentially
one block's peak plus the persistent packed hidden state -- which is what the
per-step OOM is made of. Add resident weight bytes to get the total.

    python user/minimax_vram_probe.py --budget 11
    python user/minimax_vram_probe.py --budget 11 --weights-gb 6.5 --width 1280 --height 720
    python user/minimax_vram_probe.py --budget 11 --ckpt models/diffusion_models/minimax_h3.safetensors

Without --ckpt the architecture dims below are H3 defaults; pass the checkpoint
(only its safetensors header is read, not the tensors) to confirm them.

ref2v
-----
`--mode ref2v` packs the reference rows too, in PackedLayout order:

    text | keyframes | ref images | ref audio | ref video | target audio | target video

A reference video defaults to the same length and canvas as the target, which is
the video-to-video case: it roughly *doubles* the sequence, so the frame ceiling
lands far below the t2va one. `--anchor` adds the carried first-frame keyframe
used by chunked ref2v, and `--static-refs` adds persistent reference images.

    # chunked ref2v: source video + carried anchor, canvas pinned
    python user/minimax_vram_probe.py --budget 11 --mode ref2v \
        --width 1216 --height 672 --anchor --ref-audio --ckpt <dit>

    # plus two identity references at ref_image_size=match
    ... --static-refs 2

    # calibrate the reserve against a known-good ceiling, then re-sweep
    ... --calibrate-to 90

--text-len is the Qwen text presentation only; reference rows are computed, not
folded into it.

Scope: this covers the sampling loop only. VAE encode of the source chunk and VAE
decode of the result are separate peaks and can OOM on a length that sampled
fine. In a chunked run those happen once *per chunk*, so they are a
per-iteration risk, not a tail risk -- measure them separately.
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_COMFYUI_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

if _COMFYUI_ROOT not in sys.path:
    sys.path.insert(0, _COMFYUI_ROOT)

FPS = 24
AUDIO_LATENT_FPS = 40
GB = 1024 ** 3

# Placeholder architecture -- override with --ckpt or the individual flags.
DEFAULT_ARCH = {
    "hidden_size": 5376,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "ffn_hidden_size": 14336,
    "num_layers": 50,
    "time_embed_dim": 8,
    "rope_inv_freq_len": 16,
    # latents_dim * prod(patch_size) = 24 * 1*2*2; the width of a packed video row
    "video_patch_dim": 96,
}

# model.py:544 -- modality tag per segment kind, and which timestep class it rides
SEG_TAG = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

# Under cudaMallocAsync an exhausted pool surfaces as AcceleratorError
# ("CUDA error: out of memory"), not OutOfMemoryError, so catching only the
# latter lets the sweep die instead of stopping cleanly at the ceiling.
OOM_ERRORS = tuple(e for e in (getattr(torch.cuda, "OutOfMemoryError", None),
                               getattr(torch, "AcceleratorError", None),
                               getattr(torch, "OutOfMemoryError", None)) if e is not None)


def is_oom(exc):
    """AcceleratorError covers every CUDA fault; only swallow the memory ones."""
    return "out of memory" in str(exc).lower()


def fit_ms(points):
    """Least squares ms = a*S + b*S^2 (through the origin) over resident points.

    A block forward is linear in S for the projections/FFN and quadratic for
    attention, so ns/token *rises smoothly* with length even when everything is
    resident. Comparing against a flat baseline therefore cries spill on healthy
    growth -- the deviation has to be measured against this curve instead.
    """
    if len(points) < 3:
        return None
    s11 = sum(s * s for s, _ in points)
    s12 = sum(s ** 3 for s, _ in points)
    s22 = sum(s ** 4 for s, _ in points)
    t1 = sum(s * m for s, m in points)
    t2 = sum(s * s * m for s, m in points)
    det = s11 * s22 - s12 * s12
    if det == 0:
        return None
    return ((t1 * s22 - t2 * s12) / det, (s11 * t2 - s12 * t1) / det)


# --- geometry (mirrors comfy_extras/nodes_minimax_h3.py + model_base.MiniMaxH3) ---

def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def frame_rows(width, height):
    """Rows one latent frame contributes: 2x2 DiT patches over the 16x latent grid.

    h/w round up to even first (model_base.py:2109), so this is just area/1024
    for the 32-multiple canvases the nodes actually emit.
    """
    latent_h, latent_w = height // 16, width // 16
    return ((latent_h + 1) // 2 * 2 // 2) * ((latent_w + 1) // 2 * 2 // 2)


def audio_rows(frames):
    """Audio latents pack two rows per step (model.py:_audio_grid)."""
    return round(frames / FPS * AUDIO_LATENT_FPS) * 2


def build_layout(frames, width, height, text_len, ref_frames=0, ref_width=None,
                 ref_height=None, ref_audio=False, anchor=False, static_refs=0,
                 static_ref_pixels=None):
    """Packed-sequence segment table, in PackedLayout order (model.py:297):

        text | keyframes | ref images | ref audio | ref video | tgt audio | tgt video

    Returns the row breakdown plus the mod_segments the DiT block needs, so the
    measurement runs against the same segment structure the real forward does.
    """
    ref_width = ref_width or width
    ref_height = ref_height or height
    tgt_rows = frame_rows(width, height)
    latent_t = video_latent_t(frames)

    kinds = [("text", text_len)]

    if anchor:
        # chunked ref2v's carried first frame: one latent frame on the target grid
        kinds.append(("cond", tgt_rows))

    # ref_image_size=match scales each reference to the generation's pixel area
    px = static_ref_pixels if static_ref_pixels else width * height
    for _ in range(static_refs):
        kinds.append(("ref_img", max(1, px // 1024)))

    ref_latent_t = 0
    if ref_frames:
        ref_latent_t = video_latent_t(ref_frames)
        if ref_audio:
            # a soundtrack packs immediately before its own video block
            kinds.append(("ref_audio", audio_rows(ref_frames)))
        kinds.append(("ref_img", ref_latent_t * frame_rows(ref_width, ref_height)))

    kinds.append(("audio", audio_rows(frames)))
    kinds.append(("video", latent_t * tgt_rows))

    # timestep classes: t_v and t_a always, plus one each for the visual and
    # audio condition augmentation levels when conditions are present (model.py:538)
    has_vis_cond = any(k in ("cond", "ref_img") for k, _ in kinds)
    has_aud_cond = any(k == "ref_audio" for k, _ in kinds)
    t_class = {"text": 0, "video": 0, "audio": 1, "cond": 2, "ref_img": 2, "ref_audio": 3}
    n_t_classes = 2 + int(has_vis_cond) + int(has_aud_cond)

    segments, mod_segments = [], []
    off = 0
    for kind, n in kinds:
        if n <= 0:
            continue
        segments.append((off, off + n, kind))
        mod_segments.append((off, off + n, t_class[kind] * 3 + SEG_TAG[kind]))
        off += n

    cond_video_rows = sum(n for k, n in kinds if k in ("cond", "ref_img"))
    video_rows_total = cond_video_rows + latent_t * tgt_rows
    return {
        "latent_t": latent_t,
        "ref_latent_t": ref_latent_t,
        "video_rows_total": video_rows_total,
        "cond_video_rows": cond_video_rows,
        "seq_len": off,
        "segments": segments,
        "mod_segments": mod_segments,
        "n_t_classes": n_t_classes,
    }


def cond_row_bytes(layout, arch):
    """fp32 buffers the forward builds before the block loop (model.py:469-573).

    all_video_rows is a full-size fp32 scatter target; each condition is
    patchified and, at aug < 1, mixed with a same-shape noise tensor.
    """
    pd = arch["video_patch_dim"]
    scatter = layout["video_rows_total"] * pd * 4
    cond = layout["cond_video_rows"] * pd * 4
    return scatter + 2 * cond


def latent_bytes(width, height, frames, dtype_size=2):
    """Persistent AV latent + noise/denoised copies the sampler keeps live."""
    t = video_latent_t(frames)
    video = 24 * t * (height // 16) * (width // 16)
    audio = 32 * 2 * round(frames / FPS * AUDIO_LATENT_FPS)
    return (video + audio) * dtype_size


# --- checkpoint header ---

def arch_from_ckpt(path):
    """Read safetensors header only (no tensor data) and derive DiT dims.

    Same keys comfy/model_detection.py uses.
    """
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))

    def shape(key):
        for prefix in ("", "model.diffusion_model.", "diffusion_model."):
            if prefix + key in header:
                return header[prefix + key]["shape"]
        raise KeyError(key)

    head_dim = shape("blocks.0.attn.q_norm.weight")[0]
    qkv = shape("blocks.0.attn.qkv_proj.weight")
    layers = 1 + max(
        int(k.split("blocks.")[1].split(".")[0])
        for k in header
        if "blocks." in k and "token_refiner" not in k and k != "__metadata__"
    )
    total = sum(
        math.prod(v["shape"]) * (2 if "16" in v["dtype"] else 4 if "32" in v["dtype"] else 1)
        for k, v in header.items() if k != "__metadata__"
    )
    try:
        t_dim = shape("adaln_t_table")[1]  # adaln over a precomputed curve basis
    except KeyError:
        t_dim = shape("time_embedder.proj_out.weight")[0]
    return {
        "hidden_size": shape("video_patch_proj.weight")[0],
        "attention_head_dim": head_dim,
        "num_attention_heads": qkv[0] // (3 * head_dim),
        "ffn_hidden_size": shape("blocks.0.mlp.fc1.weight")[0] // 2,
        "num_layers": layers,
        "time_embed_dim": t_dim,
        "rope_inv_freq_len": shape("rope.inv_freq")[0],
        # in_features of the video patch projection == latents_dim * patch area
        "video_patch_dim": shape("video_patch_proj.weight")[1],
    }, total


# --- measurement ---

def select_attention(use_sage):
    """Bind comfy's attention backend, then report which one got picked.

    `optimized_attention` is resolved at import time (attention.py:763-783) from
    `comfy.cli_args.args`. That module only reads argv when
    `comfy.options.args_parsing` is set, which only main.py does -- so outside
    the server it always lands on defaults and no amount of sys.argv fiddling
    changes it. Set the flag on the args object directly, before the attention
    module is first imported.
    """
    from comfy.cli_args import args as comfy_args
    if use_sage:
        comfy_args.use_sage_attention = True
    from comfy.ldm.modules.attention import optimized_attention
    name = getattr(optimized_attention, "__name__", "unknown")
    if use_sage and name != "attention_sage":
        name += "  (!! --sage requested but not selected)"
    return name


def build_block(arch, dtype, device):
    """One real comfy.ldm.minimax.model.DiTBlock with random weights.

    Layers run sequentially and free their activations, so the sampling
    transient is one block's peak -- using the actual class keeps the fused
    in-place ops (rms+rope on the qkv buffer, swiglu) that a hand-rolled
    stand-in would miss and over-allocate by a couple of GB.
    """
    import comfy.ops
    from comfy.ldm.minimax.model import DiTBlock
    block = DiTBlock(
        arch["hidden_size"], arch["num_attention_heads"], arch["attention_head_dim"],
        arch["ffn_hidden_size"], arch["time_embed_dim"], 1e-5, 1e-5,
        adaln_dtype=torch.float16, dtype=dtype, device=device,
        operations=comfy.ops.disable_weight_init,
    ).eval()
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.02)
    # the fused in-place rms+rope kernel refuses to run on autograd-tracked tensors
    block.requires_grad_(False)
    return block


def measure_transient(block, layout, arch, dtype, device):
    """Peak bytes and wall-clock for one block forward at this sequence length.

    Returns (bytes, ms). The timing is the spill detector: on a WDDM box the
    driver silently backs an oversubscribed allocation with system RAM instead
    of failing, so the ceiling shows up as a step change in time per token, not
    as an exception. Memory numbers stay correct either way -- the allocation
    succeeded, it just landed somewhere slow.
    """
    from comfy.ldm.minimax.model import rope_rotation_table
    seq_len = layout["seq_len"]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)

    x = torch.randn(seq_len, arch["hidden_size"], dtype=dtype, device=device)
    # [S, 96] angles -> [1, S, 1, 48, 2, 2] rotation table, live for the whole run
    rope = rope_rotation_table(
        torch.randn(seq_len, arch["rope_inv_freq_len"] * 6, dtype=torch.float32, device=device), dtype)
    # adaln is the checkpoint's fp16 island (see build_block's adaln_dtype).
    # One row per timestep class; adaln emits 3 modality rows per class.
    t_emb = torch.randn(layout["n_t_classes"], arch["time_embed_dim"],
                        dtype=torch.float16, device=device)

    # warm up autotune/kernel selection so the timing is steady state; the block
    # writes through x in place, so this costs no extra memory
    with torch.no_grad():
        block(x, t_emb, layout["mod_segments"], rope)
    torch.cuda.synchronize(device)
    # peak resets to *current*, so x/rope/t_emb still count toward the transient
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        y = block(x, t_emb, layout["mod_segments"], rope)
    torch.cuda.synchronize(device)
    ms = (time.perf_counter() - t0) * 1000.0
    peak = torch.cuda.max_memory_allocated(device)
    del x, y, rope, t_emb
    return peak - base, ms


def analytic_transient(seq_len, arch, dtype_size=2):
    """Fallback when CUDA is unavailable: the dominant live buffers."""
    inner = arch["num_attention_heads"] * arch["attention_head_dim"]
    qkv = seq_len * 3 * inner
    attn_out = seq_len * inner
    mlp = seq_len * 2 * arch["ffn_hidden_size"]
    hidden = seq_len * arch["hidden_size"] * 2  # x plus one residual copy
    return (qkv + attn_out + mlp + hidden) * dtype_size


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--budget", type=float, default=11.0, help="VRAM ceiling in GB")
    p.add_argument("--reserve-gb", type=float, default=None,
                   help="VRAM held outside the sampling transient: resident weights plus "
                        "ComfyUI's streaming buffers and allocator slack. Defaults to the whole "
                        "checkpoint when it fits in the budget, else one block's weights "
                        "(the floor when the DiT is streamed layer by layer).")
    p.add_argument("--text-len", type=int, default=256,
                   help="Qwen3-VL text presentation tokens; reference rows are computed, not folded in here")
    p.add_argument("--max-frames", type=int, default=396)
    p.add_argument("--ckpt", default=None, help="safetensors checkpoint; header is read for real dims")
    p.add_argument("--analytic", action="store_true", help="skip GPU measurement")
    p.add_argument("--sage", action="store_true",
                   help="use sage attention, matching --use-sage-attention in the ComfyUI launch "
                        "args. Without it the probe measures pytorch attention and the timing "
                        "column will not match production.")

    g = p.add_argument_group("ref2v")
    g.add_argument("--mode", choices=["t2va", "ref2v"], default="t2va")
    g.add_argument("--ref-frames", default="matched",
                   help="reference video length: 'matched' (same as target, the v2v case) or a frame count")
    g.add_argument("--ref-width", type=int, default=None, help="reference canvas; defaults to the target canvas")
    g.add_argument("--ref-height", type=int, default=None)
    g.add_argument("--ref-audio", action="store_true", help="the reference video carries a soundtrack")
    g.add_argument("--anchor", action="store_true",
                   help="add a carried first-frame keyframe (chunked ref2v's continuity anchor)")
    g.add_argument("--static-refs", type=int, default=0, help="number of persistent reference images")
    g.add_argument("--static-ref-pixels", type=int, default=None,
                   help="pixels per static reference; default is the generation area (ref_image_size=match). "
                        "ref_image_size=max means the source resolution, capped at a 2048 short edge")
    g.add_argument("--calibrate-to", type=int, default=None, metavar="FRAMES",
                   help="solve for the --reserve-gb that makes FRAMES the last fitting length, "
                        "then re-sweep with it. Use a length you have actually run.")
    g.add_argument("--spill-ratio", type=float, default=1.35,
                   help="flag a spill when ns/token exceeds this multiple of its minimum (default 1.35)")
    g.add_argument("--past-spill", action="store_true",
                   help="keep sweeping after the first spill instead of stopping")
    for k in DEFAULT_ARCH:
        p.add_argument("--" + k.replace("_", "-"), type=int, default=None)
    args = p.parse_args()

    arch = dict(DEFAULT_ARCH)
    ckpt_gb = None
    arch_source = "H3 defaults (pass --ckpt to confirm against your weights)"
    if args.ckpt:
        arch, total_bytes = arch_from_ckpt(args.ckpt)
        arch_source = os.path.basename(args.ckpt)
        ckpt_gb = total_bytes / GB
    for k in DEFAULT_ARCH:
        if getattr(args, k) is not None:
            arch[k] = getattr(args, k)

    streamed = False
    reserve_gb = args.reserve_gb
    if reserve_gb is None:
        if ckpt_gb is not None and ckpt_gb < args.budget:
            reserve_gb = ckpt_gb
        else:
            # DiT doesn't fit -> ComfyUI streams it; the floor is roughly one
            # block resident at a time (int8 weights, fp32 scales).
            per_block = (arch["hidden_size"] * arch["num_attention_heads"] * arch["attention_head_dim"] * 4
                         + arch["hidden_size"] * arch["ffn_hidden_size"] * 3)
            reserve_gb = per_block / GB
            streamed = ckpt_gb is not None

    device = torch.device("cuda") if torch.cuda.is_available() else None
    measured = device is not None and not args.analytic
    dtype = torch.bfloat16

    print(f"canvas      {args.width}x{args.height}  ->  latent {args.height // 16}x{args.width // 16}")
    print(f"arch        {arch_source}")
    print("            " + "  ".join(f"{k}={v}" for k, v in arch.items()))
    if device is not None:
        free, total = torch.cuda.mem_get_info(device)
        print(f"gpu         {torch.cuda.get_device_name(device)}  {total / GB:.1f} GB total, {free / GB:.1f} GB free")
    if ckpt_gb is not None:
        print(f"checkpoint  {ckpt_gb:.2f} GB on disk" + (
            f"  -- exceeds the {args.budget:.1f} GB budget, so ComfyUI streams it layer by layer"
            if streamed else "  (fits, held resident)"))
    print(f"budget      {args.budget:.1f} GB  (reserve {reserve_gb:.2f} GB outside the transient)")
    backend = select_attention(args.sage) if measured else "n/a"
    print(f"method      {'measured on GPU' if measured else 'analytic'}   attention: {backend}"
          + ("" if args.sage or not measured else "   (production uses sage -- pass --sage to match)"))

    if args.mode == "ref2v":
        bits = [f"ref video {args.ref_frames}"]
        if args.ref_width or args.ref_height:
            bits.append(f"ref canvas {args.ref_width or args.width}x{args.ref_height or args.height}")
        if args.ref_audio:
            bits.append("ref audio")
        if args.anchor:
            bits.append("anchor keyframe")
        if args.static_refs:
            px = args.static_ref_pixels or args.width * args.height
            bits.append(f"{args.static_refs} static ref(s) @ {px / 1e6:.2f} MP")
        print(f"task        ref2v -- {', '.join(bits)}")
    else:
        print("task        t2va -- target stream only")
    print()

    block = None
    if measured:
        block = build_block(arch, dtype, device)

    def layout_for(frames):
        if args.mode == "t2va":
            return build_layout(frames, args.width, args.height, args.text_len)
        ref = frames if args.ref_frames == "matched" else int(args.ref_frames)
        return build_layout(
            frames, args.width, args.height, args.text_len,
            ref_frames=ref, ref_width=args.ref_width, ref_height=args.ref_height,
            ref_audio=args.ref_audio, anchor=args.anchor,
            static_refs=args.static_refs, static_ref_pixels=args.static_ref_pixels)

    def cost(frames):
        """(transient bytes, latent bytes, cond bytes, ms, layout) for one length."""
        lay = layout_for(frames)
        if measured:
            trans, ms = measure_transient(block, lay, arch, dtype, device)
        else:
            trans, ms = analytic_transient(lay["seq_len"], arch), float("nan")
        lat = latent_bytes(args.width, args.height, frames) * 3  # latent + noise + denoised
        return trans, lat, cond_row_bytes(lay, arch), ms, lay

    if args.calibrate_to is not None:
        n = align_frame_count(max(5, args.calibrate_to))
        try:
            trans, lat, cond, _, _ = cost(n)
        except OOM_ERRORS as exc:
            if not is_oom(exc):
                raise
            torch.cuda.empty_cache()
            print(f"cannot calibrate: the probe itself OOMs at {n} frames. "
                  f"Free the card (a running ComfyUI holds its staged model) and retry.")
            return
        solved = args.budget - (trans + lat + cond) / GB
        print(f"calibration {n} frames is known to fit {args.budget:.1f} GB, so everything outside "
              f"the transient is {solved:.2f} GB")
        if solved < 0:
            print("            negative -- the measured transient alone exceeds the budget. Either the "
                  "budget is wrong or that length does not actually fit.")
            return
        print(f"            reserve {reserve_gb:.2f} GB -> {solved:.2f} GB\n")
        reserve_gb = solved
        streamed = False  # the reserve is now empirical, not a floor

    lengths = []
    n = 5
    while n <= args.max_frames:
        lengths.append(n)
        n = align_frame_count(n + 1)

    print(f"{'frames':>7} {'sec':>6} {'lat_t':>6} {'tokens':>9} "
          f"{'transient':>11} {'+cond':>8} {'total':>9} {'fwd ms':>9} {'ns/tok':>8}")
    best = None
    spill_at = None
    resident = []  # (seq_len, ms) for lengths confirmed to be on the card
    for frames in lengths:
        try:
            trans, lat, cond, ms, lay = cost(frames)
        except OOM_ERRORS as exc:
            if not is_oom(exc):
                raise
            torch.cuda.empty_cache()
            lay = layout_for(frames)
            print(f"{frames:>7} {frames / FPS:>6.2f} {lay['latent_t']:>6} "
                  f"{lay['seq_len']:>9}   hard OOM -- past this GPU's reach")
            break
        total = (trans + lat + cond) / GB + reserve_gb
        fits = total <= args.budget
        if fits:
            best = frames

        # Attention is quadratic in S, so ns/token rises smoothly even while
        # everything is resident. Spill is a step change *off that curve*.
        s = lay["seq_len"]
        rate = ms * 1e6 / s if measured else float("nan")
        flag = "ok" if fits else "OVER"
        dev = ""
        if measured:
            coef = fit_ms(resident)
            pred = coef[0] * s + coef[1] * s * s if coef else None
            if pred and ms > pred * args.spill_ratio:
                spill_at = spill_at or frames
                flag = "SPILL"
                dev = f" {ms / pred:>5.2f}x vs fit"
            else:
                resident.append((s, ms))
                if pred:
                    dev = f" {ms / pred:>5.2f}x vs fit"

        print(f"{frames:>7} {frames / FPS:>6.2f} {lay['latent_t']:>6} "
              f"{s:>9} {trans / GB:>10.3f}G {cond / GB:>7.3f}G "
              f"{total:>8.3f}G {ms:>9.1f} {rate:>8.0f}  {flag}{dev}")

        if spill_at is not None and not args.past_spill:
            print("\nstopping: time per token has stepped up, so this length is no longer "
                  "resident. Pass --past-spill to keep sweeping.")
            break

    print()
    if measured and resident:
        coef = fit_ms(resident)
        if coef:
            a, b = coef
            print(f"resident fit  ms = {a * 1e3:.4f}e-3*S + {b * 1e9:.4f}e-9*S^2   "
                  f"(linear = projections+FFN, quadratic = attention)")
            s90 = 45990
            print(f"              at S={s90}: {a * s90:.0f} ms linear + {b * s90 * s90:.0f} ms "
                  f"attention -> attention is {b * s90 * s90 / (a * s90 + b * s90 * s90) * 100:.0f}% "
                  f"of the block")
    if measured and spill_at is not None:
        prev = [f for f in lengths if f < spill_at]
        print(f"\nSPILL at {spill_at} frames ({spill_at / FPS:.2f}s) -- last fully resident length "
              f"is {prev[-1] if prev else 'none'} frames.")
        print("On this box the driver backs oversubscription with system RAM rather than failing, "
              "so the ceiling is a performance cliff, not an exception. That is the real limit.")
        print(f"Multiply the fwd ms column by {arch['num_layers']} blocks for a whole DiT step.")
    elif measured:
        print(f"\nno spill up to {lengths[-1] if lengths else 0} frames; every point stayed within "
              f"{args.spill_ratio:.2f}x of the resident fit. Raise --max-frames to find the cliff.")

    if best is None:
        print(f"nothing on the grid fits the {args.budget:.1f} GB budget at {args.width}x{args.height}.")
    else:
        print(f"max {args.mode} length under the {args.budget:.1f} GB budget at "
              f"{args.width}x{args.height}: {best} frames ({best / FPS:.2f}s)")
        if args.mode == "ref2v" and args.ref_frames == "matched":
            print(f"that is {best} frames of source and {best} of output -- "
                  f"{2 * best / FPS:.2f}s of packed video.")
        if streamed and args.calibrate_to is None:
            print("weights are streamed, so the reserve above is a floor and this budget line is an "
                  "upper bound. The fwd-ms column is the trustworthy signal; --calibrate-to pins "
                  "the reserve to a length you have actually run.")


if __name__ == "__main__":
    main()
