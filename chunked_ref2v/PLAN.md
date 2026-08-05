# MiniMax H3 Arbitrary-Length Ref2V — plan

Status: **design, pre-implementation.** Nothing here has been run on the GPU.
Every claim marked *verified* was read out of the installed source at
ComfyUI `v0.30.0-4-g9a9fdb10`; everything else is marked *estimated* or *unknown*.

## 0. Scope

Convert an arbitrarily long source video with MiniMax H3 Ref2VA while keeping
peak GPU memory bounded by a configurable chunk size. One high-level node, not a
graph of conditioning / guider / sampler / stitching nodes.

The loop, per chunk: select an overlapping source interval → build fresh Ref2VA
conditioning → anchor to a generated frame from the previous interval → sample
and decode → resolve the duplicated overlap → write finalized frames.

### The premise

A monolithic 10+ second Ref2VA run does not fit in 12 GB. Piece by piece does.
The overlap needed to prevent stitching artifacts costs extra compute, and in
exchange peak VRAM stops being a function of duration.

That trade is worth stating because runtime is **linear-dominated** in sequence
length (§1.3): attention is only 39% of the block at C=90, so chunking does not
pay for itself the way it would under pure quadratic scaling. The overlap is a
real tax of `C/S` — 1.43× at the target profile. Bounded VRAM is what it buys,
and past C≈90 the quadratic term means longer chunks stop helping at all (§4.4).

## 1. Correction log

Three corrections to the original draft, all found by reading the installed
source. Two are blocking.

### 1.1 (blocking) The anchor keyframe lands at the wrong temporal address

`PackedLayout` pins a first-frame keyframe to `cond_t = float(text_len)`
([`comfy/ldm/minimax/model.py:319`]), but when references are present it *resets*
`cursor = float(text_len)` and walks it past every reference block
([`model.py:335-377`]), and the target video grid is built from that walked cursor
([`model.py:388`]).

With keyframes alone the two coincide — that is the fl2va design. With references
also present they do not: the keyframe ends up sharing a temporal origin with the
*first reference block* rather than with target frame 0. The last-frame branch
([`model.py:321`]) has the same defect.

The original draft's "only required code change — concatenate the latent lists"
was therefore incomplete, and a Stage-0 failure would not have distinguished
"the checkpoint ignores an unseen keyframe" from "we put the keyframe in the
wrong place."

### 1.2 (blocking) The prototype patch mutates shared state

`ModelPatcher.clone()` copies the patch dict but **shares `self.model`**
([`comfy/model_patcher.py:451`]), so assigning `model.model.extra_conds = ...`
escapes the clone. Use `add_object_patch`, which `set_attr`s on patch and
restores on unpatch ([`model_patcher.py:1107-1110`, `1157-1161`]) — the same
mechanism `MiniMaxH3SigmaShift` already uses for `model_sampling`.

### 1.3 Cost is linear-dominated but not linear — and that is what makes C=90 optimal

**Superseded by measurement — see §4.4.** With sage attention enabled, the fitted
block cost is `ms = 14.83e-3·S + 205.2e-9·S²`: at C=90 that is 682 ms linear and
434 ms attention, so attention is **39% of the block** and time grows as roughly
`S^1.3` across the operating range. The linear term dominates, which is why it
reads as linear in practice, but the quadratic term is real.

The consequence is not that longer chunks are bad — it is that cost per output
frame has a **minimum** rather than falling forever, and the minimum sits exactly
at the target profile. The argument below is kept because its reasoning about
why `C/S` approximates the truth still holds; only the "no superlinear component"
claim is wrong.

### 1.3a Original argument (partially superseded)

An intermediate revision of this document argued that cost is half-quadratic in
packed sequence length, and that `C/S` therefore understated the price of longer
chunks. **That is wrong in practice**, and the draft's original `C/S` metric
survives.

Measured behaviour on this hardware, from extensive use of the model: runtime
scales almost exactly linearly with input + output tokens combined, with no
superlinear component until the working set stops fitting in VRAM.

The FLOP argument was not wrong about FLOPs — attention is ~45-55% of arithmetic
at these lengths — but the run is not FLOP-bound. Weight streaming and
bandwidth-bound attention kernels dominate, and both scale linearly in S. The
§4.3 probe confirms this **for memory**: the transient is flat to 0.3% per rung
across a 14× span, with no quadratic term at all. Time is a different story —
§4.4 measures a real quadratic component in the block forward, which is where
this argument breaks down. And
because `C/T` is nearly constant on the H3 grid (39/12 = 3.25, 56/17 = 3.29),
S is near-proportional to C, so `C/S` is a good approximation of cost per output
frame.

Consequence: **longer chunks are cheaper per output frame.** The binding
constraint on chunk size is VRAM headroom, not compute — and it is a cliff, not
a curve. See §4.

## 2. Verified constraints

| Claim | Where | Status |
| --- | --- | --- |
| Frame counts satisfy `n % 17 == 5`; `T = 2 if f<=5 else ((f-5)//17)*5+2` | [`comfy_extras/nodes_minimax_h3.py:33-46`] | verified |
| Ref video VAE-encoded at full 24 fps, Qwen sees a 2 fps sampling | [`nodes_minimax_h3.py:254`, `261-264`] | verified |
| Ref video truncated to `frame_count`, then snapped **down** to `n % 17 == 5` | [`nodes_minimax_h3.py:246-253`] | verified |
| `cond_video_latents` assigned from keyframes then **overwritten** by refs | [`comfy/model_base.py:2094`, `2098`] | verified |
| Conditioning rows consumed keyframes-first, then refs | [`model.py:329-390`], [`model.py:572`] | verified |
| Tokenizer takes `minimax_ref_items` **or** `images`, never both | [`comfy/text_encoders/minimax.py:153-178`] | verified |
| VAE: `clip_length=17`, `token_drop=3`, temporal ratio 4 | [`comfy/ldm/minimax/vae.py:335-350`] | verified |
| `encode_temporal` encodes **independent** 17-frame clips, concatenates, drops last 3 tokens | [`vae.py:522-539`] | verified |
| `payload["seed"]` also seeds conditioning noise aug (`VISUAL_COND_TIMESTEP = 0.999`) | [`model.py:31`, `473-481`] | verified |
| `_frame_grid(48, 84)` → 1008 rows per latent frame at the 768×1344 canvas | [`model.py:87-91`] | verified |
| DiT transient **memory** is exactly linear in S — 0.894 GB per rung, flat to 0.3% over a 14× span | `minimax_vram_probe.py`, §4.3 | **measured** |
| DiT **time** is linear-dominated but superlinear: `ms = 14.83e-3·S + 205.2e-9·S²` with sage, ~`S^1.3` over the operating range | probe fit, §4.4 | **measured** |
| Sage cuts the attention constant 2.83× vs pytorch attention (205.2e-9 vs 581.5e-9) | probe fit, §4.4 | **measured** |
| Spill is a performance cliff, not an exception — the driver backs oversubscription with system RAM | this hardware; probe run 1 at 209 frames | **measured** |
| ref2v packs 1.97× the t2va sequence at the same frame count | probe row breakdown, §4.3 | **measured** |
| Ref2VA trained range is ~124-362 frames | [`nodes_minimax_h3.py:177`] | verified |
| T=6/18 frames is the coherence floor, T=7/22 preferable | prior experimentation | **measured** |
| This machine runs ~4 s input + ~4 s output at 0.8 MP → C ceiling between 90 and 107 | this hardware, prior use | **measured** |
| Cost per output frame bottoms out at C=90; C=73 costs +2.7% and buys 0.90 GB margin | probe fit, §4.4 / §4.6 | **measured** |
| Whether C=73 holds up with real prompts, references and desktop load | — | **unknown — Stage 0 §4.6** |
| Whether quality improves with C independently of overlap (training-distribution effect) | — | **unknown — Stage 3** |
| Ref2VA checkpoint obeys a keyframe latent Qwen never saw | — | **unknown — Stage 0** |

### 2.1 Two different 17s

The draft conflated two grids that are offset by 5:

```
generation legality:        C % 17 == 5
VAE clip boundaries:        start % 17 == 0
```

They are independent. There is no requirement that the stride satisfy
`S % 17 == 5`.

### 2.2 Source-latent slicing is narrower than "aligned or not"

`encode_temporal` repeat-pads the tail to a clip multiple, so a 39-frame chunk
encodes clips `[0,17) [17,34) [34,51)` where frames 39-50 are copies of frame 38.
Tokens 10-11 — the two that survive `token_drop=3` — come from that padded clip
and are **not** slices of any longer encode. Only fully-real clips slice:

```
C=39: tokens 0-9  of 12 sliceable, 10-11 re-encoded per chunk   (83%)
C=73: tokens 0-19 of 22 sliceable, 20-21 re-encoded per chunk   (91%)
```

So even with `S % 17 == 0` the win is "encode 1 clip instead of 3" on the 4.9 GB
VAE — the cheapest of the three models. **Sliceability is not a profile-selection
criterion.** Choose on §4 instead.

It does yield one useful consequence for §7: within the shared overlap, the *old*
chunk's source conditioning sits in its padding-contaminated tail while the *new*
chunk's sits in its first fully-real clip. Bias the seam **early** in the overlap
rather than excluding both ends symmetrically.

## 3. Required core changes

Both are applied through `ModelPatcher`, never by forking `comfy/ldm/minimax/`.
This preserves the boundary the repo README already states: model-side behaviour
lives in core and is not forked here. These are the first changes in this repo
that need core behaviour altered, and object patches keep that reversible.

### 3.1 Concatenate the conditioning latents

```python
payload["cond_video_latents"] = [
    *[kf["latent"] for kf in keyframes],
    *[r["latent"] for r in refs if "latent" in r],
]
```

Order is load-bearing: `all_video_rows[~img_update] = cond_video_rows` consumes
them in row order, which is keyframes then refs.

### 3.2 Place the keyframe at the target origin

Two-pass layout construction. Walk the refs first to find where the target
timeline begins, then emit rows in the existing physical order with corrected
coordinates:

```python
target_cursor = calculate_cursor_after_refs(text_len=text_len, refs=refs)

for kf in keyframes:
    if kf["resolved_frame_index"] == 0:
        cond_t = target_cursor
    elif kf["resolved_frame_index"] == frame_count - 1:
        cond_t = target_cursor + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
```

Row order stays `[text | keyframes | refs | target audio | target video]` so
§3.1's concatenation order is unaffected.

### 3.3 Applying them

```python
m = model.clone()

existing = m.get_model_object("extra_conds")
if getattr(existing, "_h3_hybrid_patch", False):
    original = existing._h3_original          # already patched: do not re-wrap
else:
    original = existing

def patched_extra_conds(**kwargs):            # instance attribute — no self
    out = original(**kwargs)
    keyframes = kwargs.get("minimax_keyframes") or []
    refs = kwargs.get("minimax_refs") or []
    if keyframes and refs:
        payload = out["minimax_payload"].cond.copy()
        payload["cond_video_latents"] = [
            *[kf["latent"] for kf in keyframes],
            *[r["latent"] for r in refs if "latent" in r],
        ]
        payload["layout"] = build_corrected_layout(payload, kwargs)
        out["minimax_payload"] = comfy.conds.CONDConstant(payload)
    return out

patched_extra_conds._h3_hybrid_patch = True
patched_extra_conds._h3_original = original
m.add_object_patch("extra_conds", patched_extra_conds)
```

The idempotence guard matters because `get_model_object` returns the *patch* when
one is registered ([`model_patcher.py:756-761`]) and `clone()` copies
`object_patches` — without it, a re-run in the same graph wraps the wrapper.

If a future core release ships the fix, detect it and skip the patch.

## 4. Profile selection

### 4.1 Cost model

Runtime is linear in packed sequence length (§1.3), so cost per output frame is
`S(C) / stride`, and since S is near-proportional to C this reduces to `C / S`.

Ref2V packs the source video at the same canvas and same `T` as the target, so
the sequence roughly doubles relative to plain t2v:

```
rows per latent frame  =  pixels / 1024        (16× VAE, then 2×2 DiT patch)
S  ≈  2 · T · rows      + text presentation + audio rows
```

Resolution and chunk length therefore trade directly against one another for the
same token budget:

| canvas | rows / latent frame |
| --- | ---: |
| 1344×768 (~1.03 MP, node default) | 1008 |
| ~0.8 MP (current working canvas) | ~800 |

Running at 0.8 MP rather than the node's default buys roughly **one full rung**
on the C ladder below. Text presentation and audio add order 1-2k; the video rows
dominate.

### 4.2 Candidates

The ceiling is **measured, not estimated**: this machine runs ~4 s of input plus
~4 s of output at 0.8 MP. That is ~96 frames each way, which brackets the grid
between C=90 and C=107.

At 0.8 MP (~800 rows per latent frame), with the ceiling marked:

| C | T | duration | ≈ S | vs. trained min | fits? |
| ---: | ---: | ---: | ---: | ---: | --- |
| 39 | 12 | 1.6 s | ~21k | 0.31× | yes, wasteful |
| 56 | 17 | 2.3 s | ~29k | 0.45× | yes |
| 73 | 22 | 3.0 s | ~37k | 0.59× | yes |
| **90** | **27** | **3.75 s** | **~45k** | **0.73×** | **yes — target** |
| 107 | 32 | 4.46 s | ~53k | 0.86× | at or over the line |
| 124 | 37 | 5.17 s | ~61k | 1.00× | no |

**Target profile: C=73, O=22, S=51.** `C/S` = 1.43×, 24 chunks for a 50-second
source, `S % 17 == 0`, 0.59× the trained minimum.

C=90 is the compute optimum (§4.4) and C=73 costs **+2.7%** against it — but C=90
sits exactly at the measured ceiling with *zero* margin, and the run is
unattended for hours. C=73 buys **0.90 GB** of headroom for that 2.7%. See §4.6
for why that is the right trade.

Overlap alternatives at C=73, to be settled after Stage 0:

| O | S | `C/S` | seam window | chunks / 1200 frames |
| ---: | ---: | ---: | ---: | ---: |
| 22 | 51 | 1.43× | 0.9 s | 24 |
| 39 | 34 | 2.15× | 1.6 s | 34 |

Start at O=22 and raise it only if seams are visible. Both are clip-aligned
(51 = 3×17, 34 = 2×17).

C=90 remains the profile to use on a quiet card with `match`-sized references and
a short prompt; C=107 should not be used at all, since it is past the ceiling
*and* past the compute optimum.

### 4.3 The ceiling is a token budget — spend it deliberately

The 4 s + 4 s measurement was taken on a plain Ref2VA run. **The chunked design
adds rows on top of that**, and they come out of the same budget:

| item | rows at 0.8 MP | ≈ % of a C=90 sequence |
| --- | ---: | ---: |
| anchor keyframe (1 latent frame) | ~800 | 1.8% |
| static image ref, `ref_image_size="match"` | ~800 each | 1.8% each |
| static image ref, `ref_image_size="max"` | up to ~7,300 each | 16% each |

`match` scales a reference to the generation's pixel area; `max` only scales
*down* to a 2048 short edge, so a large source image stays large — a 2048×3641
reference is ~7,300 rows, nine latent frames' worth. In a monolithic run you pay
that once; here it rides through **every chunk**, so its total cost is multiplied
by the chunk count. Default to `match`, and treat `max` as a deliberate purchase
of roughly one C rung.

#### The canvas mismatch is a VRAM leak, not just a geometry bug

§8 already requires pinning the canvas for correctness. At the ceiling it also
costs tokens. `adapt_canvas` sizes the reference video from *its own* dimensions
([`nodes_minimax_h3.py:241-244`]) and always lands near 1.03 MP for any source
larger than that — so a 1080p source fed to a 0.8 MP target gets a 1344×768
reference against an 800-row target:

```
pinned at 0.8 MP:   S ≈ 27 × (800 + 800)  = 43.2k
unpinned:           S ≈ 27 × (800 + 1008) = 48.8k     +13%
```

That 13% is most of the margin between C=90 and the measured ceiling. Pin it.

#### Remaining probe

Not "what is the ceiling" — that is measured — but **does C=90 still fit once the
anchor, the static refs and the pinned canvas are all in place.**

`user/minimax_vram_probe.py` answers this without a full sampling run. It sweeps
the `17k+5` grid and measures the real peak allocation of one
`comfy.ldm.minimax.model.DiTBlock` forward at each packed-sequence length, using
the actual class so the fused in-place rms+rope and swiglu kernels are counted.
Blocks run sequentially and free their activations, so the sampling transient is
one block's peak plus the persistent packed hidden state.

It now packs reference rows natively via `--mode ref2v`, in PackedLayout order,
so nothing has to be hand-folded into `--text-len`:

```
python user/minimax_vram_probe.py --budget 11 --mode ref2v \
    --width 1216 --height 672 --text-len 1500 \
    --anchor --ref-audio --calibrate-to 90 \
    --ckpt models/diffusion_models/hf_minimax_h3/minimax_h3_ref2va_pruned_int8_convrot.safetensors
```

`--ref-frames` defaults to `matched` — the reference video takes the target's
length and canvas, which is the v2v case and what makes ref2v roughly double the
sequence. Verified row breakdown at C=90, 1216×672 (798 rows per latent frame,
~0.82 MP):

```
text        1,500      cond (anchor)     798      ref_audio     300
ref_img    21,546      audio             300      video      21,546
                                                  seq_len    45,990
```

That is **1.97× the t2va sequence at the same frame count** — the concrete form
of "ref2v uses a lot more." The full sweep reproduces §4.2's estimates exactly
(C=107 → 54,082; C=124 → 62,178).

**Calibrate the reserve before trusting any absolute number.** With the DiT
streamed, `--reserve-gb` defaults to a single block's weights (~0.36 GB) and
excludes ComfyUI's streaming cache, the conditioning tensor, attention workspace
and `cudaMallocAsync` slack. Run analytically it puts C=90 at ~6.2 GB and C=124
at ~8.3 GB — i.e. it predicts the trained minimum fits in 11 GB, which
contradicts the measured 4 s + 4 s ceiling.

`--calibrate-to <frames>` solves for the reserve that makes a length you have
actually run come out as the last fitting one, then re-sweeps with it. The
probe's *relative* ordering across the grid is trustworthy either way; its
absolute ceiling is not, until the reserve is pinned to reality.

Static references are priced too, and confirm §4.3's earlier estimate: at
`ref_image_size=match` one reference costs +1.7% of the sequence, at `max`
(2048×3641) +15.8%.

```
C=90 fits with margin        →  ship 73/22/51 anyway, per the §4.6 headroom budget
C=90 tips                    →  73/22/51  (C/S = 1.43×, 24 chunks / 1200 frames)
C=90 fits with room to spare →  probe 107/22/85
```

#### Measured result

Run at 1216×672, `--mode ref2v --anchor --ref-audio --calibrate-to 90`,
budget 11 GB, against the real ref2va checkpoint header:

| C | tokens | transient | +latent | +cond | total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 39 | 21,710 | 2.401 G | 0.005 | 0.015 | 8.283 G |
| 56 | 29,802 | 3.295 G | 0.007 | 0.020 | 9.186 G |
| 73 | 37,898 | 4.189 G | 0.009 | 0.026 | 10.087 G |
| **90** | **45,990** | **5.085 G** | 0.012 | 0.032 | **10.991 G** |
| 107 | 54,082 | 5.978 G | 0.014 | 0.037 | 11.892 G — over |

Calibrated reserve: **5.86 GB** outside the transient, against a 0.36 GB floor
default. The floor under-predicts by 16×, which is the whole reason the raw probe
said C=124 would fit.

**The transient is exactly linear in sequence length.** Successive 17-frame rungs
cost 0.896, 0.894, 0.894, 0.896, 0.893, 0.895, 0.893, 0.896 GB — flat to within
0.3% across a 14× span of sequence length. There is no quadratic term in memory
at all: sage attention is O(S), and this is direct confirmation of §1.3 from the
memory side to match the timing side.

Two usable constants fall out, both at 0.82 MP:

```
~116 KB per token
~0.894 GB per 17-frame rung of C
```

And an exchange rate, since rows = pixels/1024 and S ≈ 2,100 + 55·rows at T=27:

```
one C rung (17 frames)  ≈  0.15 MP of canvas
```

So C=107 fits at ~0.69 MP for what C=90 costs at 0.82 MP. Probably not worth the
16% pixel loss for one rung, but it is the lever if C=107 is wanted.

**Caveat on the ceiling itself.** `--calibrate-to 90` *defines* 90 as the last
fitting length, so "max = 90 frames" is by construction, not an independent
result. What the sweep establishes independently is the per-rung cost and the
linearity — and those say reaching C=107 needs 0.9 GB more headroom than C=90,
whatever the true absolute ceiling turns out to be.

#### What the probe does not cover

Its docstring flags the first of these; the second matters here too.

- **VAE decode.** A separate peak that can OOM on a length that sampled fine. In
  a monolithic run you hit it once; Pass C hits it **every chunk**, so it is not a
  tail risk here but a per-iteration one. Measure it separately and tile or
  chunk-decode if that is where the ceiling actually lands.
- **VAE encode** of the source chunk in Pass A — smaller, but also per chunk.

Confirm peak reserved VRAM on the real Stage-0 run regardless; `vram_guard.py`
already does this check. Under `cudaMallocAsync` trust `GPU reserved memory`, not
the Tot Alloc/Freed columns.

Note that ~46k tokens is well into the range the local sage int64 offset patch
exists to fix. Confirm that patch is applied before probing.

### 4.4 Measured: C=90 is the compute optimum, not just the VRAM ceiling

Fitted from the sage-enabled probe (`ms = 14.83e-3·S + 205.2e-9·S²`), cost per
output frame at O=22:

| C | S | block ms | stride | ms / output frame |
| ---: | ---: | ---: | ---: | ---: |
| 39 | 22,050 | 427 | 22 | 19.40 |
| 56 | 30,030 | 630 | 34 | 18.54 |
| 73 | 38,010 | 860 | 51 | 16.86 |
| **90** | **45,990** | **1,116** | **68** | **16.41** ← minimum |
| 107 | 53,970 | 1,398 | 85 | 16.45 |
| 124 | 61,950 | 1,706 | 102 | 16.73 |
| 141 | 69,930 | 2,040 | 119 | 17.14 |

The curve has a real minimum. Below C=90 the overlap tax dominates; above it the
quadratic attention term starts to bite. **The VRAM ceiling and the compute
optimum land on the same rung**, which is a convenient accident and worth not
disturbing — pushing to C=107 for lower `C/S` would cost 0.9 GB of headroom to
buy nothing.

Attention backend matters to this result. The same sweep on pytorch attention
fits `581.5e-9·S²` — 2.83× the sage constant, attention 69% of the block — and
shifts the optimum downward. Any re-measurement must pass `--sage`; the probe
defaults to pytorch because `comfy.cli_args` only reads argv under
`comfy.options.args_parsing`, which only `main.py` sets.

Multiply the block column by 50 layers for a DiT step: ~56 s per step at C=90.

### 4.6 Headroom budget — why C=73 and not the optimum

The §4.4 minimum assumes the probe's exact conditions: a 1,500-token prompt, no
static references, and whatever the desktop happened to be holding. Every one of
those varies in production, and at ~116 KB per token they convert straight into
VRAM:

| variable | cost |
| --- | ---: |
| **C=90 → C=73 headroom gained** | **0.90 G** |
| desktop VRAM swing observed in one session (0.73 → 2.99 G) | 2.26 G |
| 1 static ref at `ref_image_size=max` | 0.81 G |
| +2,500 prompt tokens | 0.28 G |
| 3 static refs at `match` | 0.26 G |
| 1 static ref at `match` | 0.09 G |

A single `max`-sized reference nearly consumes the entire C=90 → C=73 gap on its
own, and the desktop baseline alone moved 2.26 GB during one working session.
C=90 has no room for any of it.

The asymmetry decides it: **+2.7% certain compute versus a spill risk on an
unattended 5-6 hour run**, where one spilled chunk out of 24 costs more than the
2.7% saved across all of them. Take the margin.

This also argues for an `auto` chunk size (§12): the node knows the exact row
count before it samples, so it can read free VRAM, predict the transient at
116 KB/token, and pick the largest grid rung that fits with a configured safety
margin. That converts the whole question from a constant into a pre-flight check.

### 4.5 Validation rules

```
C % 17 == 5                 generation legality (hard)
O = C - S
actual_frames > O           invariant, see §5.2
S % 17 == 0                 optional, buys partial source-latent slicing only
```

## 5. Chunk planning

```python
@dataclass(frozen=True)
class ChunkSpec:
    index: int
    global_start: int
    model_frames: int      # always C
    actual_frames: int     # real source frames; < C only on the final chunk
    overlap_frames: int
    stride_frames: int
    is_first: bool
    is_final: bool
    seed: int
```

### 5.1 Counts and intervals

```python
chunk_count = 1 if N <= C else 1 + ceil((N - C) / S)
global_start  = i * S
actual_frames = min(C, N - global_start)
```

The model always receives `C` frames. When `actual_frames < C`, repeat-pad the
last real source frame to `C` and discard the generated padding after sampling.

### 5.2 Invariant

`actual_frames > overlap_frames` always holds: chunk `i > 0` exists only when
`N > C + (i-1)·S`, hence `N - i·S > C - S = O`. The stitching in §7 silently
depends on this — assert it and test it, because it is the first thing to break
if `overlap_frames` is raised.

### 5.3 Anchor mapping

For chunk `i > 0`, the anchor is the previous chunk's output at **local frame
`S`** — the first frame of its pending tail, corresponding to the new chunk's
global start. Not its final frame — local frame 51 at the 73/22/51 target profile.

## 6. Architecture — three passes

Per §9 the anchor is never shown to Qwen, so chunk conditioning depends only on
information available before sampling begins. Only the sample/decode loop is
inherently serial.

### Pass A — source VAE preprocessing

VAE resident once. Encode every source chunk and every static image reference.
Store source video latents, reference audio latents and chunk metadata on
CPU/disk (~4.6 MB per chunk latent at the default canvas).

### Pass B — Qwen conditioning preprocessing

VAE unloaded, Qwen resident once. Encode all chunk presentations; store
cross-attention conditioning and token modality tags (~7-13 MB per chunk).

**`cond_cache.py` in this repo is already most of Pass B** — the blake2b digest
over the tokenized stream plus TE fingerprint, with safetensors entries in
`<user-directory>/h3_cond_cache/`. Reuse it rather than reimplementing.

### Pass C — sequential generation

```
sample chunk → decode → take next anchor → encode anchor as keyframe
             → stitch/finalize → next chunk
```

Still alternates DiT and VAE, because each anchor is unknown until the preceding
chunk finishes. But it removes 53 of 54 Qwen residency cycles.

Only these survive between chunks: previous overlap tail, next anchor, static
reference cache, writer state, manifest state. Peak VRAM is therefore independent
of total video duration.

## 7. Stitching

`A` = previous chunk's retained tail (length `O`); `B` = current chunk's first
`O` frames. Both cover the same global timestamps.

```
first non-final chunk:   write A[0:S];              retain A[S:C]
later non-final chunk:   write prev_tail[0:k];      write cur[k:S];  retain cur[S:C]
final chunk:             write prev_tail[0:k];      write cur[k:actual_frames]
```

Every written section contains exactly `S` new global frames: `k` from the old
result, `S-k` from the new.

Default `best_cut`, no blending — crossfading mismatched geometry produces double
edges. Score on downscaled CPU frames:

```
cost[k] = appearance_cost(A, B, k)
        + edge_weight   * edge_cost(A, B, k)
        + motion_weight * motion_cost(A, B, k)
```

Bias the search **early** in the overlap (§2.2). Optional short cosine blend
around `k`, gated on correspondence error below a threshold.

## 8. Canvas, seeds, audio

**Canvas.** The stock node computes the reference-video canvas from the video's
own dimensions via `adapt_canvas`, independently of the target latent
([`nodes_minimax_h3.py:241-244`]). For v2v these must be identical and constant
across every chunk. Derive both from the source; do not expose free
`width`/`height` by default; assert equality per chunk.

**Seeds.** Separate the two roles — `payload["seed"]` drives conditioning noise
augmentation as well as sampling noise:

```
sampling_noise_seed            fixed | increment | hashed (SplitMix64, not Python hash())
conditioning_augmentation_seed fixed always
```

At `VISUAL_COND_TIMESTEP = 0.999` the augmentation is only 0.1% amplitude, but
varying it per chunk perturbs the source and anchor conditioning and makes a
fixed-vs-hashed comparison unclean.

**Audio.** MVP is `preserve_source`: feed the correct source-audio chunk to
Ref2VA for visual sync, discard the generated audio, remux the untouched source
after stitching, trimmed to `N / 24` seconds. This sidesteps music restarting per
chunk, dialogue timbre shifts, phase discontinuities and 24 fps ↔ 40 Hz overlap
alignment. Generated-audio overlap-add is experimental and must not block the
video path.

## 9. Qwen presentation limitation

The tokenizer takes `minimax_ref_items` **or** `images`, never both. So the MVP:

- presents the source video and normal reference assets to Qwen;
- supplies the anchor **only** as a DiT first-frame keyframe latent;
- gives the anchor no `<Picture N>` label;
- preserves the user's existing `<Picture N>` / `<Video N>` / `<Audio N>` numbering;
- keeps the source as `<Video 1>` in every chunk.

Whether the checkpoint responds sufficiently to a keyframe Qwen never saw is the
Stage-0 question.

## 10. Stage 0 — feasibility gate

Two hard-coded chunks at the **target profile** — C=73, O=22, S=51, 0.8 MP,
canvas pinned. Chunk A source `0-72`, chunk B source `51-123`, chunk B anchored
from chunk A output frame 51. No generated audio, fixed seeds, manual hard seam,
frame dumps for manual comparison.

Run at 73 rather than at a cheaper 39 for two reasons: it is the geometry that
will ship, and 39 frames is 0.31× the trained minimum, so a poor result there
would not distinguish a broken anchor from an out-of-distribution chunk length.
The §4.3 headroom probe needs a 90-frame run anyway — same geometry, both
questions answered in one session.

Use an anchor whose effect is **unmistakable** — strong recolour or an obvious
graphic edit — rather than judging subtle identity preservation on the first run.

Three arms:

| Arm | Latents | Keyframe `cond_t` |
| --- | --- | --- |
| A. Corrected | concatenated | target origin after refs |
| B. Stock position | concatenated | `text_len` |
| C. Anchor disabled | refs only | — |

Record peak reserved VRAM on every arm — that is the §4.3 probe, obtained for
free. If C=73 tips, drop to 56/22/34 and rerun; the conditioning question is
unaffected by the step down.

This diagnoses: whether hybrid keyframe conditioning works at all; whether
correct RoPE placement matters; whether an anchor unseen by Qwen suffices;
whether Ref2VA still obeys source motion once an anchor is introduced; and
whether the packed layout runs without row-count or latent-order errors.

### Fallbacks, in order

Explicit modes, never silent fallbacks:

1. Add the anchor to Qwen as an additional hidden reference image, keeping the DiT keyframe.
2. Supply the previous generated overlap as an additional reference-video block.
3. A stronger latent or inpainting-style first-frame constraint.
4. Overlap-only Ref2VA with no hard anchor, relying on seam selection.

## 11. Stages

**Stage 0** — the three-arm proof at 73/22/51, with peak VRAM recorded per arm.
Settles both the architectural unknown and whether the target profile fits.

**Stage 1 — core chunk engine.** Chunk planner; final padding and trimming;
hybrid conditioning builder; repeated sampling; CPU frame lifecycle; fixed
midpoint seam; exact output-length accounting. One source video, static image
references.

**Stage 2 — production node.** Single high-level node; sampler/scheduler
controls; progress and cancellation; best-cut seam search; source-audio
preservation; streaming writer; run manifest; resume.

**Stage 3 — quality controls.** Fixed vs hashed seeds; overlap sizes; short seam
blends; edge/motion-aware scoring; source-cut-aware placement; reference-latent
caching (bounded by §2.2 — only fully-real clips slice).

**Stage 4 — advanced conditioning.** Qwen-visible continuation anchor; previous
generated overlap as an extra video reference; dynamic overlap on motion or
detected cuts; prompt timeline segmentation; generated-audio stitching;
latent-space seam selection.

## 12. Node interface (Stage 2)

```
required   model, clip, video_vae, audio_vae, source_video, prompt, seed
reference  ref_images, extra_ref_videos, extra_ref_video_audios, ref_audios,
           initial_target_frame, source_audio
sampling   steps, sampler_name, scheduler, denoise=1.0,
           shift_video=12.0, shift_audio=3.0, seed_mode
chunking   chunk_frames=73 | auto, overlap_frames=22, strict_h3_grid=true,
           final_padding=repeat_last, vram_margin_gb=1.0
           `auto` reads free VRAM pre-flight and picks the largest grid rung whose
           predicted transient (116 KB/token over the real row count) fits the margin
ref_size   ref_image_size=match — `max` costs 0.81 G per reference (§4.6)
stitching  seam_mode=best_cut, seam_search_margin=2, blend_frames=0,
           motion_weight, edge_weight
output     audio_output=preserve_source, output_mode=stream_to_file,
           output_format, resume, temporary_codec
outputs    video, audio, manifest, debug_frames (debug mode)
```

`initial_target_frame` is a deliberately edited opening frame. **Never** default
it to the original source frame — that would defeat an intended subject or
appearance replacement. Absent it, chunk 0 establishes the target appearance
itself.

Production streams to a writer. Returning one enormous `IMAGE` tensor would swap
a VRAM limit for a system-RAM limit.

Defaults for `steps` / `sampler_name` / `scheduler` come from the validated
workflow, and the node applies the sigma-shift model clone and patch internally.

## 13. Streaming, manifest, resume

Persistent writer receiving finalized frames only — a single continuously open
encoder, or lossless temporary segments concatenated at the end. Do not reopen a
lossy encoder per chunk.

Persist after every completed chunk: source and settings hashes, fps, total
frames, C/O/S, completed chunks, chunk seeds, seam indices, pending tail file,
next anchor file, frames written. Store the pending tail and next anchor
**losslessly** — regenerating them from a lossy intermediate would alter later
chunks. Resume rejects changed settings unless the user starts a new run.

## 14. Testing

**Unit.** Grid calculations; C/O/S constraints; chunk count; global↔local frame
mapping; no duplicated or missing output frames; final padding; final trimming;
deterministic seeds; manifest resume state; conditioning-latent ordering; seam
output length; the §5.2 invariant. Boundary lengths: 1, 21, 22, 23, 38, 39, 40,
60, 61, 62, 100.

**End-to-end clips.** Static close-up face; fast head movement; walking subject;
rapid hand movement; hair or loose clothing; camera pan; camera shake; foreground
occlusion; scene cut inside overlap; scene cut near a chunk boundary; background
replacement; identity replacement.

**Comparisons,** where a monolithic run fits in memory: monolithic Ref2VA;
chunked without overlap; one-frame overlap; the §4.2 profiles; anchor on vs off.

**Measurements.** Peak VRAM; total and per-chunk runtime; boundary pixel and edge
disagreement; optical-flow discontinuity; identity drift; reference adherence;
human seam visibility.

## 15. Acceptance criteria

1. Output contains exactly the requested number of 24 fps frames.
2. No frames duplicated or omitted by chunk bookkeeping.
3. Peak VRAM does not grow with source duration.
4. A long run resumes after interruption without regenerating completed chunks.
5. Source audio stays synchronized in preservation mode.
6. Chunk boundaries are not consistently obvious across the evaluation set.
7. Ref2VA source motion remains substantially intact after hybrid keyframe conditioning.
8. Reference identity/appearance does not reset at every chunk.
9. Every run records enough metadata to reproduce its chunk seeds and seams.

A numeric threshold for acceptable seam visibility should be chosen only once a
baseline dataset exists.

## 16. Risks

**Hybrid conditioning is unsupported behaviour.** The model code can represent
keyframes and references together, but no shipped workflow exercises the
combination. Quality under it is unknown — this is what Stage 0 exists for.

**The carried frame fixes position, not velocity.** One frame cannot encode motion
direction, acceleration, blur, cloth state or expression trajectory. The
duplicated overlap and seam selection are what compensate.

**Ref2VA may pull identity back toward the source.** Every chunk reintroduces the
original video; static replacement references must be re-supplied every chunk.

**Drift accumulates.** The carried frame propagates any previous appearance error,
once per chunk boundary — so exposure scales with **chunk count**, not duration.
Maximizing C (§4.3) is therefore a quality lever as much as a compute one: 1200
frames is 54 hops at C=39 and 24 at C=73. Heavy overlap and repeated static
references reduce per-hop error without eliminating it.

This is the one axis on which "arbitrarily large input" is not literally true.
Peak VRAM genuinely is O(1) in duration, which is the hard part and what the
design achieves. Quality is not: drift grows with chunk count, so the practical
ceiling on duration is empirical, and characterizing it is Stage 3's job.

**Global prompt timelines reset.** Each chunk is a new invocation with local
timestamps; prompts with absolute timing across the whole source will need
timeline segmentation.

**Core drift.** This work depends on `PackedLayout`, `extra_conds`, the minimax
tokenizer and the VAE's temporal chunking. The repo README already flags that
these live in core and are not forked; the object patches in §3 must be
re-validated whenever core moves.

## 17. Module layout

```
chunked_ref2v/
├── PLAN.md              this document
├── __init__.py
├── nodes.py             the single high-level node
├── frame_source.py      PyAV / VIDEO / IMAGE-batch sources, normalized to 24 fps
├── chunk_plan.py        ChunkSpec, ChunkPlanner, invariants
├── h3_conditioning.py   per-chunk builder (reuses the forked ref2v node)
├── h3_layout_patch.py   §3.1 + §3.2, applied via add_object_patch
├── sampler_runner.py    model clone, sigma shift, guider, sample
├── seam.py              cost scoring, cut selection, optional blend
├── audio.py             preserve / experimental generated
├── writer.py            SegmentWriter
├── manifest.py          RunManifest, resume
└── tests/
```

The UI is one node; the implementation is not one class.

### Reuse from the parent repo

| Need | Existing |
| --- | --- |
| Pass B conditioning cache | `cond_cache.py` |
| Per-chunk conditioning builder | `nodes_minimax_h3.py` (the `Zi` fork) |
| 12 GB headroom checks at runtime | `vram_guard.py` |
| Per-run state | `run_context.py` |
| Attention structure measurement | `h3_probe/` |
| Chunk-size ceiling prediction (§4.3) | `user/minimax_vram_probe.py` (outside this repo) |

`minimax_vram_probe.py` now carries `--mode ref2v` with `--ref-frames` /
`--ref-audio` / `--anchor` / `--static-refs` and a `--calibrate-to` mode. It
lives in `user/` (gitignored by both repos) and is worth moving in here once the
node exists, so the layout arithmetic has one home and the tests can cover it.

[`comfy/ldm/minimax/model.py:319`]: ../../../comfy/ldm/minimax/model.py
[`model.py:335-377`]: ../../../comfy/ldm/minimax/model.py
[`model.py:388`]: ../../../comfy/ldm/minimax/model.py
[`model.py:321`]: ../../../comfy/ldm/minimax/model.py
[`model.py:329-390`]: ../../../comfy/ldm/minimax/model.py
[`model.py:572`]: ../../../comfy/ldm/minimax/model.py
[`model.py:87-91`]: ../../../comfy/ldm/minimax/model.py
[`model.py:31`]: ../../../comfy/ldm/minimax/model.py
[`model.py:473-481`]: ../../../comfy/ldm/minimax/model.py
[`comfy/model_patcher.py:451`]: ../../../comfy/model_patcher.py
[`model_patcher.py:1107-1110`]: ../../../comfy/model_patcher.py
[`1157-1161`]: ../../../comfy/model_patcher.py
[`model_patcher.py:756-761`]: ../../../comfy/model_patcher.py
[`comfy_extras/nodes_minimax_h3.py:33-46`]: ../../../comfy_extras/nodes_minimax_h3.py
[`nodes_minimax_h3.py:177`]: ../../../comfy_extras/nodes_minimax_h3.py
[`nodes_minimax_h3.py:254`]: ../../../comfy_extras/nodes_minimax_h3.py
[`261-264`]: ../../../comfy_extras/nodes_minimax_h3.py
[`nodes_minimax_h3.py:246-253`]: ../../../comfy_extras/nodes_minimax_h3.py
[`nodes_minimax_h3.py:241-244`]: ../../../comfy_extras/nodes_minimax_h3.py
[`comfy/model_base.py:2094`]: ../../../comfy/model_base.py
[`2098`]: ../../../comfy/model_base.py
[`comfy/text_encoders/minimax.py:153-178`]: ../../../comfy/text_encoders/minimax.py
[`comfy/ldm/minimax/vae.py:335-350`]: ../../../comfy/ldm/minimax/vae.py
[`vae.py:522-539`]: ../../../comfy/ldm/minimax/vae.py
