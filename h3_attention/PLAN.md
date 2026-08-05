# MiniMax H3 attention backend — implementation plan

**Status:** design, pre-implementation
**Basis:** revision of the WebGPT draft plan, scoped to this hardware.
**Supersedes:** §2.1 backend table, §4.6 architecture list, §7 stages, §11 shapes,
§13 matrix, and §16 commits of that draft. Everything not contradicted here stands.

## 0. Two objectives, and which patch each one retires

The work exists to close two distinct problems that the draft plan treats as one.

| # | goal | closed by |
| --- | --- | --- |
| 1 | The overflow crash must not return when an update reverts a patch | vendored int64 Q/K quantizers **and** a V path that never enters `_fused.pyd` |
| 2 | Reclaim the VRAM that the fused QKV holds through the attention peak | H3-owned block forward with explicit activation lifetime |

The two current patches fail on **different triggers**, which is why goal 1 needs
both halves:

| patch | site | limit | reverts when | reachable on 12 GB? |
| --- | --- | ---: | --- | --- |
| site-packages `tl.int64` | Q/K Triton quant, **signed** i32 | S = 99,864 | **sageattention** upgrade | **yes** — OOM lands ~120-150k |
| in-repo `contiguous()` | V `TransposePadPermuteKernel` in `_fused.pyd`, **unsigned** u32 | S = 199,728 | **ComfyUI** update (`git pull` conflict) | **no** — OOM arrives first |

Read that table carefully, because it inverts the draft's emphasis:

- The **Q/K** limit is live on this card. A long single-shot ref2va run can cross
  S=99,864 before it OOMs. Vendoring those kernels is a real fix, not insurance.
- The **V** limit is not reachable here and never will be. But its guard is the one
  living in the ComfyUI tree, so it is the one that breaks on update — which makes
  it exactly the patch goal 1 names. It cannot be retired by testing; only by
  proving the custom path never calls that kernel (§2.3).

## 1. Correction log — what changed from the draft

### 1.1 The memory win is the business case, not a footnote

The draft calls the peak-VRAM reduction "slight" and Gate D declines to set a
target. Both understate it. The fused QKV projection is one allocation that Q, K
and V are views into, and it is held live across the attention call:

| profile | S | fused QKV bf16 | measured transient | QKV share |
| ---: | ---: | ---: | ---: | ---: |
| C=22 | 13,617 | 0.545 GB | ~1.507 GB *(extrapolated)* | 36% |
| C=73 | 37,898 | 1.518 GB | 4.189 GB *(measured, chunked_ref2v §4.3)* | 36% |
| C=90 | 45,990 | 1.842 GB | 5.085 GB *(measured)* | 36% |

Set that against `chunked_ref2v/PLAN.md` §4.6, which spends **+2.7% compute to buy
0.90 GB** of headroom by stepping C=90 down to C=73. The releasable QKV at C=73 is
**1.518 GB — 1.7× the entire headroom budget.**

So the target is not "some reduction." It is:

```
recover >= 0.90 GB at C=90  ->  the 73-vs-90 trade dissolves and C=90 (the §4.4
                                compute optimum) becomes the shipping profile
```

That is the outcome worth building for. State it as the goal and measure against it.

**Upper bound, not a promise.** Under `cudaMallocAsync` with ComfyUI's streaming
cache, freeing a block need not shrink `GPU reserved memory`. The 36% is what is
*eligible* for release, not what the allocator returns. Which is why:

### 1.2 The go/no-go measurement moves to the front

The draft defers this to Gate D, after three commits of kernel work. Invert it.
Commit 2 delegates attention to PyTorch and exists purely to answer one question
(§2.1). If reserved memory does not move, Commits 3-4 are not worth writing, and
that is knowable for the cost of a `del` and a memory read.

### 1.3 NHD layout is not itself a win

The draft's item 1 — "keeps the fused QKV in strided NHD layout, avoiding
unnecessary layout materialization" — does not survive reading the core forward.
At [`model.py:177-179`], the three calls are:

```python
q = q.transpose(0, 1).unsqueeze(0)
```

`transpose` and `unsqueeze` are **views. Zero copy.** And `rms_rope_split_half_`
already mutates the fused buffer in place, so core is not materializing anything.

NHD therefore saves nothing at the H3 level. It matters *only* if SageAttention's
NHD entry point skips an internal copy that its HND entry point makes. **Verify
that before treating layout as a pillar.** If it does not hold, the draft's item 1
collapses into item 3 and the entire win is activation lifetime — which is fine,
but it changes what the code needs to look like.

### 1.4 SM89 only

Cut SM80/86, SM90, SM120 entirely — not "defer", cut. The draft's own rule is that
each architecture needs a hardware-backed result, and this machine is one RTX 4070
(SM89). Those paths could only ever be written blind and shipped untested.

Configuration must raise a clear, specific error on any other compute capability.
Someone with the hardware can add a path later; a speculative branch helps nobody
and rots silently.

### 1.5 Validation is 4 runs, not 21

The draft's §13 matrix is 7 cases × 3 backends, several of them multi-hour, on the
card whose scarcity motivated the whole headroom analysis. Replaced by §4.

### 1.6 The forked forward is a real, accepted maintenance cost

Core's `Attention.forward` is **25 lines** ([`model.py:156-181`]), which is small
enough that owning a copy is defensible. It is also unavoidable: an
`optimized_attention_override` **cannot** release the fused QKV, because the
caller's `forward` still holds `q`, `k`, `v` as locals until the override returns.
The draft is correct about this.

But it should be recorded as what it is: **we now own a copy of a core method that
can drift silently.** Stage 0's characterization tests turn silent drift into a
loud test failure. They do not prevent drift. Accepted deliberately.

## 2. The three questions, in the order they should be answered

### 2.1 Does freeing the QKV actually reduce memory? — **ANSWERED: yes, in full**

`benchmarks/measure_qkv_release.py` reproduces the allocation pattern at H3's real
shapes with no model loaded, so the question is settled in seconds instead of by
instrumenting a multi-hour run. Measured on the RTX 4070, 2026-08-05:

| C | S | predicted releasable | live at attention, hold → release | **realized** |
| ---: | ---: | ---: | ---: | ---: |
| 22 | 13,617 | 0.545 GB | 0.824 → 0.278 GB | **0.545 GB** |
| 73 | 37,898 | 1.518 GB | 2.293 → 0.775 GB | **1.518 GB** |
| 90 | 45,990 | 1.842 GB | 2.782 → 0.940 GB | **1.842 GB** |

Realized equals predicted to three decimals at every rung: dropping the views
returns **100%** of the fused QKV. §1.1's worry that `cudaMallocAsync` might not
give it back does not hold — see the metric traps below for why it looked like it
might.

**Gate: PASSED.** 1.518 GB at C=73 and 1.842 GB at C=90 against a 0.90 GB
threshold. Commits 3-5 are worth writing.

The consequence §1.1 predicted now has a number behind it: at C=90 the release is
**1.842 GB, 2× the 0.90 GB** that `chunked_ref2v/PLAN.md` §4.6 buys with +2.7%
compute. If it survives end to end, the 73-vs-90 trade dissolves and C=90 — the
§4.4 compute optimum — becomes the shipping profile.

**What this does not yet show.** This is the allocation pattern in isolation. A
real run also has the streaming DiT, ComfyUI's cache and the attention kernel's
own workspace competing for the same pool. The block-level release is proven; that
it moves the *end-to-end* transient by the same amount is still §4's job.

#### Two metric traps, both of which produced a false STOP

Anyone re-running this will hit both.

- **`torch.cuda.memory_reserved` is identically zero under `cudaMallocAsync`**, so
  every reserved-based delta is 0.000 GB and reads as "no saving." Use
  `memory_allocated`. `mem_get_info` is no better: the async pool retains freed
  blocks rather than returning them to the driver, so driver-free does not move
  either. The blocks stay reusable in-process, which is what actually matters.
- **A whole-tensor `x.float()` in the quantization stand-in** materializes a 4-byte
  copy of the input — 1.01 GB at C=73 — which dominates the peak *in both modes
  equally* and hides the effect entirely. The measurement quantizes in 4,096-row
  chunks to keep that temporary bounded.

Peak-allocated saving (0.506 GB at C=73) is smaller than the live-at-attention
saving because the only thing allocated after the release in this script is the
output tensor. In the real forward the attention kernel claims a much larger
workspace, so more of the freed space is reused; live-at-attention is the better
predictor of end-to-end benefit.

### 2.2 Does the NHD entry point avoid a copy? — **ANSWERED: no, it gains nothing**

Confirmed by reading the dispatch, not by benchmarking.

H3 calls attention with `skip_reshape=True` ([`model.py:181`]), so core's
`attention_sage` takes the HND branch and `tensor_layout = "HND"`. The NHD branch
exists only for callers that did *not* pre-shape, and it is the one that calls
`_reshape_qkv_to_heads`. H3 never touches it.

Inside SageAttention, `tensor_layout` is reduced to an integer flag (`0`/`1`)
handed to the kernels — it selects indexing, not whether a copy happens. And the
V path copies unconditionally either way: `per_channel_fp8` allocates a full
`v_transposed_permutted` buffer, padded to a multiple of 64, and fills it.

**§1.3 is confirmed.** Layout was never the win; activation lifetime is. The
custom forward does not need to preserve NHD, which removes a constraint from
Commit 3.

### 2.3 Can the V path avoid `_fused.pyd`? — **ANSWERED: not through sage's API**

The confirmed H3 path on this card:

```
attention_sage (skip_reshape=True -> HND, smooth_k=False)
  -> sageattn -> arch sm89
  -> sageattn_qk_int8_pv_fp8_cuda(qk_quant_gran="per_thread")   [core.py:612, 618]
       Q/K -> per_thread_int8_triton                            [core.py:747]
       V   -> per_channel_fp8 -> _fused.transpose_pad_permute_cuda  [quant.py:282]
```

Two things this pins down:

- **The Triton `per_thread` kernels really are the H3 Q/K path.** `qk_quant_gran`
  defaults to `"per_thread"` and the sm89 branch does not override it (sm120
  explicitly passes `"per_warp"`, which is what makes the default visible). So the
  site-packages `tl.int64` patch is on the live path, as `sage-int64-offset-patch`
  claims.
- **`per_channel_fp8` calls `transpose_pad_permute_cuda` unconditionally.** There
  is no flag, and both fp8 entry points (sm89 at `core.py:763`, sm90 at `924`)
  route through it. No sage API gives fp8 V without the closed kernel.

`sageattn_qk_int8_pv_fp16_cuda` — the sm80/86 path — does avoid `per_channel_fp8`
entirely, so fp16 V is a genuinely `_fused`-free option. It costs a byte per
element on V (0.253 -> 0.506 GB at C=73) and gives up SM89's fp8 tensor cores.

#### This makes goal 1 much cheaper than the draft assumed

The draft's implicit plan was: replace the closed V kernel, then delete the guard.
That means writing a transpose+pad+permute+quantize kernel to retire a guard that
is **inert on this card** — S=199,728 is unreachable when OOM arrives at 120-150k.

The guard does not need to be replaced. It needs to be **relocated**:

```
today:  comfy/ldm/modules/attention.py   <- reverts on ComfyUI update   (the problem)
move:   h3_attention/forward.py          <- lives in this repo          (goal 1 closed)
```

Applying the same `contiguous()` check inside our own forward is a few lines,
survives every ComfyUI update by construction, stays correct on a larger GPU, and
costs nothing at H3's real sequence lengths because it never fires.

**Do not delete the ComfyUI-tree edit until `sage_mem_eff` is selectable
(Commit 4), and understand what deleting it gives up.** The in-repo guard only
runs when the custom forward is patched in; `sage`, `comfy` and `pytorch` still
reach attention through core's path. Deleting the upstream edit therefore leaves
those three backends unguarded. On a 12 GB card that is academic - S=199,728 is
unreachable - but on a larger GPU it is a real narrowing, and it should be a
deliberate choice rather than a side effect of tidying.

Goal 1 then reduces to:

| half | fix | effort |
| --- | --- | --- |
| Q/K, signed i32, reverts on **sage** upgrade | vendor `per_thread_int8_triton` with int64 offsets | moderate |
| V, unsigned u32, reverts on **ComfyUI** update | move the existing guard into `forward.py` | trivial |

Vendoring a V kernel becomes a Commit 3 *performance* question — whether owning
the fp8 V transpose beats `per_channel_fp8`'s unconditional padded copy — and is
no longer entangled with surviving updates. That is a much better place for it.

## 3. Facts

| claim | source | status |
| --- | --- | --- |
| Core H3 `Attention.forward` is 25 lines; q/k/v are views into one fused projection | [`model.py:156-181`] | **verified** |
| Core's HND transposes are views, zero copy | [`model.py:177-179`] | **verified** |
| An override cannot free the QKV — caller holds the locals | draft §1.2, confirmed by reading | **verified** |
| Fused QKV is 36% of the measured sampling transient, flat across C | arithmetic over `chunked_ref2v` §4.3 | **verified** |
| Q/K signed-i32 row limit S=99,864; V unsigned-u32 limit S=199,728 | `sage-int64-offset-patch`, SASS disassembly | **verified** |
| OOM arrives on this card around S=120-150k | prior experimentation | **measured** |
| C=22 → S=13,617 and C=73 → S=37,898; both satisfy `C % 17 == 5` | rung step 8,093 from §4.3 probe table | **verified** |
| `optimized_attention_override` is a supported core seam (`wrap_attn`) | [`attention.py:158`] | **verified** |
| Custom forward is bit-identical to core on both the rotary and no-rope paths | `tests/test_h3_attention_forward.py` | **verified** |
| Dropping the QKV views returns 100% of the fused buffer: 1.518 GB at C=73, 1.842 GB at C=90 | `benchmarks/measure_qkv_release.py`, §2.1 | **measured** |
| `memory_reserved` is identically 0 under `cudaMallocAsync`; the pool also withholds freed blocks from `mem_get_info` | §2.1 metric traps | **measured** |
| H3 reaches sage as HND (`skip_reshape=True`); `tensor_layout` is an indexing flag, not a copy switch | [`model.py:181`], `core.py`, §2.2 | **verified** |
| SM89 uses `qk_quant_gran="per_thread"` by default, so the Triton kernels are the live Q/K path | `core.py:618`, `747`, §2.3 | **verified** |
| `per_channel_fp8` calls `_fused.transpose_pad_permute_cuda` unconditionally; no sage API gives fp8 V without it | `quant.py:282`, §2.3 | **verified** |
| `sageattn_qk_int8_pv_fp16_cuda` (sm80/86) avoids `per_channel_fp8` entirely — an `_fused`-free option at +1 byte/element on V | `core.py:436-599`, §2.3 | **verified** |
| Peak-memory and latency delta of `sage_mem_eff` vs `sage` | — | **unknown — §4** |

## 4. Validation — 4 runs

Two profiles × two backends. Fixed seed, identical conditioning, identical
sampling settings.

| profile | S | backend | purpose |
| --- | ---: | --- | --- |
| C=22 | 13,617 | `sage` | baseline |
| C=22 | 13,617 | `sage_mem_eff` | correctness + small-shape regression |
| C=73 | 37,898 | `sage` | baseline at the shipping profile |
| C=73 | 37,898 | `sage_mem_eff` | **the run that decides everything** |

Collect per run: completion status, wall-clock, **peak reserved VRAM**, packed
sequence length, output checksum, decoded contact sheet, and a log line confirming
which backend path actually executed.

Two things this matrix deliberately does **not** do:

- **It does not judge generation quality at C=22.** 22 frames is 0.18× the Ref2VA
  trained minimum, so the output will be poor on *both* backends. C=22 is a memory
  and numerical-agreement probe. Quality is judged at C=73 only. Do not read a bad
  C=22 clip as a backend regression.
- **It does not test goal 1 at all.** S=37,898 is 38% of the Q/K limit and 19% of
  the V limit. Neither overflow is reachable from any run in this table. Goal 1
  needs §5.

`pytorch` is dropped from end-to-end entirely; it stays as the unit-level numerical
reference where it belongs.

## 5. Goal 1 is tested in seconds, not hours

The overflow test needs no sampling. The fault happens in **block 0's first
attention call**, before step 0 produces anything, so an oversized input either
crashes or survives within seconds of the model reaching the DiT. This is a cheap,
repeatable check that belongs early and can be run often — not a scheduled event.

### 5.1 Primary: the first legal grid rung past the limit

```
C = 209 frames  ->  S = 102,640     (limit is 99,864)
C = 192 frames  ->  S =  94,547     nearest legal rung below — the control
```

C=209 is the first `C % 17 == 5` rung that crosses, at 0.82 MP. Run both: 192 must
complete block 0, 209 must complete block 0 **only** with int64 quantization.

Two things make this the right test rather than a synthetic one:

- **It exercises the real integrated path** — actual fused QKV, actual strides,
  actual node wiring — instead of a hand-built tensor that might not reproduce the
  stride that matters.
- **209 frames is already a known quantity here.** `chunked_ref2v/PLAN.md` §4.3
  cites "probe run 1 at 209 frames" for the spill measurement. That run sat at
  S=102,640, past the limit, and did *not* crash — because the site-packages int64
  patch was already applied (2026-08-04). That is direct evidence the Q/K patch is
  load-bearing on this card, and it means C=209 has a known-good precedent to
  compare against.

Cost at block 0: ~4.11 GB for the fused QKV on top of the streaming DiT. Well
inside 12 GB, and it never reaches the memory profile of a real run because it
never samples.

### 5.2 Secondary: subprocess kernel test, no model

Keep a synthetic variant for the case where the model is not loadable, or for CI.
Same stride (21,504), no DiT resident:

```
S = 99,865   fused QKV 4.00 + Q,K int8 1.33 + V fp8 0.67 + out bf16 1.33  =  ~7.33 GB
```

Run it in a **subprocess** — the one piece of the draft's §11.3 worth preserving. A
defective kernel produces a Windows access violation that kills the interpreter
rather than raising, so the parent asserts normal exit code plus an expected stdout
marker rather than catching an exception.

### 5.3 Pass condition

Both rungs behave as predicted (192 fine, 209 fine only with int64), no contiguous
copy in the custom path, no access violation, and `_fused` never entered (§2.3).

## 6. Commit sequence

| # | commit | contents | gate |
| ---: | --- | --- | --- |
| 1 | Shared attention observer seam | `h3_attention/observer.py`, refactor `h3_probe/capture.py`, extend `tests/test_probe.py` | existing probe tests still pass; no backend behavior change |
| 2 | Custom forward skeleton **+ the measurement** | `h3_attention/{patch,forward}.py`, `tests/test_h3_attention_forward.py`; attention delegates to PyTorch | §2.1 ≥ 0.90 GB **and** §2.2 answered — **stop here if not** |
| 2a | Guard lives in this repo | `forward.py` applies the `contiguous()` check itself | **goal 1, V half closed for the custom path** |
| 3 | SM89 int64 Q/K quantizer + dense Sage | `h3_attention/{triton_i64,sage_mem_eff,stats}.py`, `tests/test_sage_mem_eff.py` | §5 C=209 rung passes; **goal 1, Q/K half closed** |
| 4 | Node wiring | `h3_attention/config.py`, `nodes_minimax_h3.py`, extend `tests/test_attention_backend.py` | §4 matrix; `sage` stays default |
| 5 | Re-evaluate the shipping profile | `chunked_ref2v/PLAN.md` §4.6 revision if C=90 holds end to end | §4 results |

Commit 1 before Commit 2 is not optional — a custom block forward bypasses the
module-global `optimized_attention` binding the probe currently hooks, so the
observer seam has to exist first or the probe goes dark. The draft gets this right.

Commit 5 is new, and it is the one that actually delivers goal 1. The draft never
removes the patch it set out to remove.

## 7. Cut from the draft

| cut | reason |
| --- | --- |
| SM80/86, SM90, SM120 paths (draft Stage 4, Commit 6) | one GPU; unbuildable and untestable here (§1.4) |
| §13 seven-case × three-backend matrix | infeasible; replaced by §4 |
| §8 `H3AttentionCall` sparse interface | speculative API for deferred work; let the real need shape it |
| §9 exact conditioning sinks | belongs with the sparse backend, not dense |
| §10 Morton | correctly deferred by the draft; clones conflict with the entire point |
| Commit 7 sparse groundwork | out of scope for both goals |

**Kept from Sol-Attn**, because both are cheap and useful independent of sparsity:

- **Override chaining.** `_set_h3_attention_backend` currently *replaces*
  `optimized_attention_override`. A first-refusal chain that delegates rejected
  calls to the previous override removes most node-ordering ambiguity.
- **Dispatch counters.** Per-path call counts and log-once diagnostics. These
  directly answer "did Sage silently fall back to PyTorch," which is otherwise only
  discoverable by being confused about a benchmark.

## 8. Licensing

KJNodes is **GPLv3**. The Sol-Attn snapshot inspected for the draft carried **no
license file**. This repository also has **no license file** and describes itself as
a private fork.

Clean-room from behavior and public interface. Do not copy source from either. The
mechanisms at stake — int64 offset promotion, in-place K smoothing, ordered
release of activations — are simple enough to implement independently from a
description, so the safe route costs almost nothing. Add a LICENSE to this repo
before it is ever published.

## 9. Open questions

§2.1, §2.2 and §2.3 are all answered above; what is left is downstream of them.

1. **Does the block-level release move the end-to-end transient?** §2.1 proves the
   allocator returns 1.518 GB at C=73 in isolation. A real run also has the
   streaming DiT, ComfyUI's cache and the attention workspace competing for the
   same pool. **This is the remaining risk**, and §4's four runs are what settle it.
2. **Does C=90 become the shipping profile?** If the release survives end to end,
   the 1.842 GB at C=90 is 2× what `chunked_ref2v` §4.6 buys with +2.7% compute,
   and the 73-vs-90 trade dissolves. That is a `chunked_ref2v/PLAN.md` revision,
   not an attention change — but it is the reason this work is worth doing.
3. **Is owning the fp8 V transpose worth it?** `per_channel_fp8` allocates a full
   padded `v_transposed_permutted` copy unconditionally (§2.3). A vendored kernel
   could fuse that, or fp16 V could sidestep it at +1 byte/element. Purely a
   Commit 3 performance question now that goal 1 no longer depends on it.
4. **Does the int64 Q/K quantizer match the stock one bit-for-bit below the
   overflow?** It must, or every existing measurement is invalidated. Cheap to
   check at S=4096, as the original patch was.

[`model.py:156-181`]: ../../../comfy/ldm/minimax/model.py
[`model.py:177-179`]: ../../../comfy/ldm/minimax/model.py
[`attention.py:158`]: ../../../comfy/ldm/modules/attention.py
