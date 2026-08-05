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

### 2.1 Does freeing the QKV actually reduce reserved memory? (Commit 2)

Custom forward installed, attention delegated to stock PyTorch. Drop the QKV
references before the attention call, read `GPU reserved memory` — the allocator
counter that is meaningful under `cudaMallocAsync`.

```
measure at C=73 (S=37,898), 50 blocks, sampling transient
expected eligible release: 1.518 GB
decision threshold:        >= 0.90 GB realized  ->  proceed to Commit 3
                           <  0.90 GB realized  ->  stop, write up, keep `sage`
```

No kernels written yet at this point. This is the cheapest possible answer to the
most expensive question in the plan.

### 2.2 Does the NHD entry point avoid a copy? (Commit 2, same sitting)

Per §1.3. Instrument SageAttention's NHD vs HND entry and compare allocations on
identical strided inputs. Cheap, and it determines whether the custom forward needs
to preserve NHD at all.

### 2.3 Does the custom V path enter `_fused.pyd`? (Commit 3, blocking for goal 1)

This is the requirement the draft never states, and goal 1 fails without it.

```
REQUIREMENT: the sage_mem_eff V path must never reach
             _fused.transpose_pad_permute_cuda
```

Gate C in the draft — "no forced contiguous copy appears in the custom path" — is
about *copies*, not about *which kernel runs*, and does not establish this.

**It cannot be verified by running a long sequence.** S=199,728 is unreachable on
12 GB. Verify by code path instead: instrument or symbol-trace the V branch and
assert `_fused` is never entered. Only once that holds can the in-repo
`comfy/ldm/modules/attention.py` guard be deleted — which is the actual deliverable
of goal 1.

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
| Whether freeing the QKV reduces *reserved* memory under `cudaMallocAsync` | — | **unknown — §2.1, blocking** |
| Whether Sage's NHD path avoids a copy the HND path makes | — | **unknown — §2.2** |
| Whether an FP8 V path can avoid `_fused.pyd` on SM89 | — | **unknown — §2.3, blocking for goal 1** |
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
| 3 | SM89 int64 quantizers + dense Sage | `h3_attention/{triton_i64,sage_mem_eff,stats}.py`, `tests/test_sage_mem_eff.py` | §2.3 holds; §5 subprocess test passes |
| 4 | Node wiring | `h3_attention/config.py`, `nodes_minimax_h3.py`, extend `tests/test_attention_backend.py` | §4 matrix; `sage` stays default |
| 5 | Retire the ComfyUI-tree guard | delete the `contiguous()` edit from `comfy/ldm/modules/attention.py`; document in README | **goal 1 closed** |

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

1. §2.1 — does reserved memory actually drop? **Blocks Commits 3-5.**
2. §2.2 — does Sage's NHD entry avoid a copy? Shapes the forward.
3. §2.3 — can FP8 V avoid `_fused.pyd` on SM89? **Blocks goal 1.**
4. If §2.1 succeeds, does C=90 become the shipping profile? That is a
   `chunked_ref2v/PLAN.md` §4.6 revision, not an attention change — but it is the
   reason this work is worth doing.

[`model.py:156-181`]: ../../../comfy/ldm/minimax/model.py
[`model.py:177-179`]: ../../../comfy/ldm/minimax/model.py
[`attention.py:158`]: ../../../comfy/ldm/modules/attention.py
