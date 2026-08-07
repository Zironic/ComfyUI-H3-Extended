"""Seam and landing measurements for the bridge experiment.

The bridge does not need to reproduce the held-out interval pixel for pixel - it
is a stochastic generator and MAE against ground truth is not a success
criterion. What the arms are graded on is the two seams, and on one signed
scalar: which way the ending is moving.

That signed quantity is the point of the counterfactual arm. "B landed near C3"
proves little, because C3 is C1's natural future and arm A drifts there on its
own. "C's ending moved toward the counterfactual instead" cannot happen unless
the model is reading <Video 2> as a future constraint.
"""

import torch


def _gray(frames):
    """[N, H, W, C] in 0..1 -> [N, H, W] luma."""
    frames = frames[..., :3].to(torch.float32)
    weights = torch.tensor([0.299, 0.587, 0.114], device=frames.device)
    return (frames * weights).sum(-1)


def mae(a, b):
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().mean())


def horizontal_shift(a, b, max_shift=24):
    """Signed horizontal translation from frame `a` to frame `b`, in pixels.

    A brute-force 1-D search rather than optical flow: the discriminating
    quantity only has to be signed, stable and cheap. Positive means content
    moved to the right.
    """
    a, b = _gray(a[None] if a.ndim == 3 else a), _gray(b[None] if b.ndim == 3 else b)
    a, b = a[0], b[0]
    width = a.shape[-1]
    max_shift = int(min(max_shift, width // 4))
    best, best_err = 0, None
    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            err = (a - b).abs().mean()
        elif shift > 0:
            err = (a[:, :-shift] - b[:, shift:]).abs().mean()
        else:
            err = (a[:, -shift:] - b[:, :shift]).abs().mean()
        err = float(err)
        if best_err is None or err < best_err:
            best, best_err = shift, err
    return best


def mean_horizontal_velocity(frames, max_shift=24, limit=None):
    """Average signed per-frame horizontal motion over a clip."""
    n = int(frames.shape[0])
    if n < 2:
        return 0.0
    if limit is not None:
        frames = frames[-int(limit):] if limit > 0 else frames
        n = int(frames.shape[0])
    shifts = [horizontal_shift(frames[i], frames[i + 1], max_shift)
              for i in range(n - 1)]
    return sum(shifts) / max(1, len(shifts))


def _consecutive_deltas(frames):
    g = _gray(frames)
    return (g[1:] - g[:-1]).abs().mean(dim=(-1, -2))


def seam_ratio(before, after, window=8):
    """How much the frame-to-frame delta spikes at a join.

    `before` ends where `after` begins. 1.0 means the join is as smooth as the
    surrounding motion; large means a visible jump. Comparing against the local
    median rather than a fixed threshold keeps the number meaningful for both a
    static shot and a fast pan.
    """
    if before.shape[0] < 2 or after.shape[0] < 2:
        return None
    tail = before[-window:]
    head = after[:window]
    across = float(_consecutive_deltas(torch.stack([tail[-1], head[0]]))[0])
    local = torch.cat([_consecutive_deltas(tail), _consecutive_deltas(head)])
    baseline = float(local.median())
    if baseline <= 1e-8:
        return None
    return across / baseline


def audio_latent_discontinuity(a_latent, b_latent):
    """Normalized L2 between the last column of `a` and the first of `b`.

    Audio latents, not decoded waveform: the audio VAE decode costs more than
    this measurement is worth on a 12 GB card, and a latent-space jump at the
    join is already enough to rank the arms.
    """
    if a_latent is None or b_latent is None:
        return None
    a = a_latent.to(torch.float32).reshape(-1, a_latent.shape[-1])[:, -1]
    b = b_latent.to(torch.float32).reshape(-1, b_latent.shape[-1])[:, 0]
    scale = float(torch.linalg.vector_norm(a)) + float(torch.linalg.vector_norm(b))
    if scale <= 1e-8:
        return None
    return float(torch.linalg.vector_norm(a - b)) * 2.0 / scale


def collect(*, generated, left_context, right_natural, right_counterfactual=None,
            ground_truth=None, generated_audio=None, right_natural_audio=None,
            window=8, max_shift=24):
    """Every number for one arm.

    `right_natural` is always supplied, including for arms that were not shown
    it - the landing has to be measured against the same yardstick in every arm
    or the comparison is meaningless.
    """
    out = {}

    out["left_seam_mae"] = mae(left_context[-1], generated[0])
    out["left_seam_ratio"] = seam_ratio(left_context, generated, window)

    out["right_seam_mae_natural"] = mae(generated[-1], right_natural[0])
    out["right_seam_ratio_natural"] = seam_ratio(generated, right_natural, window)

    if right_counterfactual is not None:
        out["right_seam_mae_counterfactual"] = mae(
            generated[-1], right_counterfactual[0])

    # The signed discriminator. Compare the ending's direction of travel with
    # each candidate future's opening direction.
    ending_dx = mean_horizontal_velocity(generated[-window:], max_shift)
    out["ending_dx"] = ending_dx
    out["natural_dx"] = mean_horizontal_velocity(right_natural[:window], max_shift)
    out["dx_error_natural"] = abs(ending_dx - out["natural_dx"])
    if right_counterfactual is not None:
        out["counterfactual_dx"] = mean_horizontal_velocity(
            right_counterfactual[:window], max_shift)
        out["dx_error_counterfactual"] = abs(ending_dx - out["counterfactual_dx"])

    out["right_seam_audio_natural"] = audio_latent_discontinuity(
        generated_audio, right_natural_audio)

    if ground_truth is not None:
        # Secondary. Reported, never optimized against.
        out["ground_truth_mae"] = mae(generated, ground_truth[:generated.shape[0]])

    return out


def decisive_comparison(results):
    """The B-vs-C readout, in one dict.

    Positive `dx_swing` means C's ending moved toward the counterfactual future
    relative to B's - the signature of a real future constraint. Near zero means
    the second reference is ordinary conditioning and the middle-splice branch
    of the regeneration design does not have a mechanism behind it.
    """
    b = results.get("B_natural", {}).get("metrics")
    c = results.get("C_counterfactual", {}).get("metrics")
    if not b or not c:
        return None
    if "counterfactual_dx" not in c or "natural_dx" not in c:
        return None

    spread = abs(c["counterfactual_dx"] - c["natural_dx"])
    swing = c["ending_dx"] - b["ending_dx"]
    toward = c["counterfactual_dx"] - c["natural_dx"]
    normalized = (swing / spread) if spread > 1e-6 else None
    if normalized is not None and toward < 0:
        normalized = -normalized

    return {
        "b_ending_dx": b["ending_dx"],
        "c_ending_dx": c["ending_dx"],
        "natural_dx": c["natural_dx"],
        "counterfactual_dx": c["counterfactual_dx"],
        "target_spread_px": spread,
        "dx_swing_px": swing,
        "dx_swing_normalized": normalized,
        "left_seam_preserved": (
            None if b.get("left_seam_ratio") is None
            or c.get("left_seam_ratio") is None
            else abs(c["left_seam_ratio"] - b["left_seam_ratio"]) < 0.5),
    }
