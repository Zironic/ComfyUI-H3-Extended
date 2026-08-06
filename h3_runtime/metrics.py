"""Bounded numerical and final-output metrics for acceleration experiments."""

from __future__ import annotations

import math
import torch


@torch.no_grad()
def tensor_error_metrics(candidate: torch.Tensor, reference: torch.Tensor, chunk_rows: int = 1024) -> dict:
    if candidate.shape != reference.shape:
        return {
            "shape_match": False,
            "candidate_shape": tuple(candidate.shape),
            "reference_shape": tuple(reference.shape),
        }
    a = candidate.reshape(-1, candidate.shape[-1])
    b = reference.reshape(-1, reference.shape[-1])
    abs_sum = sq_diff = sq_ref = dot = sq_a = 0.0
    max_abs = 0.0
    count = 0
    nonfinite = 0
    for start in range(0, a.shape[0], max(1, int(chunk_rows))):
        stop = min(a.shape[0], start + max(1, int(chunk_rows)))
        af = a[start:stop].float()
        bf = b[start:stop].float()
        diff = af - bf
        nonfinite += int((~torch.isfinite(af)).sum().item())
        abs_sum += float(diff.abs().sum().item())
        sq_diff += float(diff.square().sum().item())
        sq_ref += float(bf.square().sum().item())
        sq_a += float(af.square().sum().item())
        dot += float((af * bf).sum().item())
        max_abs = max(max_abs, float(diff.abs().max().item()))
        count += diff.numel()
    denom = math.sqrt(max(sq_ref, 1e-24))
    cosine = dot / max(math.sqrt(max(sq_a, 1e-24) * max(sq_ref, 1e-24)), 1e-24)
    return {
        "shape_match": True,
        "max_abs": max_abs,
        "mean_abs": abs_sum / max(1, count),
        "relative_l2": math.sqrt(max(sq_diff, 0.0)) / denom,
        "cosine": cosine,
        "nonfinite_count": nonfinite,
        "elements": count,
    }


def video_psnr(reference, candidate) -> dict:
    """Framewise PSNR for PIL images or uint8 tensors/arrays."""
    if reference is None or candidate is None or len(reference) != len(candidate):
        return {"frames": None, "psnr_db": None, "note": "frame count mismatch"}
    import numpy as np

    values = []
    for ref, cand in zip(reference, candidate):
        r = np.asarray(ref, dtype=np.float64)
        c = np.asarray(cand, dtype=np.float64)
        if r.shape != c.shape:
            return {"frames": None, "psnr_db": None, "note": "frame shape mismatch"}
        mse = float(((r - c) ** 2).mean())
        values.append(float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse))
    finite = [x for x in values if math.isfinite(x)]
    return {
        "frames": len(values),
        "psnr_db": None if not finite else sum(finite) / len(finite),
        "min_frame_psnr_db": None if not finite else min(finite),
        "identical_frames": sum(1 for x in values if not math.isfinite(x)),
    }


def audio_mse(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    if reference is None or candidate is None:
        return {"samples": None, "mse": None}
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "samples": None,
            "mse": None,
            "note": "shape mismatch %s vs %s" % (tuple(reference.shape), tuple(candidate.shape)),
        }
    ref = reference.detach().to("cpu", torch.float64)
    cand = candidate.detach().to("cpu", torch.float64)
    return {
        "samples": int(ref.numel()),
        "mse": float((ref - cand).square().mean().item()),
    }
