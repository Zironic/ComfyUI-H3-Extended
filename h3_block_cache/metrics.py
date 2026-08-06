import torch


def tensor_metrics(approx, exact):
    a = approx.detach().float()
    e = exact.detach().float()
    d = a - e
    denom = e.square().mean().sqrt().clamp_min(1e-12)
    return {
        "mae": float(d.abs().mean().cpu()),
        "max_abs": float(d.abs().max().cpu()),
        "relative_l2": float(d.square().mean().sqrt().div(denom).cpu()),
        "cosine": float(torch.nn.functional.cosine_similarity(
            a.reshape(1, -1), e.reshape(1, -1), dim=1).cpu()),
        "nan": int(torch.isnan(a).sum().cpu()),
        "inf": int(torch.isinf(a).sum().cpu()),
    }


def residual_summary(residual):
    r = residual.detach().float()
    return {
        "rms": float(r.square().mean().sqrt().cpu()),
        "mean_abs": float(r.abs().mean().cpu()),
        "max_abs": float(r.abs().max().cpu()),
    }
