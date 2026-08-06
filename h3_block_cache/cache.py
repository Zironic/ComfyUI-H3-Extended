from dataclasses import dataclass
import torch

@dataclass
class CacheEntry:
    residual: torch.Tensor | None = None
    last_refresh_step: int = -1
    reuse_count: int = 0

    @property
    def valid(self):
        return self.residual is not None

    def store(self, residual, step):
        self.residual = residual.detach().clone()
        self.last_refresh_step = int(step)
        self.reuse_count = 0

    def apply(self, x):
        if self.residual is None:
            raise RuntimeError("attempted to apply an invalid block cache")
        if tuple(self.residual.shape) != tuple(x.shape):
            raise RuntimeError(
                f"cache shape {tuple(self.residual.shape)} != input shape {tuple(x.shape)}")
        self.reuse_count += 1
        return x + self.residual.to(device=x.device, dtype=x.dtype)
