"""CPU tests for run-scoped H3 AdaLN trajectory precomputation."""

import os
import sys
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import h3_adaln.provider as provider_module  # noqa: E402
from h3_adaln.config import AdaLNPrecomputeConfig  # noqa: E402
from h3_adaln.provider import AdaLNProvider  # noqa: E402
from h3_runtime.context import RuntimeSnapshot  # noqa: E402


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = 4
        self.modalities = 3
        self.expand = 6
        self.apply_silu = False
        self.linear = torch.nn.Linear(2, self.expand * self.hidden * self.modalities)

    def forward(self, t_emb):
        x = self.linear(t_emb)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


class Block:
    def __init__(self):
        self.adaln_proj = Projection()


def main():
    blocks = [Block(), Block()]
    model = SimpleNamespace(
        sigma_shift_video=12.0,
        sigma_shift_audio=3.0,
        use_adaln_curves=False,
    )
    originals = [block.adaln_proj.forward for block in blocks]
    provider = AdaLNProvider(
        model,
        blocks,
        originals,
        AdaLNPrecomputeConfig(mode="on", max_table_gib=0.1, strict=True),
    )

    old_values = provider_module._step_t_values
    old_embed = provider_module._embed_t_values
    provider_module._step_t_values = lambda model, sigma, options, payload, layout: (float(sigma),)
    provider_module._embed_t_values = lambda model, values, device, dtype: torch.tensor(
        [[values[0], values[0] * 0.5]], dtype=torch.float32
    )
    try:
        layout = SimpleNamespace(segments=[(0, 1, "text")])
        snapshot = RuntimeSnapshot(
            request_id=0,
            step_index=0,
            total_steps=2,
            sigma=1.0,
            branch=(0,),
            layout=layout,
            layout_signature=(1,),
            compute_dtype=torch.float32,
            device=torch.device("cpu"),
        )
        options = {"sample_sigmas": torch.tensor([1.0, 0.5, 0.0])}
        provider.before_forward(snapshot, options, {})
        check(provider.tables is not None and provider.stats["steps"] == 2, "trajectory table is built once")
        embedding = torch.tensor([[1.0, 0.5]])
        want = originals[0](embedding)
        got = provider.lookup(0, embedding)
        check(all(torch.equal(a, b) for a, b in zip(got, want)), "step lookup is bit-identical to original projection")

        snapshot1 = RuntimeSnapshot(**{**snapshot.__dict__, "step_index": 1, "sigma": 0.5})
        provider.before_forward(snapshot1, options, {})
        got1 = provider.lookup(0, torch.tensor([[0.5, 0.25]]))
        want1 = originals[0](torch.tensor([[0.5, 0.25]]))
        check(all(torch.equal(a, b) for a, b in zip(got1, want1)), "cursor selects the next denoising step")
        check(provider.as_status()["hits"] == 2, "provider records table hits")
    finally:
        provider_module._step_t_values = old_values
        provider_module._embed_t_values = old_embed
    print("\nall AdaLN precompute tests passed")


if __name__ == "__main__":
    with torch.no_grad():
        main()
