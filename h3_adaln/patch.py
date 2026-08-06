"""Reversible object patches for run-scoped AdaLN lookup."""

from __future__ import annotations

import logging

from .config import AdaLNPrecomputeConfig
from .provider import AdaLNProvider

BLOCKS_ATTR = "diffusion_model.blocks"


class AdaLNPatchError(RuntimeError):
    pass


def _key(index):
    return "%s.%d.adaln_proj.forward" % (BLOCKS_ATTR, index)


def install(model_patcher, config=None):
    config = config or AdaLNPrecomputeConfig()
    if not config.enabled:
        return None, 0
    try:
        model = model_patcher.get_model_object("diffusion_model")
        blocks = model_patcher.get_model_object(BLOCKS_ATTR)
    except Exception as exc:
        raise AdaLNPatchError("AdaLN precompute only applies to MiniMax H3") from exc
    if not blocks:
        raise AdaLNPatchError("MiniMax H3 has no blocks")

    existing = getattr(model_patcher, "object_patches", {})
    originals = []
    for index, block in enumerate(blocks):
        if not hasattr(block, "adaln_proj"):
            raise AdaLNPatchError("block %d has no adaln_proj" % index)
        current = existing.get(_key(index))
        if current is not None:
            if getattr(current, "_h3_adaln_lookup", False):
                raise AdaLNPatchError("AdaLN lookup is already installed")
            raise AdaLNPatchError("another patch owns %s" % _key(index))
        originals.append(block.adaln_proj.forward)

    provider = AdaLNProvider(model, blocks, originals, config)
    for index in range(len(blocks)):
        def forward(t_emb, _index=index):
            return provider.lookup(_index, t_emb)
        forward._h3_adaln_lookup = True
        forward._h3_adaln_layer = index
        model_patcher.add_object_patch(_key(index), forward)

    logging.info(
        "[H3 AdaLN] armed lookup patches on %d blocks (mode=%s, max_table=%.2f GiB)",
        len(blocks),
        config.mode,
        config.max_table_gib,
    )
    return provider, len(blocks)
