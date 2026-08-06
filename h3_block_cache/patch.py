"""Composable block-forward wrappers for H3 FirstBlockCache."""

from __future__ import annotations

import logging

from .config import FirstBlockCacheConfig
from .coordinator import FirstBlockCacheCoordinator

BLOCKS_ATTR = "diffusion_model.blocks"
LOG_PREFIX = "[H3 FirstBlockCache]"


class FirstBlockCachePatchError(RuntimeError):
    pass


def _key(index):
    return "%s.%d.forward" % (BLOCKS_ATTR, index)


def _make_wrapper(original, layer_index, last_layer, coordinator):
    def forward(x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        if layer_index == 0:
            # H3 mutates its residual stream in place, so retain the pre-block
            # input required by the canonical FirstBlockCache residual metric.
            original_input = x.detach().clone()
            out = original(
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )
            skipped = coordinator.after_head(
                original_input, out, transformer_options
            )
            if skipped:
                coordinator.apply_cached_tail(out, transformer_options)
            if last_layer == 0:
                if skipped:
                    coordinator.finish_skip(transformer_options)
                else:
                    coordinator.finish_compute(out, transformer_options)
            return out

        if coordinator.should_skip(transformer_options):
            out = x
            if layer_index == last_layer:
                coordinator.finish_skip(transformer_options)
            return out

        out = original(
            x,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options=transformer_options,
        )
        if layer_index == last_layer:
            coordinator.finish_compute(out, transformer_options)
        return out

    forward._h3_first_block_cache = True
    forward._h3_first_block_cache_layer = layer_index
    forward._h3_first_block_cache_original = original
    return forward


def install(model_patcher, config=None):
    config = config or FirstBlockCacheConfig()
    if not config.enabled:
        return None, 0
    try:
        blocks = model_patcher.get_model_object(BLOCKS_ATTR)
    except Exception as exc:
        raise FirstBlockCachePatchError("FirstBlockCache only applies to MiniMax H3") from exc
    if len(blocks) < 2:
        raise FirstBlockCachePatchError("FirstBlockCache requires at least two H3 blocks")

    existing = getattr(model_patcher, "object_patches", {})
    originals = []
    for index, block in enumerate(blocks):
        current = existing.get(_key(index))
        if current is not None and getattr(current, "_h3_first_block_cache", False):
            raise FirstBlockCachePatchError("FirstBlockCache is already installed")
        if current is not None and not getattr(current, "_h3_activation_memory", False):
            raise FirstBlockCachePatchError(
                "another patch owns %s; only the H3 activation-memory wrapper is composable"
                % _key(index)
            )
        originals.append(current if current is not None else block.forward)

    coordinator = FirstBlockCacheCoordinator(config)
    last = len(blocks) - 1
    for index, original in enumerate(originals):
        model_patcher.add_object_patch(
            _key(index),
            _make_wrapper(original, index, last, coordinator),
        )

    # Core builds its prefetch queue before block 0 can make the skip decision.
    # Disable dynamic prefetch in this cloned model so skipped blocks do not
    # pull weights into VRAM merely because the outer loop still visits them.
    options = model_patcher.model_options["transformer_options"] = (
        model_patcher.model_options.get("transformer_options", {}).copy()
    )
    options["prefetch_dynamic_vbars"] = False
    logging.info(
        "%s armed on %d blocks: threshold=%.4f warmup_steps=%d; dynamic prefetch disabled",
        LOG_PREFIX,
        len(blocks),
        config.threshold,
        config.warmup_steps,
    )
    return coordinator, len(blocks)
