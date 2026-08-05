"""Reversible H3 DiT-block forward patches for activation-memory execution."""

import logging

from .config import ActivationMemoryConfig
from .forward import make_forward

BLOCKS_ATTR = "diffusion_model.blocks"
REQUIRED_BLOCK_ATTRS = (
    "norm1",
    "norm2",
    "attn",
    "mlp",
    "adaln_proj",
)
REQUIRED_MLP_ATTRS = ("fc1", "fc2")


class H3ActivationPatchError(RuntimeError):
    pass


def _diffusion_model(model_patcher):
    try:
        return model_patcher.get_model_object("diffusion_model")
    except Exception:
        return None


def _blocks(model_patcher):
    try:
        blocks = model_patcher.get_model_object(BLOCKS_ATTR)
    except Exception as exc:
        raise H3ActivationPatchError(
            "This model has no '%s'; activation-memory execution only applies "
            "to MiniMax H3." % BLOCKS_ATTR
        ) from exc
    if not len(blocks):
        raise H3ActivationPatchError("'%s' is empty" % BLOCKS_ATTR)
    return blocks


def validate(model_patcher):
    diffusion_model = _diffusion_model(model_patcher)
    if diffusion_model is not None:
        try:
            from comfy.ldm.minimax.model import MiniMaxH3Model
        except ImportError:
            MiniMaxH3Model = None
        if MiniMaxH3Model is not None and not isinstance(
            diffusion_model, MiniMaxH3Model
        ):
            raise H3ActivationPatchError(
                "activation-memory execution can only patch MiniMaxH3Model; got %s"
                % type(diffusion_model).__name__
            )

    blocks = _blocks(model_patcher)
    for index, block in enumerate(blocks):
        missing = [name for name in REQUIRED_BLOCK_ATTRS if not hasattr(block, name)]
        if missing:
            raise H3ActivationPatchError(
                "block %d is missing %s; MiniMax H3 core changed and the "
                "activation-memory forward must be reviewed"
                % (index, ", ".join(missing))
            )
        missing_mlp = [
            name for name in REQUIRED_MLP_ATTRS if not hasattr(block.mlp, name)
        ]
        if missing_mlp:
            raise H3ActivationPatchError(
                "block %d MLP is missing %s"
                % (index, ", ".join(missing_mlp))
            )
    return blocks


def key_for(index):
    return "%s.%d.forward" % (BLOCKS_ATTR, index)


def install(model_patcher, config=None):
    """Patch every main H3 block. Reinstalling the same config is idempotent."""
    config = config or ActivationMemoryConfig()
    if not isinstance(config, ActivationMemoryConfig):
        raise TypeError("config must be ActivationMemoryConfig")
    blocks = validate(model_patcher)
    existing = getattr(model_patcher, "object_patches", {})

    foreign = [
        key_for(i)
        for i in range(len(blocks))
        if key_for(i) in existing
        and not getattr(existing[key_for(i)], "_h3_activation_memory", False)
    ]
    if foreign:
        raise H3ActivationPatchError(
            "another patch already owns %s (and %d more); activation-memory "
            "execution cannot safely replace a foreign DiT-block forward"
            % (foreign[0], len(foreign) - 1)
        )

    ours = [
        index
        for index in range(len(blocks))
        if getattr(existing.get(key_for(index)), "_h3_activation_memory", False)
    ]
    if ours:
        if len(ours) != len(blocks):
            raise H3ActivationPatchError(
                "only %d of %d H3 blocks carry the activation-memory patch; "
                "refusing a mixed state" % (len(ours), len(blocks))
            )
        installed = {
            getattr(existing[key_for(index)], "_h3_activation_config", None)
            for index in ours
        }
        if installed == {config.signature}:
            logging.info(
                "[H3 activation memory] block forwards already patched (%d)",
                len(blocks),
            )
            return 0
        raise H3ActivationPatchError(
            "H3 activation memory is already patched for %s; requested %s. "
            "Remove the earlier activation-memory node instead of relying on "
            "node order."
            % (sorted(str(item) for item in installed), config.signature)
        )

    for index, block in enumerate(blocks):
        model_patcher.add_object_patch(
            key_for(index),
            make_forward(
                block,
                index,
                config,
                original_forward=getattr(block, "forward"),
            ),
        )

    logging.info(
        "[H3 activation memory] patched %d blocks: mode=%s chunk_rows=%d "
        "strict=%s held_weights=%s",
        len(blocks),
        config.mode,
        config.chunk_rows,
        config.strict,
        config.prefer_held_weights,
    )
    return len(blocks)
