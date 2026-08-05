"""Install the H3 block-attention forward as reversible object patches.

Only the main DiT blocks are patched. The token refiner shares core's `Attention`
class but runs on the text span alone, where the packed-sequence machinery buys
nothing - it keeps whatever backend the rest of the model is using.

`add_object_patch` is the reversible mechanism: it `set_attr`s on the patched
clone and restores on unpatch, the same way `MiniMaxH3SigmaShift` swaps
`model_sampling`. Nothing is mutated on the shared underlying model.
"""

import logging

import comfy.quant_ops

from .forward import make_forward

BLOCKS_ATTR = "diffusion_model.blocks"

# every attribute the forward reads off the attention module
REQUIRED_ATTRS = ("qkv_proj", "q_norm", "k_norm", "out_proj", "heads", "head_dim")


class H3PatchError(RuntimeError):
    """Configuration failed before sampling. Never a silent fallback."""


def _blocks(model_patcher):
    try:
        blocks = model_patcher.get_model_object(BLOCKS_ATTR)
    except Exception as exc:
        raise H3PatchError(
            "This model has no '%s'; the H3 attention backend only applies to "
            "MiniMax H3. Select 'sage', 'comfy' or 'pytorch'." % BLOCKS_ATTR) from exc
    if not len(blocks):
        raise H3PatchError("'%s' is empty; nothing to patch." % BLOCKS_ATTR)
    return blocks


def validate(model_patcher):
    """Check every assumption the forward makes. Returns the attention modules.

    Raises rather than degrading: a violated assumption means core drifted, and
    silently falling back would hide that until a benchmark looked wrong.
    """
    if not hasattr(comfy.quant_ops.ck, "rms_rope_split_half_"):
        raise H3PatchError(
            "comfy_kitchen does not expose rms_rope_split_half_; this ComfyUI "
            "build cannot run the H3 attention backend.")

    blocks = _blocks(model_patcher)
    modules = []
    for index, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is None:
            raise H3PatchError("block %d has no 'attn' module." % index)
        missing = [a for a in REQUIRED_ATTRS if not hasattr(attn, a)]
        if missing:
            raise H3PatchError(
                "block %d attention is missing %s - core's MiniMax Attention "
                "has changed shape and the H3 forward must be re-checked "
                "against it." % (index, ", ".join(missing)))
        expected = attn.heads * attn.head_dim * 3
        actual = getattr(attn.qkv_proj, "out_features", None)
        if actual is not None and actual != expected:
            raise H3PatchError(
                "block %d qkv_proj projects to %d, expected 3 * heads * head_dim "
                "= %d; the fused-QKV split this forward relies on no longer "
                "holds." % (index, actual, expected))
        modules.append(attn)
    return modules


def key_for(index):
    return "%s.%d.attn.forward" % (BLOCKS_ATTR, index)


def install(model_patcher, attention=None):
    """Patch every main block's attention forward. Idempotent.

    Raises on a conflicting patch rather than overwriting it - another node
    owning the same forward is a real incompatibility, and silently winning the
    race would make the result depend on node order.
    """
    modules = validate(model_patcher)

    existing = getattr(model_patcher, "object_patches", {})
    ours = [i for i in range(len(modules))
            if getattr(existing.get(key_for(i)), "_h3_attention", False)]
    if len(ours) == len(modules):
        logging.info("[H3 attention] block forwards already patched (%d)", len(modules))
        return 0

    conflicts = [key_for(i) for i in range(len(modules))
                 if key_for(i) in existing
                 and not getattr(existing[key_for(i)], "_h3_attention", False)]
    if conflicts:
        raise H3PatchError(
            "another patch already owns %s (and %d more); two nodes cannot both "
            "replace the H3 attention forward. Remove one, or select a backend "
            "that uses the attention override instead."
            % (conflicts[0], len(conflicts) - 1))

    for index, attn in enumerate(modules):
        model_patcher.add_object_patch(key_for(index), make_forward(attn, index, attention))

    logging.info("[H3 attention] patched %d block forwards (token refiner untouched)",
                 len(modules))
    return len(modules)
