"""Install H3-owned block-attention forwards as reversible object patches."""

import logging

import comfy.quant_ops

from .forward import make_forward

BLOCKS_ATTR = "diffusion_model.blocks"
REQUIRED_ATTRS = ("qkv_proj", "q_norm", "k_norm", "out_proj", "heads", "head_dim")


class H3PatchError(RuntimeError):
    """Configuration failed before sampling."""


def _diffusion_model(model_patcher):
    try:
        return model_patcher.get_model_object("diffusion_model")
    except Exception:
        return None


def _blocks(model_patcher):
    try:
        blocks = model_patcher.get_model_object(BLOCKS_ATTR)
    except Exception as exc:
        raise H3PatchError(
            "This model has no '%s'; efficient Sage only applies to MiniMax H3."
            % BLOCKS_ATTR) from exc
    if not len(blocks):
        raise H3PatchError("'%s' is empty; nothing to patch." % BLOCKS_ATTR)
    return blocks


def validate(model_patcher):
    """Validate model identity and every attribute read by the copied forward."""
    if not hasattr(comfy.quant_ops.ck, "rms_rope_split_half_"):
        raise H3PatchError(
            "comfy_kitchen does not expose rms_rope_split_half_; this ComfyUI "
            "build cannot run the H3-owned attention forward.")

    diffusion_model = _diffusion_model(model_patcher)
    if diffusion_model is not None:
        try:
            from comfy.ldm.minimax.model import MiniMaxH3Model
        except ImportError:
            MiniMaxH3Model = None
        if MiniMaxH3Model is not None and not isinstance(diffusion_model, MiniMaxH3Model):
            raise H3PatchError(
                "efficient Sage can only patch MiniMaxH3Model; got %s"
                % type(diffusion_model).__name__)

    blocks = _blocks(model_patcher)
    modules = []
    for index, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is None:
            raise H3PatchError("block %d has no 'attn' module." % index)
        missing = [name for name in REQUIRED_ATTRS if not hasattr(attn, name)]
        if missing:
            raise H3PatchError(
                "block %d attention is missing %s; core MiniMax Attention drifted"
                % (index, ", ".join(missing)))
        expected = attn.heads * attn.head_dim * 3
        actual = getattr(attn.qkv_proj, "out_features", None)
        if actual is not None and actual != expected:
            raise H3PatchError(
                "block %d qkv_proj projects to %d, expected %d"
                % (index, actual, expected))
        modules.append(attn)
    return modules


def key_for(index):
    return "%s.%d.attn.forward" % (BLOCKS_ATTR, index)


def installation_signature(value):
    if value is None:
        return None
    signature = getattr(value, "installation_signature", None)
    if callable(signature):
        signature = signature()
    if signature is not None:
        return signature
    if callable(value):
        function = getattr(value, "__func__", value)
        return (
            getattr(function, "__module__", type(function).__module__),
            getattr(function, "__qualname__", type(function).__qualname__),
            id(function),
        )
    return (
        type(value).__module__,
        type(value).__qualname__,
        getattr(value, "name", None),
    )


def install(model_patcher, backend=None, attention=None, projector=None):
    """Patch every main block. Idempotent; foreign ownership is an error."""
    modules = validate(model_patcher)
    existing = getattr(model_patcher, "object_patches", {})

    desired_backend = getattr(backend, "name", None)
    desired_projector = getattr(projector, "name", None)
    desired_signature = (
        installation_signature(backend),
        installation_signature(projector),
        installation_signature(attention),
    )
    ours = [
        index for index in range(len(modules))
        if getattr(existing.get(key_for(index)), "_h3_attention", False)
    ]
    if len(ours) == len(modules):
        installed_backends = {
            getattr(existing[key_for(index)], "_h3_backend", None)
            for index in ours
        }
        installed_projectors = {
            getattr(existing[key_for(index)], "_h3_projector", None)
            for index in ours
        }
        installed_signatures = {
            getattr(existing[key_for(index)], "_h3_installation_signature", None)
            for index in ours
        }
        if (installed_backends == {desired_backend}
                and installed_projectors == {desired_projector}
                and installed_signatures == {desired_signature}):
            logging.info("[H3 attention] block forwards already patched (%d)", len(modules))
            return 0
        raise H3PatchError(
            "H3 attention is already patched for %s; requested backend is %s"
            % (sorted(str(x) for x in installed_backends), desired_backend))
    if ours:
        raise H3PatchError(
            "only %d of %d H3 attention blocks carry this patch; refusing a mixed state"
            % (len(ours), len(modules)))

    conflicts = [
        key_for(index) for index in range(len(modules))
        if key_for(index) in existing
        and not getattr(existing[key_for(index)], "_h3_attention", False)
    ]
    if conflicts:
        raise H3PatchError(
            "another patch already owns %s (and %d more); remove one attention-forward patch"
            % (conflicts[0], len(conflicts) - 1))

    for index, attn in enumerate(modules):
        forward = make_forward(
            attn,
            index,
            backend=backend,
            attention=attention,
            projector=projector,
        )
        forward._h3_installation_signature = desired_signature
        model_patcher.add_object_patch(
            key_for(index),
            forward,
        )

    logging.info(
        "[H3 attention] patched %d main-block forwards with %s (token refiner untouched)",
        len(modules),
        getattr(backend, "name", "legacy attention"),
    )
    return len(modules)
