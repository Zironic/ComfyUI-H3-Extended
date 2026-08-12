"""Reconcile one package-owned H3 attention forward across both nodes."""

from __future__ import annotations

import logging

from .model import get_h3_blocks, is_minimax_h3

OWNER_MARKER = "_h3_sage_optimizations_attention"
SIGNATURE_MARKER = "_h3_sage_optimizations_signature"
ORIGINAL_MARKER = "_h3_sage_optimizations_original"
BLOCKS_ATTR = "diffusion_model.blocks"
REQUIRED_ATTRS = (
    "qkv_proj",
    "q_norm",
    "k_norm",
    "out_proj",
    "heads",
    "head_dim",
)


class H3SagePatchError(RuntimeError):
    pass


def key_for(index):
    return "%s.%d.attn.forward" % (BLOCKS_ATTR, int(index))


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
            getattr(
                function,
                "__qualname__",
                type(function).__qualname__,
            ),
            id(function),
        )
    return (
        type(value).__module__,
        type(value).__qualname__,
        getattr(value, "name", None),
    )


def validate(model_patcher):
    if not is_minimax_h3(model_patcher):
        raise H3SagePatchError(
            "H3 Sage attention can only patch MiniMaxH3Model"
        )
    if not hasattr(comfy_quant_ops(), "rms_rope_split_half_"):
        raise H3SagePatchError(
            "comfy_kitchen does not expose rms_rope_split_half_"
        )
    blocks = get_h3_blocks(model_patcher)
    if not blocks:
        raise H3SagePatchError("MiniMax H3 has no main blocks")
    modules = []
    for index, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is None:
            raise H3SagePatchError(
                "H3 block %d has no attention module" % index
            )
        missing = [
            name for name in REQUIRED_ATTRS if not hasattr(attn, name)
        ]
        if missing:
            raise H3SagePatchError(
                "H3 block %d attention is missing %s"
                % (index, ", ".join(missing))
            )
        expected = int(attn.heads) * int(attn.head_dim) * 3
        actual = getattr(attn.qkv_proj, "out_features", None)
        if actual is not None and int(actual) != expected:
            raise H3SagePatchError(
                "H3 block %d qkv_proj projects to %d, expected %d"
                % (index, actual, expected)
            )
        modules.append(attn)
    return tuple(modules)


def comfy_quant_ops():
    import comfy.quant_ops

    return comfy.quant_ops.ck


def _pin_token_refiner_to_sage(model_patcher):
    from comfy.ldm.modules.attention import get_attention_function

    sage = get_attention_function("sage", default=None)
    if sage is None:
        raise H3SagePatchError(
            "prepared H3 Sage attention requires ComfyUI's registered "
            "'sage' backend for the small token refiner"
        )
    sage_impl = getattr(sage, "__wrapped__", sage)

    def override(_original, *args, **kwargs):
        return sage_impl(*args, **kwargs)

    override._h3_backend = "sage"
    options = model_patcher.model_options["transformer_options"] = (
        model_patcher.model_options.get(
            "transformer_options", {}
        ).copy()
    )
    options["optimized_attention_override"] = override


def configure_backend(model_patcher, backend, projector=None):
    """Install or replace the package-owned H3 attention transaction."""

    if backend is None:
        raise TypeError("backend must not be None")
    from .attention_forward import make_forward
    if bool(getattr(backend, "requires_registered_sage", True)):
        _pin_token_refiner_to_sage(model_patcher)

    modules = validate(model_patcher)
    existing = getattr(model_patcher, "object_patches", {})
    desired = (
        installation_signature(backend),
        installation_signature(projector),
    )
    owned = [
        index
        for index in range(len(modules))
        if getattr(
            existing.get(key_for(index)),
            OWNER_MARKER,
            False,
        )
    ]
    if owned and len(owned) != len(modules):
        raise H3SagePatchError(
            "only %d of %d H3 attention blocks carry the Sage "
            "optimization patch; refusing a mixed state"
            % (len(owned), len(modules))
        )

    if owned:
        installed = {
            getattr(
                existing[key_for(index)],
                SIGNATURE_MARKER,
                None,
            )
            for index in owned
        }
        if installed == {desired}:
            logging.info(
                "[H3 Sage optimizations] attention forwards already "
                "resolved (%d)",
                len(modules),
            )
            return backend, 0
        originals = [
            getattr(
                existing[key_for(index)],
                ORIGINAL_MARKER,
                None,
            )
            for index in range(len(modules))
        ]
        if any(original is None for original in originals):
            raise H3SagePatchError(
                "installed H3 Sage attention patch has no "
                "recoverable original forward"
            )
    else:
        conflicts = [
            key_for(index)
            for index in range(len(modules))
            if key_for(index) in existing
        ]
        if conflicts:
            raise H3SagePatchError(
                "another patch already owns %s; remove one H3 "
                "attention patch" % conflicts[0]
            )
        originals = [module.forward for module in modules]

    for index, module in enumerate(modules):
        forward = make_forward(
            module,
            index,
            backend=backend,
            projector=projector,
        )
        setattr(forward, OWNER_MARKER, True)
        setattr(forward, SIGNATURE_MARKER, desired)
        setattr(forward, ORIGINAL_MARKER, originals[index])
        model_patcher.add_object_patch(key_for(index), forward)

    options = model_patcher.model_options["transformer_options"] = (
        model_patcher.model_options.get(
            "transformer_options", {}
        ).copy()
    )
    options["minimax_h3_attention_backend"] = getattr(
        backend, "name", type(backend).__name__
    )
    logging.info(
        "[H3 Sage optimizations] resolved %d attention forwards: "
        "backend=%s projector=%s",
        len(modules),
        getattr(backend, "name", type(backend).__name__),
        getattr(projector, "name", "standard_qkv"),
    )
    return backend, len(modules)
