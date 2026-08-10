"""Reconcile the shared H3 attention forward for two composable nodes."""

from __future__ import annotations

import logging

OWNER_MARKER = "_h3_sage_optimizations_attention"
SIGNATURE_MARKER = "_h3_sage_optimizations_signature"
ORIGINAL_MARKER = "_h3_sage_optimizations_original"


class H3SagePatchError(RuntimeError):
    pass


def _pin_token_refiner_to_sage(model_patcher):
    from comfy.ldm.modules.attention import get_attention_function

    sage = get_attention_function("sage", default=None)
    if sage is None:
        raise H3SagePatchError(
            "prepared H3 Sage attention requires ComfyUI's registered 'sage' "
            "backend for the small token refiner"
        )
    sage_impl = getattr(sage, "__wrapped__", sage)

    def override(_original, *args, **kwargs):
        return sage_impl(*args, **kwargs)

    override._h3_backend = "sage"
    options = model_patcher.model_options["transformer_options"] = (
        model_patcher.model_options.get("transformer_options", {}).copy()
    )
    options["optimized_attention_override"] = override


def configure_backend(model_patcher, backend, projector=None):
    """Install or replace this package's complete H3 attention transaction.

    The first node may install dense Sage and a later node may resolve Sparse
    Sage, or vice versa. Replacing only forwards owned by this package makes the
    two node orders converge without accepting foreign patch ownership.
    """

    if backend is None:
        raise TypeError("backend must not be None")
    try:
        from ..h3_attention.forward import make_forward
        from ..h3_attention.patch import (
            installation_signature,
            key_for,
            validate,
        )
    except ImportError:
        from h3_attention.forward import make_forward
        from h3_attention.patch import (
            installation_signature,
            key_for,
            validate,
        )

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
        if getattr(existing.get(key_for(index)), OWNER_MARKER, False)
    ]
    if owned and len(owned) != len(modules):
        raise H3SagePatchError(
            "only %d of %d H3 attention blocks carry the Sage optimization "
            "patch; refusing a mixed state" % (len(owned), len(modules))
        )

    if owned:
        installed = {
            getattr(existing[key_for(index)], SIGNATURE_MARKER, None)
            for index in owned
        }
        if installed == {desired}:
            logging.info(
                "[H3 Sage optimizations] attention forwards already resolved (%d)",
                len(modules),
            )
            return backend, 0
        originals = [
            getattr(existing[key_for(index)], ORIGINAL_MARKER, None)
            for index in range(len(modules))
        ]
        if any(original is None for original in originals):
            raise H3SagePatchError(
                "the installed H3 Sage attention patch has no recoverable "
                "original forward"
            )
    else:
        conflicts = [
            key_for(index)
            for index in range(len(modules))
            if key_for(index) in existing
        ]
        if conflicts:
            raise H3SagePatchError(
                "another patch already owns %s; remove one H3 attention patch"
                % conflicts[0]
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
        model_patcher.model_options.get("transformer_options", {}).copy()
    )
    options["minimax_h3_attention_backend"] = getattr(
        backend, "name", type(backend).__name__
    )
    logging.info(
        "[H3 Sage optimizations] resolved %d attention forwards: backend=%s "
        "projector=%s",
        len(modules),
        getattr(backend, "name", type(backend).__name__),
        getattr(projector, "name", "standard_qkv"),
    )
    return backend, len(modules)
