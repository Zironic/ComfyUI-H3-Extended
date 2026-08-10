"""CPU contracts for order-independent H3 attention patch reconciliation."""

import os
import sys
from types import ModuleType, SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_sage_optimizations.patch import (  # noqa: E402
    H3SagePatchError,
    ORIGINAL_MARKER,
    OWNER_MARKER,
    SIGNATURE_MARKER,
    configure_backend,
)


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


class FakePatcher:
    def __init__(self, modules):
        self.modules = modules
        self.object_patches = {}
        self.model_options = {"transformer_options": {}}

    def add_object_patch(self, key, value):
        self.object_patches[key] = value


class Backend:
    requires_registered_sage = False

    def __init__(self, name):
        self.name = name

    @property
    def installation_signature(self):
        return ("backend", self.name)


class Projector:
    def __init__(self, name):
        self.name = name

    @property
    def installation_signature(self):
        return ("projector", self.name)


def install_fake_attention_modules(modules):
    package = ModuleType("h3_attention")
    forward_module = ModuleType("h3_attention.forward")
    patch_module = ModuleType("h3_attention.patch")

    def key_for(index):
        return "diffusion_model.blocks.%d.attn.forward" % index

    def validate(model_patcher):
        return model_patcher.modules

    def installation_signature(value):
        if value is None:
            return None
        signature = getattr(value, "installation_signature", None)
        return signature() if callable(signature) else signature

    def make_forward(module, layer_index, backend=None, projector=None):
        def forward(*args, **kwargs):
            return module.forward(*args, **kwargs)

        forward._h3_attention = True
        forward._h3_backend = getattr(backend, "name", None)
        forward._h3_projector = getattr(projector, "name", None)
        forward._h3_layer_index = int(layer_index)
        return forward

    forward_module.make_forward = make_forward
    patch_module.key_for = key_for
    patch_module.validate = validate
    patch_module.installation_signature = installation_signature
    package.forward = forward_module
    package.patch = patch_module
    sys.modules["h3_attention"] = package
    sys.modules["h3_attention.forward"] = forward_module
    sys.modules["h3_attention.patch"] = patch_module
    return key_for


def main():
    print("H3 Sage attention reconciliation")
    modules = [
        SimpleNamespace(forward=lambda value, index=index: (index, value))
        for index in range(3)
    ]
    key_for = install_fake_attention_modules(modules)
    patcher = FakePatcher(modules)

    dense = Backend("dense")
    dense_projector = Projector("dense_qkv")
    _, count = configure_backend(patcher, dense, projector=dense_projector)
    check(count == 3, "first resolved backend patches every attention block")
    originals = [
        getattr(patcher.object_patches[key_for(i)], ORIGINAL_MARKER)
        for i in range(3)
    ]
    check(
        all(getattr(patcher.object_patches[key_for(i)], OWNER_MARKER) for i in range(3)),
        "all forwards carry package ownership",
    )

    sparse = Backend("sparse")
    sparse_projector = Projector("sparse_qkv")
    _, count = configure_backend(patcher, sparse, projector=sparse_projector)
    check(count == 3, "a later node reconciles the complete backend transaction")
    check(
        [getattr(patcher.object_patches[key_for(i)], ORIGINAL_MARKER) for i in range(3)]
        == originals,
        "reconciliation preserves the real original forwards rather than nesting",
    )
    signatures = {
        getattr(patcher.object_patches[key_for(i)], SIGNATURE_MARKER)
        for i in range(3)
    }
    check(len(signatures) == 1, "every layer carries one identical resolved signature")

    _, count = configure_backend(patcher, sparse, projector=sparse_projector)
    check(count == 0, "reapplying the same resolved backend is idempotent")

    foreign = FakePatcher(modules)
    foreign.object_patches[key_for(0)] = lambda *args: None
    try:
        configure_backend(foreign, dense, projector=dense_projector)
    except H3SagePatchError as exc:
        check("another patch already owns" in str(exc), "foreign ownership fails before mutation")
    else:
        raise AssertionError("foreign attention ownership must fail")

    print("\nall H3 Sage attention reconciliation tests passed")


if __name__ == "__main__":
    main()
