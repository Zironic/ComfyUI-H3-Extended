"""CPU contracts for order-independent H3 attention reconciliation."""

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_sage_optimizations.patch import (  # noqa: E402
    H3SagePatchError,
    ORIGINAL_MARKER,
    OWNER_MARKER,
    SIGNATURE_MARKER,
    configure_backend,
    key_for,
)


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


class FakePatcher:
    def __init__(self):
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


def attention_module(index):
    return SimpleNamespace(
        forward=lambda value, index=index: (index, value),
        qkv_proj=SimpleNamespace(out_features=12),
        q_norm=object(),
        k_norm=object(),
        out_proj=object(),
        heads=1,
        head_dim=4,
    )


def fake_forward_module():
    module = ModuleType(
        "h3_sage_optimizations.attention_forward"
    )

    def make_forward(
        attention,
        layer_index,
        backend=None,
        projector=None,
    ):
        def forward(*args, **kwargs):
            return attention.forward(*args, **kwargs)

        forward._h3_attention = True
        forward._h3_backend = getattr(
            backend, "name", None
        )
        forward._h3_projector = getattr(
            projector, "name", None
        )
        forward._h3_layer_index = int(layer_index)
        return forward

    module.make_forward = make_forward
    return module


def main():
    print("H3 Sage attention reconciliation")
    attentions = [attention_module(index) for index in range(3)]
    blocks = [
        SimpleNamespace(attn=attention)
        for attention in attentions
    ]
    patcher = FakePatcher()

    patches = (
        mock.patch(
            "h3_sage_optimizations.patch.is_minimax_h3",
            return_value=True,
        ),
        mock.patch(
            "h3_sage_optimizations.patch.get_h3_blocks",
            return_value=blocks,
        ),
        mock.patch(
            "h3_sage_optimizations.patch.comfy_quant_ops",
            return_value=SimpleNamespace(
                rms_rope_split_half_=object()
            ),
        ),
        mock.patch.dict(
            sys.modules,
            {
                "h3_sage_optimizations.attention_forward":
                    fake_forward_module()
            },
        ),
    )
    with patches[0], patches[1], patches[2], patches[3]:
        dense = Backend("dense")
        dense_projector = Projector("dense_qkv")
        _, count = configure_backend(
            patcher, dense, projector=dense_projector
        )
        check(
            count == 3,
            "first resolved backend patches every attention block",
        )
        originals = [
            getattr(
                patcher.object_patches[key_for(index)],
                ORIGINAL_MARKER,
            )
            for index in range(3)
        ]
        check(
            all(
                getattr(
                    patcher.object_patches[key_for(index)],
                    OWNER_MARKER,
                )
                for index in range(3)
            ),
            "all forwards carry package ownership",
        )

        sparse = Backend("sparse")
        sparse_projector = Projector("sparse_qkv")
        _, count = configure_backend(
            patcher,
            sparse,
            projector=sparse_projector,
        )
        check(
            count == 3,
            "a later node reconciles the complete backend transaction",
        )
        check(
            [
                getattr(
                    patcher.object_patches[key_for(index)],
                    ORIGINAL_MARKER,
                )
                for index in range(3)
            ]
            == originals,
            "reconciliation preserves real original forwards",
        )
        signatures = {
            getattr(
                patcher.object_patches[key_for(index)],
                SIGNATURE_MARKER,
            )
            for index in range(3)
        }
        check(
            len(signatures) == 1,
            "every layer carries one resolved signature",
        )

        _, count = configure_backend(
            patcher,
            sparse,
            projector=sparse_projector,
        )
        check(
            count == 0,
            "reapplying the same backend is idempotent",
        )

        foreign = FakePatcher()
        foreign.object_patches[key_for(0)] = (
            lambda *args: None
        )
        try:
            configure_backend(
                foreign,
                dense,
                projector=dense_projector,
            )
        except H3SagePatchError as exc:
            check(
                "another patch already owns" in str(exc),
                "foreign ownership fails before mutation",
            )
        else:
            raise AssertionError(
                "foreign attention ownership must fail"
            )

    print("\nall H3 Sage attention reconciliation tests passed")


if __name__ == "__main__":
    main()
