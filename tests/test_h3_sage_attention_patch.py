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
            projector=sparse_projector,Bˆ
BˆÚXÚÊˆÛİ[OHËˆ˜H]\ˆ›ÙH™XÛÛ˜Ú[\ÈHÛÛ\]H˜XÚÙ[™˜[œØXİ[Ûˆ‹ˆ
BˆÚXÚÊˆÂˆÙ]]Šˆ]Ú\‹›Øš™XİÜ]Ú\ÖÚÙ^WÙ›ÜŠ[™^
WKˆÔ’QÒSSÓPT’ÑT‹ˆ
Bˆ›Üˆ[™^[ˆ˜[™ÙJÊBˆBˆOHÜšYÚ[˜[Ëˆœ™XÛÛ˜Ú[X][Ûˆ™\Ù\™\È™X[ÜšYÚ[˜[›ÜØ\™È‹ˆ
BˆÚYÛ˜]\™\ÈHÂˆÙ]]Šˆ]Ú\‹›Øš™XİÜ]Ú\ÖÚÙ^WÙ›ÜŠ[™^
WKˆÒQÓUT‘WÓPT’ÑT‹ˆ
Bˆ›Üˆ[™^[ˆ˜[™ÙJÊBˆBˆÚXÚÊˆ[ŠÚYÛ˜]\™\ÊHOHKˆ™]™\H^Y\ˆØ\œšY\ÈÛ™H™\ÛÛ™YÚYÛ˜]\™H‹ˆ
B‚ˆËÛİ[HÛÛ™šYİ\™WØ˜XÚÙ[™
ˆ]Ú\‹ˆÜ\œÙKˆ›Ú™XİÜ\Ü\œÙWÜ›Ú™XİÜ‹ˆ
BˆÚXÚÊˆÛİ[OHˆœ™X\Z[™ÈHØ[YH˜XÚÙ[™\ÈY[\İ[‹ˆ
B‚ˆ›Ü™ZYÛˆH˜ZÙT]Ú\Š
Bˆ›Ü™ZYÛ‹›Øš™XİÜ]Ú\ÖÚÙ^WÙ›ÜŠ
WHH
ˆ[X™H
˜\™ÜÎˆ›Û™Bˆ
BˆN‚ˆÛÛ™šYİ\™WØ˜XÚÙ[™
ˆ›Ü™ZYÛ‹ˆ[œÙKˆ›Ú™XİÜY[œÙWÜ›Ú™XİÜ‹ˆ
Bˆ^Ù\ÔØYÙT]Ú\œ›Üˆ\È^Î‚ˆÚXÚÊˆ˜[›İ\ˆ]Ú[™XYHİÛœÈˆ[ˆİŠ^ÊKˆ™›Ü™ZYÛˆİÛ™\œÚ\˜Z[È™Y›Ü™H]]][Ûˆ‹ˆ
Bˆ[ÙN‚ˆ˜Z\ÙH\ÜÙ\[Û‘\œ›ÜŠˆ™›Ü™ZYÛˆ][[ÛˆİÛ™\œÚ\]\İ˜Z[‚ˆ
B‚ˆš[
—˜[ÈØYÙH][[Ûˆ™XÛÛ˜Ú[X][Ûˆ\İÈ\ÜÙYŠB‚‚šYˆ×Û˜[YW×ÈOH—×ÛXZ[—×È‚ˆXZ[Š
B