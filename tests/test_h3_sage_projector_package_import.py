"""Regression test for custom-node package-relative sparse projector imports."""

from pathlib import Path
import sys
from types import ModuleType


def _package(name):
    module = ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def main():
    root = "comfy_custom_node_test"
    created = []
    try:
        for name in (
            root,
            root + ".h3_sage_optimizations",
            root + ".h3_sage_optimizations.qkv",
            root + ".h3_attention",
            root + ".h3_attention.hybrid",
        ):
            _package(name)
            created.append(name)

        formats_name = root + ".h3_sage_optimizations.qkv.formats"
        formats = ModuleType(formats_name)
        formats.describe_linear = lambda module: None
        formats.is_fused_weight_format_error = lambda exc: False
        sys.modules[formats_name] = formats
        created.append(formats_name)

        implementation_name = root + ".h3_attention.hybrid.fused_qkv"
        implementation = ModuleType(implementation_name)

        class FakeImplementation:
            installation_signature = ("fake",)

            def __init__(self, tensor_core=None):
                self.tensor_core = tensor_core

        implementation.FusedQKVProjector = FakeImplementation
        sys.modules[implementation_name] = implementation
        created.append(implementation_name)

        module_name = root + ".h3_sage_optimizations.qkv.projectors"
        module = ModuleType(module_name)
        module.__package__ = root + ".h3_sage_optimizations.qkv"
        module.__file__ = str(
            Path(__file__).resolve().parents[1]
            / "h3_sage_optimizations"
            / "qkv"
            / "projectors.py"
        )
        sys.modules[module_name] = module
        created.append(module_name)
        source = Path(module.__file__).read_text(encoding="utf-8")
        exec(compile(source, module.__file__, "exec"), module.__dict__)

        projector = module.SparseFusedQKVProjector(
            tensor_core="sentinel"
        )
        if not isinstance(projector._implementation, FakeImplementation):
            raise AssertionError(
                "Sparse projector did not import the sibling package implementation"
            )
        if projector._implementation.tensor_core != "sentinel":
            raise AssertionError("tensor_core was not forwarded")
        print("  ok: Sparse projector resolves sibling h3_attention package")
    finally:
        for name in reversed(created):
            sys.modules.pop(name, None)


if __name__ == "__main__":
    main()
