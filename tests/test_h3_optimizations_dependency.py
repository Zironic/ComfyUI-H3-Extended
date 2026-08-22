"""Dependency and compatibility contracts for production H3 optimizations."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest


PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations_dependency import (  # noqa: E402
    PACKAGE,
    SIBLING_ROOT,
    dependency_module,
)
from h3_sage_optimizations.plan import (  # noqa: E402
    ATTENTION_EXISTING,
    FUSED_QKV_REQUIRED,
    H3SageOptimizationPlan,
    MLP_MEMORY_EPILOGUE,
    MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
    MemoryRequest,
    SparseRequest,
)


class DependencyTests(unittest.TestCase):
    def test_sibling_checkout_is_the_canonical_package(self):
        package_file = Path(PACKAGE.__file__).resolve()
        self.assertTrue(package_file.is_relative_to(SIBLING_ROOT.resolve()))
        self.assertIs(
            H3SageOptimizationPlan,
            dependency_module("plan").H3OptimizationPlan,
        )

    def test_custom_node_entrypoint_reuses_the_canonical_package(self):
        spec = importlib.util.spec_from_file_location(
            "h3_optimizations_dependency_entrypoint_test",
            SIBLING_ROOT / "__init__.py",
            submodule_search_locations=[str(SIBLING_ROOT)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
        self.assertIs(
            module.H3OptimizationsExtension,
            dependency_module("nodes").H3OptimizationsExtension,
        )

    def test_legacy_requests_share_the_dependency_plan(self):
        memory = MemoryRequest(
            attention=ATTENTION_EXISTING,
            fused_qkv=FUSED_QKV_REQUIRED,
            mlp_memory=MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
            chunk_rows=2048,
            prefer_held_weights=False,
            mlp_strict=True,
        )
        sparse = SparseRequest(
            video_budget=0.3,
            early_steps=2,
            early_kv=0.5,
            late_steps=2,
            late_kv=0.5,
        )
        plan = H3SageOptimizationPlan().with_memory(memory).with_sparse(sparse)
        self.assertIsInstance(plan, dependency_module("plan").H3OptimizationPlan)
        self.assertTrue(plan.sparse.advanced_schedule)
        self.assertFalse(plan.memory.prefer_held_weights)
        self.assertEqual(MLP_MEMORY_EPILOGUE, "epilogue_prototype")

    def test_duplicate_implementation_files_are_removed(self):
        duplicate_paths = (
            "attention_forward.py",
            "dense_fused_qkv.py",
            "dense_resolver.py",
            "environment.py",
            "patch.py",
            "qkv/providers.py",
        )
        package = PACK / "h3_sage_optimizations"
        for relative in duplicate_paths:
            with self.subTest(relative=relative):
                self.assertFalse((package / relative).exists())


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
