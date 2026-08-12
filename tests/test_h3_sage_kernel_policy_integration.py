"""Static guards against leaking bucket-2 kernels into production auto."""

from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PACKAGE = _HERE.parent / "h3_sage_optimizations"


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def main():
    providers = (_PACKAGE / "qkv" / "providers.py").read_text(encoding="utf-8")
    plan = (_PACKAGE / "plan.py").read_text(encoding="utf-8")
    policy = (_PACKAGE / "kernel_policy.py").read_text(encoding="utf-8")
    docs = (_PACKAGE / "KERNEL_POLICY.md").read_text(encoding="utf-8")

    auto_guard = providers.index("if request != FUSED_QKV_REQUIRED")
    format_probe = providers.index("if not inventory.qkv:")
    check(
        auto_guard < format_probe,
        "QKV auto exits to standard dispatch before custom-kernel preflight",
    )
    check(
        "policy.require_research(FUSED_QKV_TRITON)" in providers
        and "policy.require_research(MLP_EPILOGUE_TRITON)" in providers,
        "both custom GEMM prototypes require the research policy",
    )
    strict_block = plan.split("if bool(self.strict):", 1)[1].split(
        "@property", 1
    )[0]
    check(
        "fused_qkv" not in strict_block,
        "strict fallback semantics cannot unlock bucket-2 QKV",
    )
    check(
        'RESEARCH_KERNELS_ENV = "H3_SAGE_ENABLE_RESEARCH_KERNELS"' in policy,
        "the research gate is one explicit package-level policy",
    )
    check(
        "Bucket 1" in docs and "Bucket 2" in docs,
        "the two optimization buckets are documented as repository policy",
    )
    print("\nall H3 Sage kernel-policy integration guards passed")


if __name__ == "__main__":
    main()
