"""Ensure the two production nodes and compatibility adapter have UI docs."""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_DOCS = os.path.join(_ROOT, "web", "docs")

EXPECTED = {
    "MiniMaxH3SageMemoryOptimizerZi.md": "Format and carrier distinction",
    "MiniMaxH3SparseSageAttentionZi.md": "Video KV budget",
    "MiniMaxH3HybridSparseAttentionZi.md": "Deprecated Compatibility",
}


def main():
    for name, marker in EXPECTED.items():
        path = os.path.join(_DOCS, name)
        if not os.path.isfile(path):
            raise AssertionError("missing node documentation %s" % path)
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        if marker not in text:
            raise AssertionError("%s is missing %r" % (name, marker))
        print("  ok: %s" % name)
    print("\nall H3 Sage node documentation tests passed")


if __name__ == "__main__":
    main()
