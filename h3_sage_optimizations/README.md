# H3 Optimizations integration

The `(Zi)` production nodes in this package are compatibility adapters over
the sibling `H3-Optimizations` custom-node pack. Sparse fallback routing,
chunked Kitchen and Sparse QKV, FP8 checkpoint handling, V-layout repair,
bounded MLP execution, and provider status have one implementation in that
dependency.

The adapters preserve the existing H3-Extended node ids, input order, defaults,
and disabled pass-through behavior:

- `MiniMaxH3SageMemoryOptimizerZi`
- `MiniMaxH3SparseSageAttentionZi`
- `MiniMaxH3SparseSageAttentionAdvancedZi`

The deprecated `MiniMaxH3HybridSparseAttentionZi` keeps its H3-Extended-owned
adaptive-density path. Its fixed-density path uses the same dependency plan as
the production adapters, so the nodes remain order-independent.

`epilogue_prototype` remains accepted for saved workflows but resolves to the
production ConvRot two-slice path. The experimental H3-Extended epilogue code
remains available through its dedicated activation-memory tooling; it is no
longer duplicated in the production Sage optimizer.
