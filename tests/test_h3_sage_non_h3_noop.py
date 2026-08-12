"""Unknown models must be exact no-ops before CUDA or format probing."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_sage_optimizations.apply import apply_plan  # noqa: E402
from h3_sage_optimizations.plan import (  # noqa: E402
    H3SageOptimizationPlan,
    MemoryRequest,
)


class NotH3:
    def __init__(self):
        self.model_options = {}
        self.clone_calls = 0

    def get_model_object(self, name):
        return object()

    def clone(self):
        self.clone_calls += 1
        raise AssertionError("non-H3 pass-through must not clone")


def main():
    model = NotH3()
    result = apply_plan(
        model,
        H3SageOptimizationPlan().with_memory(
            MemoryRequest()
        ),
    )
    if result is not model or model.clone_calls:
        raise AssertionError("non-H3 model was modified")
    print("  ok: non-H3 model is an exact pass-through")


if __name__ == "__main__":
    main()
