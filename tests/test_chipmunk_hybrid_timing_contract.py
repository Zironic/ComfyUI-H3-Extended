"""Regression test for Chipmunk + Hybrid Sparse timing integration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_chipmunk_forward_uses_only_shared_timing_stages():
    source = (ROOT / "h3_chipmunk" / "forward.py").read_text(encoding="utf-8")
    assert 'timed_stage(transformer_options, "chipmunk_mlp")' not in source
    assert 'timed_stage(transformer_options, "total_dit_block")' in source
    assert 'timed_stage(transformer_options, "norm2_modulation")' in source
    assert 'timed_stage(transformer_options, "final_mlp_gate")' in source


if __name__ == "__main__":
    test_chipmunk_forward_uses_only_shared_timing_stages()
    print("Chipmunk hybrid timing contract test passed")
