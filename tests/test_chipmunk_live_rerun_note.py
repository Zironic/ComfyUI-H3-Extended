from pathlib import Path


def test_live_rerun_note_mentions_restart_and_activation_off():
    root = Path(__file__).resolve().parents[1]
    text = (root / "h3_chipmunk" / "README_LIVE_TEST.md").read_text(encoding="utf-8").lower()
    assert "restart comfyui" in text
    assert "activation memory disabled" in text


if __name__ == "__main__":
    test_live_rerun_note_mentions_restart_and_activation_off()
    print("Chipmunk live rerun note test passed")
