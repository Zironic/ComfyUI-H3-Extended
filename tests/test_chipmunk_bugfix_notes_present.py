from pathlib import Path


def test_chipmunk_bugfix_notes_present():
    root = Path(__file__).resolve().parents[1]
    text = (root / "h3_chipmunk" / "BUGFIX_NOTES.md").read_text(encoding="utf-8")
    assert "chipmunk_mlp" in text
    assert "runtime" in text.lower()


if __name__ == "__main__":
    test_chipmunk_bugfix_notes_present()
    print("Chipmunk bugfix notes test passed")
