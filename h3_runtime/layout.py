"""Packed-layout helpers shared by Sol-Attn and quality diagnostics."""

from __future__ import annotations

SINK_PREFIX = "prefix"
SINK_TEXT = "text"
SINK_TEXT_AUDIO = "text_audio"
SINK_OFF = "off"
SINK_MODES = (SINK_PREFIX, SINK_TEXT, SINK_TEXT_AUDIO, SINK_OFF)


def sink_range(layout, mode: str) -> tuple[int, int]:
    """Return the one contiguous exact-KV range accepted by Sol-Attn.

    H3 packs ``text | references | target audio | target video``.  A requested
    text+audio sink cannot exclude intervening reference rows while remaining
    contiguous, so ``text_audio`` intentionally resolves to the complete prefix.
    """
    if mode not in SINK_MODES:
        raise ValueError("unknown Sol sink mode %r" % mode)
    if mode == SINK_OFF:
        return 0, 0
    if mode == SINK_TEXT:
        start, stop = layout.text_range
        return int(start), int(stop - start)
    # prefix and text_audio: everything before the target-video tail.
    return 0, int(layout.video_range[0])


def sink_fraction(layout, mode: str, block_size: int = 64) -> float:
    start, tokens = sink_range(layout, mode)
    if not tokens:
        return 0.0
    blocks = (int(layout.seq_len) + block_size - 1) // block_size
    first = start // block_size
    last = (start + tokens + block_size - 1) // block_size
    return float(last - first) / max(1, blocks)
