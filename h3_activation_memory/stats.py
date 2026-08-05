"""Cheap per-invocation dispatch counters for activation-memory execution."""

from collections import Counter
from dataclasses import dataclass, field

STATS_KEY = "_h3_activation_memory_stats"


@dataclass
class ActivationStats:
    config_signature: tuple
    blocks: int = 0
    chunks: int = 0
    rows: int = 0
    max_chunk_rows: int = 0
    held_sessions: int = 0
    module_fallback_sessions: int = 0
    paths: Counter = field(default_factory=Counter)
    fallback_reasons: Counter = field(default_factory=Counter)

    def record_chunk(self, rows):
        rows = int(rows)
        self.chunks += 1
        self.rows += rows
        self.max_chunk_rows = max(self.max_chunk_rows, rows)

    def record_path(self, name):
        self.paths[str(name)] += 1

    def record_fallback(self, reason):
        self.module_fallback_sessions += 1
        self.fallback_reasons[str(reason)] += 1

    def summary(self):
        return {
            "config": list(self.config_signature),
            "blocks": self.blocks,
            "chunks": self.chunks,
            "rows": self.rows,
            "max_chunk_rows": self.max_chunk_rows,
            "held_sessions": self.held_sessions,
            "module_fallback_sessions": self.module_fallback_sessions,
            "paths": dict(self.paths),
            "fallback_reasons": dict(self.fallback_reasons),
        }


def get_stats(transformer_options, config):
    stats = transformer_options.get(STATS_KEY)
    if stats is None or stats.config_signature != config.signature:
        stats = ActivationStats(config.signature)
        transformer_options[STATS_KEY] = stats
    return stats
