"""H3-owned attention: observation seam, block forward, and backends.

See PLAN.md for scope and the measurement that gates the kernel work.
"""

from .observer import OBSERVER_KEY, notify_attention, observing

__all__ = ["OBSERVER_KEY", "notify_attention", "observing"]
