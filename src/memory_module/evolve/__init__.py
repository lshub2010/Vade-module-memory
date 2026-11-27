"""Reserved for M4b (evolver). The directory skeleton is M4a's deliverable.

M4b owns everything inside this package; the interface it may use is
`memory_module.store.MemoryBackend`, never SQLite directly. `consolidate_run`
is already created by `store.py` (`get_last_consolidate` /
`record_consolidate`) so the startup backfill in W6 has a watermark to read.
"""

from __future__ import annotations

__all__: list[str] = []
