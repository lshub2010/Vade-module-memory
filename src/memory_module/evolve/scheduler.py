"""Consolidate scheduling: daily 03:00 UTC plus startup catch-up (AMEND-19).

A desktop box is off at 03:00 more often than not; a container gets stopped for
the weekend. Without a catch-up, `consolidate` effectively never runs outside a
server, so the scheduler reads `consolidate_run.last_run` at process start and
runs once when the watermark is older than 24 h. `consolidate` is idempotent,
so the extra run cannot double-promote.

`clock` is injectable and `sleep` is injectable, which is what makes the tests
deterministic without waiting for 03:00. When `evolve.enabled=false` the
scheduler neither runs nor records anything.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from ..store import MemoryBackend
from .config import EvolveSettings
from .consolidate import ConsolidateStats, consolidate

__all__ = ["ConsolidateScheduler", "BACKFILL_AFTER_S", "SCHEDULE_HOUR_UTC"]

BACKFILL_AFTER_S = 24 * 3600
SCHEDULE_HOUR_UTC = 3

_log = logging.getLogger("memory_module.evolve")


class ConsolidateScheduler:
    """Owns the consolidate clock; the pass itself lives in `consolidate.py`."""

    def __init__(
        self,
        backend: MemoryBackend,
        settings: EvolveSettings,
        *,
        consolidate_fn: Callable[[str], ConsolidateStats] | None = None,
        clock: Callable[[], int] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        embedder: Any = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.backend = backend
        self.settings = settings
        self.embedder = embedder
        self._log = logger or _log
        self._clock = clock or (lambda: int(time.time()))
        self._sleep = sleep or asyncio.sleep
        self._consolidate_fn = consolidate_fn
        self._next_due_at: int | None = None

    # -------------------------------------------------------------- schedule
    @staticmethod
    def next_run_at(now: int) -> int:
        """The first 03:00 UTC strictly after `now`."""
        moment = datetime.fromtimestamp(int(now), tz=timezone.utc)
        candidate = moment.replace(hour=SCHEDULE_HOUR_UTC, minute=0, second=0, microsecond=0)
        if candidate <= moment:
            candidate += timedelta(days=1)
        return int(candidate.timestamp())

    def arm(self, now: int | None = None) -> int:
        """Set the next due point; safe to call repeatedly."""
        current = int(self._clock()) if now is None else int(now)
        self._next_due_at = self.next_run_at(current)
        return self._next_due_at

    def due_at(self) -> int | None:
        return self._next_due_at

    # ----------------------------------------------------------------- runs
    def startup_backfill(self) -> ConsolidateStats | None:
        """Run once at startup when the watermark is missing or older than 24 h."""
        if not self.settings.enabled:
            self._log.info("evolve disabled: startup backfill skipped")
            return None
        now = int(self._clock())
        last = self.backend.get_last_consolidate()
        if last is not None and now - int(last.get("last_run", 0)) < BACKFILL_AFTER_S:
            return None
        return self._run("startup_backfill", now)

    def tick(self, now: int | None = None) -> ConsolidateStats | None:
        """Run the scheduled pass when the clock has crossed 03:00; else no-op."""
        if not self.settings.enabled:
            self._log.info("evolve disabled: scheduled consolidate skipped")
            return None
        current = int(self._clock()) if now is None else int(now)
        if self._next_due_at is None:
            self.arm(current)
            return None
        if current < self._next_due_at:
            return None
        self._next_due_at = self.next_run_at(current)
        return self._run("scheduled", current)

    async def run_forever(self, *, max_cycles: int | None = None) -> None:
        """Catch up, then sleep until the next 03:00 UTC, forever (or N cycles)."""
        if not self.settings.enabled:
            self._log.info("evolve disabled: scheduler not started")
            return
        self.startup_backfill()
        self.arm()
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            if self._next_due_at is None:
                self.arm()
            assert self._next_due_at is not None  # arm() always sets it
            delay = max(1, self._next_due_at - int(self._clock()))
            await self._sleep(float(delay))
            self.tick()
            cycles += 1

    # -------------------------------------------------------------- internal
    def _run(self, kind: str, now: int) -> ConsolidateStats | None:
        try:
            if self._consolidate_fn is not None:
                return self._consolidate_fn(kind)
            return consolidate(
                self.backend,
                self.settings,
                embedder=self.embedder,
                clock=(lambda: now),
                kind=kind,
            )
        except Exception as exc:  # noqa: BLE001 - a scheduled pass must not kill the loop
            self._log.warning(
                "evolve_degraded trigger=consolidate kind=%s at=%s reason=%s",
                kind,
                now,
                type(exc).__name__,
            )
            return None
