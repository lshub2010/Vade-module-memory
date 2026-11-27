"""Scheduler: daily 03:00, startup catch-up, disabled means silent (AMEND-19)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from memory_module.evolve.config import EvolveSettings
from memory_module.evolve.consolidate import ConsolidateStats
from memory_module.evolve.scheduler import ConsolidateScheduler
from memory_module.settings import MemorySettings
from memory_module.store import SqliteBackend, new_ulid


class Clock:
    def __init__(self, start: int = 1_700_000_000) -> None:
        self.now = start

    def __call__(self) -> int:
        return self.now

    def set(self, value: int) -> None:
        self.now = int(value)


class Recorder:
    def __init__(self) -> None:
        self.kinds: list[str] = []

    def __call__(self, kind: str) -> ConsolidateStats:
        self.kinds.append(kind)
        return ConsolidateStats(status="ok")


class FakeSleep:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        self.clock.now += int(delay)


def make_backend(tmp_path) -> SqliteBackend:
    settings = MemorySettings(data_dir=tmp_path, embedding_dim=8, global_capacity=100, session_capacity=50)
    return SqliteBackend(settings, clock=Clock())


def at(hour: int, minute: int = 0, second: int = 0) -> int:
    return int(datetime(2026, 11, 27, hour, minute, second, tzinfo=timezone.utc).timestamp())


def test_disabled_scheduler_never_runs(tmp_path, caplog):
    backend = make_backend(tmp_path)
    clock = Clock(at(2, 0))
    recorder = Recorder()
    scheduler = ConsolidateScheduler(
        backend, EvolveSettings(enabled=False), consolidate_fn=recorder, clock=clock
    )

    assert scheduler.startup_backfill() is None
    assert scheduler.tick() is None
    asyncio.run(scheduler.run_forever(max_cycles=1))

    assert recorder.kinds == []
    assert backend.get_last_consolidate() is None


def test_backfill_runs_once_when_no_watermark_exists(tmp_path):
    backend = make_backend(tmp_path)
    s1, s2 = new_ulid(), new_ulid()
    vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for session_id in (s1, s2):
        backend.staging_put("项目使用 SQLite 存储", "user:default", session_id, vector, 0.6)

    clock = Clock(at(9, 0))
    scheduler = ConsolidateScheduler(backend, EvolveSettings(enabled=True), clock=clock)

    first = scheduler.startup_backfill()
    second = scheduler.startup_backfill()

    assert first is not None and first.promoted == 1
    assert second is None
    assert backend.count(layer="global") == 1
    last = backend.get_last_consolidate()
    assert last["kind"] == "startup_backfill"
    assert last["last_run"] == clock.now


def test_backfill_skips_a_recent_run_and_runs_a_stale_one(tmp_path):
    backend = make_backend(tmp_path)
    clock = Clock(at(9, 0))
    recorder = Recorder()
    scheduler = ConsolidateScheduler(
        backend, EvolveSettings(enabled=True), consolidate_fn=recorder, clock=clock
    )

    backend.record_consolidate("manual", {}, last_run=clock.now - 3600)
    assert scheduler.startup_backfill() is None
    assert recorder.kinds == []

    backend.record_consolidate("manual", {}, last_run=clock.now - 25 * 3600)
    stats = scheduler.startup_backfill()
    assert stats is not None
    assert recorder.kinds == ["startup_backfill"]


def test_tick_runs_once_when_crossing_three_am(tmp_path):
    backend = make_backend(tmp_path)
    clock = Clock(at(2, 59, 59))
    recorder = Recorder()
    scheduler = ConsolidateScheduler(
        backend, EvolveSettings(enabled=True), consolidate_fn=recorder, clock=clock
    )

    assert scheduler.tick() is None  # arms 03:00, does not run
    assert scheduler.due_at() == at(3, 0)

    clock.set(at(3, 0, 1))
    stats = scheduler.tick()
    assert stats is not None
    assert recorder.kinds == ["scheduled"]
    assert scheduler.tick() is None
    assert recorder.kinds == ["scheduled"]


def test_next_run_at_is_strictly_in_the_future():
    assert ConsolidateScheduler.next_run_at(at(2, 0)) == at(3, 0)
    assert ConsolidateScheduler.next_run_at(at(3, 0)) == at(3, 0) + 86400
    assert ConsolidateScheduler.next_run_at(at(23, 59, 59)) == at(3, 0) + 86400


def test_run_forever_catches_up_then_sleeps_to_three_am(tmp_path):
    backend = make_backend(tmp_path)
    clock = Clock(at(2, 0))
    recorder = Recorder()
    sleep = FakeSleep(clock)
    scheduler = ConsolidateScheduler(
        backend,
        EvolveSettings(enabled=True),
        consolidate_fn=recorder,
        clock=clock,
        sleep=sleep,
    )

    asyncio.run(scheduler.run_forever(max_cycles=1))

    assert recorder.kinds == ["startup_backfill", "scheduled"]
    assert sleep.delays == [3600.0]
    assert clock.now == at(3, 0)
