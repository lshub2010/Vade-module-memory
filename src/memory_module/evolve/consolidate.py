"""`consolidate`: cluster staging, promote, detect conflicts, decay, evict.

The only path from candidate to fact (M4b §七). The rules are deliberately
conservative:

* group by `(kind, namespace)`; within a group, cosine > 0.92 joins a cluster;
* `fact` / `preference` need **two distinct sessions** in `seen_sessions`
  before they may enter `global`; one session stays in `staging` and is
  invisible to `recall`;
* `lesson` needs one session plus non-empty `evidence`, gets
  `confidence = min(cluster mean, lesson_confidence_floor)` and
  `verified_at = NULL`; lessons never supersede each other, they accumulate;
* a new fact/preference that is > 0.90 similar to an existing global row in
  the same namespace **and** contains a negation word supersedes the old row
  (both rows stay, recall hides the old one);
* over quota, lowest-score global rows are archived down to a 90% watermark;
* a promoted cluster is removed from `staging`. Running consolidate three
  times in a day yields the same library — `consolidate_run` records the
  watermark the scheduler reads.

Clustering only merges *display*; no global text is deleted (D44: when the
conflict call is uncertain, keep both).
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..errors import MemoryFailure
from ..store import MemoryBackend, MemoryRecord, StagingRecord, content_hash
from .config import EvolveSettings

__all__ = [
    "ConsolidateStats",
    "consolidate",
    "CONSOLIDATE_KINDS",
    "CLUSTER_THRESHOLD",
    "CONFLICT_THRESHOLD",
]

CLUSTER_THRESHOLD = 0.92
CONFLICT_THRESHOLD = 0.90
CONSOLIDATE_KINDS = ("scheduled", "startup_backfill", "manual")

_NEGATION = re.compile(
    r"(不|别|取消|改成|不是|停止|never|no longer|not anymore|stop|cancel|instead)",
    re.IGNORECASE,
)

_log = logging.getLogger("memory_module.evolve")


@dataclass
class ConsolidateStats:
    status: str = "ok"
    clusters: int = 0
    promoted: int = 0
    lessons: int = 0
    superseded: int = 0
    archived: int = 0
    staging_cleared: int = 0
    single_session: int = 0
    dropped: int = 0
    run_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "clusters": self.clusters,
            "promoted": self.promoted,
            "lessons": self.lessons,
            "superseded": self.superseded,
            "archived": self.archived,
            "staging_cleared": self.staging_cleared,
            "single_session": self.single_session,
            "dropped": self.dropped,
        }


def _norm(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def _cosine(left: Sequence[float] | None, right: Sequence[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    norm_left = math.sqrt(sum(float(a) * float(a) for a in left))
    norm_right = math.sqrt(sum(float(b) * float(b) for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _group(rows: Sequence[StagingRecord]) -> dict[tuple[str, str], list[StagingRecord]]:
    groups: dict[tuple[str, str], list[StagingRecord]] = {}
    for row in rows:
        groups.setdefault((row.kind, row.namespace), []).append(row)
    return groups


def _cluster(
    rows: Sequence[StagingRecord], threshold: float = CLUSTER_THRESHOLD
) -> list[list[StagingRecord]]:
    """Greedy leader clustering; input order is `created_at, id` (deterministic)."""
    clusters: list[tuple[Sequence[float] | None, list[StagingRecord]]] = []
    for row in rows:
        vector = row.embedding
        best: int | None = None
        best_similarity = threshold
        if vector is not None:
            for index, (leader, _members) in enumerate(clusters):
                if leader is None:
                    continue
                similarity = _cosine(vector, leader)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best = index
        if best is None:
            clusters.append((vector, [row]))
        else:
            clusters[best][1].append(row)
    return [members for _leader, members in clusters]


def _representative_fact(members: Sequence[StagingRecord]) -> StagingRecord:
    # Longest text wins; ties go to the earliest row, then id (stable output).
    return max(members, key=lambda row: (len(row.text), -int(row.created_at), row.id))


def _evidence_size(row: StagingRecord) -> int:
    evidence = row.evidence if isinstance(row.evidence, dict) else None
    if not evidence:
        return 0
    items = evidence.get("items")
    return len(items) if isinstance(items, list) and items else 1


def _representative_lesson(members: Sequence[StagingRecord]) -> StagingRecord:
    # A lesson's value is its evidence, not its word count (M4b §七).
    return max(
        members,
        key=lambda row: (_evidence_size(row), len(row.text), -int(row.created_at), row.id),
    )


def _merge_evidence(members: Sequence[StagingRecord]) -> dict[str, Any] | None:
    items: list[dict[str, Any]] = []
    for row in sorted(members, key=lambda item: (int(item.created_at), item.id)):
        evidence = row.evidence if isinstance(row.evidence, dict) else None
        if not evidence:
            continue
        if evidence not in items:
            items.append(evidence)
    if not items:
        return None
    if len(items) == 1:
        return dict(items[0])
    merged = dict(items[0])
    merged["items"] = items
    return merged


def _hydrate(
    backend: MemoryBackend, rows: Sequence[StagingRecord], embedder: Any, log: logging.Logger
) -> None:
    """Backfill missing staging vectors through the public `staging_put` merge."""
    if embedder is None:
        return
    for row in rows:
        if row.embedding is not None:
            continue
        try:
            vector = [float(value) for value in embedder.embed(row.text)]
        except Exception:  # noqa: BLE001 - an un-embeddable candidate stays a singleton
            continue
        try:
            backend.staging_put(
                row.text,
                row.namespace,
                "",
                vector,
                row.confidence,
                kind=row.kind,
                evidence=row.evidence,
            )
        except MemoryFailure as exc:
            log.warning("evolve_degraded trigger=consolidate reason=%s", exc.code)
            continue
        row.embedding = vector


def _conflict(
    backend: MemoryBackend, rep: StagingRecord, namespace: str, threshold: float
) -> str | None:
    """Return the id of the global row this candidate contradicts, if any."""
    if rep.embedding is None or not _NEGATION.search(rep.text):
        return None
    try:
        hits = backend.search(list(rep.embedding), 5, namespace, include_superseded=False)
    except MemoryFailure:
        return None
    for hit in hits:
        if hit.layer != "global":
            continue
        if float(hit.score) <= threshold:
            continue
        if _norm(hit.text) == _norm(rep.text):
            continue
        return hit.id
    return None


def _promote_lesson(
    backend: MemoryBackend,
    rep: StagingRecord,
    namespace: str,
    evidence: Mapping[str, Any],
    confidence: float,
    now: int,
    log: logging.Logger,
) -> tuple[str | None, bool]:
    """Write the lesson directly so confidence/verified_at are exact.

    `staging_promote` forces importance 3.0 and clamps confidence to 0.9, and
    has no way to express "unverified"; a lesson needs both. The row still
    passes through `MemoryBackend.put`, never SQL.
    """
    digest = content_hash("global", "", namespace, rep.text)
    existing = backend.find_by_hash("global", "", namespace, digest)
    if existing is not None:
        return existing.id, False
    rec = MemoryRecord(
        id="",
        layer="global",
        namespace=namespace,
        session_id="",
        text=rep.text,
        tags=[],
        content_hash=digest,
        importance=4.0,
        confidence=float(confidence),
        source="evolve",
        kind="lesson",
        evidence=dict(evidence),
        verified_at=None,
        created_at=now,
        promoted_by="evolve",
        embedding=list(rep.embedding or []),
    )
    try:
        new_id = backend.put(rec)
    except MemoryFailure as exc:
        log.warning("evolve_degraded trigger=consolidate reason=%s", exc.code)
        return None, False
    return new_id, True


def _drop_members(
    backend: MemoryBackend, members: Sequence[StagingRecord], keep: str | None
) -> int:
    dropped = 0
    for row in members:
        if keep is not None and row.id == keep:
            continue
        try:
            backend.staging_drop(row.id)
        except MemoryFailure:
            continue
        dropped += 1
    return dropped


def _active_globals(backend: MemoryBackend) -> list[MemoryRecord]:
    rows: list[MemoryRecord] = []
    cursor: str | None = None
    while True:
        items, cursor = backend.admin_list(
            layer="global", status="active", limit=200, cursor=cursor
        )
        rows.extend(items)
        if not cursor:
            return rows


def _decay_and_evict(
    backend: MemoryBackend, now: int, capacity: int | None, log: logging.Logger
) -> int:
    rows = _active_globals(backend)
    if capacity is None:
        capacity = int(getattr(getattr(backend, "settings", None), "global_capacity", 0) or 0)
    if capacity <= 0 or len(rows) <= capacity:
        return 0
    target = max(0, int(capacity * 0.9))
    score_fn = getattr(backend, "score", None)

    def value(rec: MemoryRecord) -> float:
        if callable(score_fn):
            return float(score_fn(rec, now))
        age = max(0, now - int(rec.created_at))
        return (
            0.45 * float(rec.importance)
            + 0.30 * math.exp(-(age / (30 * 86400)))
            + 0.20 * math.log1p(max(0, int(rec.hit_count)))
        )

    rows.sort(key=lambda rec: (value(rec), int(rec.created_at), rec.id))
    archived = 0
    for rec in rows[: len(rows) - target]:
        try:
            backend.archive_row(rec.id, "evolve_decay")
        except MemoryFailure as exc:
            log.warning("evolve_degraded trigger=consolidate reason=%s", exc.code)
            continue
        archived += 1
    return archived


def consolidate(
    backend: MemoryBackend,
    settings: EvolveSettings,
    *,
    embedder: Any = None,
    clock: Any = None,
    kind: str = "manual",
    capacity: int | None = None,
    logger: logging.Logger | None = None,
    cluster_threshold: float = CLUSTER_THRESHOLD,
    conflict_threshold: float = CONFLICT_THRESHOLD,
) -> ConsolidateStats:
    """One consolidation pass. Idempotent: a second pass finds an empty staging."""
    log = logger or _log
    if kind not in CONSOLIDATE_KINDS:
        raise ValueError(f"consolidate kind must be one of {list(CONSOLIDATE_KINDS)}, got {kind!r}")
    if not settings.enabled:
        log.info("evolve disabled: consolidate skipped")
        return ConsolidateStats(status="disabled")

    now = int(clock()) if clock is not None else int(time.time())
    stats = ConsolidateStats()
    rows = backend.staging_list()
    if rows:
        _hydrate(backend, rows, embedder, log)
        for (member_kind, namespace), group in _group(rows).items():
            for members in _cluster(group, cluster_threshold):
                stats.clusters += 1
                if member_kind == "lesson":
                    _consolidate_lesson(
                        backend, settings, members, namespace, now, stats, log
                    )
                else:
                    _consolidate_fact(
                        backend, members, namespace, conflict_threshold, stats, log
                    )

    stats.archived = _decay_and_evict(backend, now, capacity, log)
    stats.run_id = backend.record_consolidate(kind, stats.as_dict(), last_run=now)
    log.info(
        "evolve_consolidate kind=%s clusters=%d promoted=%d lessons=%d "
        "superseded=%d archived=%d staging_cleared=%d",
        kind,
        stats.clusters,
        stats.promoted,
        stats.lessons,
        stats.superseded,
        stats.archived,
        stats.staging_cleared,
    )
    return stats


def _consolidate_fact(
    backend: MemoryBackend,
    members: Sequence[StagingRecord],
    namespace: str,
    conflict_threshold: float,
    stats: ConsolidateStats,
    log: logging.Logger,
) -> None:
    sessions = {session for row in members for session in row.seen_sessions if session}
    if len(sessions) < 2:
        # D43: one session is an observation, not a fact. Leave it in staging;
        # `recall` cannot see staging rows.
        stats.single_session += 1
        return
    rep = _representative_fact(members)
    if rep.embedding is None:
        stats.dropped += 1
        return

    old_id = _conflict(backend, rep, namespace, conflict_threshold)
    try:
        new_id = backend.staging_promote(rep.id)
    except MemoryFailure as exc:
        log.warning("evolve_degraded trigger=consolidate reason=%s", exc.code)
        stats.dropped += 1
        return
    stats.promoted += 1
    if old_id is not None and old_id != new_id:
        try:
            backend.supersede(old_id, new_id)
        except MemoryFailure as exc:
            log.warning("evolve_degraded trigger=consolidate reason=%s", exc.code)
        else:
            stats.superseded += 1
    # staging_promote deleted the representative; drop the merged duplicates.
    stats.staging_cleared += 1 + _drop_members(backend, members, keep=rep.id)


def _consolidate_lesson(
    backend: MemoryBackend,
    settings: EvolveSettings,
    members: Sequence[StagingRecord],
    namespace: str,
    now: int,
    stats: ConsolidateStats,
    log: logging.Logger,
) -> None:
    evidence = _merge_evidence(members)
    rep = _representative_lesson(members)
    if not evidence or rep.embedding is None:
        # No locatable evidence means no lesson (M4b §七); keep the row around
        # so a later pass with embeddings can still use it.
        stats.dropped += 1
        return
    confidence = min(
        _mean(row.confidence for row in members), settings.lesson_confidence_floor
    )
    new_id, created = _promote_lesson(
        backend, rep, namespace, evidence, confidence, now, log
    )
    if new_id is None:
        stats.dropped += 1
        return
    if created:
        stats.lessons += 1
        stats.promoted += 1
    # The lesson row was written by `_promote_lesson`, so every staging row in
    # the cluster is now redundant (including the representative).
    stats.staging_cleared += _drop_members(backend, members, keep=None)
