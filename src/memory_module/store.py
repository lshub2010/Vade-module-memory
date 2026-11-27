"""SQLite backing store for the two-layer memory module.

`MemoryBackend` is the interface; `SqliteBackend` is the only v1 implementation
(M4a §五). `tools.py` and the evolver (`evolve/`, M4b) talk to the interface and
never receive a `sqlite3.Connection` — that is what keeps a future vector store
or a remote shared service from changing the tool layer.

Design points that are easy to get wrong, and are pinned by tests:

* `UNIQUE(layer, session_id, namespace, content_hash)` — dropping `session_id`
  lets session A read session B's memory and merges their quotas.
* `session_id` is `NOT NULL DEFAULT ''`; SQLite's UNIQUE treats NULLs as
  distinct, so NULL would silently disable global dedupe.
* Eviction archives rather than deletes; only `forget` and session TTL ever
  issue a physical `DELETE`.
* The vector dimension and provider model live in `meta`; opening a database
  with a different configured model/dimension raises `StartupError` instead of
  silently re-embedding.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
from array import array
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

from .embed import EmbeddingError, check_vector
from .errors import MemoryFailure, StartupError
from .settings import MemorySettings

__all__ = [
    "LAYERS",
    "KINDS",
    "SOURCES",
    "STATUSES",
    "MemoryRecord",
    "StagingRecord",
    "ArchiveRecord",
    "MemoryBackend",
    "SqliteBackend",
    "VectorIndex",
    "normalize_text",
    "content_hash",
    "new_ulid",
    "ULID_RE",
    "MAX_TEXT_BYTES",
]

LAYERS = ("session", "global")
KINDS = ("fact", "preference", "lesson")
SOURCES = ("user", "evolve", "import")
STATUSES = ("active", "archived", "superseded")
MAX_TEXT_BYTES = 8 * 1024
MAX_TAGS = 16
MAX_TAG_CHARS = 32

ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9:_-]{0,63}$")

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_LITTLE_ENDIAN = sys.byteorder == "little"


# --------------------------------------------------------------------------- ids
def new_ulid(timestamp_ms: int | None = None) -> str:
    """A 26-character Crockford-base32 ULID (48-bit ms + 80 random bits)."""
    value = int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms)
    chars = ["0"] * 26
    for index in range(9, -1, -1):  # 10 chars for 50 bits; top 2 stay zero
        chars[index] = _CROCKFORD[value & 0x1F]
        value >>= 5
    random_bits = int.from_bytes(os.urandom(10), "big")
    for index in range(25, 9, -1):
        chars[index] = _CROCKFORD[random_bits & 0x1F]
        random_bits >>= 5
    return "".join(chars)


# -------------------------------------------------------------------- text/hash
def normalize_text(text: str) -> str:
    """`strip()` plus collapse every whitespace run to one ASCII space.

    v1 deliberately does not do semantic normalization: two paraphrases are two
    rows until M4b's consolidate merges them (and even then only merges, never
    deletes the source text).
    """
    return " ".join(text.strip().split())


def content_hash(layer: str, session_id: str, namespace: str, text: str) -> str:
    import hashlib

    material = f"{layer}\0{session_id}\0{namespace}\0{normalize_text(text)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------- records
@dataclass
class MemoryRecord:
    """One row of `memories` (without the vector, which stays in the index).

    `score` and `status` are derived, never persisted: recall fills `score` with
    cosine similarity, the admin surface fills `status` with
    active/archived/superseded. `to_dict()` excludes both so archive payloads
    stay stable across restarts.
    """

    id: str
    layer: str
    namespace: str
    session_id: str
    text: str
    tags: list[str] = field(default_factory=list)
    content_hash: str = ""
    importance: float = 5.0
    confidence: float = 1.0
    source: str = "user"
    kind: str = "fact"
    evidence: dict[str, Any] | None = None
    verified_at: int | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    hit_count: int = 0
    last_hit_at: int | None = None
    created_at: int = 0
    expires_at: int | None = None
    deleted_at: int | None = None
    promoted_by: str | None = None
    #: Transient input for `put()`; never persisted in `memories` as a column
    #: (it lives in the `embedding` BLOB) and never serialized into `archive`.
    embedding: list[float] | None = None
    score: float = 0.0
    status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "namespace": self.namespace,
            "session_id": self.session_id,
            "text": self.text,
            "tags": list(self.tags),
            "content_hash": self.content_hash,
            "importance": self.importance,
            "confidence": self.confidence,
            "source": self.source,
            "kind": self.kind,
            "evidence": self.evidence,
            "verified_at": self.verified_at,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "hit_count": self.hit_count,
            "last_hit_at": self.last_hit_at,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "deleted_at": self.deleted_at,
            "promoted_by": self.promoted_by,
        }


@dataclass
class StagingRecord:
    id: str
    text: str
    namespace: str
    kind: str = "fact"
    evidence: dict[str, Any] | None = None
    content_hash: str = ""
    seen_sessions: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    confidence: float = 0.0
    created_at: int = 0

    @property
    def layer(self) -> str:
        return "staging"

    @property
    def status(self) -> str:
        return "staging"


@dataclass
class ArchiveRecord:
    id: str
    payload: dict[str, Any]
    archived_at: int
    reason: str
    archived_by: str = ""


class MemoryBackend(Protocol):
    """The only surface `tools.py` / `evolve/` may use."""

    def put(self, rec: MemoryRecord) -> str: ...
    def put_with_vector(self, rec: MemoryRecord, vec: Sequence[float]) -> str: ...
    def find_by_hash(
        self, layer: str, session_id: str, namespace: str, digest: str
    ) -> MemoryRecord | None: ...
    def record_hits(self, ids: Sequence[str]) -> None: ...
    def search(
        self,
        vec: list[float],
        k: int,
        namespace: str,
        include_session: str | None = None,
        include_superseded: bool = False,
    ) -> list[MemoryRecord]: ...
    def delete(self, mem_id: str) -> bool: ...
    def all_vectors(self) -> tuple[list[str], Any]: ...
    def get(self, mem_id: str, *, include_deleted: bool = False) -> MemoryRecord | None: ...
    def count(self, *, layer: str | None = None, include_deleted: bool = False) -> int: ...
    def close(self) -> None: ...
    def read_session(self, session_id: str, limit: int = 50) -> list[MemoryRecord]: ...
    def admin_list(self, **kwargs: Any) -> tuple[list[MemoryRecord], str | None]: ...
    def admin_get(self, mem_id: str) -> tuple[MemoryRecord, list[MemoryRecord]]: ...
    def admin_archive(self, mem_id: str, reason: str, by: str) -> bool: ...
    def admin_restore(self, mem_id: str, by: str) -> bool: ...
    def admin_stats(self) -> dict[str, Any]: ...
    def staging_put(
        self,
        text: str,
        namespace: str,
        session_id: str,
        embedding: list[float] | None,
        confidence: float,
        *,
        kind: str = "fact",
        evidence: dict[str, Any] | None = None,
    ) -> str: ...
    def staging_list(self, namespace: str | None = None) -> list[StagingRecord]: ...
    def staging_promote(self, staging_id: str, by: str = "evolve") -> str: ...
    def staging_drop(self, staging_id: str) -> None: ...
    def supersede(self, old_id: str, new_id: str) -> None: ...
    def archive_row(self, mem_id: str, reason: str) -> None: ...
    def close_session(self, session_id: str, now: int | None = None) -> int: ...
    def get_last_consolidate(self) -> dict[str, Any] | None: ...
    def record_consolidate(
        self, kind: str, stats: Mapping[str, Any] | None = None, *, last_run: int | None = None
    ) -> int: ...


# ------------------------------------------------------------------ vector index
try:  # optional acceleration; the module must work without numpy
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - covered by the pure-Python path
    _np = None


class VectorIndex:
    """Normalized vectors keyed by memory id.

    Vectors are L2-normalized at insertion, so cosine similarity is a dot
    product. With numpy present the matrix is materialized lazily and reused
    until the next mutation; without it the same `add`/`remove`/`scores`
    contract is served by a pure-Python loop. `array("f")` keeps the fallback at
    4 bytes per component instead of ~24 for a Python float list.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._vectors: dict[str, array] = {}
        self._dirty = True
        self._ids: list[str] = []
        self._matrix: Any = None

    def __len__(self) -> int:
        return len(self._vectors)

    def ids(self) -> list[str]:
        return list(self._vectors)

    def vectors(self) -> list[Sequence[float]]:
        return [list(value) for value in self._vectors.values()]

    def _normalize(self, vec: Sequence[float]) -> array:
        if len(vec) != self.dim:
            raise MemoryFailure(
                "INVALID_ARGUMENT",
                f"vector has dimension {len(vec)} but the index requires {self.dim}",
            )
        values = array("f", (float(x) for x in vec))
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise MemoryFailure("INVALID_ARGUMENT", "embedding vector has zero norm")
        return array("f", (value / norm for value in values))

    def add(self, mem_id: str, vec: Sequence[float]) -> None:
        self._vectors[mem_id] = self._normalize(vec)
        self._dirty = True

    def remove(self, mem_id: str) -> None:
        if self._vectors.pop(mem_id, None) is not None:
            self._dirty = True

    def _ensure_matrix(self) -> None:
        if _np is None or not self._dirty:
            return
        self._ids = list(self._vectors)
        if not self._ids:
            self._matrix = _np.zeros((0, self.dim), dtype=_np.float32)
        else:
            self._matrix = _np.asarray([self._vectors[key] for key in self._ids], dtype=_np.float32)
        self._dirty = False

    def scores(self, query: Sequence[float], ids: Sequence[str] | None = None) -> dict[str, float]:
        if len(query) != self.dim:
            raise MemoryFailure(
                "INVALID_ARGUMENT",
                f"query vector has dimension {len(query)} but the index requires {self.dim}",
            )
        norm = math.sqrt(sum(float(value) * float(value) for value in query))
        if norm == 0.0:
            return {}
        wanted = set(self._vectors) if ids is None else {key for key in ids if key in self._vectors}
        if not wanted:
            return {}
        if _np is not None:
            self._ensure_matrix()
            if not self._ids:
                return {}
            query_arr = _np.asarray(query, dtype=_np.float32) / _np.float32(norm)
            sims = self._matrix @ query_arr
            return {key: float(sims[index]) for index, key in enumerate(self._ids) if key in wanted}
        q = [float(value) / norm for value in query]
        return {
            key: sum(a * b for a, b in zip(q, self._vectors[key]))
            for key in wanted
        }


# ---------------------------------------------------------------------- backend
class SqliteBackend:
    """SQLite + in-memory index. One instance per process (concurrency=exclusive)."""

    def __init__(
        self,
        settings: MemorySettings,
        *,
        model: str | None = None,
        clock: Any = None,
    ) -> None:
        self.settings = settings
        self.model = model or settings.effective_model
        self.dim = settings.embedding_dim
        self._clock = clock or (lambda: int(time.time()))
        self.path = Path(settings.db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema()
            self._check_meta()
            now = int(self._clock())
            self.purge_expired(now=now)
            self.purge_deleted(now=now)
            self._records: dict[str, MemoryRecord] = {}
            self._index = VectorIndex(self.dim)
            self._load_index()
        except Exception:
            self._conn.close()
            raise

    # ------------------------------------------------------------------ setup
    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories(
              id TEXT PRIMARY KEY,
              layer TEXT NOT NULL,
              namespace TEXT NOT NULL,
              session_id TEXT NOT NULL DEFAULT '',
              text TEXT NOT NULL,
              tags TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              embedding BLOB NOT NULL,
              importance REAL NOT NULL,
              confidence REAL NOT NULL,
              source TEXT NOT NULL,
              kind TEXT NOT NULL DEFAULT 'fact',
              evidence TEXT,
              verified_at INTEGER,
              supersedes TEXT,
              superseded_by TEXT,
              hit_count INTEGER DEFAULT 0,
              last_hit_at INTEGER,
              created_at INTEGER NOT NULL,
              expires_at INTEGER,
              deleted_at INTEGER,
              promoted_by TEXT,
              UNIQUE(layer, session_id, namespace, content_hash)
            );
            CREATE TABLE IF NOT EXISTS staging(
              id TEXT PRIMARY KEY,
              text TEXT NOT NULL,
              namespace TEXT NOT NULL,
              kind TEXT NOT NULL DEFAULT 'fact',
              evidence TEXT,
              content_hash TEXT NOT NULL,
              seen_sessions TEXT NOT NULL,
              embedding BLOB,
              confidence REAL,
              created_at INTEGER,
              UNIQUE(kind, namespace, content_hash)
            );
            CREATE TABLE IF NOT EXISTS archive(
              id TEXT PRIMARY KEY,
              payload TEXT NOT NULL,
              archived_at INTEGER,
              reason TEXT,
              archived_by TEXT
            );
            CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
            CREATE TABLE IF NOT EXISTS consolidate_run(
              run_id INTEGER PRIMARY KEY AUTOINCREMENT,
              last_run INTEGER NOT NULL,
              kind TEXT NOT NULL,
              stats TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_layer_ns
              ON memories(layer, namespace, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_session
              ON memories(session_id, expires_at);
            CREATE INDEX IF NOT EXISTS idx_memories_live
              ON memories(deleted_at, superseded_by);
            CREATE INDEX IF NOT EXISTS idx_archive_at ON archive(archived_at);
            CREATE INDEX IF NOT EXISTS idx_consolidate_run ON consolidate_run(run_id);
            """
        )

    def _meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        return None if row is None else str(row["v"])

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (key, value),
        )

    def _check_meta(self) -> None:
        stored_dim = self._meta("dim")
        stored_model = self._meta("embedding_model")
        if stored_dim is None and stored_model is None:
            self._set_meta("dim", str(self.dim))
            self._set_meta("embedding_model", self.model)
            return
        if stored_dim is None or stored_model is None:
            raise StartupError(
                "memory database is corrupt: meta has only one of embedding_model/dim; "
                "refusing to guess the index shape"
            )
        try:
            dim = int(stored_dim)
        except ValueError:
            raise StartupError(f"memory database meta.dim is not an integer: {stored_dim!r}") from None
        if dim != self.dim or stored_model != self.model:
            raise StartupError(
                "embedding configuration mismatch: database has "
                f"dim={dim}, model={stored_model!r}; configured is "
                f"dim={self.dim}, model={self.model!r}. Refusing to start: re-embedding "
                "silently would corrupt the existing index. Use the original "
                "MEMORY_EMBEDDING_MODEL/MEMORY_EMBEDDING_DIM, or point MEMORY_DATA_DIR at a new database."
            )

    def _load_index(self) -> None:
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE deleted_at IS NULL ORDER BY created_at, id"
        ).fetchall()
        for row in rows:
            vec = _unpack_vector(row["embedding"], self.dim, str(row["id"]))
            rec = _row_to_record(row)
            self._records[rec.id] = rec
            self._index.add(rec.id, vec)

    # ------------------------------------------------------------------ txn
    @contextmanager
    def _tx(self) -> Iterator[None]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    # -------------------------------------------------------------- utilities
    def _fetch_row(self, mem_id: str) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()

    def _find_id_hash_locked(
        self, layer: str, session_id: str, namespace: str, digest: str
    ) -> str | None:
        row = self._conn.execute(
            "SELECT id FROM memories WHERE layer=? AND session_id=? AND namespace=? AND content_hash=?",
            (layer, session_id, namespace, digest),
        ).fetchone()
        return None if row is None else str(row["id"])

    def find_by_hash(
        self, layer: str, session_id: str, namespace: str, digest: str
    ) -> MemoryRecord | None:
        with self._lock:
            mem_id = self._find_id_hash_locked(layer, session_id, namespace, digest)
            return None if mem_id is None else self._records.get(mem_id)

    def get(self, mem_id: str, *, include_deleted: bool = False) -> MemoryRecord | None:
        with self._lock:
            rec = self._records.get(mem_id)
            if rec is not None:
                return rec
            row = self._fetch_row(mem_id)
            if row is None:
                return None
            if row["deleted_at"] is not None and not include_deleted:
                return None
            return _row_to_record(row)

    def count(self, *, layer: str | None = None, include_deleted: bool = False) -> int:
        clauses = []
        params: list[Any] = []
        if layer is not None:
            clauses.append("layer=?")
            params.append(layer)
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        row = self._conn.execute(f"SELECT COUNT(*) AS n FROM memories{where}", tuple(params)).fetchone()
        return int(row["n"])

    # --------------------------------------------------------------- scoring
    def score(self, rec: MemoryRecord, now: int | None = None) -> float:
        now = int(self._clock()) if now is None else int(now)
        age = max(0, now - int(rec.created_at))
        score = (
            0.45 * float(rec.importance)
            + 0.30 * math.exp(-(age / (30 * 86400)))
            + 0.20 * math.log1p(max(0, int(rec.hit_count)))
        )
        if rec.superseded_by:
            score -= 1.00
        return score

    def _evict_locked(self, target: int) -> list[MemoryRecord]:
        """Archive lowest-score global rows until `target` remain (caller holds tx)."""
        now = int(self._clock())
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE layer='global' AND deleted_at IS NULL"
        ).fetchall()
        live = [_row_to_record(row) for row in rows]
        live.sort(key=lambda rec: (self.score(rec, now), rec.created_at, rec.id))
        removed: list[MemoryRecord] = []
        while len(live) > max(0, target):
            rec = live.pop(0)
            archived = self._archive_sql_locked(rec.id, "quota", "", now)
            if archived is not None:
                removed.append(archived)
        return removed

    def _ensure_global_space_locked(self) -> list[MemoryRecord]:
        count = int(
            self._conn.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE layer='global' AND deleted_at IS NULL"
            ).fetchone()["n"]
        )
        if count < self.settings.global_capacity:
            return []
        target = min(self.settings.global_capacity - 1, int(self.settings.global_capacity * 0.9))
        removed = self._evict_locked(max(0, target))
        remaining = int(
            self._conn.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE layer='global' AND deleted_at IS NULL"
            ).fetchone()["n"]
        )
        if remaining >= self.settings.global_capacity:
            raise MemoryFailure(
                "OVERLOADED",
                f"global memory is at its {self.settings.global_capacity}-entry quota and "
                "eviction could not free a slot",
            )
        return removed

    def _ensure_session_space_locked(self, rec: MemoryRecord) -> None:
        now = int(self._clock())
        rows = self._conn.execute(
            "SELECT text FROM memories WHERE layer='session' AND session_id=? AND deleted_at IS NULL "
            "AND (expires_at IS NULL OR expires_at > ?)",
            (rec.session_id, now),
        ).fetchall()
        if len(rows) >= self.settings.session_capacity:
            raise MemoryFailure(
                "MEMORY_FULL",
                f"session memory has {len(rows)} entries (limit {self.settings.session_capacity}); "
                "merge or forget something in this turn and retry",
            )
        used = sum(len(str(row["text"])) for row in rows)
        # The write gate checks the *already stored* budget (M4a §六 test 14:
        # 4 x 40 chars under a 100-char limit fails on the 4th write). The
        # injection side is strict: read_session never returns more than the
        # configured budget, so a final overshoot cannot inflate the prompt.
        if len(rec.text) > self.settings.effective_session_write_char_limit or used >= self.settings.effective_session_write_char_limit:
            raise MemoryFailure(
                "MEMORY_FULL",
                f"session memory already uses {used} characters "
                f"(limit {self.settings.effective_session_write_char_limit}); "
                "merge or forget something in this turn and retry",
            )

    # ------------------------------------------------------------------- put
    def _validate_record(self, rec: MemoryRecord) -> MemoryRecord:
        if rec.layer not in LAYERS:
            raise MemoryFailure("INVALID_ARGUMENT", f"layer must be one of {list(LAYERS)}")
        if rec.kind not in KINDS:
            raise MemoryFailure("INVALID_ARGUMENT", f"kind must be one of {list(KINDS)}")
        if rec.source not in SOURCES:
            raise MemoryFailure("INVALID_ARGUMENT", f"source must be one of {list(SOURCES)}")
        if not isinstance(rec.text, str) or not rec.text.strip():
            raise MemoryFailure("INVALID_ARGUMENT", "text must be a non-empty string")
        if len(rec.text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise MemoryFailure("PAYLOAD_TOO_LARGE", f"text exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
        if rec.kind == "lesson" and not rec.evidence:
            raise MemoryFailure("INVALID_ARGUMENT", "kind=lesson requires non-empty evidence")
        if rec.evidence is not None and not isinstance(rec.evidence, dict):
            raise MemoryFailure("INVALID_ARGUMENT", "evidence must be an object")
        if not (1.0 <= float(rec.importance) <= 5.0):
            raise MemoryFailure("INVALID_ARGUMENT", "importance must be within [1, 5]")
        if not (0.0 <= float(rec.confidence) <= 1.0):
            raise MemoryFailure("INVALID_ARGUMENT", "confidence must be within [0, 1]")
        if not isinstance(rec.tags, list) or len(rec.tags) > MAX_TAGS:
            raise MemoryFailure("INVALID_ARGUMENT", f"tags must be a list of at most {MAX_TAGS} strings")
        for tag in rec.tags:
            if not isinstance(tag, str) or len(tag) > MAX_TAG_CHARS:
                raise MemoryFailure("INVALID_ARGUMENT", f"each tag must be a string of at most {MAX_TAG_CHARS} characters")
        if rec.layer == "global":
            if rec.session_id != "":
                raise MemoryFailure("INVALID_ARGUMENT", "global records must have an empty session_id")
            if not NAMESPACE_RE.match(rec.namespace or ""):
                raise MemoryFailure("INVALID_ARGUMENT", "namespace violates ^[a-z0-9][a-z0-9:_-]{0,63}$")
        else:
            if rec.namespace != "":
                raise MemoryFailure("INVALID_ARGUMENT", "session records must have an empty namespace")
            if not ULID_RE.match(rec.session_id or ""):
                raise MemoryFailure("INVALID_ARGUMENT", "session records require a core-injected ULID session_id")
        if rec.id and not ULID_RE.match(rec.id):
            raise MemoryFailure("INVALID_ARGUMENT", "id must be a ULID")
        return rec

    def put(self, rec: MemoryRecord) -> str:
        """Store a record; `rec.embedding` is the vector, computed by the caller.

        The vector is not a separate argument because the frozen `MemoryRecord`
        contract (C-12) carries it as a transient field. Keeping embedding out
        of `to_dict()` is deliberate: archive payloads must not grow a second
        copy of a 1536-float BLOB.
        """
        if rec.embedding is None:
            raise MemoryFailure(
                "INVALID_ARGUMENT",
                "put() requires rec.embedding; a memory row cannot exist without its vector",
            )
        return self.put_with_vector(rec, rec.embedding)

    def put_with_vector(self, rec: MemoryRecord, vec: Sequence[float]) -> str:
        """Store a record and its already-computed vector as one operation."""
        rec = self._validate_record(rec)
        try:
            vector = check_vector(vec, self.dim, where="memory vector")
        except EmbeddingError as exc:
            raise MemoryFailure("INVALID_ARGUMENT", exc.message) from None
        if not any(value != 0.0 for value in vector):
            raise MemoryFailure("INVALID_ARGUMENT", "embedding vector must not be all zeros")
        now = int(self._clock())
        if not rec.id:
            rec.id = new_ulid()
        if not rec.content_hash:
            rec.content_hash = content_hash(rec.layer, rec.session_id, rec.namespace, rec.text)
        if not rec.created_at:
            rec.created_at = now
        if rec.layer == "session" and rec.expires_at is None:
            rec.expires_at = now + self.settings.ttl_days * 86400
        blob = _pack_vector(vec)
        with self._lock:
            existing = self._find_id_hash_locked(rec.layer, rec.session_id, rec.namespace, rec.content_hash)
            if existing is not None and existing in self._records:
                return existing
            removed: list[MemoryRecord] = []
            with self._tx():
                existing = self._find_id_hash_locked(
                    rec.layer, rec.session_id, rec.namespace, rec.content_hash
                )
                if existing is not None:
                    if existing in self._records:
                        return existing
                    # A forgotten row keeps its UNIQUE key until the 7-day
                    # purge. Re-remembering the same text before then must not
                    # return the tombstone, so drop the dead row first (the
                    # archive payload already preserves the audit copy).
                    self._conn.execute("DELETE FROM memories WHERE id=? AND deleted_at IS NOT NULL", (existing,))
                if rec.layer == "global":
                    removed = self._ensure_global_space_locked()
                else:
                    self._ensure_session_space_locked(rec)
                self._insert_locked(rec, blob)
            for old in removed:
                self._forget_from_memory(old.id)
            self._records[rec.id] = rec
            self._index.add(rec.id, vector)
            return rec.id

    def _insert_locked(self, rec: MemoryRecord, blob: bytes | None = None) -> None:
        if blob is None:
            blob = _pack_vector([0.0] * self.dim)
        self._conn.execute(
            """
            INSERT INTO memories(
              id, layer, namespace, session_id, text, tags, content_hash, embedding,
              importance, confidence, source, kind, evidence, verified_at,
              supersedes, superseded_by, hit_count, last_hit_at, created_at,
              expires_at, deleted_at, promoted_by
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(layer, session_id, namespace, content_hash) DO NOTHING
            """,
            (
                rec.id,
                rec.layer,
                rec.namespace,
                rec.session_id,
                rec.text,
                json.dumps(list(rec.tags), ensure_ascii=False),
                rec.content_hash,
                blob,
                float(rec.importance),
                float(rec.confidence),
                rec.source,
                rec.kind,
                None if rec.evidence is None else json.dumps(rec.evidence, ensure_ascii=False),
                rec.verified_at,
                rec.supersedes,
                rec.superseded_by,
                int(rec.hit_count),
                rec.last_hit_at,
                int(rec.created_at),
                rec.expires_at,
                rec.deleted_at,
                rec.promoted_by,
            ),
        )

    # ---------------------------------------------------------------- search
    def search(
        self,
        vec: list[float],
        k: int,
        namespace: str,
        include_session: str | None = None,
        include_superseded: bool = False,
    ) -> list[MemoryRecord]:
        with self._lock:
            now = int(self._clock())
            candidates: list[MemoryRecord] = []
            for rec in self._records.values():
                if rec.deleted_at is not None:
                    continue
                if rec.layer == "session":
                    if include_session is None or rec.session_id != include_session:
                        continue
                    if rec.expires_at is not None and rec.expires_at <= now:
                        continue
                elif rec.layer != "global":
                    continue
                if namespace and rec.layer == "global" and rec.namespace != namespace:
                    continue
                if rec.superseded_by and not include_superseded:
                    continue
                candidates.append(rec)
            if not candidates:
                return []
            ids = [rec.id for rec in candidates]
            scores = self._index.scores(vec, ids)
            ranked: list[MemoryRecord] = []
            for rec in candidates:
                value = scores.get(rec.id)
                if value is None:
                    continue
                ranked.append(replace(rec, score=float(value)))
            ranked.sort(key=lambda rec: (-rec.score, -int(rec.created_at), rec.id))
            return ranked[: max(0, int(k))]

    # ---------------------------------------------------------------- delete
    def _archive_sql_locked(
        self, mem_id: str, reason: str, by: str, now: int
    ) -> MemoryRecord | None:
        row = self._fetch_row(mem_id)
        if row is None or row["deleted_at"] is not None:
            return None
        rec = _row_to_record(row)
        rec.deleted_at = now
        self._conn.execute("UPDATE memories SET deleted_at=? WHERE id=?", (now, mem_id))
        payload = json.dumps(rec.to_dict(), ensure_ascii=False)
        self._conn.execute(
            "INSERT INTO archive(id,payload,archived_at,reason,archived_by) VALUES(?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, archived_at=excluded.archived_at, "
            "reason=excluded.reason, archived_by=excluded.archived_by",
            (mem_id, payload, now, reason, by),
        )
        rec.status = "archived"
        return rec

    def _forget_from_memory(self, mem_id: str) -> None:
        self._records.pop(mem_id, None)
        self._index.remove(mem_id)

    def delete(self, mem_id: str) -> bool:
        with self._lock:
            now = int(self._clock())
            with self._tx():
                rec = self._archive_sql_locked(mem_id, "forget", "", now)
            if rec is None:
                return False
            self._forget_from_memory(mem_id)
            return True

    # ------------------------------------------------------------- core read
    def read_session(self, session_id: str, limit: int = 50) -> list[MemoryRecord]:
        if not ULID_RE.match(session_id or ""):
            raise MemoryFailure("INVALID_ARGUMENT", "session_id must be a ULID")
        if not (1 <= int(limit) <= 500):
            raise MemoryFailure("INVALID_ARGUMENT", "limit must be within [1, 500]")
        with self._lock:
            now = int(self._clock())
            rows = [
                rec
                for rec in self._records.values()
                if rec.layer == "session"
                and rec.session_id == session_id
                and rec.deleted_at is None
                and (rec.expires_at is None or rec.expires_at > now)
            ]
            window = max(1, self.settings.ttl_days * 86400)
            rows.sort(
                key=lambda rec: (
                    -(float(rec.importance) * math.exp(-max(0, now - rec.created_at) / window)),
                    -int(rec.created_at),
                    rec.id,
                )
            )
            budget = self.settings.session_inject_char_limit
            used = 0
            out: list[MemoryRecord] = []
            for rec in rows:
                size = len(rec.text)
                if used + size > budget:
                    # Budget injection is a prefix, not "all 200 every turn"
                    # (AMEND-19/D79). The cut items remain reachable through
                    # recall(include_session=true).
                    break
                out.append(rec)
                used += size
                if len(out) >= int(limit):
                    break
            return out

    # ------------------------------------------------------------ admin list
    def admin_list(
        self,
        *,
        layer: str | None = None,
        namespace: str | None = None,
        status: str | None = None,
        t_from: int | None = None,
        t_to: int | None = None,
        q: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[MemoryRecord], str | None]:
        if not (1 <= int(limit) <= 200):
            raise MemoryFailure("INVALID_ARGUMENT", "limit must be within [1, 200]")
        if layer is not None and layer not in LAYERS + ("staging",):
            raise MemoryFailure("INVALID_ARGUMENT", f"layer must be one of {list(LAYERS + ('staging',))}")
        if status is not None and status not in STATUSES + ("staging",):
            raise MemoryFailure("INVALID_ARGUMENT", f"status must be one of {list(STATUSES + ('staging',))}")
        if namespace is not None and namespace and not NAMESPACE_RE.match(namespace):
            raise MemoryFailure("INVALID_ARGUMENT", "namespace violates ^[a-z0-9][a-z0-9:_-]{0,63}$")
        offset = _decode_cursor(cursor)
        if t_from is not None and not isinstance(t_from, int):
            raise MemoryFailure("INVALID_ARGUMENT", "from must be a unix timestamp")
        if t_to is not None and not isinstance(t_to, int):
            raise MemoryFailure("INVALID_ARGUMENT", "to must be a unix timestamp")
        q = q.strip() if isinstance(q, str) else None
        if q and len(q) > 256:
            raise MemoryFailure("INVALID_ARGUMENT", "q must be at most 256 characters")

        staging_only = layer == "staging" or status == "staging"
        with self._lock:
            if staging_only:
                items = self._admin_list_staging(namespace, t_from, t_to, q, limit, offset)
                more = len(items) > int(limit)
                items = items[: int(limit)]
                next_cursor = _encode_cursor(offset + int(limit)) if more else None
                return items, next_cursor

            clauses = ["1=1"]
            params: list[Any] = []
            if layer is not None:
                clauses.append("layer=?")
                params.append(layer)
            if namespace:
                clauses.append("namespace=?")
                params.append(namespace)
            if status == "active":
                clauses.append("deleted_at IS NULL AND superseded_by IS NULL")
            elif status == "archived":
                clauses.append("deleted_at IS NOT NULL")
            elif status == "superseded":
                clauses.append("superseded_by IS NOT NULL")
            if t_from is not None:
                clauses.append("created_at >= ?")
                params.append(int(t_from))
            if t_to is not None:
                clauses.append("created_at <= ?")
                params.append(int(t_to))
            if q:
                clauses.append("(text LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\')")
                pattern = f"%{_escape_like(q)}%"
                params.extend([pattern, pattern])
            sql = (
                "SELECT * FROM memories WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
            )
            params.extend([int(limit) + 1, offset])
            rows = self._conn.execute(sql, tuple(params)).fetchall()
            more = len(rows) > int(limit)
            items = [_row_to_record(row) for row in rows[: int(limit)]]
            for rec in items:
                rec.status = derive_status(rec)
            next_cursor = _encode_cursor(offset + int(limit)) if more else None
            return items, next_cursor

    def _admin_list_staging(
        self,
        namespace: str | None,
        t_from: int | None,
        t_to: int | None,
        q: str | None,
        limit: int,
        offset: int,
    ) -> list[MemoryRecord]:
        clauses = ["1=1"]
        params: list[Any] = []
        if namespace:
            clauses.append("namespace=?")
            params.append(namespace)
        if t_from is not None:
            clauses.append("created_at >= ?")
            params.append(int(t_from))
        if t_to is not None:
            clauses.append("created_at <= ?")
            params.append(int(t_to))
        if q:
            clauses.append("text LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(q)}%")
        sql = (
            "SELECT * FROM staging WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        )
        params.extend([int(limit) + 1, offset])
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        out: list[MemoryRecord] = []
        for row in rows:
            staging = _row_to_staging(row)
            out.append(
                MemoryRecord(
                    id=staging.id,
                    layer="staging",
                    namespace=staging.namespace,
                    session_id="",
                    text=staging.text,
                    tags=[],
                    content_hash=staging.content_hash,
                    importance=0.0,
                    confidence=staging.confidence,
                    source="evolve",
                    kind=staging.kind,
                    evidence=staging.evidence,
                    created_at=staging.created_at,
                    status="staging",
                )
            )
        return out

    def admin_get(self, mem_id: str) -> tuple[MemoryRecord, list[MemoryRecord]]:
        with self._lock:
            row = self._fetch_row(mem_id)
            if row is None:
                raise MemoryFailure("NOT_FOUND", f"memory {mem_id!r} does not exist")
            current = _row_to_record(row)
            current.status = derive_status(current)
            chain: list[MemoryRecord] = []
            seen = {current.id}
            previous = current.supersedes
            for _ in range(50):
                if not previous or previous in seen:
                    break
                prow = self._fetch_row(previous)
                if prow is None:
                    break
                item = _row_to_record(prow)
                item.status = derive_status(item)
                chain.append(item)
                seen.add(item.id)
                previous = item.supersedes
            following = current.superseded_by
            for _ in range(50):
                if not following or following in seen:
                    break
                frow = self._fetch_row(following)
                if frow is None:
                    break
                item = _row_to_record(frow)
                item.status = derive_status(item)
                chain.append(item)
                seen.add(item.id)
                following = item.superseded_by
            chain.sort(key=lambda rec: (rec.created_at, rec.id))
            return current, chain

    def admin_archive(self, mem_id: str, reason: str, by: str) -> bool:
        if not isinstance(reason, str) or not reason.strip():
            raise MemoryFailure("INVALID_ARGUMENT", "reason must be a non-empty string")
        if not isinstance(by, str) or not by.strip():
            raise MemoryFailure("INVALID_ARGUMENT", "by must be a non-empty string")
        with self._lock:
            now = int(self._clock())
            with self._tx():
                rec = self._archive_sql_locked(mem_id, reason.strip()[:200], by.strip()[:64], now)
            if rec is None:
                return False
            self._forget_from_memory(mem_id)
            return True

    def admin_restore(self, mem_id: str, by: str) -> bool:
        if not isinstance(by, str) or not by.strip():
            raise MemoryFailure("INVALID_ARGUMENT", "by must be a non-empty string")
        with self._lock:
            row = self._fetch_row(mem_id)
            if row is None or row["deleted_at"] is None:
                return False
            reason_row = self._conn.execute(
                "SELECT reason FROM archive WHERE id=?", (mem_id,)
            ).fetchone()
            if reason_row is not None and str(reason_row["reason"]) == "forget":
                # forget is the one non-recoverable path: admin_restore is for
                # soft archive/quota, not for undoing a model-requested delete.
                raise MemoryFailure(
                    "PERMISSION_DENIED",
                    "a forgotten memory cannot be restored; the archive copy is audit-only",
                )
            with self._tx():
                self._conn.execute("UPDATE memories SET deleted_at=NULL WHERE id=?", (mem_id,))
            rec = _row_to_record(row)
            rec.deleted_at = None
            rec.status = derive_status(rec)
            self._records[mem_id] = rec
            self._index.add(mem_id, _unpack_vector(row["embedding"], self.dim, mem_id))
            return True

    def admin_stats(self) -> dict[str, Any]:
        with self._lock:
            global_count = int(
                self._conn.execute(
                    "SELECT COUNT(*) AS n FROM memories WHERE layer='global' AND deleted_at IS NULL"
                ).fetchone()["n"]
            )
            staging_count = int(self._conn.execute("SELECT COUNT(*) AS n FROM staging").fetchone()["n"])
            archived_count = int(
                self._conn.execute("SELECT COUNT(*) AS n FROM memories WHERE deleted_at IS NOT NULL").fetchone()["n"]
            )
            row = self._conn.execute(
                "SELECT MIN(created_at) AS oldest FROM memories WHERE layer='global' AND deleted_at IS NULL"
            ).fetchone()
            oldest = row["oldest"] if row is not None else None
            return {
                "global_count": global_count,
                "global_limit": self.settings.global_capacity,
                "staging_count": staging_count,
                "archived_count": archived_count,
                "oldest_at": None if oldest is None else int(oldest),
            }

    # --------------------------------------------------------------- staging
    def staging_put(
        self,
        text: str,
        namespace: str,
        session_id: str,
        embedding: list[float] | None,
        confidence: float,
        *,
        kind: str = "fact",
        evidence: dict[str, Any] | None = None,
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            raise MemoryFailure("INVALID_ARGUMENT", "text must be a non-empty string")
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise MemoryFailure("PAYLOAD_TOO_LARGE", f"text exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
        if not NAMESPACE_RE.match(namespace or ""):
            raise MemoryFailure("INVALID_ARGUMENT", "namespace violates ^[a-z0-9][a-z0-9:_-]{0,63}$")
        if kind not in KINDS:
            raise MemoryFailure("INVALID_ARGUMENT", f"kind must be one of {list(KINDS)}")
        if kind == "lesson" and not evidence:
            raise MemoryFailure("INVALID_ARGUMENT", "kind=lesson requires non-empty evidence")
        if evidence is not None and not isinstance(evidence, dict):
            raise MemoryFailure("INVALID_ARGUMENT", "evidence must be an object")
        if not (0.0 <= float(confidence) <= 1.0):
            raise MemoryFailure("INVALID_ARGUMENT", "confidence must be within [0, 1]")
        if embedding is not None and len(embedding) != self.dim:
            raise MemoryFailure(
                "INVALID_ARGUMENT",
                f"embedding has dimension {len(embedding)} but memory.embedding_dim is {self.dim}",
            )
        if session_id and not ULID_RE.match(session_id):
            raise MemoryFailure("INVALID_ARGUMENT", "session_id must be a ULID")
        digest = content_hash("staging", "", namespace, text)
        now = int(self._clock())
        blob = None if embedding is None else _pack_vector(embedding)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM staging WHERE kind=? AND namespace=? AND content_hash=?",
                (kind, namespace, digest),
            ).fetchone()
            if row is not None:
                existing = _row_to_staging(row)
                if session_id and session_id not in existing.seen_sessions:
                    existing.seen_sessions.append(session_id)
                existing.confidence = max(float(existing.confidence), float(confidence))
                if existing.embedding is None and embedding is not None:
                    existing.embedding = [float(x) for x in embedding]
                with self._tx():
                    self._conn.execute(
                        "UPDATE staging SET seen_sessions=?, confidence=?, embedding=? WHERE id=?",
                        (
                            json.dumps(existing.seen_sessions, ensure_ascii=False),
                            existing.confidence,
                            None if existing.embedding is None else _pack_vector(existing.embedding),
                            existing.id,
                        ),
                    )
                return existing.id
            new_id = new_ulid()
            with self._tx():
                self._conn.execute(
                    "INSERT INTO staging(id,text,namespace,kind,evidence,content_hash,seen_sessions,"
                    "embedding,confidence,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        new_id,
                        text,
                        namespace,
                        kind,
                        None if evidence is None else json.dumps(evidence, ensure_ascii=False),
                        digest,
                        json.dumps([session_id] if session_id else [], ensure_ascii=False),
                        blob,
                        float(confidence),
                        now,
                    ),
                )
            return new_id

    def staging_list(self, namespace: str | None = None) -> list[StagingRecord]:
        with self._lock:
            if namespace:
                rows = self._conn.execute(
                    "SELECT * FROM staging WHERE namespace=? ORDER BY created_at, id", (namespace,)
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM staging ORDER BY created_at, id").fetchall()
            return [_row_to_staging(row) for row in rows]

    def staging_promote(self, staging_id: str, by: str = "evolve") -> str:
        with self._lock:
            row = self._conn.execute("SELECT * FROM staging WHERE id=?", (staging_id,)).fetchone()
            if row is None:
                raise MemoryFailure("NOT_FOUND", f"staging record {staging_id!r} does not exist")
            staging = _row_to_staging(row)
            if staging.embedding is None:
                raise MemoryFailure(
                    "INVALID_ARGUMENT",
                    "staging record has no embedding; consolidate must supply one before promotion",
                )
            importance = 5.0 if by.startswith("admin:") else 3.0
            rec = MemoryRecord(
                id=new_ulid(),
                layer="global",
                namespace=staging.namespace,
                session_id="",
                text=staging.text,
                tags=[],
                content_hash=content_hash("global", "", staging.namespace, staging.text),
                importance=importance,
                confidence=min(0.9, float(staging.confidence or 0.6)),
                source="evolve",
                kind=staging.kind,
                evidence=staging.evidence,
                created_at=int(self._clock()),
                promoted_by=by,
                embedding=list(staging.embedding),
            )
            existing = self._find_id_hash_locked("global", "", rec.namespace, rec.content_hash)
            if existing is not None and existing in self._records:
                with self._tx():
                    self._conn.execute("DELETE FROM staging WHERE id=?", (staging_id,))
                return existing
            removed: list[MemoryRecord] = []
            with self._tx():
                if existing is not None:
                    self._conn.execute("DELETE FROM memories WHERE id=? AND deleted_at IS NOT NULL", (existing,))
                removed = self._ensure_global_space_locked()
                self._insert_locked(rec)
                self._conn.execute("DELETE FROM staging WHERE id=?", (staging_id,))
            for old in removed:
                self._forget_from_memory(old.id)
            self._records[rec.id] = rec
            self._index.add(rec.id, staging.embedding)
            return rec.id

    def staging_drop(self, staging_id: str) -> None:
        with self._lock:
            with self._tx():
                self._conn.execute("DELETE FROM staging WHERE id=?", (staging_id,))

    # ------------------------------------------------------------- supersede
    def supersede(self, old_id: str, new_id: str) -> None:
        if old_id == new_id:
            raise MemoryFailure("INVALID_ARGUMENT", "a memory cannot supersede itself")
        with self._lock:
            with self._tx():
                old = self._fetch_row(old_id)
                new = self._fetch_row(new_id)
                if old is None:
                    raise MemoryFailure("NOT_FOUND", f"memory {old_id!r} does not exist")
                if new is None:
                    raise MemoryFailure("NOT_FOUND", f"memory {new_id!r} does not exist")
                self._conn.execute("UPDATE memories SET superseded_by=? WHERE id=?", (new_id, old_id))
                self._conn.execute("UPDATE memories SET supersedes=? WHERE id=?", (old_id, new_id))
            if old_id in self._records:
                self._records[old_id].superseded_by = new_id
                self._records[old_id].status = "superseded"
            if new_id in self._records:
                self._records[new_id].supersedes = old_id

    def archive_row(self, mem_id: str, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise MemoryFailure("INVALID_ARGUMENT", "reason must be a non-empty string")
        if not self.delete_internal(mem_id, reason.strip()[:200]):
            raise MemoryFailure("NOT_FOUND", f"memory {mem_id!r} does not exist")

    def delete_internal(self, mem_id: str, reason: str) -> bool:
        with self._lock:
            now = int(self._clock())
            with self._tx():
                rec = self._archive_sql_locked(mem_id, reason, "", now)
            if rec is None:
                return False
            self._forget_from_memory(mem_id)
            return True

    # ---------------------------------------------------------------- hits
    def record_hits(self, ids: Sequence[str]) -> None:
        unique = [mem_id for mem_id in dict.fromkeys(ids) if mem_id]
        if not unique:
            return
        now = int(self._clock())
        with self._lock:
            with self._tx():
                self._conn.executemany(
                    "UPDATE memories SET hit_count=COALESCE(hit_count,0)+1, last_hit_at=? WHERE id=?",
                    [(now, mem_id) for mem_id in unique],
                )
            for mem_id in unique:
                rec = self._records.get(mem_id)
                if rec is not None:
                    rec.hit_count += 1
                    rec.last_hit_at = now

    # --------------------------------------------------------------- session
    def close_session(self, session_id: str, now: int | None = None) -> int:
        if not ULID_RE.match(session_id or ""):
            raise MemoryFailure("INVALID_ARGUMENT", "session_id must be a ULID")
        now = int(self._clock()) if now is None else int(now)
        expires = now + self.settings.ttl_days * 86400  # from close, not "now" as expiry
        with self._lock:
            with self._tx():
                cursor = self._conn.execute(
                    "UPDATE memories SET expires_at=? WHERE layer='session' AND session_id=? AND deleted_at IS NULL",
                    (expires, session_id),
                )
                changed = cursor.rowcount
            for rec in self._records.values():
                if rec.layer == "session" and rec.session_id == session_id and rec.deleted_at is None:
                    rec.expires_at = expires
            return int(changed)

    def purge_expired(self, now: int | None = None) -> int:
        now = int(self._clock()) if now is None else int(now)
        with self._lock:
            with self._tx():
                cursor = self._conn.execute(
                    "DELETE FROM memories WHERE layer='session' AND expires_at IS NOT NULL AND expires_at < ?",
                    (now,),
                )
                removed = int(cursor.rowcount)
            if removed and hasattr(self, "_records"):
                expired = [
                    mem_id
                    for mem_id, rec in self._records.items()
                    if rec.layer == "session" and rec.expires_at is not None and rec.expires_at < now
                ]
                for mem_id in expired:
                    self._forget_from_memory(mem_id)
            return removed

    def purge_deleted(self, now: int | None = None) -> int:
        """Physically remove rows whose `deleted_at` is older than 7 days."""
        now = int(self._clock()) if now is None else int(now)
        cutoff = now - 7 * 86400
        with self._lock:
            with self._tx():
                cursor = self._conn.execute(
                    "DELETE FROM memories WHERE deleted_at IS NOT NULL AND deleted_at < ?", (cutoff,)
                )
                return int(cursor.rowcount)

    # ------------------------------------------------------------- consolidate
    def get_last_consolidate(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM consolidate_run ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        stats: Any = row["stats"]
        if stats:
            try:
                stats = json.loads(stats)
            except (TypeError, ValueError):
                stats = {}
        return {
            "run_id": int(row["run_id"]),
            "last_run": int(row["last_run"]),
            "kind": str(row["kind"]),
            "stats": stats or {},
        }

    def record_consolidate(
        self, kind: str, stats: Mapping[str, Any] | None = None, *, last_run: int | None = None
    ) -> int:
        if kind not in ("scheduled", "startup_backfill", "manual"):
            raise MemoryFailure(
                "INVALID_ARGUMENT",
                "kind must be one of ['scheduled', 'startup_backfill', 'manual']",
            )
        payload = json.dumps(dict(stats or {}), ensure_ascii=False)
        with self._lock:
            with self._tx():
                cursor = self._conn.execute(
                    "INSERT INTO consolidate_run(last_run,kind,stats) VALUES(?,?,?)",
                    (int(self._clock()) if last_run is None else int(last_run), kind, payload),
                )
                return int(cursor.lastrowid)

    # ------------------------------------------------------------------ debug
    def all_vectors(self) -> tuple[list[str], Any]:
        """Return the in-memory matrix: `(ids, ndarray)` when numpy exists.

        The fallback is a list of float lists, which is what the pure-Python
        deployment has; callers must use `len(ids)` for the row count rather
        than assuming a `.shape` attribute.
        """
        with self._lock:
            ids = self._index.ids()
            vectors = self._index.vectors()
            if _np is not None:
                return ids, _np.asarray(vectors, dtype=_np.float32)
            return ids, vectors

    def record_count(self) -> int:
        return len(self._records)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# -------------------------------------------------------------------- helpers
def derive_status(rec: MemoryRecord) -> str:
    if rec.superseded_by:
        return "superseded"
    if rec.deleted_at is not None:
        return "archived"
    return "active"


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    tags_raw = row["tags"]
    try:
        tags = json.loads(tags_raw) if tags_raw else []
    except (TypeError, ValueError):
        tags = []
    evidence_raw = row["evidence"]
    evidence = None
    if evidence_raw:
        try:
            evidence = json.loads(evidence_raw)
        except (TypeError, ValueError):
            evidence = None
    return MemoryRecord(
        id=str(row["id"]),
        layer=str(row["layer"]),
        namespace=str(row["namespace"]),
        session_id=str(row["session_id"]),
        text=str(row["text"]),
        tags=[str(tag) for tag in tags] if isinstance(tags, list) else [],
        content_hash=str(row["content_hash"]),
        importance=float(row["importance"]),
        confidence=float(row["confidence"]),
        source=str(row["source"]),
        kind=str(row["kind"]),
        evidence=evidence,
        verified_at=None if row["verified_at"] is None else int(row["verified_at"]),
        supersedes=None if row["supersedes"] is None else str(row["supersedes"]),
        superseded_by=None if row["superseded_by"] is None else str(row["superseded_by"]),
        hit_count=int(row["hit_count"] or 0),
        last_hit_at=None if row["last_hit_at"] is None else int(row["last_hit_at"]),
        created_at=int(row["created_at"]),
        expires_at=None if row["expires_at"] is None else int(row["expires_at"]),
        deleted_at=None if row["deleted_at"] is None else int(row["deleted_at"]),
        promoted_by=None if "promoted_by" not in row.keys() or row["promoted_by"] is None else str(row["promoted_by"]),
    )


def _row_to_staging(row: sqlite3.Row) -> StagingRecord:
    try:
        seen = json.loads(row["seen_sessions"] or "[]")
    except (TypeError, ValueError):
        seen = []
    evidence = None
    if row["evidence"]:
        try:
            evidence = json.loads(row["evidence"])
        except (TypeError, ValueError):
            evidence = None
    embedding = None
    if row["embedding"] is not None:
        raw = row["embedding"]
        values = array("f")
        values.frombytes(raw)
        if not _LITTLE_ENDIAN:
            values.byteswap()
        embedding = [float(v) for v in values]
    return StagingRecord(
        id=str(row["id"]),
        text=str(row["text"]),
        namespace=str(row["namespace"]),
        kind=str(row["kind"]),
        evidence=evidence,
        content_hash=str(row["content_hash"]),
        seen_sessions=[str(x) for x in seen] if isinstance(seen, list) else [],
        embedding=embedding,
        confidence=float(row["confidence"] or 0.0),
        created_at=int(row["created_at"] or 0),
    )


def _pack_vector(vec: Sequence[float]) -> bytes:
    values = array("f", (float(x) for x in vec))
    if not _LITTLE_ENDIAN:
        values.byteswap()
    return values.tobytes()


def _unpack_vector(blob: bytes | None, dim: int, mem_id: str) -> list[float]:
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise StartupError(f"memory {mem_id!r} has no embedding blob")
    raw = bytes(blob)
    if len(raw) != dim * 4:
        raise StartupError(
            f"memory {mem_id!r} embedding is {len(raw)} bytes; expected {dim * 4} for dim={dim}. "
            "Refusing to start with a corrupt/foreign index."
        )
    values = array("f")
    values.frombytes(raw)
    if not _LITTLE_ENDIAN:
        values.byteswap()
    return [float(value) for value in values]


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"o": int(offset)}).encode("utf-8")).decode("ascii")


def _decode_cursor(cursor: str | None) -> int:
    if cursor is None or cursor == "":
        return 0
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
        offset = int(payload.get("o", 0))
    except Exception:
        raise MemoryFailure("INVALID_ARGUMENT", "cursor is not a memory admin cursor") from None
    if offset < 0:
        raise MemoryFailure("INVALID_ARGUMENT", "cursor offset must be >= 0")
    return offset
