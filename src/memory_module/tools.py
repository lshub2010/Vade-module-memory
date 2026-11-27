"""The three memory tools, implemented as pure logic (ZERO SDK imports).

`server.py` binds these methods to `ToolSpec`s; tests instantiate `MemoryTools`
directly. The one non-obvious mechanism is the session context variable: the
`remember`/`recall` signatures deliberately have **no** `session_id` parameter,
so a model cannot forge one (M4a §一·五 / §七). Core injects the id out of band;
`session_scope()` is the adapter for that, and `server.py` reads the same
context when it builds a tool call.

No function here logs text. A failure message may name a length, a limit or a
field, never the memory body.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Sequence

from .embed import EmbeddingError
from .errors import MemoryFailure
from .settings import MemorySettings
from .store import (
    KINDS,
    MAX_TAG_CHARS,
    MAX_TAGS,
    MAX_TEXT_BYTES,
    NAMESPACE_RE,
    ULID_RE,
    MemoryBackend,
    MemoryRecord,
    content_hash,
)

__all__ = [
    "MemoryTools",
    "ToolFailure",
    "validate_text",
    "validate_tags",
    "validate_namespace",
    "validate_scope",
    "estimate_tokens",
]

#: `tools.py` must not import the SDK, so it raises its own structured error.
#: `server.py` maps this to `vade_sdk.ToolError` with the same C-04 code.
ToolFailure = MemoryFailure


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ToolFailure("INVALID_ARGUMENT", f"{field} must be a string, got {type(value).__name__}")
    return value


def validate_text(value: Any, *, field: str = "text") -> str:
    text = _require_str(value, field)
    if not text.strip():
        raise ToolFailure("INVALID_ARGUMENT", f"{field} must not be empty")
    size = len(text.encode("utf-8"))
    if size > MAX_TEXT_BYTES:
        raise ToolFailure(
            "PAYLOAD_TOO_LARGE",
            f"{field} is {size} UTF-8 bytes; the per-memory limit is {MAX_TEXT_BYTES}",
        )
    return text


def validate_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ToolFailure("INVALID_ARGUMENT", f"tags must be a list of strings, got {type(value).__name__}")
    out: list[str] = []
    seen: set[str] = set()
    supplied = 0
    for index, tag in enumerate(value):
        if not isinstance(tag, str):
            raise ToolFailure("INVALID_ARGUMENT", f"tags[{index}] must be a string")
        clean = tag.strip()
        if not clean:
            continue
        supplied += 1
        if supplied > MAX_TAGS:
            raise ToolFailure("INVALID_ARGUMENT", f"at most {MAX_TAGS} tags are allowed")
        if len(clean) > MAX_TAG_CHARS:
            raise ToolFailure(
                "INVALID_ARGUMENT", f"tags[{index}] is longer than {MAX_TAG_CHARS} characters"
            )
        if clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def validate_namespace(value: Any, *, allow_empty: bool = False) -> str:
    if value is None:
        value = ""
    namespace = _require_str(value, "namespace").strip()
    if allow_empty and namespace == "":
        return ""
    if not NAMESPACE_RE.match(namespace):
        raise ToolFailure(
            "INVALID_ARGUMENT",
            "namespace violates ^[a-z0-9][a-z0-9:_-]{0,63}$ (e.g. user:default, agent:reviewer, project:orion)",
        )
    return namespace


def validate_scope(value: Any) -> str:
    scope = _require_str(value, "scope")
    if scope not in ("global", "session"):
        raise ToolFailure("INVALID_ARGUMENT", "scope must be 'global' or 'session'")
    return scope


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ToolFailure("INVALID_ARGUMENT", f"{field} must be a boolean, got {type(value).__name__}")
    return value


def _require_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolFailure("INVALID_ARGUMENT", f"{field} must be an integer, got {type(value).__name__}")
    if not (minimum <= value <= maximum):
        raise ToolFailure("INVALID_ARGUMENT", f"{field} must be within [{minimum}, {maximum}], got {value}")
    return value


def estimate_tokens(text: str) -> int:
    """Cheap provider-independent estimate: ~4 characters per token.

    `recall` budgets in tokens (C-18 `memory.recall_token_limit`, default 800);
    using a character approximation avoids shipping a tokenizer and errs high
    for CJK, which is the safe direction for a cost cap.
    """
    return max(1, math.ceil(len(text) / 4))


class MemoryTools:
    """Tool bodies for `remember` / `recall` / `forget`.

    `session_id` is read from a `ContextVar`, never from tool arguments. The
    context manager is process-local and short-lived (one tool call), which is
    enough because the module is `concurrency=exclusive`.
    """

    def __init__(
        self,
        backend: MemoryBackend,
        embedder: Any,
        settings: MemorySettings,
        *,
        clock: Any = None,
    ) -> None:
        self.backend = backend
        self.embedder = embedder
        self.settings = settings
        self._clock = clock or (lambda: int(__import__("time").time()))
        self._session: ContextVar[str | None] = ContextVar("memory_session_id", default=None)

    # --------------------------------------------------------------- context
    @contextmanager
    def session_scope(self, session_id: str | None) -> Iterator[None]:
        """Set the core-injected session id for exactly one call."""
        if session_id is not None and not ULID_RE.match(session_id):
            raise ToolFailure("INVALID_ARGUMENT", "core-injected session_id is not a ULID")
        token = self._session.set(session_id)
        try:
            yield
        finally:
            self._session.reset(token)

    @property
    def current_session(self) -> str | None:
        return self._session.get()

    # ----------------------------------------------------------------- tools
    def remember(
        self,
        text: str,
        tags: list[str] = [],
        scope: str = "global",
        namespace: str = "user:default",
    ) -> dict[str, Any]:
        body = validate_text(text)
        clean_tags = validate_tags(tags)
        layer_scope = validate_scope(scope)

        if layer_scope == "session":
            # A non-default namespace here is a caller mistake, not something
            # to silently discard: session rows have no namespace at all.
            if namespace not in ("", "user:default"):
                raise ToolFailure(
                    "INVALID_ARGUMENT",
                    "namespace is only valid for scope='global'; session memories are partitioned by session_id",
                )
            session_id = self._session.get()
            if session_id is None:
                raise ToolFailure(
                    "PERMISSION_DENIED",
                    "scope='session' requires a core-injected session_id; the tool accepts no session_id argument",
                )
            layer = "session"
            ns = ""
        else:
            layer = "global"
            ns = validate_namespace(namespace)
            session_id = ""

        digest = content_hash(layer, session_id or "", ns, body)
        existing = self.backend.find_by_hash(layer, session_id or "", ns, digest)
        if existing is not None:
            return {
                "id": existing.id,
                "deduped": True,
                "layer": existing.layer,
                "namespace": existing.namespace,
            }

        vec = self._embed(body)
        rec = MemoryRecord(
            id="",
            layer=layer,
            namespace=ns,
            session_id=session_id or "",
            text=body,
            tags=clean_tags,
            content_hash=digest,
            importance=5.0,  # explicit remember is always 5 (D44)
            confidence=1.0,
            source="user",
            created_at=int(self._clock()),
            embedding=vec,
        )
        mem_id = self.backend.put(rec)
        return {
            "id": mem_id,
            "deduped": mem_id != rec.id,
            "layer": layer,
            "namespace": ns,
        }

    def recall(
        self,
        query: str,
        k: int = 5,
        namespace: str = "",
        include_session: bool = False,
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        body = validate_text(query, field="query")
        limit = _require_int(k, "k", 1, min(20, self.settings.recall_k_max))
        ns = validate_namespace(namespace, allow_empty=True)
        with_session = _require_bool(include_session, "include_session")
        with_superseded = _require_bool(include_superseded, "include_superseded")
        session_id = self._session.get() if with_session else None
        if with_session and session_id is None:
            # No injection means no session layer: returning global results is
            # safer than inventing a session. The caller can tell from `layer`.
            session_id = None

        vec = self._embed(body)
        records = self.backend.search(
            vec,
            limit,
            ns,
            include_session=session_id,
            include_superseded=with_superseded,
        )
        items, used = self._budget(records)
        if items:
            self.backend.record_hits([item["id"] for item in items])
        return {"items": items, "count": len(items)}

    def forget(self, id: str) -> dict[str, Any]:
        mem_id = _require_str(id, "id").strip()
        if not mem_id:
            raise ToolFailure("INVALID_ARGUMENT", "id must not be empty")
        if not self.backend.delete(mem_id):
            raise ToolFailure("NOT_FOUND", f"memory {mem_id!r} does not exist or was already forgotten")
        return {"ok": True, "id": mem_id}

    # --------------------------------------------------------------- helpers
    def _embed(self, text: str) -> list[float]:
        try:
            vec = self.embedder.embed(text)
        except EmbeddingError as exc:
            # No write has happened yet; the caller gets UNAVAILABLE and the
            # database has no half record.
            raise ToolFailure(exc.code, exc.message) from None
        if not isinstance(vec, (list, tuple)) or len(vec) != self.settings.embedding_dim:
            raise ToolFailure(
                "UNAVAILABLE",
                f"embedding provider returned dimension "
                f"{len(vec) if isinstance(vec, (list, tuple)) else 'non-vector'}; "
                f"memory.embedding_dim is {self.settings.embedding_dim}",
            )
        return [float(value) for value in vec]

    def _budget(self, records: Sequence[MemoryRecord]) -> tuple[list[dict[str, Any]], int]:
        """Apply `recall_token_limit`; clip only the returned copy, never the row."""
        budget_chars = max(1, int(self.settings.recall_token_limit) * 4)
        used = 0
        items: list[dict[str, Any]] = []
        for rec in records:
            text = rec.text
            remaining = budget_chars - used
            if remaining <= 0:
                break
            clipped = False
            if len(text) > remaining:
                text = text[: max(0, remaining - 1)].rstrip() + "…"
                clipped = True
            items.append(
                {
                    "id": rec.id,
                    "text": text,
                    "tags": list(rec.tags),
                    "layer": rec.layer,
                    "namespace": rec.namespace,
                    "confidence": round(float(rec.confidence), 3),
                    "score": round(float(rec.score), 3),
                }
            )
            used += len(text)
            if clipped:
                break
        return items, used
