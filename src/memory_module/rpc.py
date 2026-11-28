"""Private RPCs: core → module, never in the tool table.

`memory/read_session` is what C3's `MemoryPort.read_session` calls to inject
the session layer into the prompt. The six `memory/admin_*` methods are the
config page's only path to the store (AMEND-16); the page never opens SQLite.

Three rules from M4a §四 are enforced here:

1. No permission checks. Only core can connect (FR-5.7); roles live in core.
2. No raw transcript. Items carry the memory body and the `session_id`/`turn_id`
   pointer, never `user_text` / `assistant_text` (design §7.9).
3. No `admin_update`. Editing is `archive` + `remember`; the supersedes chain is
   the audit trail.

The SDK 0.1 transport only dispatches `tools/call`, `tools/cancel`,
`module/drain`, `prompts/get` and `events/deliver`, so `server.py` exposes this
dispatcher separately; see README "Known limitations".
"""

from __future__ import annotations

from typing import Any, Mapping

from .errors import MemoryFailure
from .settings import MemorySettings
from .store import MemoryBackend, MemoryRecord, StagingRecord, ULID_RE

__all__ = ["RpcDispatcher", "handle_event", "admin_item", "MEMORY_RPC_METHODS"]

MEMORY_RPC_METHODS = (
    "memory/read_session",
    "memory/admin_list",
    "memory/admin_get",
    "memory/admin_promote",
    "memory/admin_archive",
    "memory/admin_restore",
    "memory/admin_stats",
)


def _params_object(params: Any) -> dict[str, Any]:
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise MemoryFailure("INVALID_ARGUMENT", "params must be an object")
    return dict(params)


def _require_id(value: Any, field: str = "id") -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryFailure("INVALID_ARGUMENT", f"{field} must be a non-empty string")
    return value.strip()


def _optional_bool(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise MemoryFailure("INVALID_ARGUMENT", f"{field} must be a boolean")
    return value


def _optional_int(value: Any, field: str) -> int | None:
    """Accept an integral JSON number; unix seconds are still integer-valued.

    C-08 declares `from`/`to` as `number`, and a client sending `1.7e9` produced
    a float that used to be rejected. The contract wins: accept int or float as
    long as it is finite, and normalise to integer seconds.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise MemoryFailure("INVALID_ARGUMENT", f"{field} must be a number")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value == value and value not in (float("inf"), float("-inf")):
        return int(value)
    raise MemoryFailure("INVALID_ARGUMENT", f"{field} must be a number")


def admin_item(rec: MemoryRecord) -> dict[str, Any]:
    """The one admin item shape. Deliberately has no transcript fields."""
    return {
        "id": rec.id,
        "layer": rec.layer,
        "namespace": rec.namespace,
        "status": rec.status or _derive(rec),
        "text": rec.text,
        "tags": list(rec.tags),
        "kind": rec.kind,
        "confidence": round(float(rec.confidence), 3),
        "source": rec.source,
        "session_id": rec.session_id,
        "created_at": int(rec.created_at),
        "expires_at": rec.expires_at,
        "hit_count": int(rec.hit_count),
        "supersedes": rec.supersedes,
        "superseded_by": rec.superseded_by,
        "evidence": rec.evidence,
        "verified_at": rec.verified_at,
        "promoted_by": rec.promoted_by,
    }


def _derive(rec: MemoryRecord) -> str:
    if rec.superseded_by:
        return "superseded"
    if rec.deleted_at is not None:
        return "archived"
    return "active"


def _staging_item(staging: StagingRecord) -> dict[str, Any]:
    return {
        "id": staging.id,
        "layer": "staging",
        "namespace": staging.namespace,
        "status": "staging",
        "text": staging.text,
        "tags": [],
        "kind": staging.kind,
        "confidence": round(float(staging.confidence), 3),
        "source": "evolve",
        "session_id": "",
        "created_at": int(staging.created_at),
        "expires_at": None,
        "hit_count": 0,
        "supersedes": None,
        "superseded_by": None,
        "evidence": staging.evidence,
        "verified_at": None,
        "promoted_by": None,
    }


class RpcDispatcher:
    """Validate params, call the backend, return the C-16 response shape."""

    def __init__(self, backend: MemoryBackend, settings: MemorySettings) -> None:
        self.backend = backend
        self.settings = settings

    def dispatch(self, method: str, params: Any = None) -> dict[str, Any]:
        if not isinstance(method, str) or method not in MEMORY_RPC_METHODS:
            raise MemoryFailure("NOT_FOUND", f"unknown private memory method {method!r}")
        payload = _params_object(params)
        handler = getattr(self, "_" + method.replace("/", "_").replace(".", "_"))
        return handler(payload)

    # ------------------------------------------------------------- core read
    def _memory_read_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = _require_id(params.get("session_id"), "session_id")
        if not ULID_RE.match(session_id):
            raise MemoryFailure("INVALID_ARGUMENT", "session_id must be a ULID")
        limit = params.get("limit", 50)
        if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= 500):
            raise MemoryFailure("INVALID_ARGUMENT", "limit must be an integer within [1, 500]")
        records = self.backend.read_session(session_id, limit)
        return {
            "items": [
                {
                    "id": rec.id,
                    "text": rec.text,
                    "tags": list(rec.tags),
                    "created_at": int(rec.created_at),
                }
                for rec in records
            ]
        }

    # --------------------------------------------------------------- admin
    def _memory_admin_list(self, params: dict[str, Any]) -> dict[str, Any]:
        layer = params.get("layer")
        if layer is not None and not isinstance(layer, str):
            raise MemoryFailure("INVALID_ARGUMENT", "layer must be a string")
        namespace = params.get("namespace")
        if namespace is not None and not isinstance(namespace, str):
            raise MemoryFailure("INVALID_ARGUMENT", "namespace must be a string")
        status = params.get("status")
        if status is not None and not isinstance(status, str):
            raise MemoryFailure("INVALID_ARGUMENT", "status must be a string")
        q = params.get("q")
        if q is not None and not isinstance(q, str):
            raise MemoryFailure("INVALID_ARGUMENT", "q must be a string")
        limit = params.get("limit", 50)
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise MemoryFailure("INVALID_ARGUMENT", "limit must be an integer")
        cursor = params.get("cursor")
        if cursor is not None and not isinstance(cursor, str):
            raise MemoryFailure("INVALID_ARGUMENT", "cursor must be a string")
        items, next_cursor = self.backend.admin_list(
            layer=layer or None,
            namespace=namespace or None,
            status=status or None,
            t_from=_optional_int(params.get("from"), "from"),
            t_to=_optional_int(params.get("to"), "to"),
            q=q or None,
            limit=limit,
            cursor=cursor or None,
        )
        return {"items": [admin_item(item) for item in items], "next_cursor": next_cursor}

    def _memory_admin_get(self, params: dict[str, Any]) -> dict[str, Any]:
        mem_id = _require_id(params.get("id"))
        record, chain = self.backend.admin_get(mem_id)
        return {
            "item": admin_item(record),
            "chain": [
                {
                    "id": item.id,
                    "status": item.status or _derive(item),
                    "created_at": int(item.created_at),
                }
                for item in chain
            ],
        }

    def _memory_admin_promote(self, params: dict[str, Any]) -> dict[str, Any]:
        staging_id = _require_id(params.get("id"))
        by = _require_id(params.get("by"), "by")[:64]
        actor = by if by.startswith("admin:") else f"admin:{by}"
        mem_id = self.backend.staging_promote(staging_id, by=actor)
        return {"ok": True, "id": mem_id}

    def _memory_admin_archive(self, params: dict[str, Any]) -> dict[str, Any]:
        mem_id = _require_id(params.get("id"))
        reason = _require_id(params.get("reason"), "reason")
        if len(reason) > 300:
            # C-16 maxLength is a bound, not a truncation instruction: silently
            # keeping 300 of 301 would make the audit and the stored row differ.
            raise MemoryFailure("INVALID_ARGUMENT", "reason must be at most 300 characters")
        by = _require_id(params.get("by"), "by")[:64]
        if not self.backend.admin_archive(mem_id, reason, by):
            raise MemoryFailure("NOT_FOUND", f"memory {mem_id!r} does not exist or is already archived")
        return {"ok": True}

    def _memory_admin_restore(self, params: dict[str, Any]) -> dict[str, Any]:
        mem_id = _require_id(params.get("id"))
        by = _require_id(params.get("by"), "by")[:64]
        if not self.backend.admin_restore(mem_id, by):
            raise MemoryFailure("NOT_FOUND", f"memory {mem_id!r} does not exist or is not archived")
        return {"ok": True}

    def _memory_admin_stats(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.backend.admin_stats()


def handle_event(backend: MemoryBackend, topic: str, payload: Any = None) -> dict[str, Any]:
    """Apply the one event M4a owns: `session.closed` re-bases the session TTL.

    `session.turn` is accepted but ignored in M4a; M4b will subscribe and do the
    extraction. Returning `handled: false` for an unknown topic keeps the
    dispatcher honest without turning a stray event into a module failure.
    """
    if topic == "session.closed":
        data = payload if isinstance(payload, Mapping) else {}
        session_id = data.get("session_id")
        if session_id is None and isinstance(data.get("session"), Mapping):
            session_id = data["session"].get("id")
        if not isinstance(session_id, str) or not ULID_RE.match(session_id):
            raise MemoryFailure("INVALID_ARGUMENT", "session.closed payload has no valid session_id")
        closed_at = data.get("closed_at")
        closed_at_s: int | None = None
        if isinstance(closed_at, int) and not isinstance(closed_at, bool) and closed_at > 0:
            # C-14 carries unix milliseconds; `close_session` takes seconds.
            closed_at_s = closed_at // 1000 if closed_at > 10_000_000_000 else closed_at
        changed = backend.close_session(session_id, now=closed_at_s)  # 7 days from close, not from now
        return {"ok": True, "handled": True, "session_id": session_id, "updated": changed}
    if topic == "session.turn":
        return {"ok": True, "handled": False}
    return {"ok": True, "handled": False}
