"""The event consumer M4b was missing: `session.turn` / `session.closed` in.

`pipeline.py` holds the rules; this module is the only place that knows how a
core event becomes those calls. Two constraints shape it:

* **No transcript retention.** A `session.turn` contributes a counter entry
  `{turn_id, state, tool_calls, negative, self_corrected}` and nothing else.
  `user_text` goes to `rules` for extraction and is then dropped; it is never
  stored on the worker. `session.closed` uses core's structured `turns[]`, not
  a remembered copy.
* **A session must never be blocked.** Every handler is wrapped: an exception
  becomes `evolve_degraded` in the log and an `{"status": "degraded"}` reply.
  The event transport must not see a callback raise.

`enabled=false` is checked before anything else — no subscription, no write, no
model call, one `evolve disabled` log line (M4b §九).
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Mapping

from ..store import ULID_RE
from .config import EvolveSettings
from .pipeline import EvolvePipeline
from .skillgen import SkillGenerator

__all__ = ["EvolveWorker", "ProposalSink"]

#: `async def sink(payload: dict) -> Any` — `server.py` wires this to core's
#: `skill/proposal` RPC. A sink that raises is logged, never retried here.
ProposalSink = Callable[[Mapping[str, Any]], Awaitable[Any]]

_log = logging.getLogger("memory_module.evolve")

SUPPORTED_TOPICS = ("session.turn", "session.closed")


class EvolveWorker:
    """Routes the two subscribed topics into rules/review/summary/skillgen."""

    def __init__(
        self,
        pipeline: EvolvePipeline,
        settings: EvolveSettings,
        *,
        skillgen: SkillGenerator | None = None,
        propose: ProposalSink | None = None,
        catalog: Callable[[], Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.settings = settings
        self.skillgen = skillgen
        self._propose = propose
        self._catalog = catalog or (lambda: ())
        self._log = logger or _log
        #: Per-session counters only — `turn_id`, state and tool tallies.
        self._counters: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------ gate
    @property
    def enabled(self) -> bool:
        return bool(self.settings.enabled)

    def topics(self) -> tuple[str, ...]:
        return SUPPORTED_TOPICS if self.enabled else ()

    async def on_event(self, topic: str, payload: Any = None) -> dict[str, Any]:
        if not self.enabled:
            # One line per event would be noise; the gate itself is logged once
            # at construction, this stays debug.
            self._log.debug("evolve disabled: dropping %s", topic)
            return {"status": "disabled", "topic": topic}
        if topic == "session.turn":
            return await self._guard("turn_end", self._on_turn, payload)
        if topic == "session.closed":
            return await self._guard("session_close", self._on_closed, payload)
        self._log.warning("evolve ignored unsupported topic=%s", topic)
        return {"status": "ignored", "topic": topic}

    # --------------------------------------------------------------- handlers
    async def _on_turn(self, payload: Any) -> dict[str, Any]:
        data = payload if isinstance(payload, Mapping) else {}
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not ULID_RE.match(session_id):
            self._log.warning("evolve_degraded trigger=turn_end reason=invalid_session_id")
            return {"status": "ignored"}
        self._remember_counter(session_id, data)
        result = await self.pipeline.on_turn(data)
        return dict(result or {})

    async def _on_closed(self, payload: Any) -> dict[str, Any]:
        data = payload if isinstance(payload, Mapping) else {}
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not ULID_RE.match(session_id):
            self._log.warning("evolve_degraded trigger=session_close reason=invalid_session_id")
            return {"status": "ignored"}
        turns = data.get("turns")
        if not isinstance(turns, list):
            turns = self._counters.get(session_id, [])
        result = await self.pipeline.on_session_closed({**dict(data), "turns": turns})

        proposal = None
        if self.skillgen is not None:
            proposal = await self._skill_evolve(session_id, turns)
        self._counters.pop(session_id, None)
        return {**dict(result or {}), "skill_proposal": proposal}

    async def _skill_evolve(self, session_id: str, turns: Any) -> dict[str, Any] | None:
        generator = self.skillgen
        if generator is None:
            return None
        try:
            payload = await generator.maybe_propose(
                session_id=session_id, turns=turns, catalog=self._catalog() or ()
            )
        except Exception as exc:  # noqa: BLE001 - never block session close
            self._log.warning("evolve_degraded trigger=skill_evolve reason=%s", type(exc).__name__)
            return None
        if payload is None:
            return None
        if self._propose is None:
            # No sink: honestly report "produced but not delivered" instead of
            # pretending a pending skill exists.
            self._log.warning("evolve_degraded trigger=skill_evolve reason=no_sink")
            return {"status": "not_delivered", "name": payload.get("name")}
        try:
            reply = await self._propose(payload)
        except Exception as exc:  # noqa: BLE001 - core rejection is not a session failure
            self._log.warning("evolve_degraded trigger=skill_evolve reason=proposal_rejected")
            return {"status": "rejected", "name": payload.get("name"), "error": type(exc).__name__}
        # C-13: a rejected proposal is logged, never retried (no proposal loop).
        status = "proposed"
        if isinstance(reply, Mapping) and reply.get("ok") is False:
            status = "rejected"
        return {"status": status, "name": payload.get("name"), "reply": reply}

    # ---------------------------------------------------------------- helpers
    def _remember_counter(self, session_id: str, data: Mapping[str, Any]) -> None:
        turn_id = data.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            return
        raw_calls = data.get("tool_calls")
        calls: list[dict[str, Any]] = []
        if isinstance(raw_calls, list):
            for call in raw_calls:
                if isinstance(call, Mapping):
                    calls.append({"tool": str(call.get("tool") or ""), "ok": call.get("ok") is not False})
        bucket = self._counters.setdefault(session_id, [])
        bucket.append(
            {
                "turn_id": turn_id,
                "state": str(data.get("state") or ""),
                "tool_calls": calls,
                "negative": bool(data.get("negative")),
                "self_corrected": bool(data.get("self_corrected")),
            }
        )
        # Bound the buffer: only the tail can still matter, and an unbounded
        # per-session list is exactly the transcript retention we promised not
        # to do.
        if len(bucket) > 200:
            del bucket[:-200]

    async def _guard(self, trigger: str, handler: Callable[[Any], Awaitable[dict[str, Any]]],
                     payload: Any) -> dict[str, Any]:
        try:
            return await handler(payload)
        except Exception as exc:  # noqa: BLE001 - the event path must not raise
            self._log.warning("evolve_degraded trigger=%s reason=%s", trigger, type(exc).__name__)
            return {"status": "degraded", "trigger": trigger}
