"""Composition root for the five evolve triggers that do not need the SDK.

`worker.py` (the event consumer) and the CLI are thin shells around this class;
all the rules live in the modules it wires together. Keeping the glue here
means the "enabled=false ⇒ zero writes, zero model calls" test has one obvious
entry point instead of relying on every module to be called correctly.

Trigger map (M4b §四):

* `session.turn` → `rules.extract_from_payload` (session layer) then
  `reviewer.on_turn` (ring buffer, every `nudge_interval` turns a review);
* `session.closed` → `summarizer.on_session_close` (gated by `flush_min_turns`);
* `consolidate` → called by the scheduler, the CLI, or `consolidate()` below.

`skill_evolve` is not in this part: it needs the skill catalog/event contract
(C3b/C3d) and is delivered with `skillgen.py`.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from ..errors import MemoryFailure
from ..store import ULID_RE, MemoryBackend, MemoryRecord, content_hash
from . import rules
from .config import DEFAULT_NAMESPACE, EvolveSettings
from .consolidate import ConsolidateStats, consolidate
from .llm import LlmCallable
from .review import TurnReviewer
from .scheduler import ConsolidateScheduler
from .summarize import SessionSummarizer, SummaryOutcome

__all__ = ["EvolvePipeline"]

_log = logging.getLogger("memory_module.evolve")


class EvolvePipeline:
    """Wires rules + review + summary + consolidate for one memory backend."""

    def __init__(
        self,
        backend: MemoryBackend,
        settings: EvolveSettings,
        *,
        llm: LlmCallable | None = None,
        embedder: Any = None,
        memory_settings: Any = None,
        clock: Any = None,
        logger: logging.Logger | None = None,
        namespace: str = DEFAULT_NAMESPACE,
        timeout_s: float = 30.0,
    ) -> None:
        self.backend = backend
        self.settings = settings
        self.embedder = embedder
        self.namespace = namespace
        self._log = logger or _log
        self._clock = clock or (lambda: int(time.time()))
        memory_limit = int(
            getattr(memory_settings, "effective_session_write_char_limit", settings.session_char_limit)
        )
        self.write_limit = min(settings.session_char_limit, memory_limit)
        self.reviewer = TurnReviewer(
            backend,
            settings,
            llm=llm,
            embedder=embedder,
            clock=clock,
            namespace=namespace,
            logger=self._log,
            timeout_s=timeout_s,
        )
        self.summarizer = SessionSummarizer(
            backend,
            settings,
            llm=llm,
            embedder=embedder,
            reviewer=self.reviewer,
            clock=clock,
            namespace=namespace,
            logger=self._log,
            timeout_s=timeout_s,
            memory_write_limit=self.write_limit,
        )
        self.scheduler = ConsolidateScheduler(
            backend,
            settings,
            consolidate_fn=self.consolidate,
            clock=clock,
            embedder=embedder,
            logger=self._log,
        )
        self._last_turn: dict[str, str] = {}
        self._negative_turns: dict[str, set[str]] = {}

    # --------------------------------------------------------------- events
    async def on_event(self, topic: str, payload: Any = None) -> dict[str, Any]:
        """Dispatch the two subscribed topics; anything else is dropped loudly."""
        if not self.settings.enabled:
            self._log.info("evolve disabled: event %s dropped", topic)
            return {"status": "disabled", "topic": topic}
        if topic == "session.turn":
            return await self.on_turn(payload)
        if topic == "session.closed":
            return await self.on_session_closed(payload)
        self._log.warning("evolve ignored unsupported topic=%s", topic)
        return {"status": "ignored", "topic": topic}

    async def on_turn(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        """`turn_end` + `turn_review` for one `session.turn` event."""
        if not self.settings.enabled:
            self._log.info("evolve disabled: turn dropped")
            return {"status": "disabled", "written": 0, "review": {"status": "disabled"}}
        data = payload if isinstance(payload, Mapping) else {}
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not ULID_RE.match(session_id):
            self._log.warning("evolve_degraded trigger=turn_end reason=invalid_session_id")
            return {"status": "ignored", "written": 0, "review": {"status": "ignored"}}

        extraction = rules.extract_from_payload(data)
        written = 0
        dropped = 0
        for candidate in extraction.candidates:
            text = candidate.text[: self.write_limit].strip()
            if not text:
                continue
            digest = content_hash("session", session_id, "", text)
            if self.backend.find_by_hash("session", session_id, "", digest) is not None:
                continue
            vector = self._embed(text)
            if vector is None:
                # A session row cannot exist without a vector; skipping is a
                # degradation, not a half write.
                dropped += 1
                self._log.warning("evolve_degraded trigger=turn_end reason=embedding")
                continue
            rec = MemoryRecord(
                id="",
                layer="session",
                namespace="",
                session_id=session_id,
                text=text,
                tags=[candidate.signal],
                content_hash=digest,
                importance=float(candidate.importance),
                confidence=1.0,
                source="evolve",
                kind="preference" if candidate.signal == "preference" else "fact",
                created_at=int(self._clock()),
                embedding=vector,
            )
            try:
                self.backend.put(rec)
            except MemoryFailure as exc:
                dropped += 1
                self._log.warning("evolve_degraded trigger=turn_end reason=%s", exc.code)
                continue
            written += 1

        previous = self._last_turn.get(session_id)
        if extraction.negative_feedback and previous is not None:
            self._negative_turns.setdefault(session_id, set()).add(previous)
        turn_id = data.get("turn_id")
        if isinstance(turn_id, str) and turn_id:
            self._last_turn[session_id] = turn_id

        review = await self.reviewer.on_turn(data)
        return {
            "status": "ok",
            "written": written,
            "dropped": dropped,
            "matched": extraction.matched,
            "negative_feedback": extraction.negative_feedback,
            "review": review.as_dict(),
        }

    async def on_session_closed(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        """`session_close`: summary + staging candidates, gated by turn count."""
        if not self.settings.enabled:
            self._log.info("evolve disabled: session close dropped")
            return {"status": "disabled"}
        data = dict(payload) if isinstance(payload, Mapping) else {}
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not ULID_RE.match(session_id):
            self._log.warning("evolve_degraded trigger=session_close reason=invalid_session_id")
            return {"status": "ignored"}

        turns = data.get("turns")
        flagged = self._negative_turns.get(session_id, set())
        if isinstance(turns, list) and flagged:
            merged: list[Any] = []
            for turn in turns:
                if isinstance(turn, Mapping):
                    item = dict(turn)
                    if item.get("turn_id") in flagged and "negative" not in item:
                        item["negative"] = True
                    merged.append(item)
                else:
                    merged.append(turn)
            data["turns"] = merged

        try:
            outcome: SummaryOutcome = await self.summarizer.on_session_close(data)
        finally:
            # The skeleton buffer and rolling summary exist only for the open
            # session; dropping them keeps memory bounded and satisfies "不保留
            # 转录超过一次处理周期".
            self.reviewer.forget(session_id)
            self._last_turn.pop(session_id, None)
            self._negative_turns.pop(session_id, None)
        return outcome.as_dict()

    # ----------------------------------------------------------- consolidate
    def consolidate(self, kind: str = "manual") -> ConsolidateStats:
        return consolidate(
            self.backend,
            self.settings,
            embedder=self.embedder,
            clock=self._clock,
            kind=kind,
        )

    def startup(self) -> ConsolidateStats | None:
        """Startup catch-up; also arms the daily timer."""
        stats = self.scheduler.startup_backfill()
        if self.settings.enabled:
            self.scheduler.arm()
        return stats

    # -------------------------------------------------------------- helpers
    def _embed(self, text: str) -> list[float] | None:
        if self.embedder is None:
            return None
        try:
            return [float(value) for value in self.embedder.embed(text)]
        except Exception:  # noqa: BLE001 - degrade to rules only
            return None
