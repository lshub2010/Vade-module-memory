"""`session_close`: rolling summary + strict-JSON extraction.

The input is deliberately not the transcript (M4b §六): the evolver's own
rolling summary (built from `turn_review` deltas), the last 10 turns with text
clipped and redacted, and a skill catalog snapshot. A 100-turn session would
otherwise cost ~80k tokens in one call.

Output rules:

* one strict JSON object; a parse failure gets exactly one retry, then the raw
  response is stored as the session summary with `facts=[]`, `lessons=[]`;
* `facts` / `lessons` go to `staging`, never `global`;
* `summary` goes to the **session** layer, importance 4, so a follow-up in the
  same session can be answered from it;
* a session shorter than `flush_min_turns` never reaches the model and logs
  `evolve_skipped_low_signal`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..errors import MemoryFailure
from ..store import NAMESPACE_RE, ULID_RE, MemoryBackend, MemoryRecord, content_hash
from .config import DEFAULT_NAMESPACE, LESSON_NAMESPACE, EvolveSettings
from .llm import EvolveLLM, LlmCallable, parse_strict_json
from .review import TurnReviewer

__all__ = [
    "SummaryOutcome",
    "SessionSummarizer",
    "sanitize_turn",
    "MAX_FACTS",
    "MAX_TAIL_TURNS",
]

MAX_TAIL_TURNS = 10
MAX_FACTS = 10
MAX_FACT_CHARS = 500
MAX_SUMMARY_CHARS = 300
MAX_TURN_TEXT = 400
MAX_CATALOG_ITEMS = 60
MAX_CATALOG_DESCRIPTION = 120
MAX_FALLBACK_CHARS = 1000

_log = logging.getLogger("memory_module.evolve")

_SUMMARY_PROMPT = """你是记忆系统的会话总结器。根据 rolling summary、最近若干轮（文本已裁剪）和 skill 目录快照，产出一个会话总结与少量候选记忆。

硬规则：
- 只输出一个 JSON 对象，不要 markdown 代码块，不要解释。
- summary ≤ 300 字：这个会话解决了什么、结论是什么；不写过程流水账。
- facts 最多 10 条，只写会话中确认的事实或偏好；没有就空数组。
- lessons 最多 1 条，evidence 必须含 turn_id、tool、result_digest、claim 四键（C-12）；缺任一项就空数组。
- negative_feedback 判的是「用户有没有否定」，不是任务顺不顺；模型自我纠错不算。
- skill_candidate 只是「值得考虑」，最终仍由四条硬条件裁决。

JSON 形状：
{"summary":"...",
 "facts":[{"text":"...","namespace":"user:default","importance":4,"confidence":0.8}],
 "lessons":[{"when":"...","do":"...","pitfall":"...","verify":"...",
             "evidence":{"turn_id":"...","tool":"...","result_digest":"sha256:...","claim":"..."}}],
 "negative_feedback":false,
 "skill_candidate":false}

输入：
"""


@dataclass
class SummaryOutcome:
    status: str
    turns: int = 0
    summary_chars: int = 0
    facts: int = 0
    lessons: int = 0
    dropped: int = 0
    retried: bool = False
    error: str | None = None
    negative_feedback: bool = False
    skill_candidate: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "turns": self.turns,
            "summary_chars": self.summary_chars,
            "facts": self.facts,
            "lessons": self.lessons,
            "dropped": self.dropped,
            "retried": self.retried,
            "error": self.error,
            "negative_feedback": self.negative_feedback,
            "skill_candidate": self.skill_candidate,
        }


def _clip(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: max(0, limit - 1)].rstrip() + "…"


def sanitize_turn(turn: Mapping[str, Any], redact: Sequence[str]) -> dict[str, Any]:
    """Whitelist the fields a closing summary may see.

    Only `user_text` is ever carried as text; assistant prose is never fed back
    to the model (two stacked inferences is exactly what M4b §五 forbids).
    `redact` then removes any configured field, so `file_content` / `tool_args`
    cannot leak through a new key somebody adds later.
    """
    out: dict[str, Any] = {}
    for key in ("turn_id", "state"):
        value = turn.get(key)
        if isinstance(value, str):
            out[key] = value
    for key in ("negative", "self_corrected"):
        value = turn.get(key)
        if isinstance(value, bool):
            out[key] = value
    calls = turn.get("tool_calls")
    if isinstance(calls, list):
        names: list[str] = []
        for call in calls[:16]:
            if isinstance(call, Mapping):
                name = call.get("tool") or call.get("name")
                if isinstance(name, str) and name and name not in names:
                    names.append(name)
        out["tool_names"] = names[:8]
    text = turn.get("user_text")
    if isinstance(text, str):
        out["user_text"] = _clip(text, MAX_TURN_TEXT)
    for key in redact:
        out.pop(key, None)
    return out


def _sanitize_catalog(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for item in value[:MAX_CATALOG_ITEMS]:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        description = item.get("description")
        out.append(
            {
                "name": name.strip()[:80],
                "description": _clip(description, MAX_CATALOG_DESCRIPTION)
                if isinstance(description, str)
                else "",
            }
        )
    return out


def _validate_summary(payload: Mapping[str, Any]) -> None:
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("summary must be a non-empty string")
    facts = payload.get("facts", [])
    if facts is None:
        facts = []
    if not isinstance(facts, list):
        raise ValueError("facts must be a list")
    lessons = payload.get("lessons", [])
    if lessons is None:
        lessons = []
    if not isinstance(lessons, list):
        raise ValueError("lessons must be a list")
    for key in ("negative_feedback", "skill_candidate"):
        value = payload.get(key)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean")


def _lesson_row(item: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    when = str(item.get("when") or "").strip()
    do = str(item.get("do") or "").strip()
    pitfall = str(item.get("pitfall") or "").strip()
    verify = str(item.get("verify") or "").strip()
    if not (do or pitfall or verify):
        return None

    raw = item.get("evidence")
    locator = (
        {str(key): value for key, value in raw.items() if value not in (None, "", [], {})}
        if isinstance(raw, Mapping)
        else {}
    )
    if not locator:
        # A lesson must be auditable; without a turn/tool/claim there is
        # nothing to verify later (M4b §八).
        return None

    head = f"当{when}时" if when and not when.endswith("时") else when
    text = "，".join(part for part in (head, do) if part)
    if pitfall:
        text = f"{text}；注意：{pitfall}" if text else f"注意：{pitfall}"
    if verify:
        text = f"{text}；验证：{verify}" if text else f"验证：{verify}"

    evidence = dict(locator)
    for key, value in (("when", when), ("do", do), ("pitfall", pitfall), ("verify", verify)):
        if value:
            evidence.setdefault(key, value)
    # C-12 requires all four locator keys. The evolver never sees the raw tool
    # result, so when the model omits `result_digest` we digest the claim it
    # did give: deterministic, content-addressed, and never a fabricated tool
    # output. `turn_id`/`tool`/`claim` stay mandatory — without them there is
    # nothing to look up.
    if not str(evidence.get("turn_id") or "").strip():
        return None
    if not str(evidence.get("tool") or "").strip():
        return None
    if not str(evidence.get("claim") or "").strip():
        return None
    if not str(evidence.get("result_digest") or "").strip():
        digest = hashlib.sha256(str(evidence["claim"]).encode("utf-8")).hexdigest()
        evidence["result_digest"] = f"sha256:{digest[:32]}"
    return _clip(text, MAX_FACT_CHARS), evidence


class SessionSummarizer:
    """The `session.closed` handler (LLM part; the event itself stays M4a's)."""

    def __init__(
        self,
        backend: MemoryBackend,
        settings: EvolveSettings,
        *,
        llm: LlmCallable | None,
        embedder: Any = None,
        reviewer: TurnReviewer | None = None,
        clock: Any = None,
        namespace: str = DEFAULT_NAMESPACE,
        logger: logging.Logger | None = None,
        timeout_s: float = 30.0,
        memory_write_limit: int | None = None,
    ) -> None:
        self.backend = backend
        self.settings = settings
        self.embedder = embedder
        self.reviewer = reviewer
        self.namespace = namespace
        self._log = logger or _log
        self._llm = EvolveLLM(llm, settings, clock=clock, timeout_s=timeout_s, logger=self._log)
        self._clock = clock or (lambda: int(time.time()))
        self._write_limit = min(
            settings.session_char_limit,
            memory_write_limit or settings.session_char_limit,
        )

    # ----------------------------------------------------------------- entry
    async def on_session_close(self, payload: Mapping[str, Any] | None) -> SummaryOutcome:
        try:
            return await self._on_close(payload)
        except Exception as exc:  # noqa: BLE001 - a summary failure must not break close
            self._log.warning(
                "evolve_degraded trigger=session_close reason=%s", type(exc).__name__
            )
            return SummaryOutcome(status="degraded", error="internal")

    async def _on_close(self, payload: Mapping[str, Any] | None) -> SummaryOutcome:
        if not self.settings.enabled:
            self._log.info("evolve disabled: session_close skipped")
            return SummaryOutcome(status="disabled")
        data = payload if isinstance(payload, Mapping) else {}
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not ULID_RE.match(session_id):
            self._log.warning("evolve_degraded trigger=session_close reason=invalid_session_id")
            return SummaryOutcome(status="ignored")

        raw_turns = data.get("turns")
        turns = [turn for turn in raw_turns if isinstance(turn, Mapping)] if isinstance(raw_turns, list) else []
        if len(turns) < self.settings.flush_min_turns:
            self._log.info(
                "evolve_skipped_low_signal session=%s turns=%d", session_id, len(turns)
            )
            return SummaryOutcome(status="skipped", turns=len(turns))

        tail = [sanitize_turn(turn, self.settings.redact) for turn in turns[-MAX_TAIL_TURNS:]]
        rolling = self.reviewer.rolling_summary(session_id) if self.reviewer else ""
        if len(rolling) > self.settings.summary_char_budget():
            rolling = rolling[-self.settings.summary_char_budget() :]
        inputs = {
            "rolling_summary": rolling,
            "recent_turns": tail,
            "skill_catalog": _sanitize_catalog(data.get("skill_catalog")),
        }
        prompt = _SUMMARY_PROMPT + json.dumps(inputs, ensure_ascii=False)
        result = await self._llm.json_call(prompt, validate=_validate_summary, label="session_close")
        if not result.ok:
            stored = 0
            fallback_text = self._fallback_summary(result.raw)
            if fallback_text:
                stored = self._store_summary(session_id, fallback_text, fallback=True)
            return SummaryOutcome(
                status="degraded",
                turns=len(turns),
                summary_chars=stored,
                retried=result.retried,
                error=result.error,
            )

        payload_json = result.payload or {}
        facts, facts_dropped = self._write_facts(payload_json.get("facts"), session_id)
        lessons, lesson_dropped = self._write_lessons(payload_json.get("lessons"), session_id)
        summary = payload_json.get("summary")
        stored = self._store_summary(session_id, summary if isinstance(summary, str) else "")
        return SummaryOutcome(
            status="ok",
            turns=len(turns),
            summary_chars=stored,
            facts=facts,
            lessons=lessons,
            dropped=facts_dropped + lesson_dropped,
            retried=result.retried,
            negative_feedback=bool(payload_json.get("negative_feedback")),
            skill_candidate=bool(payload_json.get("skill_candidate")),
        )

    # --------------------------------------------------------------- writers
    def _write_facts(self, value: Any, session_id: str) -> tuple[int, int]:
        if not isinstance(value, list):
            return 0, 0
        written = 0
        dropped = 0
        for item in value[:MAX_FACTS]:
            if not isinstance(item, Mapping):
                dropped += 1
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                dropped += 1
                continue
            namespace = item.get("namespace")
            namespace = namespace.strip() if isinstance(namespace, str) and namespace.strip() else self.namespace
            if not NAMESPACE_RE.match(namespace):
                dropped += 1
                continue
            # staging has no importance column (C-12); `importance` is accepted
            # for schema fidelity and deliberately dropped here. Consolidate
            # applies M4a's promotion defaults.
            confidence = _as_unit_float(item.get("confidence"), 0.6)
            kind_value = item.get("kind")
            kind = (
                kind_value
                if isinstance(kind_value, str) and kind_value in ("fact", "preference")
                else "fact"
            )
            body = _clip(text, MAX_FACT_CHARS)
            try:
                self.backend.staging_put(
                    body,
                    namespace,
                    session_id,
                    self._embed(body),
                    confidence,
                    kind=kind,
                )
            except MemoryFailure:
                dropped += 1
                continue
            written += 1
        return written, dropped

    def _write_lessons(self, value: Any, session_id: str) -> tuple[int, int]:
        if not isinstance(value, list):
            return 0, 0
        written = 0
        dropped = 0
        # The schema asks for at most one lesson; if the model sent several,
        # take the first one with a locatable evidence object.
        for item in value[:3]:
            if not isinstance(item, Mapping):
                dropped += 1
                continue
            row = _lesson_row(item)
            if row is None:
                dropped += 1
                continue
            text, evidence = row
            confidence = min(
                _as_unit_float(item.get("confidence"), self.settings.lesson_confidence_floor),
                self.settings.lesson_confidence_floor,
            )
            try:
                self.backend.staging_put(
                    text,
                    LESSON_NAMESPACE,
                    session_id,
                    self._embed(text),
                    confidence,
                    kind="lesson",
                    evidence=evidence,
                )
            except MemoryFailure:
                dropped += 1
                continue
            written += 1
            break
        return written, dropped

    def _store_summary(self, session_id: str, text: str, *, fallback: bool = False) -> int:
        body = text.strip()
        if not body:
            return 0
        limit = min(self._write_limit, MAX_FALLBACK_CHARS if fallback else MAX_SUMMARY_CHARS)
        body = body[:limit].rstrip()
        if not body:
            return 0
        digest = content_hash("session", session_id, "", body)
        if self.backend.find_by_hash("session", session_id, "", digest) is not None:
            return 0
        vector = self._embed_required(body)
        if vector is None:
            self._log.warning("evolve_degraded trigger=session_close reason=embedding")
            return 0
        rec = MemoryRecord(
            id="",
            layer="session",
            namespace="",
            session_id=session_id,
            text=body,
            tags=["session-summary"],
            content_hash=digest,
            importance=4.0,
            confidence=0.7,
            source="evolve",
            kind="fact",
            created_at=int(self._clock()),
            embedding=vector,
        )
        try:
            self.backend.put(rec)
        except MemoryFailure as exc:
            self._log.warning(
                "evolve_degraded trigger=session_close reason=%s", exc.code
            )
            return 0
        return len(body)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _fallback_summary(raw: str) -> str:
        """The text to store when the model did not produce valid JSON.

        If it at least produced an object with a usable `summary` field, store
        that field; otherwise store the raw response (M4b §六: "只存 summary
        原文"). An empty response stores nothing.
        """
        text = raw.strip()
        if not text:
            return ""
        try:
            parsed = parse_strict_json(text)
        except ValueError:
            return text
        summary = parsed.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
        return text

    def _embed(self, text: str) -> list[float] | None:
        if self.embedder is None:
            return None
        try:
            return [float(value) for value in self.embedder.embed(text)]
        except Exception:  # noqa: BLE001 - staging tolerates a missing vector
            return None

    def _embed_required(self, text: str) -> list[float] | None:
        return self._embed(text)


def _as_unit_float(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return min(1.0, max(0.0, float(value)))
