"""`turn_review`: side-channel reflection over skeletons, staging only.

Runs every `nudge_interval` turns on the same `session.turn` stream (M4b §六·五
— no new event topic). The input is the per-session ring buffer of *skeletons*,
never the transcript: one turn contributes a state, tool names, and the first
and last sentence clipped to 80 characters. That is enough to see "third retry
succeeded" or "switched from plan A to B" without keeping raw text around.

Every output goes to `staging`, never `session` and never `global`: a review is
a model inference, and inferences must pass consolidate's promotion gate. The
call runs in a 30 s `wait_for`; timeout, provider error or two parse failures
log `evolve_degraded` and return — `turn_end` is never blocked.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..errors import MemoryFailure
from ..store import ULID_RE, MemoryBackend, content_hash
from ..tools import estimate_tokens
from .config import DEFAULT_NAMESPACE, LESSON_NAMESPACE, EvolveSettings
from .llm import EvolveLLM, LlmCallable

__all__ = [
    "Skeleton",
    "ReviewOutcome",
    "TurnReviewer",
    "build_skeleton",
    "MAX_CANDIDATES",
    "MAX_PROMPT_TOKENS",
]

MAX_CANDIDATES = 3
MAX_PROMPT_TOKENS = 1500
MAX_TOOL_NAMES = 6
MAX_SKELETON_SENTENCE = 80
MAX_CANDIDATE_CHARS = 500
#: A single review is one session's observation; it is not verified evidence.
#: Consolidate still requires two distinct sessions for fact/preference.
REVIEW_CONFIDENCE = 0.55

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")
_log = logging.getLogger("memory_module.evolve")

_REVIEW_PROMPT = """你是记忆系统的旁路复盘器。输入只有最近若干轮的结构化骨架（没有原文）。
任务：从状态变化里识别值得长期记住的事实/偏好，以及至多一条可复用教训。

硬规则：
- 只输出一个 JSON 对象，不要 markdown 代码块，不要解释。
- 多数情况下应当什么都不产出；宁少勿滥，产出率高说明阈值太低。
- candidates 最多 3 条；lesson 最多 1 条，且 evidence.turn_id 必须来自输入骨架，否则 lesson 为 null。
- 不要复述过程，只写可复用的结论。
- summary_delta 只写这几轮新增的结论或转折，不重复已有摘要。

JSON 形状：
{"candidates":[{"text":"...","kind":"fact|preference","importance":4}],
 "lesson":{"when":"...","do":"...","pitfall":"...","verify":"...",
           "evidence":{"turn_id":"...","tool":"...","result_digest":"sha256:...","claim":"..."}},
 "summary_delta":"..."}

轮次骨架：
"""


@dataclass(frozen=True)
class Skeleton:
    """One turn's structural residue (M4b §六·五)."""

    turn_id: str
    state: str
    tool_names: tuple[str, ...] = ()
    first_sentence: str = ""
    last_sentence: str = ""
    error_flag: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "state": self.state,
            "tool_names": list(self.tool_names),
            "first_sentence": self.first_sentence,
            "last_sentence": self.last_sentence,
            "error_flag": self.error_flag,
        }


@dataclass
class ReviewOutcome:
    status: str
    turns: int = 0
    candidates: int = 0
    lessons: int = 0
    deduped: int = 0
    dropped: int = 0
    retried: bool = False
    error: str | None = None
    summary_chars: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "turns": self.turns,
            "candidates": self.candidates,
            "lessons": self.lessons,
            "deduped": self.deduped,
            "dropped": self.dropped,
            "retried": self.retried,
            "error": self.error,
            "summary_chars": self.summary_chars,
        }


def _clip(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: max(0, limit - 1)].rstrip() + "…"


def _sentence_edges(text: str) -> tuple[str, str]:
    if not text or not text.strip():
        return "", ""
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text)]
    parts = [part for part in parts if part]
    if not parts:
        return "", ""
    return _clip(parts[0], MAX_SKELETON_SENTENCE), _clip(parts[-1], MAX_SKELETON_SENTENCE)


def build_skeleton(payload: Mapping[str, Any]) -> Skeleton:
    """Turn a `session.turn` payload into the only shape review may see."""
    user_text = payload.get("user_text")
    first, last = _sentence_edges(user_text if isinstance(user_text, str) else "")

    tool_names: list[str] = []
    error_flag = False
    calls = payload.get("tool_calls")
    if isinstance(calls, list):
        for call in calls[:16]:
            if not isinstance(call, Mapping):
                continue
            name = call.get("tool") or call.get("name")
            if isinstance(name, str) and name and name not in tool_names and len(tool_names) < MAX_TOOL_NAMES:
                tool_names.append(name)
            if call.get("ok") is False:
                error_flag = True

    state = payload.get("state")
    state = state if isinstance(state, str) and state else "unknown"
    error_flag = error_flag or state in ("failed", "interrupted") or bool(payload.get("error_flag"))
    turn_id = payload.get("turn_id")
    return Skeleton(
        turn_id=turn_id if isinstance(turn_id, str) else "",
        state=state,
        tool_names=tuple(tool_names),
        first_sentence=first,
        last_sentence=last,
        error_flag=error_flag,
    )


def _validate_review(payload: Mapping[str, Any]) -> None:
    candidates = payload.get("candidates", [])
    if candidates is None:
        candidates = []
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    lesson = payload.get("lesson")
    if lesson is not None and not isinstance(lesson, Mapping):
        raise ValueError("lesson must be an object or null")
    delta = payload.get("summary_delta", "")
    if delta is not None and not isinstance(delta, str):
        raise ValueError("summary_delta must be a string")


def _lesson_text(item: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Flatten a lesson into row text while keeping the locator in evidence.

    `when/do/pitfall/verify` are the lesson body; `turn_id/tool/claim` are what
    makes it auditable. A lesson without a locator is dropped (M4b §八: "证据
    可定位"), which is why the four sections are added *after* the locator
    check instead of making every lesson look evidenced.
    """
    when = str(item.get("when") or "").strip()
    do = str(item.get("do") or "").strip()
    pitfall = str(item.get("pitfall") or "").strip()
    verify = str(item.get("verify") or "").strip()
    if not (do or pitfall or verify):
        return "", {}

    raw = item.get("evidence")
    locator = (
        {str(key): value for key, value in raw.items() if value not in (None, "", [], {})}
        if isinstance(raw, Mapping)
        else {}
    )
    if not locator:
        return "", {}

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
    # C-12 `Evidence` requires all four keys; the reviewer never receives the
    # raw tool result, so a missing digest is derived from the claim (stable,
    # content-addressed) rather than invented. turn/tool/claim stay mandatory.
    for key in ("turn_id", "tool", "claim"):
        if not str(evidence.get(key) or "").strip():
            return "", {}
    if not str(evidence.get("result_digest") or "").strip():
        digest = hashlib.sha256(str(evidence["claim"]).encode("utf-8")).hexdigest()
        evidence["result_digest"] = f"sha256:{digest[:32]}"
    return _clip(text, MAX_CANDIDATE_CHARS), evidence


class TurnReviewer:
    """Per-session skeleton ring buffer + the review call."""

    def __init__(
        self,
        backend: MemoryBackend,
        settings: EvolveSettings,
        *,
        llm: LlmCallable | None,
        embedder: Any = None,
        clock: Any = None,
        namespace: str = DEFAULT_NAMESPACE,
        logger: logging.Logger | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.backend = backend
        self.settings = settings
        self.embedder = embedder
        self.namespace = namespace
        self._log = logger or _log
        self._llm = EvolveLLM(llm, settings, clock=clock, timeout_s=timeout_s, logger=self._log)
        self._buffers: dict[str, deque[Skeleton]] = {}
        self._counts: dict[str, int] = {}
        self._summaries: dict[str, str] = {}

    # -------------------------------------------------------------- inspect
    def turns_seen(self, session_id: str) -> int:
        return int(self._counts.get(session_id, 0))

    def buffered(self, session_id: str) -> list[Skeleton]:
        return list(self._buffers.get(session_id, ()))

    def rolling_summary(self, session_id: str) -> str:
        return self._summaries.get(session_id, "")

    def forget(self, session_id: str) -> None:
        self._buffers.pop(session_id, None)
        self._counts.pop(session_id, None)
        self._summaries.pop(session_id, None)

    # ----------------------------------------------------------------- entry
    async def on_turn(self, payload: Mapping[str, Any] | None) -> ReviewOutcome:
        try:
            return await self._on_turn(payload)
        except Exception as exc:  # noqa: BLE001 - the evolver never breaks a session
            self._log.warning(
                "evolve_degraded trigger=turn_review reason=%s", type(exc).__name__
            )
            return ReviewOutcome(status="degraded", error="internal")

    async def _on_turn(self, payload: Mapping[str, Any] | None) -> ReviewOutcome:
        if not self.settings.enabled:
            self._log.info("evolve disabled: turn_review skipped")
            return ReviewOutcome(status="disabled")
        data = payload if isinstance(payload, Mapping) else {}
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not ULID_RE.match(session_id):
            self._log.warning("evolve_degraded trigger=turn_review reason=invalid_session_id")
            return ReviewOutcome(status="ignored")

        buffer = self._buffers.get(session_id)
        if buffer is None:
            buffer = deque(maxlen=self.settings.nudge_interval)
            self._buffers[session_id] = buffer
        buffer.append(build_skeleton(data))
        count = self._counts.get(session_id, 0) + 1
        self._counts[session_id] = count
        if count % self.settings.nudge_interval != 0:
            return ReviewOutcome(status="buffered", turns=len(buffer))
        return await self._review(session_id, list(buffer))

    # ------------------------------------------------------------------ call
    async def _review(self, session_id: str, skeletons: Sequence[Skeleton]) -> ReviewOutcome:
        # The hard budget is 1.5k prompt tokens (M4b §六·五). At the default
        # interval ten skeletons fit; a raised interval trims the oldest
        # skeletons rather than blowing the budget (the doc's "don't cut turns"
        # rule assumes the default).
        kept = list(skeletons)
        prompt = self._build_prompt(kept)
        while len(kept) > 1 and estimate_tokens(prompt) > MAX_PROMPT_TOKENS:
            kept = kept[1:]
            prompt = self._build_prompt(kept)
        result = await self._llm.json_call(prompt, validate=_validate_review, label="turn_review")
        if not result.ok:
            return ReviewOutcome(
                status="degraded",
                turns=len(kept),
                retried=result.retried,
                error=result.error,
            )

        payload = result.payload or {}
        existing = {
            (row.kind, row.namespace, row.content_hash) for row in self.backend.staging_list()
        }
        written = 0
        deduped = 0
        dropped = 0

        raw_candidates = payload.get("candidates") or []
        for item in list(raw_candidates)[:MAX_CANDIDATES]:
            if not isinstance(item, Mapping):
                dropped += 1
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                dropped += 1
                continue
            text = _clip(text, MAX_CANDIDATE_CHARS)
            digest = content_hash("staging", "", self.namespace, text)
            if (("fact", self.namespace, digest)) in existing or (
                ("preference", self.namespace, digest)
            ) in existing:
                deduped += 1
                continue
            kind_value = item.get("kind")
            kind = (
                kind_value
                if isinstance(kind_value, str) and kind_value in ("fact", "preference")
                else "fact"
            )
            vector = self._embed(text)
            try:
                self.backend.staging_put(
                    text,
                    self.namespace,
                    session_id,
                    vector,
                    REVIEW_CONFIDENCE,
                    kind=kind,
                )
            except MemoryFailure:
                dropped += 1
                continue
            existing.add((kind, self.namespace, digest))
            written += 1

        lessons = 0
        lesson = payload.get("lesson")
        if isinstance(lesson, Mapping):
            text, evidence = _lesson_text(lesson)
            if not text or not evidence:
                dropped += 1
            else:
                digest = content_hash("staging", "", LESSON_NAMESPACE, text)
                if ("lesson", LESSON_NAMESPACE, digest) in existing:
                    deduped += 1
                else:
                    try:
                        self.backend.staging_put(
                            text,
                            LESSON_NAMESPACE,
                            session_id,
                            self._embed(text),
                            min(REVIEW_CONFIDENCE, self.settings.lesson_confidence_floor),
                            kind="lesson",
                            evidence=evidence,
                        )
                    except MemoryFailure:
                        dropped += 1
                    else:
                        existing.add(("lesson", LESSON_NAMESPACE, digest))
                        lessons = 1

        delta = payload.get("summary_delta")
        if isinstance(delta, str):
            self._append_summary(session_id, delta)

        return ReviewOutcome(
            status="reviewed",
            turns=len(kept),
            candidates=written,
            lessons=lessons,
            deduped=deduped,
            dropped=dropped,
            retried=result.retried,
            summary_chars=len(self._summaries.get(session_id, "")),
        )

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _build_prompt(skeletons: Sequence[Skeleton]) -> str:
        return _REVIEW_PROMPT + json.dumps(
            [item.to_dict() for item in skeletons], ensure_ascii=False
        )

    def _embed(self, text: str) -> list[float] | None:
        if self.embedder is None:
            return None
        try:
            return [float(value) for value in self.embedder.embed(text)]
        except Exception:  # noqa: BLE001 - staging allows a NULL embedding
            return None

    def _append_summary(self, session_id: str, delta: str) -> None:
        clean = " ".join(delta.split())
        if not clean:
            return
        current = self._summaries.get(session_id, "")
        merged = f"{current}\n{clean}".strip() if current else clean
        budget = self.settings.summary_char_budget()
        # Over budget: drop the oldest process detail, keep the latest
        # conclusions (M4b §六·五). Never truncate from the tail.
        while len(merged) > budget and "\n" in merged:
            merged = merged.split("\n", 1)[1].lstrip()
        if len(merged) > budget:
            merged = merged[-budget:]
        self._summaries[session_id] = merged
