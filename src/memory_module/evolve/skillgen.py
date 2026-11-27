"""`skill_evolve`: a successful trajectory → a *proposal*, never an install.

M4b §八. The four hard gates are conjunctive — a trajectory that fails any one
of them produces no skill at all:

1. `turn.state == "completed"` for the whole trajectory;
2. effective tool calls >= 5 **and** at least two distinct tools (reading five
   files is not a complex procedure);
3. `negative == false` — the user never rejected the approach;
4. `self_corrected == false` — a trajectory that overturned its own earlier
   conclusion is a debugging story, not a best practice.

A failed or self-corrected trajectory may still yield a *lesson*, and that is
`pipeline`'s job; this module only decides whether a skill proposal exists.

The output is a C-13 payload: `{name, action, content, base_revision,
derived_from}`. It is handed to core's `skill/proposal` RPC by `worker.py`; it
never writes `skills/` itself, so a proposal cannot become active without a
human approving it.

Nothing here reads the transcript: the caller passes the structured `turns[]`
from `session.closed` (tool names and counts only). Absolute paths, ULIDs and
e-mail addresses in a generated body are rejected because the body must be
reusable procedure, not this session's parameters (M4b §八).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..store import ULID_RE
from .config import EvolveSettings
from .llm import EvolveLLM, LlmCallable

__all__ = [
    "SkillDraft",
    "SkillBudget",
    "Trajectory",
    "SkillGenerator",
    "REQUIRED_SECTIONS",
    "MIN_TOOL_CALLS",
    "MIN_DISTINCT_TOOLS",
    "SkillRejected",
]

_log = logging.getLogger("memory_module.evolve")

REQUIRED_SECTIONS = ("When to Use", "Procedure", "Pitfalls", "Verification")
MIN_TOOL_CALLS = 5
MIN_DISTINCT_TOOLS = 2
MAX_BODY_CHARS = 8192
MAX_CONTENT_CHARS = 16384
NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,31}$")
_ABSOLUTE_PATH = re.compile(r"(?<![\w.])/(?:[\w.-]+/)*[\w.-]+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

_SKILL_PROMPT = """你是 Vade 的技能提炼器。输入是一次**成功**任务的轨迹（只有工具名、调用次数和状态，没有原文）。
任务：写一份可复用的 SKILL.md 正文，帮助下次遇到同类任务时少走弯路。

硬规则：
- 只输出一个 JSON 对象，不要 markdown 代码块，不要解释。
- 字段：name（小写连字符，2–32 字符）、description（一行，≤200 字）、
  when_to_use（何时使用）、procedure（步骤数组，每项一句）、
  pitfalls（至少 1 条）、verification（可执行的验证命令或可观察结果）。
- 正文只写可复用的步骤与判据，**不要**出现本次会话的绝对路径、ID、邮箱、URL 参数。
- Pitfalls 写不出说明任务本来没难度；Verification 写不出说明成功不可判定——那就应当输出 {"skip": true}。

轨迹：
"""


class SkillRejected(ValueError):
    """The proposal cannot be made; `reason` is a log-friendly code."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        self.message = message
        super().__init__(f"{reason}: {message}")


# --------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Trajectory:
    """The structured `turns[]` of one closed session (no raw text)."""

    turns: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_payload(cls, turns: Any) -> "Trajectory":
        if not isinstance(turns, list):
            return cls(())
        return cls(tuple(turn for turn in turns if isinstance(turn, Mapping)))

    # ------------------------------------------------------------- gates
    def tool_calls(self) -> list[Mapping[str, Any]]:
        calls: list[Mapping[str, Any]] = []
        for turn in self.turns:
            raw = turn.get("tool_calls")
            if isinstance(raw, list):
                calls.extend(call for call in raw if isinstance(call, Mapping))
        return calls

    def effective_tool_calls(self) -> int:
        """Calls that actually ran. A failed call is not a step of the recipe."""
        count = 0
        for call in self.tool_calls():
            if call.get("ok") is False:
                continue
            count += 1
        return count

    def distinct_tools(self) -> set[str]:
        names = set()
        for call in self.tool_calls():
            if call.get("ok") is False:
                continue
            name = str(call.get("tool") or "").strip()
            if name:
                names.add(name)
        return names

    def gate_failure(self) -> str | None:
        """Return the first failed gate name, or None when all four hold."""
        if not self.turns:
            return "no_turns"
        for turn in self.turns:
            if str(turn.get("state") or "") != "completed":
                return "not_completed"
        if self.effective_tool_calls() < MIN_TOOL_CALLS:
            return "too_few_tool_calls"
        if len(self.distinct_tools()) < MIN_DISTINCT_TOOLS:
            return "too_few_distinct_tools"
        if any(bool(turn.get("negative")) for turn in self.turns):
            return "negative_feedback"
        if any(bool(turn.get("self_corrected")) for turn in self.turns):
            return "self_corrected"
        return None

    def turn_ids(self) -> list[str]:
        return [str(turn.get("turn_id")) for turn in self.turns if turn.get("turn_id")]


# -------------------------------------------------------------------- budgets
@dataclass
class SkillBudget:
    """Per-session ≤ `session_budget_skills`, per-UTC-day ≤ `daily_budget_skills`.

    State is in memory for the life of the worker. That is deliberate: the
    budget exists to stop a runaway loop inside one session, and a restart is a
    new process with a human watching it. Persistent counters would need a
    schema change (`meta` is M4a's table), which this task may not make.
    """

    session_limit: int
    daily_limit: int
    clock: Any = None
    _day: str = field(default="", repr=False)
    _daily: int = field(default=0, repr=False)
    _sessions: dict[str, int] = field(default_factory=dict, repr=False)

    def _roll(self) -> None:
        now = int(self.clock() if self.clock is not None else time.time())
        today = time.strftime("%Y-%m-%d", time.gmtime(now))
        if today != self._day:
            self._day = today
            self._daily = 0

    def allow(self, session_id: str) -> bool:
        self._roll()
        if self.daily_limit > 0 and self._daily >= self.daily_limit:
            return False
        if self.session_limit > 0 and self._sessions.get(session_id, 0) >= self.session_limit:
            return False
        return True

    def spend(self, session_id: str) -> None:
        self._roll()
        self._daily += 1
        self._sessions[session_id] = self._sessions.get(session_id, 0) + 1


# --------------------------------------------------------------------- draft
@dataclass(frozen=True)
class SkillDraft:
    """A validated SKILL.md, ready to render into a C-13 payload."""

    name: str
    description: str
    when_to_use: str
    procedure: tuple[str, ...]
    pitfalls: tuple[str, ...]
    verification: tuple[str, ...]
    triggers: tuple[str, ...] = ()

    def body(self) -> str:
        lines: list[str] = [f"## {REQUIRED_SECTIONS[0]}", self.when_to_use.strip(), "",
                            f"## {REQUIRED_SECTIONS[1]}"]
        lines.extend(f"{index}. {step}" for index, step in enumerate(self.procedure, 1))
        lines.extend(["", f"## {REQUIRED_SECTIONS[2]}"])
        lines.extend(f"- {item}" for item in self.pitfalls)
        lines.extend(["", f"## {REQUIRED_SECTIONS[3]}"])
        for item in self.verification:
            lines.append(f"- `{item}`" if _looks_like_command(item) else f"- {item}")
        return "\n".join(lines).strip() + "\n"

    def render(self) -> str:
        """A complete SKILL.md: YAML frontmatter + the four required sections."""
        frontmatter = [
            "---",
            f"name: {json.dumps(self.name, ensure_ascii=False)}",
            f"description: {json.dumps(self.description, ensure_ascii=False)}",
            f"when_to_use: {json.dumps(self.when_to_use, ensure_ascii=False)}",
        ]
        if self.triggers:
            frontmatter.append("triggers: " + json.dumps(list(self.triggers), ensure_ascii=False))
        frontmatter.append("---")
        return "\n".join(frontmatter) + "\n\n" + self.body()


def _looks_like_command(text: str) -> bool:
    return bool(re.match(r"^(?:\w+ |python |pytest|npm |cargo |go |make )", text.strip()))


def _section_items(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = payload.get(key)
    if isinstance(raw, str):
        raw = [line for line in raw.splitlines() if line.strip()]
    if not isinstance(raw, (list, tuple)):
        return ()
    out: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip()[:300])
    return tuple(out)


def _validate(payload: Mapping[str, Any]) -> SkillDraft:
    """Turn the model's JSON into a `SkillDraft` or raise `SkillRejected`."""
    if payload.get("skip") is True:
        raise SkillRejected("skip", "the model judged the trajectory not reusable")
    name = str(payload.get("name") or "").strip().lower()
    if not NAME_RE.match(name):
        raise SkillRejected("schema_invalid", f"name {name!r} violates ^[a-z][a-z0-9-]{{1,31}}$")
    description = str(payload.get("description") or "").strip()
    if not description or len(description) > 200:
        raise SkillRejected("schema_invalid", "description must be 1..200 characters")
    when_to_use = str(payload.get("when_to_use") or "").strip()
    if not when_to_use:
        raise SkillRejected("schema_invalid", "when_to_use is empty")
    procedure = _section_items(payload, "procedure")
    pitfalls = _section_items(payload, "pitfalls")
    verification = _section_items(payload, "verification")
    if not procedure:
        raise SkillRejected("schema_invalid", "Procedure is empty")
    if not pitfalls:
        raise SkillRejected("schema_invalid", "Pitfalls is empty")
    if not verification:
        raise SkillRejected("schema_invalid", "Verification is empty")
    triggers = tuple(str(item).strip() for item in (payload.get("triggers") or []) if str(item).strip())
    draft = SkillDraft(
        name=name,
        description=description,
        when_to_use=when_to_use[:500],
        procedure=procedure,
        pitfalls=pitfalls,
        verification=verification,
        triggers=triggers[:8],
    )
    body = draft.body()
    if len(body) > MAX_BODY_CHARS:
        raise SkillRejected("schema_invalid", f"body exceeds {MAX_BODY_CHARS} characters")
    if len(draft.render()) > MAX_CONTENT_CHARS:
        raise SkillRejected("schema_invalid", f"SKILL.md exceeds {MAX_CONTENT_CHARS} characters")
    return draft


def leaks(draft: SkillDraft) -> list[str]:
    """Absolute paths / ULIDs / e-mails that must not be baked into a manual."""
    body = draft.body()
    found: list[str] = []
    if _ABSOLUTE_PATH.search(body):
        found.append("absolute_path")
    if ULID_RE.search(body):
        found.append("ulid")
    if _EMAIL.search(body):
        found.append("email")
    return found


# ----------------------------------------------------------------- generator
class SkillGenerator:
    """Gate → generate → self-check → C-13 payload. Never writes a skill."""

    def __init__(
        self,
        settings: EvolveSettings,
        *,
        llm: LlmCallable | None,
        clock: Any = None,
        logger: logging.Logger | None = None,
        timeout_s: float = 30.0,
        budget: SkillBudget | None = None,
        model: str = "",
    ) -> None:
        self.settings = settings
        self._log = logger or _log
        self._llm = EvolveLLM(llm, settings, clock=clock, timeout_s=timeout_s, logger=self._log)
        self._clock = clock or (lambda: int(time.time()))
        self.budget = budget or SkillBudget(
            settings.session_budget_skills, settings.daily_budget_skills, clock=clock
        )
        self.model = model

    async def maybe_propose(
        self,
        *,
        session_id: str,
        turns: Any,
        catalog: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any] | None:
        """Return a C-13 payload, or None when no proposal is warranted.

        A `SkillRejected` is a decision, not a failure: it is logged and turned
        into None so `session.closed` never fails because of it.
        """
        if not self.settings.enabled:
            return None
        trajectory = Trajectory.from_payload(turns)
        failure = trajectory.gate_failure()
        if failure is not None:
            self._log.info("evolve_skill_skipped reason=%s", failure)
            return None
        if not self.budget.allow(session_id):
            self._log.warning("evolve_budget_exceeded trigger=skill_evolve session=%s", session_id)
            return None

        prompt = _SKILL_PROMPT + json.dumps(
            [
                {
                    "turn_id": turn.get("turn_id"),
                    "state": turn.get("state"),
                    "tools": [
                        str(call.get("tool"))
                        for call in (turn.get("tool_calls") or [])
                        if isinstance(call, Mapping) and call.get("ok") is not False
                    ],
                }
                for turn in trajectory.turns
            ],
            ensure_ascii=False,
        )

        draft: SkillDraft | None = None
        for attempt in (1, 2):
            prompt_now = prompt if attempt == 1 else (
                prompt + "\n\n[纠正] 上一版缺少必需段落、名字不合法或含具体路径/ID/邮箱，修正后重写。"
            )
            result = await self._llm.json_call(prompt_now, label="skill_evolve")
            if not result.ok:
                return None
            try:
                draft = _validate(result.payload or {})
            except SkillRejected as exc:
                if exc.reason == "skip":
                    self._log.info("evolve_skill_skipped reason=model_skip")
                    return None
                self._log.warning(
                    "evolve_schema_invalid trigger=skill_evolve attempt=%d reason=%s",
                    attempt, exc.reason,
                )
                if attempt == 2:
                    return None
                draft = None
                continue
            bad = leaks(draft)
            if not bad:
                break
            self._log.warning("evolve_skill_leak trigger=skill_evolve kinds=%s", ",".join(bad))
            if attempt == 2:
                return None
            draft = None
        if draft is None:
            return None

        self.budget.spend(session_id)
        action, base_revision = _match_catalog(draft, catalog)
        payload: dict[str, Any] = {
            "name": draft.name,
            "action": action,
            "content": draft.render(),
            "derived_from": {
                "session_id": session_id,
                "turn_ids": trajectory.turn_ids()[:64],
            },
        }
        if self.model:
            payload["derived_from"]["model"] = self.model[:128]
        if action == "patch":
            # C-13: `create` must be absent/0; `patch` must carry the active
            # revision. A stale value is rejected by core with TOOL_CONFLICT and
            # the proposal is NOT retried (that would loop).
            payload["base_revision"] = base_revision
        return payload


def _match_catalog(
    draft: SkillDraft, catalog: Sequence[Mapping[str, Any]]
) -> tuple[str, int]:
    """`patch` when an active skill covers this task, else `create`."""
    wanted_name = draft.name
    wanted_words = set(re.findall(r"[\w\u4e00-\u9fff]+", draft.description.lower()))
    for entry in catalog:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "")
        if name == wanted_name:
            revision = int(entry.get("revision") or 0)
            return "patch", revision
        description = str(entry.get("description") or "").lower()
        overlap = wanted_words & set(re.findall(r"[\w\u4e00-\u9fff]+", description))
        if len(overlap) >= 4:
            revision = int(entry.get("revision") or 0)
            return "patch", revision
    return "create", 0
