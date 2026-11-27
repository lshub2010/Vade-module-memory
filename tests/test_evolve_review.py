"""turn_review: skeletons only, staging only, one retry, never blocks (M4b §六·五)."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from memory_module.embed import FakeEmbedding
from memory_module.evolve.config import EvolveSettings
from memory_module.evolve.review import TurnReviewer, build_skeleton
from memory_module.settings import MemorySettings
from memory_module.store import SqliteBackend, new_ulid
from memory_module.tools import estimate_tokens


class Clock:
    def __init__(self, start: int = 1_700_000_000) -> None:
        self.now = start

    def __call__(self) -> int:
        self.now += 1
        return self.now


class FakeLLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.responses:
            return "{}"
        return self.responses.pop(0)


class SlowLLM:
    def __init__(self, delay: float = 0.2) -> None:
        self.delay = delay
        self.calls = 0

    async def __call__(self, prompt: str) -> str:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return "{}"


def make_backend(tmp_path) -> SqliteBackend:
    settings = MemorySettings(
        data_dir=tmp_path,
        embedding_dim=8,
        global_capacity=100,
        session_capacity=50,
    )
    return SqliteBackend(settings, clock=Clock())


def evolve_settings(**overrides) -> EvolveSettings:
    kwargs = dict(enabled=True, nudge_interval=5, flush_min_turns=4)
    kwargs.update(overrides)
    return EvolveSettings(**kwargs)


def turn(session_id: str, index: int, user_text: str = "继续", **extra):
    payload = {
        "session_id": session_id,
        "turn_id": f"T{index:02d}",
        "state": "completed",
        "user_text": user_text,
        "assistant_text": "",
        "tool_calls": [],
    }
    payload.update(extra)
    return payload


REVIEW_OK = json.dumps(
    {
        "candidates": [
            {"text": "项目使用 SQLite 存储", "kind": "fact", "importance": 4},
            {"text": "用户偏好简洁回复", "kind": "preference", "importance": 4},
            {"text": "部署走本地回环", "kind": "fact", "importance": 3},
            {"text": "第四条不应被写入", "kind": "fact", "importance": 3},
            {"text": "第五条不应被写入", "kind": "fact", "importance": 3},
        ],
        "lesson": {
            "when": "部署失败",
            "do": "先检查端口占用",
            "pitfall": "只看最后一次日志",
            "verify": "curl -fsS /healthz",
            "evidence": {"turn_id": "T05", "tool": "shell__run", "claim": "exit 1 then 0"},
        },
        "summary_delta": "部署流程已经确定",
    },
    ensure_ascii=False,
)


def make_reviewer(backend, llm, settings=None, **kwargs) -> TurnReviewer:
    return TurnReviewer(
        backend,
        settings or evolve_settings(),
        llm=llm,
        embedder=FakeEmbedding(dim=8),
        **kwargs,
    )


def test_review_only_runs_every_nudge_interval(tmp_path):
    backend = make_backend(tmp_path)
    llm = FakeLLM(REVIEW_OK)
    reviewer = make_reviewer(backend, llm)
    sid = new_ulid()

    async def run():
        outcomes = []
        for index in range(1, 11):
            outcomes.append(await reviewer.on_turn(turn(sid, index, f"第 {index} 轮")))
        return outcomes

    outcomes = asyncio.run(run())
    assert [item.status for item in outcomes[:4]] == ["buffered"] * 4
    assert outcomes[4].status == "reviewed"
    assert [item.status for item in outcomes[5:9]] == ["buffered"] * 4
    assert outcomes[9].status == "reviewed"
    assert len(llm.calls) == 2
    assert reviewer.turns_seen(sid) == 10


def test_prompt_contains_skeletons_never_transcripts(tmp_path):
    backend = make_backend(tmp_path)
    llm = FakeLLM("{}")
    reviewer = make_reviewer(backend, llm)
    sid = new_ulid()

    long_text = "这是一段很长的用户原文" * 30
    async def run():
        for index in range(1, 6):
            await reviewer.on_turn(
                turn(
                    sid,
                    index,
                    long_text,
                    assistant_text="ASSISTANT_SECRET 结论是记住这个",
                    tool_calls=[
                        {"tool": "fs__read", "args_digest": "SECRET_DIGEST", "ok": True}
                    ],
                )
            )

    asyncio.run(run())
    assert len(llm.calls) == 1
    prompt = llm.calls[0]
    assert "first_sentence" in prompt and "last_sentence" in prompt
    assert "fs__read" in prompt
    assert "ASSISTANT_SECRET" not in prompt
    assert "SECRET_DIGEST" not in prompt
    assert long_text not in prompt  # clipped to 80 chars per edge
    assert "user_text" not in prompt


def test_output_lands_in_staging_only_and_is_capped(tmp_path):
    backend = make_backend(tmp_path)
    llm = FakeLLM(REVIEW_OK)
    reviewer = make_reviewer(backend, llm)
    sid = new_ulid()

    async def run():
        outcomes = []
        for index in range(1, 6):
            outcomes.append(await reviewer.on_turn(turn(sid, index)))
        return outcomes[-1]

    outcome = asyncio.run(run())
    assert outcome.status == "reviewed"
    assert outcome.candidates == 3
    assert outcome.lessons == 1
    assert outcome.dropped == 0
    assert backend.count(layer="session") == 0
    assert backend.count(layer="global") == 0
    rows = backend.staging_list()
    assert len(rows) == 4
    kinds = sorted(row.kind for row in rows)
    assert kinds == ["fact", "fact", "lesson", "preference"]
    lesson = next(row for row in rows if row.kind == "lesson")
    assert lesson.namespace == "agent:lessons"
    assert lesson.evidence["turn_id"] == "T05"
    assert lesson.confidence <= 0.6
    assert backend.find_by_hash("session", sid, "", rows[0].content_hash) is None


def test_lesson_without_evidence_is_dropped(tmp_path):
    backend = make_backend(tmp_path)
    payload = json.dumps(
        {
            "candidates": [],
            "lesson": {"when": "x", "do": "y", "evidence": {}},
            "summary_delta": "",
        },
        ensure_ascii=False,
    )
    reviewer = make_reviewer(backend, FakeLLM(payload))
    sid = new_ulid()

    async def run():
        outcomes = []
        for index in range(1, 6):
            outcomes.append(await reviewer.on_turn(turn(sid, index)))
        return outcomes[-1]

    outcome = asyncio.run(run())
    assert outcome.lessons == 0
    assert outcome.dropped == 1
    assert backend.staging_list() == []


def test_parse_failure_retries_once_then_degrades(tmp_path, caplog):
    backend = make_backend(tmp_path)
    llm = FakeLLM("this is not json", "still not json")
    reviewer = make_reviewer(backend, llm)
    sid = new_ulid()

    async def run():
        outcomes = []
        for index in range(1, 6):
            outcomes.append(await reviewer.on_turn(turn(sid, index)))
        return outcomes[-1]

    with caplog.at_level(logging.WARNING, logger="memory_module.evolve"):
        outcome = asyncio.run(run())

    assert outcome.status == "degraded"
    assert outcome.retried is True
    assert len(llm.calls) == 2
    assert "parse_error" in llm.calls[1]
    assert "Expecting value" in llm.calls[1]
    assert backend.staging_list() == []
    assert any("evolve_degraded" in record.message for record in caplog.records)


def test_parse_failure_then_success_uses_the_retry(tmp_path):
    backend = make_backend(tmp_path)
    llm = FakeLLM("not json", REVIEW_OK)
    reviewer = make_reviewer(backend, llm)
    sid = new_ulid()

    async def run():
        outcomes = []
        for index in range(1, 6):
            outcomes.append(await reviewer.on_turn(turn(sid, index)))
        return outcomes[-1]

    outcome = asyncio.run(run())
    assert outcome.status == "reviewed"
    assert outcome.retried is True
    assert len(llm.calls) == 2
    assert len(backend.staging_list()) == 4


def test_timeout_degrades_without_raising(tmp_path, caplog):
    backend = make_backend(tmp_path)
    llm = SlowLLM(delay=0.2)
    reviewer = make_reviewer(backend, llm, timeout_s=0.02)
    sid = new_ulid()

    async def run():
        outcomes = []
        for index in range(1, 6):
            outcomes.append(await reviewer.on_turn(turn(sid, index)))
        return outcomes[-1]

    with caplog.at_level(logging.WARNING, logger="memory_module.evolve"):
        outcome = asyncio.run(run())

    assert outcome.status == "degraded"
    assert outcome.error == "timeout"
    assert backend.staging_list() == []
    assert any("evolve_degraded" in record.message for record in caplog.records)


def test_repeated_review_output_is_deduplicated(tmp_path):
    backend = make_backend(tmp_path)
    llm = FakeLLM(REVIEW_OK, REVIEW_OK)
    reviewer = make_reviewer(backend, llm)
    sid = new_ulid()

    async def run():
        second = None
        for index in range(1, 11):
            second = await reviewer.on_turn(turn(sid, index))
        return second

    outcome = asyncio.run(run())
    assert outcome.status == "reviewed"
    assert outcome.candidates == 0
    assert outcome.deduped == 4
    assert len(backend.staging_list()) == 4


def test_rolling_summary_is_bounded_and_keeps_the_tail(tmp_path):
    backend = make_backend(tmp_path)
    settings = evolve_settings(summary_token_ratio=0.01)  # 32k * 0.01 * 4 = 1280 chars
    first = json.dumps(
        {"candidates": [], "lesson": None, "summary_delta": "A" * 700}, ensure_ascii=False
    )
    second = json.dumps(
        {"candidates": [], "lesson": None, "summary_delta": "B" * 700}, ensure_ascii=False
    )
    reviewer = make_reviewer(backend, FakeLLM(first, second), settings=settings)
    sid = new_ulid()

    async def run():
        for index in range(1, 11):
            await reviewer.on_turn(turn(sid, index))

    asyncio.run(run())
    summary = reviewer.rolling_summary(sid)
    assert len(summary) <= settings.summary_char_budget()
    assert "B" in summary
    assert "A" not in summary


def test_prompt_stays_under_1500_tokens_even_at_the_max_interval(tmp_path):
    backend = make_backend(tmp_path)
    llm = FakeLLM("{}")
    reviewer = make_reviewer(backend, llm, settings=evolve_settings(nudge_interval=50))
    sid = new_ulid()

    async def run():
        for index in range(1, 51):
            await reviewer.on_turn(
                turn(
                    sid,
                    index,
                    "这是一段很长的用户原文，用来测试骨架预算。" * 8,
                    tool_calls=[
                        {"tool": "fs__read", "ok": True},
                        {"tool": "shell__run", "ok": True},
                        {"tool": "web__fetch", "ok": False},
                    ],
                )
            )

    asyncio.run(run())
    assert len(llm.calls) == 1
    assert estimate_tokens(llm.calls[0]) <= 1500


def test_exhausted_daily_budget_degrades_without_a_model_call(tmp_path, caplog):
    backend = make_backend(tmp_path)
    llm = FakeLLM(REVIEW_OK)
    reviewer = make_reviewer(backend, llm, settings=evolve_settings(daily_token_budget=1))
    sid = new_ulid()

    async def run():
        outcomes = []
        for index in range(1, 6):
            outcomes.append(await reviewer.on_turn(turn(sid, index)))
        return outcomes[-1]

    with caplog.at_level(logging.WARNING, logger="memory_module.evolve"):
        outcome = asyncio.run(run())

    assert outcome.status == "degraded"
    assert outcome.error == "budget"
    assert llm.calls == []
    assert backend.staging_list() == []
    assert any("evolve_budget_exhausted" in record.message for record in caplog.records)


def test_disabled_reviewer_makes_no_calls_and_no_writes(tmp_path, caplog):
    backend = make_backend(tmp_path)
    llm = FakeLLM(REVIEW_OK)
    reviewer = make_reviewer(backend, llm, settings=EvolveSettings(enabled=False))
    sid = new_ulid()

    async def run():
        outcomes = []
        for index in range(1, 11):
            outcomes.append(await reviewer.on_turn(turn(sid, index)))
        return outcomes[-1]

    with caplog.at_level(logging.INFO, logger="memory_module.evolve"):
        outcome = asyncio.run(run())

    assert outcome.status == "disabled"
    assert llm.calls == []
    assert backend.staging_list() == []
    assert reviewer.turns_seen(sid) == 0
    assert any("evolve disabled" in record.message for record in caplog.records)


def test_invalid_session_id_is_ignored(tmp_path):
    backend = make_backend(tmp_path)
    llm = FakeLLM(REVIEW_OK)
    reviewer = make_reviewer(backend, llm)

    async def run():
        return await reviewer.on_turn({"session_id": "not-a-ulid", "user_text": "记住"})

    outcome = asyncio.run(run())
    assert outcome.status == "ignored"
    assert llm.calls == []


def test_build_skeleton_carries_flags_and_no_args():
    skeleton = build_skeleton(
        {
            "turn_id": "T1",
            "state": "failed",
            "user_text": "第一句。第二句。",
            "assistant_text": "不能说",
            "tool_calls": [{"tool": "shell__run", "args_digest": "SECRET", "ok": False}],
        }
    )
    assert skeleton.turn_id == "T1"
    assert skeleton.state == "failed"
    assert skeleton.tool_names == ("shell__run",)
    assert skeleton.first_sentence == "第一句。"
    assert skeleton.last_sentence == "第二句。"
    assert skeleton.error_flag is True
    assert "SECRET" not in json.dumps(skeleton.to_dict())
