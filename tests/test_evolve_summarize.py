"""session_close: gate, strict JSON, staging + session layer (M4b §六)."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from memory_module.embed import FakeEmbedding
from memory_module.evolve.config import EvolveSettings
from memory_module.evolve.summarize import SessionSummarizer
from memory_module.settings import MemorySettings
from memory_module.store import SqliteBackend, new_ulid


class Clock:
    def __init__(self, start: int = 1_700_000_000) -> None:
        self.now = start

    def __call__(self) -> int:
        self.now += 1
        return self.now


class FakeLLM:
    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        item = self.responses.pop(0) if self.responses else "{}"
        if isinstance(item, Exception):
            raise item
        return item


class StubReviewer:
    def __init__(self, text: str) -> None:
        self.text = text

    def rolling_summary(self, session_id: str) -> str:
        return self.text


def make_backend(tmp_path) -> SqliteBackend:
    return SqliteBackend(
        MemorySettings(data_dir=tmp_path, embedding_dim=8, global_capacity=100, session_capacity=50),
        clock=Clock(),
    )


def evolve_settings(**overrides) -> EvolveSettings:
    kwargs = dict(enabled=True, flush_min_turns=6)
    kwargs.update(overrides)
    return EvolveSettings(**kwargs)


def make_summarizer(backend, llm, settings=None, **kwargs) -> SessionSummarizer:
    return SessionSummarizer(
        backend,
        settings or evolve_settings(),
        llm=llm,
        embedder=FakeEmbedding(dim=8),
        **kwargs,
    )


def turns(count: int, **extra):
    return [
        {"turn_id": f"T{index:02d}", "state": "completed", "tool_calls": [], **extra}
        for index in range(1, count + 1)
    ]


SUMMARY_OK = json.dumps(
    {
        "summary": "会话确定了部署方案：本地回环 + SQLite。",
        "facts": [
            {"text": "部署在本地回环", "namespace": "user:default", "importance": 4, "confidence": 0.8},
            {"text": "用户偏好简洁回复", "namespace": "user:default", "importance": 4, "confidence": 0.7},
            {"text": "命名空间非法的候选", "namespace": "BAD SPACE", "importance": 3, "confidence": 0.5},
        ],
        "lessons": [
            {
                "when": "部署失败",
                "do": "先检查端口占用",
                "pitfall": "只看最后一次日志",
                "verify": "curl -fsS /healthz",
                "evidence": {"turn_id": "T01", "tool": "shell__run", "claim": "exit 1"},
                "confidence": 0.99,
            }
        ],
        "negative_feedback": False,
        "skill_candidate": True,
    },
    ensure_ascii=False,
)


def test_low_turn_session_skips_the_model(tmp_path, caplog):
    backend = make_backend(tmp_path)
    llm = FakeLLM(SUMMARY_OK)
    summarizer = make_summarizer(backend, llm)
    sid = new_ulid()

    with caplog.at_level(logging.INFO, logger="memory_module.evolve"):
        outcome = asyncio.run(
            summarizer.on_session_close({"session_id": sid, "turns": turns(3), "closed_at": 1})
        )

    assert outcome.status == "skipped"
    assert outcome.turns == 3
    assert llm.calls == []
    assert backend.staging_list() == []
    assert backend.count(layer="session") == 0
    assert any("evolve_skipped_low_signal" in record.message for record in caplog.records)


def test_summary_goes_to_session_and_candidates_to_staging(tmp_path):
    backend = make_backend(tmp_path)
    llm = FakeLLM(SUMMARY_OK)
    summarizer = make_summarizer(backend, llm)
    sid = new_ulid()

    outcome = asyncio.run(
        summarizer.on_session_close({"session_id": sid, "turns": turns(6), "closed_at": 1})
    )

    assert outcome.status == "ok"
    assert outcome.summary_chars > 0
    assert outcome.facts == 2
    assert outcome.lessons == 1
    assert outcome.dropped == 1  # the BAD SPACE fact
    assert outcome.skill_candidate is True

    assert backend.count(layer="global") == 0
    session_rows = backend.read_session(sid)
    assert len(session_rows) == 1
    assert session_rows[0].importance == 4.0
    assert session_rows[0].source == "evolve"
    assert "会话确定了部署方案" in session_rows[0].text
    assert "session-summary" in session_rows[0].tags

    staging = backend.staging_list()
    assert len(staging) == 3
    facts = [row for row in staging if row.kind != "lesson"]
    lessons = [row for row in staging if row.kind == "lesson"]
    assert len(facts) == 2
    assert len(lessons) == 1
    assert lessons[0].namespace == "agent:lessons"
    assert lessons[0].evidence["turn_id"] == "T01"
    assert lessons[0].confidence == pytest.approx(0.6)
    assert lessons[0].seen_sessions == [sid]


def test_parse_failure_stores_only_the_raw_summary(tmp_path):
    backend = make_backend(tmp_path)
    llm = FakeLLM("会话结论：部署完成，没有更多动作。", "会话结论：部署完成，没有更多动作。")
    summarizer = make_summarizer(backend, llm)
    sid = new_ulid()

    outcome = asyncio.run(
        summarizer.on_session_close({"session_id": sid, "turns": turns(6), "closed_at": 1})
    )

    assert outcome.status == "degraded"
    assert outcome.retried is True
    assert len(llm.calls) == 2
    assert outcome.facts == 0 and outcome.lessons == 0
    assert backend.staging_list() == []
    rows = backend.read_session(sid)
    assert len(rows) == 1
    assert rows[0].text == "会话结论：部署完成，没有更多动作。"


def test_provider_error_degrades_without_writing(tmp_path, caplog):
    backend = make_backend(tmp_path)
    llm = FakeLLM(RuntimeError("provider down"))
    summarizer = make_summarizer(backend, llm)
    sid = new_ulid()

    with caplog.at_level(logging.WARNING, logger="memory_module.evolve"):
        outcome = asyncio.run(
            summarizer.on_session_close({"session_id": sid, "turns": turns(6), "closed_at": 1})
        )

    assert outcome.status == "degraded"
    assert outcome.error == "exception"
    assert backend.staging_list() == []
    assert backend.count(layer="session") == 0
    assert any("evolve_degraded" in record.message for record in caplog.records)


def test_tail_is_redacted_and_never_carries_assistant_text(tmp_path):
    backend = make_backend(tmp_path)
    llm = FakeLLM(SUMMARY_OK)
    summarizer = make_summarizer(backend, llm)
    sid = new_ulid()
    payload_turns = turns(6)
    for index, turn in enumerate(payload_turns):
        turn["user_text"] = "用户说的内容" * 100
        turn["assistant_text"] = "ASSISTANT_SECRET"
        turn["tool_args"] = {"token": "TOOL_SECRET"}
        turn["file_content"] = "FILE_SECRET"
        turn["tool_calls"] = [{"tool": "fs__read", "args_digest": "DIGEST_SECRET", "ok": True}]

    asyncio.run(
        summarizer.on_session_close({"session_id": sid, "turns": payload_turns, "closed_at": 1})
    )

    prompt = llm.calls[0]
    assert "ASSISTANT_SECRET" not in prompt
    assert "TOOL_SECRET" not in prompt
    assert "FILE_SECRET" not in prompt
    assert "DIGEST_SECRET" not in prompt
    assert "用户说的内容" in prompt
    assert len(prompt) < 20_000


def test_summary_is_clipped_and_rolling_summary_is_an_input(tmp_path):
    backend = make_backend(tmp_path)
    long_summary = "总结" * 400
    payload = json.dumps(
        {"summary": long_summary, "facts": [], "lessons": [], "negative_feedback": False},
        ensure_ascii=False,
    )
    llm = FakeLLM(payload)
    summarizer = make_summarizer(
        backend, llm, reviewer=StubReviewer("ROLLING_MARKER_ABC")
    )
    sid = new_ulid()

    outcome = asyncio.run(
        summarizer.on_session_close({"session_id": sid, "turns": turns(6), "closed_at": 1})
    )

    assert "ROLLING_MARKER_ABC" in llm.calls[0]
    assert outcome.summary_chars <= 300
    rows = backend.read_session(sid)
    assert len(rows[0].text) <= 300


def test_repeated_close_is_idempotent(tmp_path):
    backend = make_backend(tmp_path)
    llm = FakeLLM(SUMMARY_OK, SUMMARY_OK)
    summarizer = make_summarizer(backend, llm)
    sid = new_ulid()
    payload = {"session_id": sid, "turns": turns(6), "closed_at": 1}

    first = asyncio.run(summarizer.on_session_close(payload))
    second = asyncio.run(summarizer.on_session_close(payload))

    assert first.status == "ok" and second.status == "ok"
    assert first.summary_chars > 0
    assert second.summary_chars == 0
    assert backend.count(layer="session") == 1
    assert len(backend.staging_list()) == 3
    assert backend.count(layer="global") == 0


def test_validation_failure_falls_back_to_the_summary_field(tmp_path):
    backend = make_backend(tmp_path)
    malformed = json.dumps(
        {"summary": "可用结论", "facts": "not-a-list", "lessons": []}, ensure_ascii=False
    )
    llm = FakeLLM(malformed, malformed)
    summarizer = make_summarizer(backend, llm)
    sid = new_ulid()

    outcome = asyncio.run(
        summarizer.on_session_close({"session_id": sid, "turns": turns(6), "closed_at": 1})
    )

    assert outcome.status == "degraded"
    assert outcome.facts == 0 and outcome.lessons == 0
    assert backend.staging_list() == []
    rows = backend.read_session(sid)
    assert len(rows) == 1
    assert rows[0].text == "可用结论"


def test_disabled_summarizer_does_nothing(tmp_path, caplog):
    backend = make_backend(tmp_path)
    llm = FakeLLM(SUMMARY_OK)
    summarizer = make_summarizer(backend, llm, settings=EvolveSettings(enabled=False))
    sid = new_ulid()

    with caplog.at_level(logging.INFO, logger="memory_module.evolve"):
        outcome = asyncio.run(
            summarizer.on_session_close({"session_id": sid, "turns": turns(6), "closed_at": 1})
        )

    assert outcome.status == "disabled"
    assert llm.calls == []
    assert backend.count(layer="session") == 0
    assert backend.staging_list() == []
    assert any("evolve disabled" in record.message for record in caplog.records)
