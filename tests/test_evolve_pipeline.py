"""Pipeline glue: the switch, user_text-only rules, and the full promotion path."""

from __future__ import annotations

import asyncio
import json
import logging

from memory_module.embed import FakeEmbedding
from memory_module.evolve.config import EvolveSettings
from memory_module.evolve.pipeline import EvolvePipeline
from memory_module.settings import MemorySettings
from memory_module.store import SqliteBackend, new_ulid

V_A = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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


def make_pipeline(tmp_path, llm, settings=None, mapping=None):
    memory_settings = MemorySettings(
        data_dir=tmp_path,
        embedding_dim=8,
        global_capacity=100,
        session_capacity=50,
    )
    backend = SqliteBackend(memory_settings, clock=Clock())
    pipeline = EvolvePipeline(
        backend,
        settings or EvolveSettings(enabled=True),
        llm=llm,
        embedder=FakeEmbedding(dim=8, mapping=mapping or {}),
        memory_settings=memory_settings,
        clock=Clock(),
    )
    return pipeline, backend, memory_settings


def turn_payload(session_id: str, index: int, user_text: str = "继续", **extra):
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


def session_turns(count: int, **extra):
    return [
        {"turn_id": f"T{index:02d}", "state": "completed", "tool_calls": [], **extra}
        for index in range(1, count + 1)
    ]


def summary_json(summary: str, facts=()) -> str:
    return json.dumps(
        {
            "summary": summary,
            "facts": list(facts),
            "lessons": [],
            "negative_feedback": False,
            "skill_candidate": False,
        },
        ensure_ascii=False,
    )


def test_disabled_pipeline_does_nothing(tmp_path, caplog):
    llm = FakeLLM()
    pipeline, backend, _settings = make_pipeline(tmp_path, llm, EvolveSettings(enabled=False))
    sid = new_ulid()

    async def run():
        for index in range(1, 9):
            await pipeline.on_turn(turn_payload(sid, index, "以后都用中文回答"))
        await pipeline.on_session_closed({"session_id": sid, "turns": session_turns(8)})

    with caplog.at_level(logging.INFO, logger="memory_module.evolve"):
        asyncio.run(run())
    pipeline.consolidate()
    pipeline.startup()

    assert llm.calls == []
    assert backend.count(layer="session") == 0
    assert backend.count(layer="global") == 0
    assert backend.staging_list() == []
    assert backend.get_last_consolidate() is None
    assert any("evolve disabled" in record.message for record in caplog.records)


def test_only_user_text_can_produce_a_session_memory(tmp_path):
    llm = FakeLLM()
    pipeline, backend, _settings = make_pipeline(tmp_path, llm)
    sid = new_ulid()
    payload = turn_payload(sid, 1, "你好，今天天气不错")
    payload["assistant_text"] = "结论是：记住这个项目使用 SQLite。以后都用英文回答。"

    result = asyncio.run(pipeline.on_turn(payload))

    assert result["written"] == 0
    assert result["matched"] == 0
    assert backend.count(layer="session") == 0
    assert backend.count(layer="global") == 0
    assert llm.calls == []


def test_explicit_instruction_lands_in_session_with_importance_five(tmp_path):
    llm = FakeLLM()
    pipeline, backend, _settings = make_pipeline(tmp_path, llm)
    sid = new_ulid()

    result = asyncio.run(pipeline.on_turn(turn_payload(sid, 1, "以后都用中文回答。")))

    assert result["written"] == 1
    rows = backend.read_session(sid)
    assert len(rows) == 1
    assert rows[0].layer == "session"
    assert rows[0].importance == 5.0
    assert rows[0].kind == "fact"
    assert rows[0].source == "evolve"
    assert rows[0].tags == ["memory_instruction"]
    assert backend.count(layer="global") == 0


def test_preference_signal_uses_the_preference_kind(tmp_path):
    llm = FakeLLM()
    pipeline, backend, _settings = make_pipeline(tmp_path, llm)
    sid = new_ulid()

    asyncio.run(pipeline.on_turn(turn_payload(sid, 1, "我喜欢深色主题。")))

    rows = backend.read_session(sid)
    assert len(rows) == 1
    assert rows[0].kind == "preference"
    assert rows[0].importance == 4.0


def test_turn_end_writes_at_most_two_candidates(tmp_path):
    llm = FakeLLM()
    pipeline, backend, _settings = make_pipeline(tmp_path, llm)
    sid = new_ulid()
    text = "服务跑在 /srv/app。结论是继续用 SQLite。我喜欢简洁回复。以后都用中文回答。"

    result = asyncio.run(pipeline.on_turn(turn_payload(sid, 1, text)))

    assert result["matched"] == 4
    assert result["written"] == 2
    rows = backend.read_session(sid)
    assert len(rows) == 2
    assert sorted(row.importance for row in rows) == [4.0, 5.0]


def test_session_close_to_two_sessions_to_global_end_to_end(tmp_path):
    fact_text = "团队统一使用 ruff 格式化"
    fact = {"text": fact_text, "namespace": "user:default", "importance": 4, "confidence": 0.8}
    llm = FakeLLM(
        summary_json("第一次会话的结论", [fact]),
        summary_json("第二次会话的结论", [fact]),
    )
    pipeline, backend, _settings = make_pipeline(
        tmp_path, llm, mapping={fact_text: V_A}
    )
    s1, s2 = new_ulid(), new_ulid()

    async def close(session_id: str):
        return await pipeline.on_session_closed(
            {"session_id": session_id, "turns": session_turns(6)}
        )

    first = asyncio.run(close(s1))
    assert first["status"] == "ok" and first["facts"] == 1
    assert backend.count(layer="session") == 1
    assert len(backend.staging_list()) == 1

    one = pipeline.consolidate()
    assert one.promoted == 0 and one.single_session == 1
    assert backend.count(layer="global") == 0
    assert backend.staging_list()[0].seen_sessions == [s1]
    assert pipeline.summarizer.backend.read_session(s1)[0].importance == 4.0

    second = asyncio.run(close(s2))
    assert second["status"] == "ok" and second["facts"] == 1
    assert backend.staging_list()[0].seen_sessions == [s1, s2]

    two = pipeline.consolidate()
    assert two.promoted == 1
    records, _cursor = backend.admin_list(layer="global", status="active")
    assert len(records) == 1
    assert records[0].text == fact_text
    assert backend.staging_list() == []


def test_correction_marks_the_previous_turn_negative_in_the_close_prompt(tmp_path):
    llm = FakeLLM(summary_json("会话结论"))
    pipeline, _backend, _settings = make_pipeline(tmp_path, llm)
    sid = new_ulid()

    async def run():
        await pipeline.on_turn(turn_payload(sid, 1, "继续"))
        await pipeline.on_turn(turn_payload(sid, 2, "不对，应该是 9090。"))
        return await pipeline.on_session_closed(
            {"session_id": sid, "turns": session_turns(6)}
        )

    asyncio.run(run())
    prompt = llm.calls[0]
    assert '"turn_id": "T01"' in prompt
    assert '"negative": true' in prompt
    # T02 is the correction itself, not the turn being corrected.
    assert '"turn_id": "T02"' in prompt


def test_unknown_topic_is_dropped_with_a_warning(tmp_path, caplog):
    llm = FakeLLM()
    pipeline, _backend, _settings = make_pipeline(tmp_path, llm)

    with caplog.at_level(logging.WARNING, logger="memory_module.evolve"):
        result = asyncio.run(pipeline.on_event("session.transcript", {}))

    assert result == {"status": "ignored", "topic": "session.transcript"}
    assert any("unsupported topic" in record.message for record in caplog.records)
