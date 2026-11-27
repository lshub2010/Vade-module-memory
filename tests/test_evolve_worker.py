"""The event worker: subscription gate, counters, skill proposal delivery.

These tests exercise the wiring `pipeline`/`skillgen` deliberately do not own:
which topics are consumed, that a transcript is never retained, and that a
proposal reaches a sink but never writes a skill itself.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from memory_module.embed import FakeEmbedding
from memory_module.evolve.config import EvolveSettings
from memory_module.evolve.pipeline import EvolvePipeline
from memory_module.evolve.skillgen import SkillGenerator
from memory_module.evolve.worker import EvolveWorker
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


def make_backend(tmp_path) -> SqliteBackend:
    return SqliteBackend(
        MemorySettings(data_dir=tmp_path, embedding_dim=8, global_capacity=100, session_capacity=50),
        clock=Clock(),
    )


def settings(**overrides) -> EvolveSettings:
    kwargs = dict(enabled=True, flush_min_turns=6, daily_token_budget=0)
    kwargs.update(overrides)
    return EvolveSettings(**kwargs)


def make_worker(backend, *, llm=None, propose=None, cfg=None, catalog=None) -> EvolveWorker:
    cfg = cfg or settings()
    pipeline = EvolvePipeline(
        backend, cfg, llm=llm, embedder=FakeEmbedding(dim=8), memory_settings=backend.settings
    )
    return EvolveWorker(
        pipeline,
        cfg,
        skillgen=SkillGenerator(cfg, llm=llm),
        propose=propose,
        catalog=catalog or (lambda: ()),
    )


def turn(session_id: str, *, tool_names=(), state="completed", negative=False, self_corrected=False, **extra):
    return {
        "session_id": session_id,
        "turn_id": new_ulid(),
        "state": state,
        "tool_calls": [{"tool": name, "ok": True} for name in tool_names],
        "negative": negative,
        "self_corrected": self_corrected,
        **extra,
    }


SKILL_JSON = json.dumps(
    {
        "name": "check-the-lockfile",
        "description": "Verify a lockfile before upgrading dependencies",
        "when_to_use": "Before upgrading a dependency tree",
        "procedure": ["Read the lockfile", "Run the resolver", "Diff the result"],
        "pitfalls": ["A resolver may silently pick a newer transitive version"],
        "verification": ["pytest tests -q"],
    }
)


def test_disabled_worker_has_no_topics_and_writes_nothing(tmp_path):
    backend = make_backend(tmp_path)
    worker = make_worker(backend, cfg=EvolveSettings(enabled=False))
    assert worker.topics() == ()
    result = asyncio.run(worker.on_event("session.turn", turn(new_ulid(), tool_names=["a"])))
    assert result["status"] == "disabled"
    assert backend.count() == 0
    backend.close()


def test_turn_events_keep_counters_but_never_transcript_text(tmp_path):
    backend = make_backend(tmp_path)
    worker = make_worker(backend)
    session_id = new_ulid()
    payload = turn(session_id, tool_names=["fs__read", "shell__run"], user_text="我喜欢用中文回答")
    asyncio.run(worker.on_event("session.turn", payload))

    stored = worker._counters[session_id]
    assert stored[0]["turn_id"] == payload["turn_id"]
    assert stored[0]["state"] == "completed"
    assert [call["tool"] for call in stored[0]["tool_calls"]] == ["fs__read", "shell__run"]
    serialized = json.dumps(stored, ensure_ascii=False)
    assert "我喜欢用中文回答" not in serialized
    backend.close()


def test_successful_trajectory_sends_one_create_proposal_to_the_sink(tmp_path):
    backend = make_backend(tmp_path)
    sent = []

    async def sink(payload):
        sent.append(payload)
        return {"ok": True, "name": payload["name"]}

    llm = FakeLLM(SKILL_JSON)
    worker = make_worker(backend, llm=llm, propose=sink)
    session_id = new_ulid()
    # 6 effective calls across 3 distinct tools: the §八 gate needs >=5 and >=2.
    turns = [
        turn(session_id, tool_names=["fs__read", "shell__run"]),
        turn(session_id, tool_names=["fs__read", "web__fetch"]),
        turn(session_id, tool_names=["fs__read", "shell__run"]),
    ]
    result = asyncio.run(worker.on_event("session.closed", {"session_id": session_id, "turns": turns}))

    assert result["skill_proposal"]["status"] == "proposed"
    assert len(sent) == 1
    payload = sent[0]
    assert payload["action"] == "create"
    assert payload["name"] == "check-the-lockfile"
    assert payload["derived_from"]["session_id"] == session_id
    assert payload["derived_from"]["turn_ids"] == [t["turn_id"] for t in turns]
    assert "## Pitfalls" in payload["content"]
    # A proposal is never installed by the module.
    assert not (tmp_path / "skills").exists()
    backend.close()


def test_negative_or_self_corrected_trajectory_produces_no_proposal(tmp_path):
    backend = make_backend(tmp_path)
    sent = []

    async def sink(payload):  # pragma: no cover - must never run
        sent.append(payload)

    for flags in ({"negative": True}, {"self_corrected": True}, {"state": "failed"}):
        worker = make_worker(backend, llm=FakeLLM(SKILL_JSON), propose=sink)
        session_id = new_ulid()
        turns = [
            turn(session_id, tool_names=["fs__read", "shell__run"], **flags),
            turn(session_id, tool_names=["fs__read", "web__fetch"]),
            turn(session_id, tool_names=["fs__read", "shell__run"]),
        ]
        result = asyncio.run(
            worker.on_event("session.closed", {"session_id": session_id, "turns": turns})
        )
        assert result["skill_proposal"] is None, flags
    assert sent == []
    backend.close()


def test_missing_pitfalls_is_retried_once_then_dropped(tmp_path):
    backend = make_backend(tmp_path)
    sent = []
    bad = json.dumps(
        {
            "name": "no-pitfalls",
            "description": "d",
            "when_to_use": "w",
            "procedure": ["p"],
            "verification": ["pytest -q"],
        }
    )

    async def sink(payload):  # pragma: no cover - must never run
        sent.append(payload)

    llm = FakeLLM(bad, bad)
    worker = make_worker(backend, llm=llm, propose=sink)
    session_id = new_ulid()
    turns = [
        turn(session_id, tool_names=["a", "b"]),
        turn(session_id, tool_names=["a", "c"]),
        turn(session_id, tool_names=["a", "b"]),
    ]
    result = asyncio.run(worker.on_event("session.closed", {"session_id": session_id, "turns": turns}))
    assert result["skill_proposal"] is None
    assert sent == []
    # The contract allows one rewrite; the second failure is a drop, not a loop.
    assert len(llm.calls) == 2
    backend.close()


def test_session_budget_allows_one_proposal_only(tmp_path):
    backend = make_backend(tmp_path)
    sent = []

    async def sink(payload):
        sent.append(payload)

    worker = make_worker(
        backend, llm=FakeLLM(SKILL_JSON, SKILL_JSON), propose=sink,
        cfg=settings(session_budget_skills=1, daily_budget_skills=5),
    )
    session_id = new_ulid()
    for _ in range(2):
        turns = [
            turn(session_id, tool_names=["a", "b"]),
            turn(session_id, tool_names=["a", "c"]),
            turn(session_id, tool_names=["a", "b"]),
        ]
        asyncio.run(worker.on_event("session.closed", {"session_id": session_id, "turns": turns}))
    assert len(sent) == 1
    backend.close()


def test_server_still_rebases_ttl_when_evolve_is_disabled(tmp_path):
    from memory_module.server import MemoryService

    service = MemoryService(
        MemorySettings(data_dir=tmp_path, embedding_dim=8),
        embedder=FakeEmbedding(dim=8),
        evolve=EvolveSettings(enabled=False),
    )
    session_id = new_ulid()
    stored = service.call_tool("remember", {"text": "会话记忆", "scope": "session"}, session_id=session_id)

    result = asyncio.run(
        service.handle_event_async(
            "session.closed", {"session_id": session_id, "closed_at": 1_700_000_000_000}
        )
    )
    assert result["handled"] is True
    assert result["updated"] == 1
    assert "evolve" not in result
    reopened = service.backend.get(stored["id"])
    assert reopened is not None and reopened.expires_at == 1_700_000_000 + 7 * 86400
    service.close()


def test_unsupported_topic_is_dropped_not_written(tmp_path):
    backend = make_backend(tmp_path)
    worker = make_worker(backend)
    result = asyncio.run(worker.on_event("session.transcript", {"session_id": new_ulid(), "text": "x"}))
    assert result["status"] == "ignored"
    assert backend.count() == 0
    backend.close()
