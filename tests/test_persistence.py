"""Persistence, restart, TTL, quotas and the two session gates."""

from __future__ import annotations

import sqlite3

import pytest

from memory_module.embed import FakeEmbedding
from memory_module.errors import MemoryFailure, StartupError
from memory_module.server import MemoryService
from memory_module.settings import MemorySettings
from memory_module.store import SqliteBackend, new_ulid


class Clock:
    def __init__(self, start: int = 1_700_000_000) -> None:
        self.now = start

    def __call__(self) -> int:
        self.now += 1
        return self.now


def settings_for(tmp_path, **overrides):
    kwargs = dict(
        data_dir=tmp_path,
        embedding_dim=8,
        global_capacity=100,
        session_capacity=50,
        session_inject_char_limit=4200,
        recall_token_limit=800,
        ttl_days=7,
    )
    kwargs.update(overrides)
    return MemorySettings(**kwargs)


def test_restart_returns_the_same_order_and_scores(tmp_path):
    settings = settings_for(tmp_path)
    first = MemoryService(settings, embedder=FakeEmbedding(dim=8), clock=Clock())
    for index, text in enumerate(
        ["用户偏好深色主题", "项目代号 Orion", "部署在本地回环", "评审代理使用只读模式", "记忆模块走 SQLite"]
    ):
        first.call_tool("remember", {"text": text, "tags": [f"t{index}"]})
    before = first.call_tool("recall", {"query": "主题 代号 部署", "k": 5})
    first.close()

    second = MemoryService(settings, embedder=FakeEmbedding(dim=8), clock=Clock())
    after = second.call_tool("recall", {"query": "主题 代号 部署", "k": 5})
    second.close()

    assert [item["id"] for item in before["items"]] == [item["id"] for item in after["items"]]
    assert [item["score"] for item in before["items"]] == [item["score"] for item in after["items"]]
    assert [item["text"] for item in before["items"]] == [item["text"] for item in after["items"]]


def test_dimension_mismatch_refuses_to_start(tmp_path):
    settings = settings_for(tmp_path, embedding_dim=8)
    backend = SqliteBackend(settings, model="local-hash-v1", clock=Clock())
    backend.put_with_vector(
        __import__("memory_module.store", fromlist=["MemoryRecord"]).MemoryRecord(
            id="", layer="global", namespace="user:default", session_id="", text="dimension seed"
        ),
        [1.0] + [0.0] * 7,
    )
    backend.close()

    with pytest.raises(StartupError) as exc:
        SqliteBackend(settings_for(tmp_path, embedding_dim=4), model="local-hash-v1", clock=Clock())
    message = str(exc.value)
    assert "dim=8" in message and "dim=4" in message
    assert "Refusing to start" in message


def test_model_mismatch_refuses_to_start(tmp_path):
    settings = settings_for(tmp_path, embedding_model="model-a")
    backend = SqliteBackend(settings, clock=Clock())
    backend.close()

    with pytest.raises(StartupError) as exc:
        SqliteBackend(settings_for(tmp_path, embedding_model="model-b"), clock=Clock())
    assert "model-a" in str(exc.value) and "model-b" in str(exc.value)


def test_session_ttl_is_physically_deleted_and_not_archived(tmp_path):
    settings = settings_for(tmp_path)
    service = MemoryService(settings, embedder=FakeEmbedding(dim=8), clock=Clock())
    session_id = new_ulid()
    rec = service.call_tool(
        "remember", {"text": "会话里的临时记忆", "scope": "session"}, session_id=session_id
    )
    service.close()

    conn = sqlite3.connect(settings.db_path)
    conn.execute("UPDATE memories SET expires_at=? WHERE id=?", (1, rec["id"]))
    conn.commit()
    conn.close()

    reopened = MemoryService(settings, embedder=FakeEmbedding(dim=8), clock=Clock())
    assert reopened.backend.count(layer="session") == 0
    conn = sqlite3.connect(settings.db_path)
    assert conn.execute("SELECT COUNT(*) FROM archive WHERE id=?", (rec["id"],)).fetchone()[0] == 0
    conn.close()
    reopened.close()


def test_session_closed_expires_seven_days_after_close(tmp_path):
    clock = Clock()
    settings = settings_for(tmp_path)
    service = MemoryService(settings, embedder=FakeEmbedding(dim=8), clock=clock)
    session_id = new_ulid()
    rec = service.call_tool(
        "remember", {"text": "关闭后仍然保留", "scope": "session"}, session_id=session_id
    )
    clock.now += 10 * 86400  # ten days later, the session is closed
    closed_at_ms = (clock.now - 3600) * 1000  # core's event timestamp: one hour before handling
    result = service.handle_event(
        "session.closed", {"session_id": session_id, "closed_at": closed_at_ms}
    )
    assert result["updated"] == 1
    stored = service.backend.get(rec["id"])
    assert stored is not None
    assert stored.expires_at == (clock.now - 3600) + 7 * 86400
    service.close()


def test_quota_eviction_archives_the_lowest_scores(tmp_path):
    settings = settings_for(tmp_path, global_capacity=5)
    service = MemoryService(settings, embedder=FakeEmbedding(dim=8), clock=Clock())
    ids = []
    for index in range(8):
        ids.append(service.call_tool("remember", {"text": f"记忆条目 {index}"})["id"])

    stats = service.backend.admin_stats()
    assert stats["global_count"] == 5
    assert stats["archived_count"] == 3

    conn = sqlite3.connect(settings.db_path)
    reasons = conn.execute("SELECT reason, COUNT(*) FROM archive GROUP BY reason").fetchall()
    conn.close()
    assert reasons == [("quota", 3)]

    retained = {rec.id for rec in service.backend.search([1.0] + [0.0] * 7, 10, "", include_superseded=True)}
    # All eight vectors are lexical; use ids rather than the vector path:
    live = {rec["id"] for rec in service.call_tool("recall", {"query": "记忆条目", "k": 10})["items"]}
    assert len(live) == 5
    assert live <= set(ids)
    assert retained  # search path also returns rows
    service.close()


def test_overloaded_when_eviction_cannot_free_a_slot(tmp_path):
    """The contract's "腾不出空间才 OVERLOADED" branch: no silent drop."""
    settings = settings_for(tmp_path, global_capacity=2)
    service = MemoryService(settings, embedder=FakeEmbedding(dim=8), clock=Clock())
    service.call_tool("remember", {"text": "满了 1"})
    service.call_tool("remember", {"text": "满了 2"})
    service.backend._evict_locked = lambda target: []  # type: ignore[assignment]
    with pytest.raises(MemoryFailure) as exc:
        service.call_tool("remember", {"text": "放不下 3"})
    assert exc.value.code == "OVERLOADED"
    assert service.backend.count() == 2
    assert all(
        item["text"] != "放不下 3"
        for item in service.call_tool("recall", {"query": "放不下", "k": 5})["items"]
    )
    service.close()


def test_session_char_gate_is_memory_full_and_retryable(tmp_path):
    settings = settings_for(
        tmp_path, session_inject_char_limit=100, session_write_char_limit=0, session_capacity=10
    )
    service = MemoryService(settings, embedder=FakeEmbedding(dim=8), clock=Clock())
    session_id = new_ulid()
    texts = [f"a{index}" + "x" * 38 for index in range(4)]
    first = service.call_tool("remember", {"text": texts[0], "scope": "session"}, session_id=session_id)
    service.call_tool("remember", {"text": texts[1], "scope": "session"}, session_id=session_id)
    service.call_tool("remember", {"text": texts[2], "scope": "session"}, session_id=session_id)
    assert service.backend.count(layer="session") == 3

    with pytest.raises(MemoryFailure) as exc:
        service.call_tool("remember", {"text": texts[3], "scope": "session"}, session_id=session_id)
    assert exc.value.code == "MEMORY_FULL"
    assert service.backend.count(layer="session") == 3

    service.call_tool("forget", {"id": first["id"]})
    fourth = service.call_tool("remember", {"text": texts[3], "scope": "session"}, session_id=session_id)
    assert fourth["deduped"] is False
    assert service.backend.count(layer="session") == 3
    service.close()


def test_session_count_gate_is_memory_full(tmp_path):
    settings = settings_for(tmp_path, session_capacity=2, session_inject_char_limit=4200)
    service = MemoryService(settings, embedder=FakeEmbedding(dim=8), clock=Clock())
    session_id = new_ulid()
    service.call_tool("remember", {"text": "第一条", "scope": "session"}, session_id=session_id)
    service.call_tool("remember", {"text": "第二条", "scope": "session"}, session_id=session_id)
    with pytest.raises(MemoryFailure) as exc:
        service.call_tool("remember", {"text": "第三条", "scope": "session"}, session_id=session_id)
    assert exc.value.code == "MEMORY_FULL"
    assert service.backend.count(layer="session") == 2
    service.close()


def test_read_session_budget_and_recall_reaches_cut_items(tmp_path):
    settings = settings_for(
        tmp_path,
        session_inject_char_limit=200,  # injection budget: 6 x 30 chars
        session_write_char_limit=10_000,  # write budget: all 10 fit
        session_capacity=50,
        recall_token_limit=800,
    )
    service = MemoryService(settings, embedder=FakeEmbedding(dim=8), clock=Clock())
    session_id = new_ulid()
    for index in range(10):
        text = f"记忆条目{index}" + "x" * (30 - len(f"记忆条目{index}"))
        service.call_tool("remember", {"text": text, "scope": "session"}, session_id=session_id)

    injected = service.backend.read_session(session_id, limit=50)
    assert len(injected) == 6
    assert sum(len(rec.text) for rec in injected) <= 200
    assert [rec.created_at for rec in injected] == sorted(
        [rec.created_at for rec in injected], reverse=True
    )
    injected_ids = {rec.id for rec in injected}

    recalled = service.call_tool(
        "recall", {"query": "记忆", "k": 10, "include_session": True}, session_id=session_id
    )
    assert recalled["count"] == 10
    assert len({item["id"] for item in recalled["items"]} - injected_ids) == 4
    service.close()


def test_deleted_rows_are_purged_after_seven_days_but_archive_survives(tmp_path):
    settings = settings_for(tmp_path)
    service = MemoryService(settings, embedder=FakeEmbedding(dim=8), clock=Clock())
    rec = service.call_tool("remember", {"text": "会被物理清理"})
    service.call_tool("forget", {"id": rec["id"]})
    service.close()

    conn = sqlite3.connect(settings.db_path)
    conn.execute("UPDATE memories SET deleted_at=? WHERE id=?", (1, rec["id"]))
    conn.commit()
    conn.close()

    reopened = MemoryService(settings, embedder=FakeEmbedding(dim=8), clock=Clock())
    assert reopened.backend.get(rec["id"], include_deleted=True) is None
    conn = sqlite3.connect(settings.db_path)
    assert conn.execute("SELECT COUNT(*) FROM memories WHERE id=?", (rec["id"],)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM archive WHERE id=?", (rec["id"],)).fetchone()[0] == 1
    conn.close()
    reopened.close()


def test_remember_after_forget_revives_without_physical_delete(tmp_path):
    service = MemoryService(settings_for(tmp_path), embedder=FakeEmbedding(dim=8), clock=Clock())
    first = service.call_tool("remember", {"text": "忘掉再记一次"})
    service.call_tool("forget", {"id": first["id"]})
    second = service.call_tool("remember", {"text": "忘掉再记一次"})
    assert second["deduped"] is False
    assert service.backend.count() == 1
    assert service.call_tool("recall", {"query": "忘掉再记", "k": 5})["count"] == 1
    conn = sqlite3.connect(service.settings.db_path)
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM archive WHERE id=?", (first["id"],)).fetchone()[0] == 1
    conn.close()
    service.close()


def test_embedding_failure_leaves_no_row(tmp_path):
    settings = settings_for(tmp_path)
    embedder = FakeEmbedding(dim=8, fail_times=1)
    service = MemoryService(settings, embedder=embedder, clock=Clock())
    with pytest.raises(MemoryFailure) as exc:
        service.call_tool("remember", {"text": "这次 embedding 会失败"})
    assert exc.value.code == "UNAVAILABLE"
    assert service.backend.count() == 0
    assert service.backend.all_vectors()[0] == []
    assert embedder.calls == ["这次 embedding 会失败"]
    service.close()


def test_schema_has_no_transcript_columns(tmp_path):
    settings = settings_for(tmp_path)
    service = MemoryService(settings, embedder=FakeEmbedding(dim=8), clock=Clock())
    columns = {
        row[1] for row in service.backend._conn.execute("PRAGMA table_info(memories)").fetchall()
    }
    assert "user_text" not in columns
    assert "assistant_text" not in columns
    assert "transcript" not in columns
    assert {"id", "layer", "namespace", "session_id", "text", "tags", "content_hash"} <= columns
    service.close()
