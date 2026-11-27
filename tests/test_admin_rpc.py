"""Admin pagination/stats, staging promotion, archive/restore, private RPC shapes."""

from __future__ import annotations

import pytest

from memory_module.embed import FakeEmbedding
from memory_module.errors import MemoryFailure
from memory_module.server import MemoryService
from memory_module.settings import MemorySettings
from memory_module.store import new_ulid


class Clock:
    def __init__(self, start: int = 1_700_000_000) -> None:
        self.now = start

    def __call__(self) -> int:
        self.now += 1
        return self.now


def make_service(tmp_path, **overrides):
    kwargs = dict(
        data_dir=tmp_path,
        embedding_dim=8,
        global_capacity=100,
        session_capacity=50,
        session_inject_char_limit=4200,
        recall_token_limit=800,
    )
    kwargs.update(overrides)
    settings = MemorySettings(**kwargs)
    clock = Clock()
    return MemoryService(settings, embedder=FakeEmbedding(dim=8), clock=clock), clock


def test_admin_list_paginates_every_row_exactly_once(tmp_path):
    service, _ = make_service(tmp_path)
    ids = [service.call_tool("remember", {"text": f"分页记忆 {index}"})["id"] for index in range(60)]

    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        items, cursor = service.backend.admin_list(status="active", limit=10, cursor=cursor)
        seen.extend(item.id for item in items)
        pages += 1
        if cursor is None:
            break
        assert pages < 20
    assert len(seen) == 60
    assert len(set(seen)) == 60
    assert set(seen) == set(ids)
    assert pages == 6
    service.close()


def test_admin_list_filters_and_keyword_search(tmp_path):
    service, _ = make_service(tmp_path)
    service.call_tool("remember", {"text": "alpha 主题", "tags": ["alpha"], "namespace": "user:default"})
    service.call_tool("remember", {"text": "beta 主题", "tags": ["beta"], "namespace": "agent:reviewer"})
    service.call_tool("remember", {"text": "gamma 主题", "tags": ["gamma"], "namespace": "user:default"})

    by_tag, _ = service.backend.admin_list(q="beta", limit=10)
    assert [item.text for item in by_tag] == ["beta 主题"]

    by_namespace, _ = service.backend.admin_list(namespace="user:default", limit=10)
    assert {item.text for item in by_namespace} == {"alpha 主题", "gamma 主题"}

    first = service.call_tool("remember", {"text": "时间边界"})
    created = service.backend.get(first["id"]).created_at
    before, _ = service.backend.admin_list(t_to=created - 1, limit=10)
    assert first["id"] not in {item.id for item in before}
    after, _ = service.backend.admin_list(t_from=created, limit=10)
    assert first["id"] in {item.id for item in after}
    service.close()


def test_admin_stats_and_staging_promotion(tmp_path):
    service, _ = make_service(tmp_path, global_capacity=7)
    session_id = new_ulid()
    vec = [0.1] * 8
    staging_id = service.backend.staging_put("候选结论：走本地向量", "agent:reviewer", session_id, vec, 0.55)
    again = service.backend.staging_put(
        "候选结论：走本地向量", "agent:reviewer", new_ulid(), vec, 0.7
    )
    assert again == staging_id
    staging = service.backend.staging_list("agent:reviewer")
    assert len(staging) == 1
    assert len(staging[0].seen_sessions) == 2
    assert staging[0].confidence == 0.7

    stats = service.backend.admin_stats()
    assert stats == {
        "global_count": 0,
        "global_limit": 7,
        "staging_count": 1,
        "archived_count": 0,
        "oldest_at": None,
    }

    result = service.handle_rpc("memory/admin_promote", {"id": staging_id, "by": "alice"})
    assert result["ok"] is True and len(result["id"]) == 26
    promoted = service.backend.get(result["id"])
    assert promoted is not None
    assert promoted.layer == "global"
    assert promoted.promoted_by == "admin:alice"
    assert promoted.importance == 5.0
    assert service.backend.staging_list() == []

    with pytest.raises(MemoryFailure) as exc:
        service.handle_rpc("memory/admin_promote", {"id": staging_id, "by": "alice"})
    assert exc.value.code == "NOT_FOUND"
    service.close()


def test_staging_lesson_requires_evidence(tmp_path):
    service, _ = make_service(tmp_path)
    with pytest.raises(MemoryFailure) as exc:
        service.backend.staging_put("教训", "agent:lessons", new_ulid(), [0.1] * 8, 0.5, kind="lesson")
    assert exc.value.code == "INVALID_ARGUMENT"
    staging_id = service.backend.staging_put(
        "教训",
        "agent:lessons",
        new_ulid(),
        [0.1] * 8,
        0.5,
        kind="lesson",
        evidence={"turn_id": "01J00000000000000000000000", "claim": "x"},
    )
    item, _ = service.backend.admin_list(status="staging", limit=10)
    assert [row.id for row in item] == [staging_id]
    assert item[0].layer == "staging"
    detail, chain = service.backend.admin_get(staging_id)
    assert detail.id == staging_id and detail.status == "staging" and chain == []
    service.close()


def test_admin_archive_and_restore_round_trip(tmp_path):
    service, _ = make_service(tmp_path)
    rec = service.call_tool("remember", {"text": "可归档可恢复"})
    archived = service.handle_rpc("memory/admin_archive", {"id": rec["id"], "reason": "manual", "by": "alice"})
    assert archived == {"ok": True}
    assert service.backend.get(rec["id"]) is None
    assert service.call_tool("recall", {"query": "归档", "k": 5})["count"] == 0

    archived_items, _ = service.backend.admin_list(status="archived", limit=10)
    assert [item.id for item in archived_items] == [rec["id"]]
    assert service.backend.admin_stats()["archived_count"] == 1

    restored = service.handle_rpc("memory/admin_restore", {"id": rec["id"], "by": "alice"})
    assert restored == {"ok": True}
    assert service.backend.get(rec["id"]).deleted_at is None  # type: ignore[union-attr]
    assert service.call_tool("recall", {"query": "归档", "k": 5})["count"] == 1

    with pytest.raises(MemoryFailure):
        service.handle_rpc("memory/admin_restore", {"id": rec["id"], "by": "alice"})
    service.close()


def test_forget_cannot_be_undone_through_admin_restore(tmp_path):
    service, _ = make_service(tmp_path)
    rec = service.call_tool("remember", {"text": "forget 不可恢复"})
    service.call_tool("forget", {"id": rec["id"]})
    with pytest.raises(MemoryFailure) as exc:
        service.handle_rpc("memory/admin_restore", {"id": rec["id"], "by": "alice"})
    assert exc.value.code == "PERMISSION_DENIED"
    service.close()


def test_admin_get_returns_both_sides_of_the_supersedes_chain(tmp_path):
    service, _ = make_service(tmp_path)
    old = service.call_tool("remember", {"text": "规则是 8080"})
    mid = service.call_tool("remember", {"text": "规则是 8081"})
    new = service.call_tool("remember", {"text": "规则是 8082"})
    service.backend.supersede(old["id"], mid["id"])
    service.backend.supersede(mid["id"], new["id"])

    item = service.handle_rpc("memory/admin_get", {"id": mid["id"]})
    assert item["item"]["id"] == mid["id"]
    assert item["item"]["status"] == "superseded"
    assert {row["id"] for row in item["chain"]} == {old["id"], new["id"]}

    old_detail = service.handle_rpc("memory/admin_get", {"id": old["id"]})
    assert old_detail["item"]["status"] == "superseded"
    assert old_detail["chain"][0]["id"] == mid["id"]
    service.close()


def test_read_session_rpc_shape_has_no_transcript_fields(tmp_path):
    service, _ = make_service(tmp_path)
    session_id = new_ulid()
    service.call_tool("remember", {"text": "会话条目一", "scope": "session"}, session_id=session_id)
    result = service.handle_rpc("memory/read_session", {"session_id": session_id, "limit": 10})
    assert len(result["items"]) == 1
    assert set(result["items"][0]) == {"id", "text", "tags", "created_at"}

    admin_item = service.handle_rpc("memory/admin_get", {"id": result["items"][0]["id"]})["item"]
    assert "user_text" not in admin_item
    assert "assistant_text" not in admin_item
    assert "transcript" not in admin_item

    with pytest.raises(MemoryFailure) as exc:
        service.handle_rpc("memory/read_session", {"session_id": "not-a-ulid"})
    assert exc.value.code == "INVALID_ARGUMENT"
    service.close()


def test_events_and_consolidate_watermark(tmp_path):
    service, _ = make_service(tmp_path)
    assert service.handle_event("session.turn", {"session_id": new_ulid()})["handled"] is False
    assert service.handle_event("unknown.topic", {})["handled"] is False

    with pytest.raises(MemoryFailure) as exc:
        service.handle_event("session.closed", {})
    assert exc.value.code == "INVALID_ARGUMENT"

    assert service.backend.get_last_consolidate() is None
    first = service.backend.record_consolidate("scheduled", {"promoted": 2, "merged": 1})
    second = service.backend.record_consolidate("manual", {"promoted": 0})
    assert second == first + 1
    explicit = service.backend.record_consolidate(99, "manual", {"promoted": 3})
    assert explicit == 99
    last = service.backend.get_last_consolidate()
    assert last is not None
    assert last["kind"] == "manual"
    assert last["stats"] == {"promoted": 3}
    service.close()
