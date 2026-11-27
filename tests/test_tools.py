"""Pure tool behaviour: dedupe, two-layer isolation, supersedes, validation."""

from __future__ import annotations

import pytest

from memory_module.embed import FakeEmbedding
from memory_module.errors import MemoryFailure
from memory_module.server import MemoryService
from memory_module.settings import MemorySettings
from memory_module.store import MemoryRecord, new_ulid


class Clock:
    def __init__(self, start: int = 1_700_000_000) -> None:
        self.now = start

    def __call__(self) -> int:
        self.now += 1
        return self.now


def make_service(tmp: object, **overrides: object) -> tuple[MemoryService, Clock, FakeEmbedding]:
    kwargs = dict(
        data_dir=tmp,
        embedding_dim=8,
        global_capacity=100,
        session_capacity=20,
        session_inject_char_limit=4200,
        recall_token_limit=800,
        ttl_days=7,
    )
    kwargs.update(overrides)
    settings = MemorySettings(**kwargs)  # type: ignore[arg-type]
    clock = Clock()
    embedder = FakeEmbedding(dim=settings.embedding_dim)
    return MemoryService(settings, embedder=embedder, clock=clock), clock, embedder


def test_remember_twice_is_one_row_and_second_is_deduped(tmp_path):
    service, _, _ = make_service(tmp_path)
    first = service.call_tool("remember", {"text": "用户偏好深色主题", "tags": ["pref"]})
    second = service.call_tool("remember", {"text": "  用户偏好深色主题  ", "tags": ["other"]})

    assert first["deduped"] is False
    assert second["deduped"] is True
    assert second["id"] == first["id"]
    assert service.backend.count() == 1
    item = service.backend.get(first["id"])
    assert item is not None
    assert item.tags == ["pref"]  # first write wins; a duplicate is not an update


def test_global_and_session_same_text_are_two_rows(tmp_path):
    service, _, _ = make_service(tmp_path)
    session_id = new_ulid()
    global_rec = service.call_tool("remember", {"text": "项目代号 Orion"})
    session_rec = service.call_tool(
        "remember", {"text": "项目代号 Orion", "scope": "session"}, session_id=session_id
    )

    assert global_rec["id"] != session_rec["id"]
    assert service.backend.count() == 2

    default = service.call_tool("recall", {"query": "项目代号", "k": 5})
    assert [item["layer"] for item in default["items"]] == ["global"]

    both = service.call_tool(
        "recall", {"query": "项目代号", "k": 5, "include_session": True}, session_id=session_id
    )
    assert {item["layer"] for item in both["items"]} == {"global", "session"}


def test_two_sessions_never_share_a_row(tmp_path):
    service, _, _ = make_service(tmp_path)
    a, b = new_ulid(), new_ulid()
    first = service.call_tool("remember", {"text": "我偏好深色主题", "scope": "session"}, session_id=a)
    second = service.call_tool("remember", {"text": "我偏好深色主题", "scope": "session"}, session_id=b)

    assert first["id"] != second["id"]
    assert service.backend.count(layer="session") == 2
    assert [rec.id for rec in service.backend.read_session(a)] == [first["id"]]
    assert [rec.id for rec in service.backend.read_session(b)] == [second["id"]]

    hits_a = service.call_tool(
        "recall", {"query": "深色主题", "k": 10, "include_session": True}, session_id=a
    )
    assert {item["id"] for item in hits_a["items"]} == {first["id"]}


def test_namespace_is_a_global_partition(tmp_path):
    service, _, _ = make_service(tmp_path)
    service.call_tool("remember", {"text": "默认用户偏好", "namespace": "user:default"})
    service.call_tool("remember", {"text": "评审代理偏好", "namespace": "agent:reviewer"})

    only_agent = service.call_tool("recall", {"query": "偏好", "k": 5, "namespace": "agent:reviewer"})
    assert len(only_agent["items"]) == 1
    assert only_agent["items"][0]["namespace"] == "agent:reviewer"

    all_namespaces = service.call_tool("recall", {"query": "偏好", "k": 5})
    assert len(all_namespaces["items"]) == 2


def test_forget_is_soft_and_second_forget_is_not_found(tmp_path):
    service, _, _ = make_service(tmp_path)
    rec = service.call_tool("remember", {"text": "这条会被删除"})
    assert service.call_tool("forget", {"id": rec["id"]}) == {"ok": True, "id": rec["id"]}
    assert service.call_tool("recall", {"query": "删除", "k": 5})["count"] == 0

    with pytest.raises(MemoryFailure) as exc:
        service.call_tool("forget", {"id": rec["id"]})
    assert exc.value.code == "NOT_FOUND"

    archived, cursor = service.backend.admin_list(status="archived", limit=10)
    assert [item.id for item in archived] == [rec["id"]]


def test_supersedes_keeps_both_rows_but_hides_old_from_default_recall(tmp_path):
    service, _, _ = make_service(tmp_path)
    old = service.call_tool("remember", {"text": "端口配置为 8080"})
    new = service.call_tool("remember", {"text": "端口配置改为 9090"})
    service.backend.supersede(old["id"], new["id"])

    default = service.call_tool("recall", {"query": "端口配置", "k": 10})
    assert {item["id"] for item in default["items"]} == {new["id"]}

    with_history = service.call_tool(
        "recall", {"query": "端口配置", "k": 10, "include_superseded": True}
    )
    assert {item["id"] for item in with_history["items"]} == {old["id"], new["id"]}

    current, chain = service.backend.admin_get(new["id"])
    assert current.supersedes == old["id"]
    assert old["id"] in {item.id for item in chain}
    assert service.backend.get(old["id"]).superseded_by == new["id"]  # type: ignore[union-attr]


def test_lesson_requires_evidence_and_starts_unverified(tmp_path):
    service, _, embedder = make_service(tmp_path)
    vec = embedder.embed("一条教训")
    lesson = MemoryRecord(
        id="",
        layer="global",
        namespace="agent:lessons",
        session_id="",
        text="不要在未验证时清空索引",
        kind="lesson",
        evidence=None,
        embedding=vec,
    )
    with pytest.raises(MemoryFailure) as exc:
        service.backend.put(lesson)
    assert exc.value.code == "INVALID_ARGUMENT"

    lesson.evidence = {"turn_id": "01J00000000000000000000000", "tool": "memory", "claim": "验证前不要清空"}
    lesson_id = service.backend.put(lesson)
    stored = service.backend.get(lesson_id)
    assert stored is not None
    assert stored.kind == "lesson"
    assert stored.verified_at is None
    assert stored.evidence is not None


def test_session_scope_needs_core_injection_and_rejects_namespace(tmp_path):
    service, _, _ = make_service(tmp_path)
    with pytest.raises(MemoryFailure) as exc:
        service.call_tool("remember", {"text": "无会话上下文", "scope": "session"})
    assert exc.value.code == "PERMISSION_DENIED"

    with pytest.raises(MemoryFailure) as exc:
        service.call_tool(
            "remember",
            {"text": "错误命名空间", "scope": "session", "namespace": "agent:reviewer"},
            session_id=new_ulid(),
        )
    assert exc.value.code == "INVALID_ARGUMENT"


def test_session_id_is_not_a_tool_argument(tmp_path):
    service, _, _ = make_service(tmp_path)
    with pytest.raises(MemoryFailure) as exc:
        service.call_tool(
            "remember", {"text": "伪造会话", "scope": "session", "session_id": new_ulid()}
        )
    assert exc.value.code == "INVALID_ARGUMENT"
    assert service.backend.count() == 0


@pytest.mark.parametrize(
    "args, code",
    [
        ({"text": ""}, "INVALID_ARGUMENT"),
        ({"text": "x" * 8193}, "PAYLOAD_TOO_LARGE"),
        ({"text": "ok", "tags": ["x"] * 17}, "INVALID_ARGUMENT"),
        ({"text": "ok", "tags": ["x" * 33]}, "INVALID_ARGUMENT"),
        ({"text": "ok", "namespace": "Invalid Namespace"}, "INVALID_ARGUMENT"),
        ({"text": "ok", "scope": "both"}, "INVALID_ARGUMENT"),
    ],
)
def test_remember_argument_edges(tmp_path, args, code):
    service, _, _ = make_service(tmp_path)
    with pytest.raises(MemoryFailure) as exc:
        service.call_tool("remember", args)
    assert exc.value.code == code


def test_recall_argument_edges(tmp_path):
    service, _, _ = make_service(tmp_path)
    service.call_tool("remember", {"text": "一些内容"})
    for args in (
        {"query": ""},
        {"query": "内容", "k": 0},
        {"query": "内容", "k": 21},
        {"query": "内容", "namespace": "bad namespace"},
        {"query": "内容", "include_session": "yes"},
    ):
        with pytest.raises(MemoryFailure) as exc:
            service.call_tool("recall", args)
        assert exc.value.code == "INVALID_ARGUMENT"


def test_recall_token_budget_clips_only_the_returned_copy(tmp_path):
    service, _, _ = make_service(tmp_path, recall_token_limit=40)
    service.call_tool("remember", {"text": "记忆" * 100})  # 200 chars -> over 40*4
    item = service.call_tool("recall", {"query": "记忆", "k": 5})["items"][0]
    assert len(item["text"]) <= 160
    stored = service.backend.get(item["id"])
    assert stored is not None and len(stored.text) == 200
