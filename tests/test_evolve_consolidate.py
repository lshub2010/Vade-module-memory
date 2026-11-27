"""consolidate: two-session gate, conflicts, lessons, decay, idempotency (M4b §七)."""

from __future__ import annotations

import math

import pytest

from memory_module.embed import FakeEmbedding
from memory_module.evolve.config import EvolveSettings
from memory_module.evolve.consolidate import consolidate
from memory_module.settings import MemorySettings
from memory_module.store import SqliteBackend, new_ulid
from memory_module.tools import MemoryTools

V_A = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#: cos(V_NEAR, V_A) = 0.95 — above both the 0.92 cluster gate and the 0.90 conflict gate.
V_NEAR = [0.95, math.sqrt(1 - 0.95**2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
V_FAR = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class Clock:
    def __init__(self, start: int = 1_700_000_000) -> None:
        self.now = start

    def __call__(self) -> int:
        self.now += 1
        return self.now


def make(tmp_path):
    settings = MemorySettings(
        data_dir=tmp_path,
        embedding_dim=8,
        global_capacity=100,
        session_capacity=50,
        session_inject_char_limit=4200,
    )
    backend = SqliteBackend(settings, clock=Clock())
    return backend, settings


def stage(
    backend,
    text,
    sessions,
    *,
    namespace="user:default",
    vector=V_A,
    kind="fact",
    evidence=None,
    confidence=0.6,
):
    for session_id in sessions:
        backend.staging_put(
            text,
            namespace,
            session_id,
            vector,
            confidence,
            kind=kind,
            evidence=evidence,
        )


def enabled(**overrides) -> EvolveSettings:
    kwargs = dict(enabled=True)
    kwargs.update(overrides)
    return EvolveSettings(**kwargs)


def test_two_distinct_sessions_promote_exactly_one_global_row(tmp_path):
    backend, _settings = make(tmp_path)
    s1, s2 = new_ulid(), new_ulid()
    stage(backend, "项目使用 SQLite 存储", [s1, s2])

    stats = consolidate(backend, enabled(), clock=Clock(), kind="manual")

    assert stats.status == "ok"
    assert stats.promoted == 1
    assert stats.clusters == 1
    assert stats.staging_cleared == 1
    assert stats.single_session == 0
    assert backend.count(layer="global") == 1
    assert backend.staging_list() == []
    records, _cursor = backend.admin_list(layer="global", status="active")
    assert len(records) == 1
    assert records[0].text == "项目使用 SQLite 存储"
    assert records[0].source == "evolve"
    assert records[0].promoted_by == "evolve"
    assert backend.get_last_consolidate()["kind"] == "manual"


def test_one_session_stays_in_staging_and_is_invisible_to_recall(tmp_path):
    backend, settings = make(tmp_path)
    s1 = new_ulid()
    stage(backend, "只在一次会话里出现的偏好", [s1], vector=V_A)

    stats = consolidate(backend, enabled(), clock=Clock())

    assert stats.promoted == 0
    assert stats.single_session == 1
    assert len(backend.staging_list()) == 1
    assert backend.count(layer="global") == 0

    embedder = FakeEmbedding(dim=8, mapping={"只在一次会话里出现的偏好": V_A})
    tools = MemoryTools(backend, embedder, settings)
    result = tools.recall("只在一次会话里出现的偏好")
    assert result["count"] == 0


def test_same_session_twice_does_not_pass_the_two_session_gate(tmp_path):
    backend, _settings = make(tmp_path)
    s1 = new_ulid()
    backend.staging_put("同一会话重复出现", "user:default", s1, V_A, 0.6)
    backend.staging_put("同一会话重复出现", "user:default", s1, V_A, 0.7)

    stats = consolidate(backend, enabled(), clock=Clock())
    assert stats.promoted == 0
    assert stats.single_session == 1


def test_clustering_merges_paraphrases_and_keeps_the_longest_text(tmp_path):
    backend, _settings = make(tmp_path)
    s1, s2 = new_ulid(), new_ulid()
    stage(backend, "部署在本地回环", [s1, s2], vector=V_A)
    stage(backend, "部署固定在本地回环地址上，不对外暴露", [s1, s2], vector=V_NEAR)

    stats = consolidate(backend, enabled(), clock=Clock())

    assert stats.clusters == 1
    assert stats.promoted == 1
    assert stats.staging_cleared == 2
    assert backend.staging_list() == []
    records, _cursor = backend.admin_list(layer="global", status="active")
    assert len(records) == 1
    assert records[0].text == "部署固定在本地回环地址上，不对外暴露"


def test_conflict_supersedes_the_old_global_row(tmp_path):
    backend, settings = make(tmp_path)
    embedder = FakeEmbedding(
        dim=8,
        mapping={
            "部署端口是 8080": V_A,
            "部署端口不是 8080，改成 9090": V_NEAR,
        },
    )
    tools = MemoryTools(backend, embedder, settings)
    old = tools.remember("部署端口是 8080")["id"]

    s1, s2 = new_ulid(), new_ulid()
    stage(backend, "部署端口不是 8080，改成 9090", [s1, s2], vector=V_NEAR)

    stats = consolidate(backend, enabled(), clock=Clock())
    assert stats.promoted == 1
    assert stats.superseded == 1

    current, chain = backend.admin_get(old)
    assert current.superseded_by is not None
    assert current.status == "superseded"
    new_id = current.superseded_by
    new_rec = backend.get(new_id)
    assert new_rec is not None
    assert new_rec.supersedes == old
    assert [item.id for item in chain] == [new_id]

    recents = tools.recall("部署端口不是 8080，改成 9090")
    texts = [item["text"] for item in recents["items"]]
    assert "部署端口不是 8080，改成 9090" in texts
    assert "部署端口是 8080" not in texts

    everything = tools.recall("部署端口不是 8080，改成 9090", include_superseded=True)
    all_texts = [item["text"] for item in everything["items"]]
    assert set(all_texts) == {"部署端口是 8080", "部署端口不是 8080，改成 9090"}


def test_high_similarity_without_negation_keeps_both_rows(tmp_path):
    backend, settings = make(tmp_path)
    embedder = FakeEmbedding(dim=8, mapping={"网关端口是 8080": V_A})
    tools = MemoryTools(backend, embedder, settings)
    tools.remember("网关端口是 8080")

    s1, s2 = new_ulid(), new_ulid()
    stage(backend, "网关端口是 8080，网关版本 v2", [s1, s2], vector=V_NEAR)

    stats = consolidate(backend, enabled(), clock=Clock())
    assert stats.promoted == 1
    assert stats.superseded == 0
    records, _cursor = backend.admin_list(layer="global", status="active")
    assert len(records) == 2


def test_lessons_accumulate_and_never_supersede(tmp_path):
    backend, _settings = make(tmp_path)
    s1 = new_ulid()
    evidence_one = {"turn_id": "T01", "tool": "shell__run", "claim": "exit 1"}
    evidence_two = {"turn_id": "T02", "tool": "fs__read", "claim": "ENOENT"}

    stage(
        backend,
        "当部署失败时，先检查端口占用；注意：只看最后一次日志；验证：curl -fsS /healthz",
        [s1],
        namespace="agent:lessons",
        vector=V_A,
        kind="lesson",
        evidence=evidence_one,
        confidence=0.4,
    )
    first = consolidate(backend, enabled(), clock=Clock())
    assert first.lessons == 1

    stage(
        backend,
        "当部署失败时，不要只看最后一次日志，要先检查端口占用；验证：curl -fsS /healthz",
        [s1],
        namespace="agent:lessons",
        vector=V_NEAR,
        kind="lesson",
        evidence=evidence_two,
        confidence=0.9,
    )
    second = consolidate(backend, enabled(), clock=Clock())
    assert second.lessons == 1
    assert second.superseded == 0

    rows, _cursor = backend.admin_list(layer="global", namespace="agent:lessons", status="active")
    assert len(rows) == 2
    assert all(row.kind == "lesson" for row in rows)
    assert all(row.superseded_by is None and row.supersedes is None for row in rows)
    assert all(row.verified_at is None for row in rows)
    assert all(row.confidence <= 0.6 for row in rows)
    assert {row.evidence["turn_id"] for row in rows} == {"T01", "T02"}


def test_lesson_without_embedding_is_not_promoted(tmp_path):
    backend, _settings = make(tmp_path)
    s1 = new_ulid()
    backend.staging_put(
        "当构建失败时，先清缓存",
        "agent:lessons",
        s1,
        None,
        0.5,
        kind="lesson",
        evidence={"turn_id": "T01", "claim": "cache stale"},
    )

    stats = consolidate(backend, enabled(), clock=Clock(), embedder=None)

    assert stats.lessons == 0
    assert stats.dropped == 1
    assert backend.count(layer="global") == 0
    assert len(backend.staging_list()) == 1


def test_consolidate_is_idempotent_across_three_runs(tmp_path):
    backend, _settings = make(tmp_path)
    s1, s2 = new_ulid(), new_ulid()
    stage(backend, "项目使用 SQLite 存储", [s1, s2])
    clock = Clock()

    first = consolidate(backend, enabled(), clock=clock)
    second = consolidate(backend, enabled(), clock=clock)
    third = consolidate(backend, enabled(), clock=clock)

    assert first.promoted == 1 and first.staging_cleared == 1
    assert second.promoted == 0 and second.staging_cleared == 0
    assert third.promoted == 0 and third.archived == 0
    assert second.as_dict() == third.as_dict()
    assert backend.count(layer="global") == 1
    assert backend.count(layer="global", include_deleted=True) == 1
    assert backend.staging_list() == []
    assert backend.get_last_consolidate()["run_id"] == 3


def test_decay_archives_to_the_90_percent_watermark(tmp_path):
    backend, settings = make(tmp_path)
    embedder = FakeEmbedding(dim=8)
    tools = MemoryTools(backend, embedder, settings)
    for text in ("事实一", "事实二", "事实三"):
        tools.remember(text)

    stats = consolidate(backend, enabled(), clock=Clock(), capacity=2)

    # 3 live rows, capacity 2 -> target int(2 * 0.9) = 1, archive the two lowest.
    assert stats.archived == 2
    stats_now = backend.admin_stats()
    assert stats_now["global_count"] == 1
    assert stats_now["archived_count"] == 2


def test_disabled_consolidate_writes_nothing(tmp_path):
    backend, _settings = make(tmp_path)
    s1, s2 = new_ulid(), new_ulid()
    stage(backend, "项目使用 SQLite 存储", [s1, s2])

    stats = consolidate(backend, EvolveSettings(enabled=False), clock=Clock())

    assert stats.status == "disabled"
    assert backend.count(layer="global") == 0
    assert len(backend.staging_list()) == 1
    assert backend.get_last_consolidate() is None


def test_unknown_kind_is_rejected(tmp_path):
    backend, _settings = make(tmp_path)
    with pytest.raises(ValueError):
        consolidate(backend, enabled(), clock=Clock(), kind="whenever")
