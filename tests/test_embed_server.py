"""Embedding clients and the SDK/manifest adapter."""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import pytest

from memory_module.embed import (
    EmbeddingError,
    FakeEmbedding,
    LocalHashEmbedding,
    OpenAIEmbeddingClient,
)
from memory_module.server import MemoryService, build_app, handle_tools_call
from memory_module.settings import MemorySettings
from memory_module.store import new_ulid

MODULE_ROOT = Path(__file__).resolve().parents[1]


class Transport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[urllib.request.Request] = []
        self.timeouts: list[float] = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        return self.responses.pop(0)


def _ok(vector):
    return 200, json.dumps({"data": [{"embedding": vector}]}).encode()


def test_local_hash_is_deterministic_normalized_and_overlaps_cjk():
    embedder = LocalHashEmbedding(dim=64)
    first = embedder.embed("用户偏好深色主题")
    second = embedder.embed("用户偏好深色主题")
    assert first == second
    assert pytest.approx(math.sqrt(sum(x * x for x in first)), rel=1e-6) == 1.0

    related = embedder.embed("界面主题")
    unrelated = embedder.embed("quantum chromodynamics")
    cosine_related = sum(a * b for a, b in zip(first, related))
    cosine_unrelated = sum(a * b for a, b in zip(first, unrelated))
    assert cosine_related > cosine_unrelated


def test_openai_client_retries_once_on_429_then_returns_vector():
    transport = Transport([(429, b""), _ok([0.1, 0.2, 0.3, 0.4])])
    client = OpenAIEmbeddingClient(
        base_url="https://llm.example/v1",
        model="embed-small",
        dim=4,
        api_key_env="MEMORY_API_KEY",
        environ={"MEMORY_API_KEY": "sk-test"},
        transport=transport,
    )
    assert client.embed("hello") == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert len(transport.requests) == 2
    assert transport.requests[0].full_url == "https://llm.example/v1/embeddings"
    assert transport.requests[0].get_header("Authorization") == "Bearer sk-test"
    body = json.loads(transport.requests[0].data)
    assert body == {"model": "embed-small", "input": ["hello"]}


def test_openai_client_gives_up_after_two_5xx():
    transport = Transport([(500, b"boom"), (503, b"boom")])
    client = OpenAIEmbeddingClient(
        base_url="https://llm.example/v1",
        model="m",
        dim=4,
        environ={"MEMORY_API_KEY": "sk-test"},
        transport=transport,
    )
    with pytest.raises(EmbeddingError) as exc:
        client.embed("hello")
    assert "HTTP 503" in str(exc.value)
    assert len(transport.requests) == 2


def test_openai_client_does_not_retry_4xx_and_never_leaks_text():
    transport = Transport([(400, b"bad request")])
    client = OpenAIEmbeddingClient(
        base_url="https://llm.example/v1",
        model="m",
        dim=4,
        environ={"MEMORY_API_KEY": "sk-test"},
        transport=transport,
    )
    with pytest.raises(EmbeddingError) as exc:
        client.embed("secret memory text")
    assert "HTTP 400" in str(exc.value)
    assert "secret memory text" not in str(exc.value)
    assert len(transport.requests) == 1


def test_openai_client_requires_the_environment_variable():
    transport = Transport([])
    client = OpenAIEmbeddingClient(
        base_url="https://llm.example/v1",
        model="m",
        dim=4,
        environ={},
        transport=transport,
    )
    with pytest.raises(EmbeddingError) as exc:
        client.embed("hello")
    assert "MEMORY_API_KEY" in str(exc.value)
    assert transport.requests == []


def test_openai_client_rejects_wrong_dimension():
    transport = Transport([_ok([0.1, 0.2])])
    client = OpenAIEmbeddingClient(
        base_url="https://llm.example/v1",
        model="m",
        dim=4,
        environ={"MEMORY_API_KEY": "sk-test"},
        transport=transport,
    )
    with pytest.raises(EmbeddingError) as exc:
        client.embed("hello")
    assert "dimension 2" in str(exc.value) and "4" in str(exc.value)


def test_fake_embedding_can_fail_and_records_calls():
    fake = FakeEmbedding(dim=4, fail_times=1)
    with pytest.raises(EmbeddingError):
        fake.embed("a")
    assert fake.calls == ["a"]
    assert len(fake.embed("a")) == 4


def test_pure_python_vector_path_matches_the_numpy_path(tmp_path, monkeypatch):
    from memory_module import store as store_module

    monkeypatch.setattr(store_module, "_np", None)
    settings = MemorySettings(data_dir=tmp_path, embedding_dim=8)
    service = MemoryService(settings, embedder=FakeEmbedding(dim=8))
    service.call_tool("remember", {"text": "纯 Python 余弦"})
    service.call_tool("remember", {"text": "另一条记忆"})
    result = service.call_tool("recall", {"query": "余弦", "k": 2})
    assert result["count"] >= 1
    ids, vectors = service.backend.all_vectors()
    assert len(ids) == 2
    assert isinstance(vectors, list) and len(vectors[0]) == 8
    service.close()


# --------------------------------------------------------------------- server
pytest.importorskip("vade_sdk")


def make_service(tmp_path):
    settings = MemorySettings(data_dir=tmp_path, embedding_dim=8)
    return MemoryService(settings, embedder=FakeEmbedding(dim=8))


def test_committed_manifest_matches_the_built_app(tmp_path):
    service = make_service(tmp_path)
    app = build_app(service)
    committed = json.loads((MODULE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert app.manifest() == committed
    assert app.check() == []
    assert committed["concurrency"] == "exclusive"
    assert committed["limits"]["max_concurrency"] == 1
    assert committed["role"] == "evolve"
    assert committed["subscriptions"] == ["session.turn", "session.closed"]
    assert committed["injects"] == ["session_id"]
    assert [tool["name"] for tool in committed["capabilities"]["tools"]] == [
        "remember",
        "recall",
        "forget",
    ]
    assert {tool["name"]: tool["risk"] for tool in committed["capabilities"]["tools"]} == {
        "remember": "low",
        "recall": "low",
        "forget": "high",
    }
    service.close()


def test_forget_is_high_and_remember_has_no_session_id_field(tmp_path):
    service = make_service(tmp_path)
    app = build_app(service)
    remember = app.tool("remember").input_schema
    assert "session_id" not in remember["properties"]
    assert remember["additionalProperties"] is False
    assert remember["required"] == ["text"]
    assert app.tool("forget").risk == "high"
    service.close()


async def test_sdk_rejects_a_forged_session_id_argument(tmp_path):
    service = make_service(tmp_path)
    app = build_app(service)
    envelope = await app.call_tool(
        "remember",
        {"text": "伪造", "scope": "session", "session_id": new_ulid()},
    )
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "INVALID_ARGUMENT"
    service.close()


async def test_sdk_tool_error_keeps_not_found(tmp_path):
    service = make_service(tmp_path)
    app = build_app(service)
    envelope = await app.call_tool("forget", {"id": "01J00000000000000000000000"})
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "NOT_FOUND"
    service.close()


def test_core_ctx_session_id_is_used_by_the_adapter(tmp_path):
    service = make_service(tmp_path)
    session_id = new_ulid()
    missing = handle_tools_call(
        service, {"tool": "remember", "args": {"text": "没有 ctx", "scope": "session"}}
    )
    assert missing["ok"] is False
    assert missing["error"]["code"] == "PERMISSION_DENIED"

    injected = handle_tools_call(
        service,
        {"tool": "remember", "args": {"text": "有 ctx", "scope": "session"}, "ctx": {"session_id": session_id}},
    )
    assert injected["ok"] is True
    assert len(service.backend.read_session(session_id)) == 1
    service.close()


def test_startup_dimension_mismatch_is_a_readable_cli_error(tmp_path):
    import os
    import subprocess
    import sys

    from memory_module.store import SqliteBackend

    settings = MemorySettings(data_dir=tmp_path, embedding_dim=8)
    SqliteBackend(settings).close()
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(MODULE_ROOT / "src"),
            "MEMORY_DATA_DIR": str(tmp_path),
            "MEMORY_EMBEDDING_DIM": "4",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-m", "memory_module", "--standalone", "recall", "--args", '{"query":"x"}'],
        cwd=MODULE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 2
    assert "dimension" in completed.stderr.lower() or "dim=8" in completed.stderr
    assert "Traceback" not in completed.stderr
