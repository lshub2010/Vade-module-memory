"""The last mile: a real OpenAI-compatible call behind the four model triggers.

`EvolveLLM` was already fully tested with an injected fake, so the gaps these
tests close are the ones the fake could never catch:

* does a request actually leave the process, at `/v1/chat/completions`, with
  the `Authorization` header built from the *named* environment variable?
* when nothing is configured, does the module still start and run pure rules?
* when the upstream is broken (500 / timeout), does the failure stop at
  `evolve_degraded` instead of reaching the session?

The fake upstream is stdlib `http.server` on a background thread — no network,
no extra dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from memory_module.embed import FakeEmbedding
from memory_module.evolve.config import EvolveSettings
from memory_module.evolve.llm import EvolveLLM
from memory_module.evolve.openai_compat import LlmRequestError, OpenAICompatLLM
from memory_module.evolve.pipeline import EvolvePipeline
from memory_module.settings import MemorySettings
from memory_module.store import SqliteBackend, new_ulid

SECRET = "sk-test-from-env-not-config"

SUMMARY_OK = json.dumps(
    {
        "summary": "会话确认了部署走本地回环。",
        "facts": [
            {"text": "部署走本地回环", "namespace": "user:default", "importance": 4, "confidence": 0.9}
        ],
        "lessons": [],
        "negative_feedback": False,
        "skill_candidate": False,
    },
    ensure_ascii=False,
)


class Clock:
    def __init__(self, start: int = 1_700_000_000) -> None:
        self.now = start

    def __call__(self) -> int:
        self.now += 1
        return self.now


class Upstream:
    """A scripted OpenAI-compatible upstream.

    `responses` is a list of `(status, body_dict_or_str)`; each request pops
    one, and the last entry repeats. `delay` makes the timeout test real rather
    than simulated. Every request is recorded with its path, headers and body
    so the assertions can be about the wire, not about our own abstractions.
    """

    def __init__(self, *responses, delay: float = 0.0) -> None:
        self.responses = list(responses) or [(200, {"choices": [{"message": {"content": "{}"}}]})]
        self.delay = delay
        self.requests: list[dict] = []
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - stdlib naming
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                try:
                    body = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    body = None
                upstream.requests.append(
                    {
                        "path": self.path,
                        "method": self.command,
                        "authorization": self.headers.get("Authorization"),
                        "content_type": self.headers.get("Content-Type"),
                        "body": body,
                    }
                )
                status, payload = upstream.responses[0] if len(upstream.responses) == 1 else upstream.responses.pop(0)
                if upstream.delay:
                    import time as _time

                    _time.sleep(upstream.delay)
                data = (
                    payload.encode("utf-8")
                    if isinstance(payload, str)
                    else json.dumps(payload, ensure_ascii=False).encode("utf-8")
                )
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    # The client gave up first (the timeout test). That is the
                    # behaviour under test, not a server failure. Windows raises
                    # `ConnectionAbortedError` (WSAECONNABORTED) where POSIX
                    # raises `BrokenPipeError`.
                    pass

            def log_message(self, format, *args) -> None:  # noqa: A002 - stdlib signature
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "Upstream":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"


def make_backend(tmp_path) -> SqliteBackend:
    settings = MemorySettings(
        data_dir=tmp_path,
        embedding_dim=8,
        global_capacity=100,
        session_capacity=50,
    )
    return SqliteBackend(settings, clock=Clock())


def session_turns(count: int) -> list[dict]:
    return [
        {"turn_id": f"T{index:02d}", "state": "completed", "tool_calls": []}
        for index in range(1, count + 1)
    ]


def make_pipeline(tmp_path, llm, memory_settings=None):
    memory = memory_settings or MemorySettings(
        data_dir=tmp_path,
        embedding_dim=8,
        global_capacity=100,
        session_capacity=50,
    )
    backend = SqliteBackend(memory, clock=Clock())
    pipeline = EvolvePipeline(
        backend,
        EvolveSettings(enabled=True, flush_min_turns=2),
        llm=llm,
        embedder=FakeEmbedding(dim=8),
        memory_settings=memory,
        clock=Clock(),
    )
    return pipeline, backend


# --------------------------------------------------------------------- e2e
def test_session_close_reaches_a_real_upstream(tmp_path, monkeypatch, caplog):
    """The whole chain: settings → client → wire → parsed JSON → stored summary."""
    # A real process env var, so "read from the environment" is actually tested
    # rather than an injected mapping standing in for it.
    monkeypatch.setenv("TEST_EVOLVE_KEY", SECRET)

    with Upstream((200, {"choices": [{"message": {"content": SUMMARY_OK}}]})) as upstream:
        memory = MemorySettings.from_env(
            {
                "MEMORY_DATA_DIR": str(tmp_path),
                "MEMORY_EMBEDDING_DIM": "8",
                "MEMORY_GLOBAL_CAPACITY": "100",
                "MEMORY_SESSION_CAPACITY": "50",
                "MEMORY_EVOLVE_LLM_BASE_URL": upstream.base_url,
                "MEMORY_EVOLVE_LLM_API_KEY_ENV": "TEST_EVOLVE_KEY",
                "MEMORY_EVOLVE_LLM_MODEL": "test-cheap-model",
            }
        )
        assert memory.evolve_llm_base_url == upstream.base_url
        assert memory.evolve_llm_api_key_env == "TEST_EVOLVE_KEY"
        assert memory.effective_evolve_model() == "test-cheap-model"

        # Exactly what `build_evolve_llm` constructs, but with no `environ`
        # override so the key has to come from `os.environ`.
        llm = OpenAICompatLLM(
            base_url=memory.evolve_llm_base_url,
            model=memory.effective_evolve_model(),
            api_key_env=memory.evolve_llm_api_key_env,
            timeout_s=memory.evolve_llm_timeout_s,
        )
        pipeline, backend = make_pipeline(tmp_path, llm, memory)
        session_id = new_ulid()

        with caplog.at_level(logging.INFO, logger="memory_module.evolve"):
            outcome = asyncio.run(
                pipeline.on_session_closed(
                    {"session_id": session_id, "turns": session_turns(4)}
                )
            )

    # The request really went out, once, to the documented path.
    assert len(upstream.requests) == 1
    request = upstream.requests[0]
    assert request["method"] == "POST"
    assert request["path"] == "/v1/chat/completions"
    assert request["content_type"] == "application/json"
    # The bearer token came from the env var, and never from config.
    assert request["authorization"] == f"Bearer {SECRET}"
    assert SECRET not in json.dumps(memory.__dict__, default=str)

    # Non-streaming single-shot shape.
    body = request["body"]
    assert body["model"] == "test-cheap-model"
    assert body["stream"] is False
    assert body["temperature"] == 0
    assert [m["role"] for m in body["messages"]] == ["user"]
    assert "会话总结器" in body["messages"][0]["content"]

    # And the JSON the upstream returned was parsed and used, not degraded.
    assert outcome["status"] == "ok", outcome
    assert outcome["turns"] == 4

    stored = backend.read_session(session_id)
    texts = [row.text for row in stored]
    assert any("部署走本地回环" in text for text in texts), texts


def test_default_base_url_is_empty_so_nothing_is_called(tmp_path):
    """No `MEMORY_EVOLVE_LLM_BASE_URL` ⇒ no client ⇒ pure rules, no crash."""
    from memory_module.server import build_evolve_llm

    settings = MemorySettings(data_dir=tmp_path)
    evolve = EvolveSettings(enabled=True)
    assert settings.evolve_llm_base_url == ""
    assert build_evolve_llm(settings, evolve) is None


def test_enabled_without_base_url_warns_once_and_degrades(tmp_path, caplog):
    """`enabled=true` + no upstream is a half-finished setup: say so, don't die."""
    from memory_module.server import build_evolve_llm

    settings = MemorySettings(data_dir=tmp_path)
    with caplog.at_level(logging.WARNING, logger="memory_module"):
        client = build_evolve_llm(settings, EvolveSettings(enabled=True))

    assert client is None
    warnings = [r.message for r in caplog.records if "MEMORY_EVOLVE_LLM_BASE_URL" in r.message]
    assert len(warnings) == 1
    assert "pure rules" in warnings[0]

    # And the pipeline built with `llm=None` degrades on a real trigger rather
    # than raising: this is the "no model" path end to end.
    pipeline, _backend = make_pipeline(tmp_path, None, settings)
    session_id = new_ulid()
    with caplog.at_level(logging.WARNING, logger="memory_module.evolve"):
        outcome = asyncio.run(
            pipeline.on_session_closed({"session_id": session_id, "turns": session_turns(4)})
        )
    assert outcome["status"] == "degraded"
    assert outcome["error"] == "no_llm"
    assert any("evolve_degraded" in r.message for r in caplog.records)


def test_disabled_evolve_builds_no_client_even_with_a_base_url(tmp_path, caplog):
    """`enabled=false` is silent and inert even when a URL is configured."""
    from memory_module.server import build_evolve_llm

    settings = MemorySettings(data_dir=tmp_path, evolve_llm_base_url="http://127.0.0.1:1/v1")
    with caplog.at_level(logging.DEBUG, logger="memory_module"):
        assert build_evolve_llm(settings, EvolveSettings(enabled=False)) is None
    assert caplog.records == []


# ------------------------------------------------------- upstream failures
def test_upstream_500_degrades_without_reaching_the_session(tmp_path, caplog):
    """A broken provider must be `evolve_degraded`, never an exception."""
    with Upstream((500, {"error": {"message": "boom"}})) as upstream:
        llm = OpenAICompatLLM(
            base_url=upstream.base_url,
            model="test-model",
            api_key_env="TEST_EVOLVE_KEY",
            environ={"TEST_EVOLVE_KEY": SECRET},
        )
        pipeline, backend = make_pipeline(tmp_path, llm)
        session_id = new_ulid()

        with caplog.at_level(logging.WARNING, logger="memory_module.evolve"):
            outcome = asyncio.run(
                pipeline.on_session_closed({"session_id": session_id, "turns": session_turns(4)})
            )

    assert outcome["status"] == "degraded"
    assert outcome["error"] == "exception"
    assert any("evolve_degraded" in r.message for r in caplog.records)
    # The client raises; `EvolveLLM` is the one that decides that means degrade.
    assert len(upstream.requests) == 1
    assert backend.staging_list() == []


def test_upstream_timeout_degrades_without_raising(tmp_path, caplog):
    """A hanging upstream surfaces as `reason=timeout`, not a stack trace."""
    with Upstream((200, {"choices": [{"message": {"content": SUMMARY_OK}}]}), delay=1.0) as upstream:
        # 0.1 s is the client's floor; the upstream sleeps ten times that.
        llm = OpenAICompatLLM(
            base_url=upstream.base_url,
            model="test-model",
            api_key_env="TEST_EVOLVE_KEY",
            timeout_s=0.1,
            environ={"TEST_EVOLVE_KEY": SECRET},
        )
        settings = EvolveSettings(enabled=True, flush_min_turns=2)
        evolve_llm = EvolveLLM(llm, settings, timeout_s=5.0)

        with caplog.at_level(logging.WARNING, logger="memory_module.evolve"):
            result = asyncio.run(evolve_llm.json_call("给一个 JSON", label="session_close"))

    assert result.payload is None
    assert result.error == "timeout"
    assert any("reason=timeout" in r.message for r in caplog.records)


def test_missing_api_key_env_degrades_instead_of_sending(tmp_path, caplog):
    """The key is only ever a variable *name*; unset means no request at all."""
    with Upstream((200, {"choices": [{"message": {"content": SUMMARY_OK}}]})) as upstream:
        llm = OpenAICompatLLM(
            base_url=upstream.base_url,
            model="test-model",
            api_key_env="DEFINITELY_NOT_SET_KEY",
            environ={},
        )
        settings = EvolveSettings(enabled=True, flush_min_turns=2)
        evolve_llm = EvolveLLM(llm, settings)

        with caplog.at_level(logging.WARNING, logger="memory_module.evolve"):
            result = asyncio.run(evolve_llm.json_call("{}", label="session_close"))

    assert result.payload is None
    assert result.error == "exception"
    assert upstream.requests == []
    assert any("evolve_degraded" in r.message for r in caplog.records)


# ------------------------------------------------------- client unit level
def test_endpoint_joins_base_url_and_chat_completions():
    client = OpenAICompatLLM(
        base_url="http://127.0.0.1:8080/v1/",
        model="m",
        api_key_env="K",
        environ={"K": SECRET},
    )
    # A trailing slash on base_url must not produce `//chat/completions`.
    assert client.endpoint == "http://127.0.0.1:8080/v1/chat/completions"


def test_client_raises_on_a_non_json_body():
    client = OpenAICompatLLM(
        base_url="http://127.0.0.1:1/v1", model="m", api_key_env="K", environ={"K": SECRET}
    )
    with pytest.raises(LlmRequestError):
        client._content(b"not json at all")
    with pytest.raises(LlmRequestError):
        client._content(b'{"choices": []}')
    with pytest.raises(LlmRequestError):
        client._content(b'{"choices": [{"message": {"content": null}}]}')


def test_client_rejects_a_bad_base_url_at_construction():
    with pytest.raises(ValueError):
        OpenAICompatLLM(base_url="127.0.0.1:8080/v1", model="m", api_key_env="K")
    with pytest.raises(ValueError):
        OpenAICompatLLM(base_url="http://x/v1", model="", api_key_env="K")
    with pytest.raises(ValueError):
        OpenAICompatLLM(base_url="http://x/v1", model="m", api_key_env="")
    with pytest.raises(ValueError):
        OpenAICompatLLM(base_url="http://x/v1", model="m", api_key_env="K", timeout_s=0)


def test_bad_base_url_at_startup_degrades_instead_of_failing_to_boot(tmp_path, caplog):
    """A typo in the URL must not take the module down; the rule layer stays."""
    from memory_module.server import build_evolve_llm

    settings = MemorySettings(data_dir=tmp_path, evolve_llm_base_url="http://ok/v1")
    # Force the invalid value past `MemorySettings` to model a config source
    # that does not run `__post_init__` (e.g. a direct constructor call).
    object.__setattr__(settings, "evolve_llm_base_url", "not-a-url")
    with caplog.at_level(logging.WARNING, logger="memory_module"):
        assert build_evolve_llm(settings, EvolveSettings(enabled=True)) is None
    assert any("evolve llm disabled" in r.message for r in caplog.records)


# ------------------------------------------------------ build_app wiring
def test_build_app_hands_the_real_client_to_the_service(tmp_path, monkeypatch, caplog):
    """Requirement: `build_app()` itself must pass `llm=`, not just `build_service`."""
    from memory_module import server

    captured: dict = {}

    def fake_build_module_app(service):
        captured["service"] = service
        return "app"

    monkeypatch.setattr(server, "_build_module_app", fake_build_module_app)

    with Upstream((200, {"choices": [{"message": {"content": SUMMARY_OK}}]})) as upstream:
        monkeypatch.setattr(
            server.MemorySettings,
            "from_env",
            classmethod(
                lambda cls, env=None: MemorySettings(
                    data_dir=tmp_path,
                    embedding_dim=8,
                    global_capacity=100,
                    session_capacity=50,
                    evolve_llm_base_url=upstream.base_url,
                    evolve_llm_api_key_env="TEST_EVOLVE_KEY",
                    evolve_llm_model="wired-model",
                )
            ),
        )
        monkeypatch.setattr(
            server.EvolveSettings, "from_env", classmethod(lambda cls, env=None: EvolveSettings(enabled=True))
        )
        with caplog.at_level(logging.INFO, logger="memory_module"):
            assert server.build_app() == "app"

    service = captured["service"]
    assert service.worker is not None, "evolve enabled ⇒ a worker must exist"
    # The callable reached the layer that actually calls the model.
    reviewer_llm = service.worker.pipeline.reviewer._llm
    assert reviewer_llm._call is not None, "llm=None means the model triggers are pure rules"
    assert reviewer_llm._call.endpoint == f"{upstream.base_url}/chat/completions"
    assert reviewer_llm._call.model == "wired-model"
    assert any("evolve llm configured" in r.message for r in caplog.records)


def test_build_app_without_a_base_url_keeps_llm_none(tmp_path, monkeypatch, caplog):
    """The default deployment must be byte-for-byte today's behaviour."""
    from memory_module import server

    captured: dict = {}
    monkeypatch.setattr(server, "_build_module_app", lambda service: captured.setdefault("service", service))
    monkeypatch.setattr(
        server.MemorySettings,
        "from_env",
        classmethod(
            lambda cls, env=None: MemorySettings(
                data_dir=tmp_path, embedding_dim=8, global_capacity=100, session_capacity=50
            )
        ),
    )
    monkeypatch.setattr(
        server.EvolveSettings, "from_env", classmethod(lambda cls, env=None: EvolveSettings(enabled=True))
    )
    with caplog.at_level(logging.DEBUG, logger="memory_module"):
        server.build_app()

    service = captured["service"]
    assert service.worker is not None
    assert service.worker.pipeline.reviewer._llm._call is None
    assert service.worker.skillgen._llm._call is None
    assert any("MEMORY_EVOLVE_LLM_BASE_URL is unset" in r.message for r in caplog.records)
