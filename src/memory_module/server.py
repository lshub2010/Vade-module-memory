"""Protocol adapter: ModuleApp + ToolSpec + the private-RPC surface.

`tools.py` and `store.py` never import the SDK. This file is the only place
that knows `vade_sdk` exists:

1. it turns `MemoryTools`' three methods into `ToolSpec`s (metadata here, JSON
   Schema from the method signature by the SDK);
2. it translates `MemoryFailure` into `vade_sdk.ToolError` with the same C-04
   code;
3. it owns the manifest extras M4b needs: `role=evolve`, `injects=[session_id]`
   and `subscriptions=[session.turn, session.closed]` (M4a §三).

The private RPCs (`memory/read_session`, `memory/admin_*`) are exposed through
`handle_rpc` / `handle_event`. Stock vade-sdk 0.1 does not dispatch unknown
method names, so core integration needs the small hook described in README
"Known limitations"; the dispatcher itself is complete and tested.
"""

from __future__ import annotations

import functools
import inspect
import json
import sys
from typing import Any, Callable, Mapping, Sequence

from . import tools as tools_module
from .embed import EmbeddingClient, LocalHashEmbedding, OpenAIEmbeddingClient
from .errors import MemoryFailure
from .rpc import MEMORY_RPC_METHODS, RpcDispatcher, handle_event as dispatch_event
from .settings import MemorySettings
from .store import MemoryBackend, SqliteBackend

__all__ = [
    "MODULE_ID",
    "MODULE_VERSION",
    "MemoryService",
    "build_service",
    "build_app",
    "handle_rpc",
    "handle_event",
    "handle_tools_call",
    "main",
]

MODULE_ID = "com.example.memory"
MODULE_VERSION = "1.0.0"
SUBSCRIPTIONS = ("session.turn", "session.closed")

TOOL_METADATA: dict[str, dict[str, Any]] = {
    "remember": {
        "description": "记住一条事实、偏好或结论；同一内容自动去重，不覆盖旧结论",
        "risk": "low",
        "timeout_ms": 15_000,
        "idempotent": True,
    },
    "recall": {
        "description": "按语义检索长期记忆；可选叠加当前会话层",
        "risk": "low",
        "timeout_ms": 15_000,
        "idempotent": True,
    },
    "forget": {
        "description": "软删除一条记忆并写入归档；不可恢复（危险操作）",
        "risk": "high",
        "timeout_ms": 10_000,
        "idempotent": False,
    },
}


def build_embedder(settings: MemorySettings) -> EmbeddingClient:
    if settings.embedding_base_url:
        return OpenAIEmbeddingClient(
            base_url=settings.embedding_base_url,
            model=settings.effective_model,
            dim=settings.embedding_dim,
            api_key_env=settings.api_key_env,
            timeout_s=settings.embedding_timeout_s,
        )
    return LocalHashEmbedding(dim=settings.embedding_dim, model=settings.effective_model)


class MemoryService:
    """The composition root: backend + embedder + tool bodies + RPC dispatcher."""

    def __init__(
        self,
        settings: MemorySettings | None = None,
        *,
        backend: MemoryBackend | None = None,
        embedder: EmbeddingClient | None = None,
        clock: Any = None,
    ) -> None:
        self.settings = settings or MemorySettings.from_env()
        self.backend = backend or SqliteBackend(self.settings, clock=clock)
        self.embedder = embedder or build_embedder(self.settings)
        self.tools = tools_module.MemoryTools(self.backend, self.embedder, self.settings, clock=clock)
        self.rpc = RpcDispatcher(self.backend, self.settings)

    def call_tool(
        self, name: str, args: Mapping[str, Any] | None = None, *, session_id: str | None = None
    ) -> Any:
        """Invoke one tool with the core-injected session id in scope."""
        function = {
            "remember": self.tools.remember,
            "recall": self.tools.recall,
            "forget": self.tools.forget,
        }.get(name)
        if function is None:
            raise MemoryFailure("NOT_FOUND", f"unknown tool {name!r}")
        payload = dict(args or {})
        required = {
            "remember": ["text"],
            "recall": ["query"],
            "forget": ["id"],
        }[name]
        missing = [key for key in required if key not in payload]
        if missing:
            raise MemoryFailure(
                "INVALID_ARGUMENT",
                f"missing required argument(s) {missing} for tool {name!r}",
            )
        allowed = {
            "remember": {"text", "tags", "scope", "namespace"},
            "recall": {"query", "k", "namespace", "include_session", "include_superseded"},
            "forget": {"id"},
        }[name]
        extra = sorted(set(payload) - allowed)
        if extra:
            # The SDK rejects this from the manifest's additionalProperties=false;
            # direct callers get the same code instead of a TypeError.
            raise MemoryFailure(
                "INVALID_ARGUMENT",
                f"unknown argument(s) {extra}; session_id is injected by core and is not a tool argument",
            )
        with self.tools.session_scope(session_id):
            return function(**payload)

    def handle_rpc(self, method: str, params: Any = None) -> dict[str, Any]:
        return self.rpc.dispatch(method, params)

    def handle_event(self, topic: str, payload: Any = None) -> dict[str, Any]:
        return dispatch_event(self.backend, topic, payload)

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()


def build_service(settings: MemorySettings | None = None) -> MemoryService:
    return MemoryService(settings)


def _adapt(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Translate `MemoryFailure` into the SDK's `ToolError` (same code)."""
    from vade_sdk import ToolError

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_call(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except MemoryFailure as exc:
                raise ToolError(exc.code, exc.message) from None

        return async_call

    @functools.wraps(fn)
    def sync_call(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except MemoryFailure as exc:
            raise ToolError(exc.code, exc.message) from None

    return sync_call


def _build_module_app(service: MemoryService) -> Any:
    from vade_sdk import ModuleApp, ToolSpec

    class _App(ModuleApp):
        def _manifest_with(self, tools: Mapping[str, Any]) -> dict[str, Any]:
            payload = super()._manifest_with(tools)
            # C-03 optional fields; core reads them but the 0.1 SDK constructor
            # has no parameters for them. They are part of the frozen contract
            # the evolver depends on (M4a §三).
            payload["subscriptions"] = list(SUBSCRIPTIONS)
            payload["injects"] = ["session_id"]
            return payload

    app = _App(
        module_id=MODULE_ID,
        version=MODULE_VERSION,
        concurrency="exclusive",
        display_name="记忆模块",
        priority=0,
        capabilities=["tools"],
        role="evolve",
        max_concurrency=1,
        max_payload=65519,
    )
    for name, method in (
        ("remember", service.tools.remember),
        ("recall", service.tools.recall),
        ("forget", service.tools.forget),
    ):
        meta = TOOL_METADATA[name]
        app.add_tool(
            ToolSpec(
                name=name,
                fn=_adapt(method),
                description=meta["description"],
                risk=meta["risk"],
                timeout_ms=meta["timeout_ms"],
                idempotent=meta["idempotent"],
            )
        )
    return app


class _SchemaOnlyService:
    """A service-shaped object for `--print-manifest` / `--check`.

    Manifest generation must not open SQLite: a dimension mismatch or a
    read-only data directory must not stop the module from describing itself,
    and `--check` is run in CI before any database exists.
    """

    def __init__(self, settings: MemorySettings) -> None:
        self.settings = settings
        self.tools = tools_module.MemoryTools(None, LocalHashEmbedding(settings.embedding_dim), settings)

    def close(self) -> None:
        return None


def build_app(service: MemoryService | _SchemaOnlyService | None = None) -> Any:
    """Build the `ModuleApp`. The SDK is imported here, not at module import."""
    return _build_module_app(service or build_service())


def handle_rpc(service: MemoryService, method: str, params: Any = None) -> dict[str, Any]:
    return service.handle_rpc(method, params)


def handle_event(service: MemoryService, topic: str, payload: Any = None) -> dict[str, Any]:
    return service.handle_event(topic, payload)


def handle_tools_call(
    service: MemoryService,
    params: Mapping[str, Any],
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    """The `tools/call` shape core sends, including its out-of-band `ctx`.

    Core puts the injected session id in `params["ctx"]`, not in `args`
    (`router/dispatch.py`). SDK 0.1's transport drops `ctx`; this adapter reads
    it, so a patched transport or a test can exercise the real contract.
    """
    ctx = params.get("ctx") if isinstance(params.get("ctx"), Mapping) else {}
    injected = ctx.get("session_id", session_id)
    name = str(params.get("tool") or "")
    args = params.get("args") or {}
    if not isinstance(args, Mapping):
        return {"ok": False, "error": {"code": "INVALID_ARGUMENT", "message": "args must be an object", "retriable": False}}
    try:
        result = service.call_tool(name, args, session_id=injected if isinstance(injected, str) else None)
    except MemoryFailure as exc:
        return {
            "ok": False,
            "error": {"code": exc.code, "message": exc.message[:300], "retriable": exc.code == "MEMORY_FULL" or exc.code == "UNAVAILABLE" or exc.code == "OVERLOADED"},
        }
    return {"ok": True, "result": result, "truncated": False}


def _run_rpc_cli(argv: Sequence[str]) -> int:
    method_index = list(argv).index("--rpc")
    if method_index + 1 >= len(argv):
        sys.stderr.write("INVALID_ARGUMENT: --rpc needs a method name\n")
        return 2
    method = argv[method_index + 1]
    args_json = "{}"
    if "--args" in argv:
        args_json = argv[list(argv).index("--args") + 1]
    try:
        params = json.loads(args_json)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"INVALID_ARGUMENT: --args is not valid JSON: {exc}\n")
        return 2
    service = build_service()
    try:
        result = service.handle_rpc(method, params)
    except MemoryFailure as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": {"code": exc.code, "message": exc.message}}, ensure_ascii=False) + "\n")
        return 1
    finally:
        service.close()
    sys.stdout.write(json.dumps({"ok": True, "result": result}, ensure_ascii=False) + "\n")
    return 0


def _run_event_cli(argv: Sequence[str]) -> int:
    index = list(argv).index("--event")
    if index + 1 >= len(argv):
        sys.stderr.write("INVALID_ARGUMENT: --event needs a topic\n")
        return 2
    topic = argv[index + 1]
    payload = {}
    if "--args" in argv:
        try:
            payload = json.loads(argv[list(argv).index("--args") + 1])
        except json.JSONDecodeError as exc:
            sys.stderr.write(f"INVALID_ARGUMENT: --args is not valid JSON: {exc}\n")
            return 2
    service = build_service()
    try:
        result = service.handle_event(topic, payload)
    except MemoryFailure as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": {"code": exc.code, "message": exc.message}}, ensure_ascii=False) + "\n")
        return 1
    finally:
        service.close()
    sys.stdout.write(json.dumps({"ok": True, "result": result}, ensure_ascii=False) + "\n")
    return 0


def _run_standalone_cli(argv: Sequence[str]) -> int:
    """`--standalone TOOL --args JSON` without the SDK.

    The SDK's own standalone runner needs `vade_sdk`; this path builds the
    service directly, which is what makes the image's "no SDK mounted" mode
    (and the design's "tools.py works before the SDK exists" gate) real.
    """
    index = list(argv).index("--standalone")
    if index + 1 >= len(argv):
        sys.stderr.write("INVALID_ARGUMENT: --standalone needs a tool name\n")
        return 2
    name = argv[index + 1]
    args_json = "{}"
    if "--args" in argv:
        at = list(argv).index("--args")
        if at + 1 >= len(argv):
            sys.stderr.write("INVALID_ARGUMENT: --args needs a JSON object\n")
            return 2
        args_json = argv[at + 1]
    try:
        payload = json.loads(args_json or "{}")
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"INVALID_ARGUMENT: --args is not valid JSON: {exc}\n")
        return 2
    if not isinstance(payload, dict):
        sys.stderr.write("INVALID_ARGUMENT: --args must be a JSON object\n")
        return 2
    service = build_service()
    try:
        result = service.call_tool(name, payload)
    except MemoryFailure as exc:
        sys.stdout.write(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": exc.code,
                        "message": exc.message[:300],
                        "retriable": exc.code in ("MEMORY_FULL", "UNAVAILABLE", "OVERLOADED", "TIMEOUT"),
                    },
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        return 1
    finally:
        service.close()
    sys.stdout.write(json.dumps({"ok": True, "result": result}, ensure_ascii=False) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    from .errors import StartupError

    try:
        if "--self-test" in args:
            from .selftest import run_self_test

            return run_self_test()
        if "--rpc" in args:
            return _run_rpc_cli(args)
        if "--event" in args:
            return _run_event_cli(args)
        if "--print-manifest" in args or "--check" in args:
            return build_app(_SchemaOnlyService(MemorySettings.from_env())).run(args)
        if "--standalone" in args and "--core" not in args:
            return _run_standalone_cli(args)
        return build_app().run(args)
    except StartupError as exc:
        # A dimension/model mismatch must be one readable line, not a traceback
        # (M4a §六 test 6). Exit 2 is the module's "cannot start" code.
        sys.stderr.write(f"UNAVAILABLE: {exc}\n")
        return 2
    except ValueError as exc:
        sys.stderr.write(f"UNAVAILABLE: invalid configuration: {exc}\n")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
