"""`python -m memory_module --self-test` — one-process, no-core acceptance.

This is the module's own gate (design §15.3/§15.4): it proves storage, dedupe,
two-layer isolation, recall, forget and the private RPC dispatcher work on a
bare Python with no vade-sdk, no network and no API key. It prints one JSON
object; exit code 0 only when every check passed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from .embed import LocalHashEmbedding
from .errors import MemoryFailure
from .server import MemoryService
from .settings import MemorySettings
from .store import new_ulid


def run_self_test() -> int:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(name)

    with tempfile.TemporaryDirectory(prefix="vade-memory-selftest-") as tmp:
        settings = MemorySettings(
            data_dir=Path(tmp),
            global_capacity=10,
            session_capacity=10,
            session_inject_char_limit=400,
            recall_token_limit=200,
            embedding_dim=32,
        )
        service = MemoryService(settings, embedder=LocalHashEmbedding(dim=32))
        try:
            first = service.call_tool("remember", {"text": "用户偏好深色主题", "tags": ["pref"]})
            check("remember", bool(first.get("id")) and first.get("deduped") is False, first)
            again = service.call_tool("remember", {"text": "用户偏好深色主题", "tags": ["pref"]})
            check("dedupe", again.get("id") == first["id"] and again.get("deduped") is True, again)

            session_id = new_ulid()
            session_item = service.call_tool(
                "remember",
                {"text": "本次会话正在实现 memory 模块", "tags": ["ctx"], "scope": "session"},
                session_id=session_id,
            )
            check("session write", session_item.get("layer") == "session", session_item)

            global_hits = service.call_tool("recall", {"query": "记忆模块", "k": 5})
            layers = {item["layer"] for item in global_hits["items"]}
            check("default recall is global only", "session" not in layers and "global" in layers, global_hits)

            both = service.call_tool(
                "recall", {"query": "记忆模块", "k": 5, "include_session": True}, session_id=session_id
            )
            check("include_session", {item["layer"] for item in both["items"]} == {"global", "session"}, both)

            read = service.handle_rpc("memory/read_session", {"session_id": session_id, "limit": 10})
            check("read_session rpc", len(read["items"]) == 1, read)

            stats = service.handle_rpc("memory/admin_stats", {})
            check("admin_stats rpc", stats["global_count"] == 1 and stats["global_limit"] == 10, stats)

            service.call_tool("forget", {"id": first["id"]})
            after = service.call_tool("recall", {"query": "界面主题", "k": 5})
            check("forget removes from recall", all(item["id"] != first["id"] for item in after["items"]), after)

            try:
                service.call_tool("forget", {"id": first["id"]})
            except MemoryFailure as exc:
                check("second forget is NOT_FOUND", exc.code == "NOT_FOUND", exc.code)
            else:
                check("second forget is NOT_FOUND", False, "no error")

            result = {"ok": all(item["ok"] for item in checks), "checks": checks}
            sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
            return 0 if result["ok"] else 1
        finally:
            service.close()
