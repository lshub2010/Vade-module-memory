"""CLI + SDK wiring for M4b: the manual `consolidate` entry and event routing."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from memory_module.embed import FakeEmbedding
from memory_module.evolve.config import EvolveSettings
from memory_module.settings import MemorySettings
from memory_module.store import new_ulid

pytest.importorskip("vade_sdk")

ROOT = Path(__file__).resolve().parents[1]


def _run(args, env_extra, tmp_path):
    env = {
        **os.environ,
        "PYTHONPATH": f"{ROOT / 'src'}",
        "MEMORY_DATA_DIR": str(tmp_path),
        "MEMORY_EMBEDDING_DIM": "8",
        **env_extra,
    }
    return subprocess.run(
        [sys.executable, "-m", "memory_module.evolve", *args],
        capture_output=True, text=True, env=env, timeout=60,
    )


def test_consolidate_cli_refuses_when_disabled_and_leaves_no_db(tmp_path):
    result = _run(["consolidate", "--json"], {"MEMORY_EVOLVE_ENABLED": "0"}, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "evolve disabled" in result.stderr
    assert not (tmp_path / "memory.db").exists()


def test_consolidate_cli_runs_and_reports_json(tmp_path):
    result = _run(
        ["consolidate", "--json", "--kind", "manual"],
        {"MEMORY_EVOLVE_ENABLED": "1", "MEMORY_EVOLVE_DAILY_TOKEN_BUDGET": "0"},
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["run_id"] >= 1


def test_build_app_routes_events_through_both_layers(tmp_path):
    from memory_module.server import MemoryService, build_app

    service = MemoryService(
        MemorySettings(data_dir=tmp_path, embedding_dim=8),
        embedder=FakeEmbedding(dim=8),
        evolve=EvolveSettings(enabled=True, daily_token_budget=0),
    )
    app = build_app(service)
    session_id = new_ulid()

    async def scenario():
        stored = await app.call_tool(
            "remember",
            {"text": "会话记忆", "scope": "session"},
            context={"session_id": session_id},
        )
        assert stored["ok"] is True
        await app.dispatch_event(
            "session.turn",
            {"session_id": session_id, "turn_id": new_ulid(), "state": "completed",
             "user_text": "以后都用中文回答", "tool_calls": []},
        )
        closed = await app.dispatch_event(
            "session.closed",
            {"session_id": session_id, "closed_at": 1_700_000_000_000, "turns": []},
        )
        return closed

    closed = asyncio.run(scenario())
    # M4a's TTL path answered...
    assert closed["handled"] is True
    # ...and M4b ran in the same delivery instead of replacing it.
    assert "evolve" in closed
    service.close()
