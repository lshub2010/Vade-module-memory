"""C-12 / C-16 files stay in sync with the implementation."""

from __future__ import annotations

import json
from pathlib import Path

from memory_module.rpc import MEMORY_RPC_METHODS

MODULE_ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((MODULE_ROOT / "schema" / name).read_text(encoding="utf-8"))


def test_c12_has_the_three_records_and_enums():
    schema = load("memory.schema.json")
    defs = schema["$defs"]
    assert set(defs["MemoryLayer"]["enum"]) == {"session", "global"}
    assert set(defs["MemoryKind"]["enum"]) == {"fact", "preference", "lesson"}
    assert set(defs["MemorySource"]["enum"]) == {"user", "evolve", "import"}
    assert set(defs["MemoryStatus"]["enum"]) == {"active", "archived", "superseded"}
    for name in ("MemoryRecord", "StagingRecord", "ArchiveRecord"):
        assert name in defs
    record = defs["MemoryRecord"]["properties"]
    assert "embedding" in record and record["embedding"]["x-vade-internal"] is True
    assert "promoted_by" in record
    assert "session_id" in record
    assert record["session_id"]["description"].startswith("Session layer only")


def test_c12_record_fields_match_the_sqlite_columns():
    schema = load("memory.schema.json")
    schema_fields = set(schema["$defs"]["MemoryRecord"]["properties"]) - {"embedding"}
    import sqlite3
    import tempfile

    from memory_module.store import SqliteBackend
    from memory_module.settings import MemorySettings

    with tempfile.TemporaryDirectory() as tmp:
        backend = SqliteBackend(MemorySettings(data_dir=Path(tmp), embedding_dim=4))
        columns = {
            row[1] for row in backend._conn.execute("PRAGMA table_info(memories)").fetchall()
        } - {"embedding"}
        backend.close()
    assert schema_fields == columns


def test_c16_declares_all_seven_private_methods_and_no_tool_table():
    schema = load("memory-admin.json")
    assert schema["x-vade"]["private"] is True
    assert schema["x-vade"]["tool_table"] is False
    assert tuple(schema["x-vade"]["methods"]) == MEMORY_RPC_METHODS
    assert set(schema["methods"]) == set(MEMORY_RPC_METHODS)
    item = schema["$defs"]["MemoryItem"]["properties"]
    assert "user_text" not in item and "assistant_text" not in item and "transcript" not in item
    query = schema["$defs"]["MemoryQuery"]["properties"]
    assert set(query) == {"layer", "namespace", "status", "from", "to", "q", "limit", "cursor"}
    assert query["limit"]["default"] == 50 and query["limit"]["maximum"] == 200


def test_manifest_is_valid_json_and_has_no_secret_literals():
    manifest = json.loads((MODULE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    blob = json.dumps(manifest)
    assert "sk-" not in blob
    assert "api_key" not in blob
    assert manifest["module_id"] == "com.example.memory"
    assert manifest["version"] == "1.0.0"
