"""`python -m memory_module.evolve consolidate` — the manual consolidation entry.

M4b §四 lists three ways consolidate runs: daily 03:00, startup catch-up, and
this CLI. The CLI is the one an operator can run after changing quotas or to
drain a backlog without waiting for 03:00, so it prints the stats as one JSON
object and exits non-zero only on a real failure (not on "disabled").

It opens the same SQLite file as the running module, which is safe because the
module is declared `concurrency=exclusive`: there is no second writer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import EvolveSettings
from .consolidate import consolidate
from ..embed import LocalHashEmbedding, OpenAIEmbeddingClient
from ..settings import MemorySettings
from ..store import SqliteBackend

__all__ = ["main"]


def _embedder(settings: MemorySettings):
    if settings.embedding_base_url:
        return OpenAIEmbeddingClient(
            base_url=settings.embedding_base_url,
            model=settings.effective_model,
            dim=settings.embedding_dim,
            api_key_env=settings.api_key_env,
            timeout_s=settings.embedding_timeout_s,
        )
    return LocalHashEmbedding(dim=settings.embedding_dim, model=settings.effective_model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m memory_module.evolve")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("consolidate", help="run one consolidation pass")
    run.add_argument("--kind", default="manual", choices=["manual", "startup", "daily"],
                     help="recorded in consolidate_run.kind")
    run.add_argument("--json", action="store_true", help="print machine-readable stats")
    args = parser.parse_args(argv)

    memory = MemorySettings.from_env()
    evolve = EvolveSettings.from_env()
    if not evolve.enabled:
        # A disabled evolver must not touch the database: this is the same rule
        # as the event path, not a CLI exception.
        print("evolve disabled: set MEMORY_EVOLVE_ENABLED=1 to run consolidate", file=sys.stderr)
        return 0

    data_dir = Path(memory.data_dir)
    backend = SqliteBackend(memory)
    try:
        stats = consolidate(
            backend,
            evolve,
            embedder=_embedder(memory),
            kind=args.kind,
        )
    finally:
        backend.close()
    payload = {**stats.as_dict(), "run_id": stats.run_id}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(
            f"consolidate kind={args.kind} clusters={payload['clusters']} "
            f"promoted={payload['promoted']} archived={payload['archived']} "
            f"staging_cleared={payload['staging_cleared']}"
        )
    return 0 if payload.get("status") in ("ok", "disabled") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
