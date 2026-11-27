"""M4b `evolve`: the memory module's write path.

Five triggers, all deterministic and all switchable (M4b §四):

* `turn_end`   — `rules.py`, regex extraction from `user_text` only;
* `turn_review`— `review.py`, skeletons every `nudge_interval` turns → staging;
* `session_close` — `summarize.py`, rolling summary + tail → session/staging;
* `consolidate`— `consolidate.py`, cluster/promote/conflict/decay;
* `skill_evolve`— `skillgen.py` (not in this delivery).

`pipeline.EvolvePipeline` wires them for `worker.py`; `config.EvolveSettings`
is the only reader of the C-18 `[evolve]` table. Everything here talks to
`memory_module.store.MemoryBackend`, never to SQLite.
"""

from __future__ import annotations

from .config import DEFAULT_NAMESPACE, LESSON_NAMESPACE, EvolveSettings
from .consolidate import ConsolidateStats, consolidate
from .pipeline import EvolvePipeline
from .review import ReviewOutcome, Skeleton, TurnReviewer, build_skeleton
from .rules import RuleCandidate, TurnExtraction, extract_from_payload, extract_turn
from .scheduler import ConsolidateScheduler
from .summarize import SessionSummarizer, SummaryOutcome

__all__ = [
    "EvolveSettings",
    "EvolvePipeline",
    "ConsolidateScheduler",
    "ConsolidateStats",
    "consolidate",
    "ReviewOutcome",
    "Skeleton",
    "TurnReviewer",
    "build_skeleton",
    "RuleCandidate",
    "TurnExtraction",
    "extract_from_payload",
    "extract_turn",
    "SessionSummarizer",
    "SummaryOutcome",
    "DEFAULT_NAMESPACE",
    "LESSON_NAMESPACE",
]
