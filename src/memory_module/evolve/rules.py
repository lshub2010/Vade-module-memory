"""`turn_end` rule extraction: deterministic, no model, user text only.

Five signals, each anchored to the sentence the user actually typed (M4b §五).
The extraction is intentionally boring: regexes and keywords, no inference.
Anything a regex cannot see belongs to `turn_review`, which reads skeletons in
a side channel.

Two prohibitions are structural, not conventions:

* the only input is `user_text`. `extract_from_payload` never reads
  `assistant_text` — otherwise the model's own "结论是…" would become a memory,
  then a staging candidate, then (after two sessions) a "fact";
* the output is a candidate for the **session** layer. Nothing here writes,
  and `pipeline.py` is the only caller that stores the result.

Single turn: at most two candidates, by importance desc. Extra hits are
discarded, not queued (M4b §五).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..store import normalize_text

__all__ = [
    "RuleCandidate",
    "TurnExtraction",
    "SIGNAL_IMPORTANCE",
    "MAX_CANDIDATES_PER_TURN",
    "extract_turn",
    "extract_from_payload",
]

MAX_CANDIDATES_PER_TURN = 2
MAX_CANDIDATE_CHARS = 300
MAX_SENTENCES = 200

SIGNAL_IMPORTANCE = {
    "memory_instruction": 5,
    "correction": 5,
    "preference": 4,
    "conclusion": 3,
    "environment": 2,
}


@dataclass(frozen=True)
class _Signal:
    name: str
    importance: int
    pattern: re.Pattern[str]


# Order is the tie-breaker: a sentence that is both an instruction and a
# preference is an instruction (importance 5), and a sentence that is both an
# instruction and a correction stays an instruction while still flagging the
# previous turn as negative feedback.
_SIGNALS: tuple[_Signal, ...] = (
    _Signal(
        "memory_instruction",
        5,
        re.compile(
            r"(记住|记一下|记下来|请记得|以后都|以后只|今后都|不要再|以后别|别再"
            r"|from now on|remember (?:that|to)|always use|never use)",
            re.IGNORECASE,
        ),
    ),
    _Signal(
        "correction",
        5,
        re.compile(
            r"(不对|不是这样|不是这个意思|不是这个|错了|说反了|纠正"
            r"|改成|改为|应该是|应当改为"
            r"|not right|that'?s wrong|instead|change (?:it )?to)",
            re.IGNORECASE,
        ),
    ),
    _Signal(
        "preference",
        4,
        re.compile(
            r"(我喜欢|我习惯|我偏好|我更喜欢|我倾向|我们统一用|我们统一|统一用"
            r"|倾向于|偏好|prefer|i like|i usually|we standardi[sz]e on)",
            re.IGNORECASE,
        ),
    ),
    _Signal(
        "conclusion",
        3,
        re.compile(
            r"(结论是|最终决定|最终结论|最终方案|决定了|定下来|就这么定|拍板"
            r"|conclusion is|we decided|final decision)",
            re.IGNORECASE,
        ),
    ),
    _Signal(
        "environment",
        2,
        re.compile(
            r"(?<![\w])/(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+"  # /srv/app
            r"|(?:端口|port)\s*[:=：]?\s*\d{2,5}"  # 端口 8080
            r"|:\d{2,5}\b"  # :8080
            r"|\bv?\d+\.\d+\.\d+\b"  # v1.2.3
            r"|\b[A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9_.\-]+\b",  # owner/repo
            re.IGNORECASE,
        ),
    ),
)

_CORRECTION = _SIGNALS[1].pattern
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


@dataclass(frozen=True)
class RuleCandidate:
    """One session-layer candidate, still unattached to storage."""

    text: str
    signal: str
    importance: int
    negative_feedback: bool = False


@dataclass(frozen=True)
class TurnExtraction:
    """Everything one `turn_end` produced: candidates plus audit counters.

    `matched` counts signals seen before the max-2 cut, so a test (and the
    log line) can tell "nothing matched" from "four matched, two dropped".
    """

    candidates: tuple[RuleCandidate, ...] = ()
    matched: int = 0
    negative_feedback: bool = False


def _clip(text: str, limit: int = MAX_CANDIDATE_CHARS) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)].rstrip() + "…"


def _digest(text: str) -> str:
    return hashlib.sha256(normalize_text(text).lower().encode("utf-8")).hexdigest()


def _sentences(text: str) -> list[str]:
    out = [part.strip() for part in _SENTENCE_SPLIT.split(text)]
    return [part for part in out if part][:MAX_SENTENCES]


def _match_signal(sentence: str) -> _Signal | None:
    for signal in _SIGNALS:
        if signal.pattern.search(sentence):
            return signal
    return None


def extract_turn(
    user_text: str, *, max_items: int = MAX_CANDIDATES_PER_TURN
) -> TurnExtraction:
    """Extract 0..`max_items` candidates from one user message."""
    if not isinstance(user_text, str) or not user_text.strip():
        return TurnExtraction()
    limit = max(0, min(int(max_items), MAX_CANDIDATES_PER_TURN))

    seen: set[str] = set()
    scored: list[tuple[int, int, RuleCandidate]] = []
    matched = 0
    negative = False
    for sentence in _sentences(user_text):
        signal = _match_signal(sentence)
        if signal is None:
            continue
        matched += 1
        corrected = bool(_CORRECTION.search(sentence))
        negative = negative or corrected
        text = _clip(sentence)
        digest = _digest(text)
        if digest in seen:
            continue
        seen.add(digest)
        scored.append(
            (
                signal.importance,
                len(scored),
                RuleCandidate(
                    text=text,
                    signal=signal.name,
                    importance=signal.importance,
                    negative_feedback=corrected,
                ),
            )
        )

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen: Sequence[RuleCandidate] = tuple(item[2] for item in scored[:limit])
    return TurnExtraction(candidates=tuple(chosen), matched=matched, negative_feedback=negative)


def extract_from_payload(payload: Mapping[str, Any] | None) -> TurnExtraction:
    """`turn_end` entry point. Reads `user_text` and nothing else.

    `assistant_text` is deliberately not touched (M4b §五/§十一). The rule is
    enforced here, once, instead of trusting every future caller to remember.
    """
    if not isinstance(payload, Mapping):
        return TurnExtraction()
    user_text = payload.get("user_text")
    return extract_turn(user_text if isinstance(user_text, str) else "")
