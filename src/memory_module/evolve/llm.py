"""Injectable LLM gateway: strict JSON, exactly one retry, 30 s cap.

`review.py` and `summarize.py` both call a cheap model and both must obey the
same rules (M4b §六/§六·五):

* output is a bare JSON object — a markdown fence is a parse failure;
* a parse failure is retried **once** with the error appended to the prompt;
* timeout / provider error / second parse failure degrades the trigger, logs
  `evolve_degraded` and never raises into the session;
* the four model triggers share `evolve.daily_token_budget`; exhaustion
  degrades to pure rules, it does not queue work for tomorrow.

The callable is injected (`async def call(prompt: str) -> str`), so tests and
the offline self-test use no network at all.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from ..tools import estimate_tokens
from .config import EvolveSettings

__all__ = [
    "LlmCallable",
    "JsonParseError",
    "parse_strict_json",
    "DailyBudget",
    "LlmResult",
    "EvolveLLM",
]

LlmCallable = Callable[[str], Awaitable[str]]

DEFAULT_TIMEOUT_S = 30.0

_log = logging.getLogger("memory_module.evolve")


class JsonParseError(ValueError):
    """The model returned something that is not one JSON object."""


def parse_strict_json(raw: str) -> dict[str, Any]:
    """Parse exactly one JSON object; reject fences, prose and arrays.

    Strictness is the point: `session_close` writes the `summary` field into
    the session layer, so "almost JSON" that a lenient parser rescues is not
    something we want to feed the store.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise JsonParseError("empty response")
    text = raw.strip()
    if text.startswith("```") or text.endswith("```"):
        raise JsonParseError("markdown code fences are not accepted")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JsonParseError(f"{exc.msg} at char {exc.pos}") from None
    if not isinstance(parsed, dict):
        raise JsonParseError(f"top-level JSON must be an object, got {type(parsed).__name__}")
    return parsed


@dataclass
class DailyBudget:
    """Token bucket keyed by UTC day; `limit == 0` means unlimited.

    `clock` returns unix seconds. A rollover starts a fresh day at 00:00 UTC;
    the spec says "次日 00:00 重置" and does not queue the deficit.
    """

    limit: int
    clock: Callable[[], int]
    spent: int = 0
    _day: str = field(default="", repr=False)

    @property
    def unlimited(self) -> bool:
        return self.limit <= 0

    def _roll(self) -> None:
        today = time.strftime("%Y-%m-%d", time.gmtime(int(self.clock())))
        if today != self._day:
            self._day = today
            self.spent = 0

    def remaining(self) -> int | None:
        if self.unlimited:
            return None
        self._roll()
        return max(0, self.limit - self.spent)

    def try_spend(self, tokens: int) -> bool:
        amount = max(0, int(tokens))
        if self.unlimited:
            return True
        self._roll()
        if self.spent + amount > self.limit:
            return False
        self.spent += amount
        return True


@dataclass
class LlmResult:
    """One `json_call` outcome. `payload is None` means "degrade to rules"."""

    payload: dict[str, Any] | None
    raw: str = ""
    attempts: int = 0
    error: str | None = None
    retried: bool = False

    @property
    def ok(self) -> bool:
        return self.payload is not None


class EvolveLLM:
    """The only path from the evolver to a model."""

    def __init__(
        self,
        call: LlmCallable | None,
        settings: EvolveSettings,
        *,
        clock: Callable[[], int] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        budget: DailyBudget | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        self._call = call
        self.settings = settings
        self.timeout_s = float(timeout_s)
        now = clock or (lambda: int(time.time()))
        self.budget = budget or DailyBudget(settings.daily_token_budget, now)
        self._log = logger or _log

    async def json_call(
        self,
        prompt: str,
        *,
        validate: Callable[[Mapping[str, Any]], None] | None = None,
        label: str = "llm",
    ) -> LlmResult:
        """Call the model and return parsed JSON, retrying a parse failure once.

        `validate` is the schema check; a `ValueError` it raises is treated
        like a parse error (output validation is part of "did we get usable
        JSON"), so it also gets the single retry.
        """
        if self._call is None:
            self._log.warning("evolve_degraded trigger=%s reason=no_llm", label)
            return LlmResult(None, error="no_llm")
        if not self.budget.try_spend(estimate_tokens(prompt)):
            self._log.warning("evolve_budget_exhausted trigger=%s", label)
            return LlmResult(None, error="budget")

        last_error = ""
        raw = ""
        for attempt in (1, 2):
            current = prompt
            if attempt == 2:
                current = (
                    f"{prompt}\n\n[parse_error] {last_error}\n"
                    "Return exactly one JSON object. No markdown fence, no prose."
                )
            try:
                raw = await asyncio.wait_for(self._call(current), timeout=self.timeout_s)
            except (TimeoutError, asyncio.TimeoutError):
                self._log.warning(
                    "evolve_degraded trigger=%s reason=timeout after=%ss", label, self.timeout_s
                )
                return LlmResult(None, attempts=attempt, error="timeout")
            except Exception as exc:  # provider error of any shape: degrade, never raise
                self._log.warning(
                    "evolve_degraded trigger=%s reason=%s", label, type(exc).__name__
                )
                return LlmResult(None, attempts=attempt, error="exception")

            text = raw if isinstance(raw, str) else ""
            try:
                payload = parse_strict_json(text)
                if validate is not None:
                    validate(payload)
            except ValueError as exc:
                last_error = str(exc)[:200]
                if attempt == 1:
                    continue
                self._log.warning("evolve_degraded trigger=%s reason=parse", label)
                return LlmResult(None, raw=text, attempts=2, error="parse", retried=True)

            self.budget.try_spend(estimate_tokens(text))
            return LlmResult(payload, raw=text, attempts=attempt, retried=attempt == 2)

        return LlmResult(None, raw=raw if isinstance(raw, str) else "", attempts=2,
                         error="parse", retried=True)
