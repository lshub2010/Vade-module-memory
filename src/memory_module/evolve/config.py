"""Evolve settings: the `[evolve]` table of C-18, read-only.

The authoritative defaults live in `docs/design.md` §7.9.1 (contract C-18) and
M4b §九. This module is a reader: `from_mapping` accepts exactly the keys C-18
defines and rejects anything else. Core already refuses to start on an unknown
`[evolve]` key, so silently ignoring one here would make the module disagree
with the config page.

`enabled` defaults to false. Nothing in `evolve/` writes, buffers or calls a
model until it is true (M4b §九: "enabled=false → 不订阅、不消费、不产生任何写入").
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Mapping

__all__ = [
    "EvolveSettings",
    "DEFAULT_CONTEXT_TOKENS",
    "DEFAULT_NAMESPACE",
    "LESSON_NAMESPACE",
]

#: Context window used when the caller does not pass one. C-18 budgets the
#: rolling summary as `ctx x summary_token_ratio`; the evolver never sees the
#: real context length (core owns the model), so this is the documented v1
#: assumption rather than a measured value.
DEFAULT_CONTEXT_TOKENS = 32_000

#: Where facts/preferences live when the model does not name a namespace.
DEFAULT_NAMESPACE = "user:default"

#: Lessons accumulate in their own namespace so `recall` can scope to them.
LESSON_NAMESPACE = "agent:lessons"

_KNOWN_KEYS = frozenset(
    {
        "enabled",
        "allow_transcript",
        "allowlist",
        "nudge_interval",
        "flush_min_turns",
        "session_char_limit",
        "summary_token_ratio",
        "lesson_confidence_floor",
        "failure_learning",
        "daily_token_budget",
        "daily_budget_skills",
        "session_budget_skills",
        "redact",
        "model",
    }
)
_BOOL_KEYS = frozenset({"enabled", "allow_transcript", "failure_learning"})
_INT_KEYS = frozenset(
    {
        "nudge_interval",
        "flush_min_turns",
        "session_char_limit",
        "daily_token_budget",
        "daily_budget_skills",
        "session_budget_skills",
    }
)
_FLOAT_KEYS = frozenset({"summary_token_ratio", "lesson_confidence_floor"})
_TUPLE_KEYS = frozenset({"redact", "allowlist"})
_STR_KEYS = frozenset({"model"})

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _as_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE:
            return True
        if token in _FALSE:
            return False
    raise ValueError(f"[evolve] {key} must be a boolean, got {value!r}")


def _as_int(key: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"[evolve] {key} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ValueError(f"[evolve] {key} must be an integer, got {value!r}")


def _as_float(key: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"[evolve] {key} must be a number, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise ValueError(f"[evolve] {key} must be a number, got {value!r}")


def _as_str_tuple(key: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError(f"[evolve] {key} must be a list of strings, not a bare string")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"[evolve] {key} must be a list of strings, got {type(value).__name__}")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"[evolve] {key} entries must be non-empty strings, got {item!r}")
        clean = item.strip()
        if clean not in out:
            out.append(clean)
    return tuple(out)


def _check_range(name: str, value: float, minimum: float, maximum: float) -> None:
    if not (minimum <= value <= maximum):
        raise ValueError(f"evolve.{name} must be within [{minimum}, {maximum}], got {value}")


@dataclass(frozen=True)
class EvolveSettings:
    """One immutable snapshot of `[evolve]`.

    Hot-reloaded keys (everything except `allowlist` and `redact`) are applied
    by building a new snapshot; nothing in this package mutates settings in
    place, so a running review cycle always sees one consistent policy.
    """

    enabled: bool = False
    allow_transcript: bool = False
    allowlist: tuple[str, ...] = ()
    nudge_interval: int = 10
    flush_min_turns: int = 6
    session_char_limit: int = 4200
    summary_token_ratio: float = 0.05
    lesson_confidence_floor: float = 0.6
    failure_learning: bool = True
    daily_token_budget: int = 200_000
    daily_budget_skills: int = 5
    session_budget_skills: int = 1
    redact: tuple[str, ...] = ("tool_args", "file_content")
    model: str = "cheap"

    def __post_init__(self) -> None:
        for name in ("enabled", "allow_transcript", "failure_learning"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"evolve.{name} must be a boolean")
        for name in ("nudge_interval", "flush_min_turns", "session_char_limit",
                     "daily_token_budget", "daily_budget_skills", "session_budget_skills"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"evolve.{name} must be an integer, got {type(value).__name__}")
        for name in ("summary_token_ratio", "lesson_confidence_floor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"evolve.{name} must be a number, got {type(value).__name__}")
        if not isinstance(self.redact, tuple) or not all(
            isinstance(item, str) and item for item in self.redact
        ):
            raise ValueError("evolve.redact must be a tuple of non-empty strings")
        if not isinstance(self.allowlist, tuple) or not all(
            isinstance(item, str) and item for item in self.allowlist
        ):
            raise ValueError("evolve.allowlist must be a tuple of non-empty strings")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("evolve.model must be a non-empty string")

        # Ranges are C-18 / §7.9.1. `nudge_interval` has a hard floor of 5:
        # going lower multiplies the model bill linearly and the config page
        # demands a second confirmation at exactly 5.
        _check_range("nudge_interval", self.nudge_interval, 5, 50)
        _check_range("flush_min_turns", self.flush_min_turns, 2, 20)
        _check_range("session_char_limit", self.session_char_limit, 600, 20_000)
        _check_range("summary_token_ratio", self.summary_token_ratio, 0.01, 0.5)
        _check_range("lesson_confidence_floor", self.lesson_confidence_floor, 0.0, 1.0)
        _check_range("daily_token_budget", self.daily_token_budget, 0, 1_000_000)
        _check_range("daily_budget_skills", self.daily_budget_skills, 0, 20)
        _check_range("session_budget_skills", self.session_budget_skills, 0, 3)

    # ------------------------------------------------------------------ parse
    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None = None) -> "EvolveSettings":
        """Build a snapshot from the raw `[evolve]` table.

        Unknown keys are an error, not a warning: C-18 rule 1 says a typo must
        fail loudly instead of silently running on a default.
        """
        data = dict(mapping or {})
        unknown = sorted(set(data) - _KNOWN_KEYS)
        if unknown:
            raise ValueError(
                f"unknown [evolve] key(s) {unknown}; extend C-18 before this module may read them"
            )
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in _BOOL_KEYS:
                kwargs[key] = _as_bool(key, value)
            elif key in _INT_KEYS:
                kwargs[key] = _as_int(key, value)
            elif key in _FLOAT_KEYS:
                kwargs[key] = _as_float(key, value)
            elif key in _TUPLE_KEYS:
                kwargs[key] = _as_str_tuple(key, value)
            elif key in _STR_KEYS:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"[evolve] {key} must be a non-empty string, got {value!r}")
                kwargs[key] = value.strip()
            else:  # pragma: no cover - _KNOWN_KEYS and the buckets must agree
                raise ValueError(f"unhandled [evolve] key {key!r}")
        return cls(**kwargs)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "EvolveSettings":
        """Read the `[evolve]` table from `MEMORY_EVOLVE_<KEY>` environment vars.

        The module receives no TOML path (core owns `core.toml`), so deployment
        maps the same C-18 keys with a prefix: `MEMORY_EVOLVE_ENABLED=1`,
        `MEMORY_EVOLVE_NUDGE_INTERVAL=10`, `MEMORY_EVOLVE_REDACT=tool_args,file_content`.
        Unknown `MEMORY_EVOLVE_*` names are ignored here and left to core's
        schema check; values that *are* known still go through `from_mapping`,
        so a typo in a value fails loudly.
        """
        source = dict(os.environ if env is None else env)
        raw: dict[str, Any] = {}
        for key in _KNOWN_KEYS:
            name = f"MEMORY_EVOLVE_{key.upper()}"
            if name not in source:
                continue
            value = source[name]
            if key in _TUPLE_KEYS:
                raw[key] = [part.strip() for part in value.split(",") if part.strip()]
            else:
                raw[key] = value
        return cls.from_mapping(raw)

    # ---------------------------------------------------------------- budgets
    def summary_token_budget(self, context_tokens: int = DEFAULT_CONTEXT_TOKENS) -> int:
        """`ctx x summary_token_ratio`, floored at one token."""
        if context_tokens <= 0:
            return 1
        return max(1, int(context_tokens * self.summary_token_ratio))

    def summary_char_budget(self, context_tokens: int = DEFAULT_CONTEXT_TOKENS) -> int:
        """The same budget in characters (~4 chars/token, the tools.py estimate)."""
        return self.summary_token_budget(context_tokens) * 4

    def with_overrides(self, **changes: Any) -> "EvolveSettings":
        """Validate a hot-reload patch through the same `__post_init__`."""
        return replace(self, **changes)  # type: ignore[arg-type]
