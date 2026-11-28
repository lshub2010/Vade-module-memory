"""Configuration for the memory module.

The authoritative defaults live in `docs/design.md` §7.9.1 / contract C-18.
This file is a *reader*, not a second source of truth: every default below is
copied verbatim from C-18 and the README maps each environment variable back to
its `core.toml` key. Core validates `core.toml` before the module starts; the
module must not invent additional keys, so `from_env` only looks at the names
listed here.

Secrets are never values here: `api_key_env` is the *name* of an environment
variable, resolved by `embed.py` at call time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

__all__ = ["MemorySettings"]


def _env_int(env: Mapping[str, str], name: str, default: int, minimum: int, maximum: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if not (minimum <= value <= maximum):
        raise ValueError(f"{name} must be within [{minimum}, {maximum}], got {value}")
    return value


def _env_float(env: Mapping[str, str], name: str, default: float, minimum: float, maximum: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not (minimum <= value <= maximum):
        raise ValueError(f"{name} must be within [{minimum}, {maximum}], got {value}")
    return value


@dataclass(frozen=True)
class MemorySettings:
    """Runtime policy for one memory instance.

    The field names are the C-18 key suffixes. `data_dir` is the one item that
    is a deployment concern rather than a `[memory]` key; it follows the same
    `VADE_DATA_DIR` convention as the other modules.
    """

    data_dir: Path = Path("data")
    global_capacity: int = 5000
    session_capacity: int = 200
    session_inject_char_limit: int = 4200
    #: Write-side budget. C-18 splits the two: `memory.session_inject_char_limit`
    #: caps what core injects, `evolve.session_char_limit` caps what the evolver
    #: stores. 0 means "same as the injection limit", which is the M4a default
    #: and keeps a single knob for explicit `remember(scope='session')`.
    session_write_char_limit: int = 0
    recall_k_max: int = 20
    recall_token_limit: int = 800
    ttl_days: int = 7
    embedding_model: str = ""
    embedding_dim: int = 1536
    embedding_base_url: str = ""
    api_key_env: str = "MEMORY_API_KEY"
    embedding_timeout_s: float = 10.0
    #: Evolve's own model connection (M4b §九 `model = "cheap" 走 M4a 的 LLM
    #: 配置`). These follow `embedding_base_url` exactly: module-private
    #: environment variables, not C-18 keys. `[evolve]` in
    #: `schema/config.schema.json` has 14 keys and no upstream address, so a
    #: new field there would be rejected by core as unknown.
    #:
    #: Empty `evolve_llm_base_url` means "no model": the four model triggers
    #: degrade to pure rules exactly as they do today. The key is never stored,
    #: only the *name* of the variable holding it.
    evolve_llm_base_url: str = ""
    evolve_llm_api_key_env: str = "MEMORY_API_KEY"
    evolve_llm_timeout_s: float = 30.0
    #: Provider model id sent as `model` in the chat request. Empty means
    #: `EvolveSettings.model` (the `"cheap"` label) is used, which is only
    #: correct against a router that understands it.
    evolve_llm_model: str = ""

    def __post_init__(self) -> None:
        if not (1 <= self.global_capacity <= 100_000):
            raise ValueError(f"global_capacity must be within [1, 100000], got {self.global_capacity}")
        if not (1 <= self.session_capacity <= 10_000):
            raise ValueError(f"session_capacity must be within [1, 10000], got {self.session_capacity}")
        if not (100 <= self.session_inject_char_limit <= 100_000):
            raise ValueError(
                f"session_inject_char_limit must be within [100, 100000], got {self.session_inject_char_limit}"
            )
        if self.session_write_char_limit != 0 and not (100 <= self.session_write_char_limit <= 100_000):
            raise ValueError(
                f"session_write_char_limit must be 0 (follow inject limit) or within [100, 100000], "
                f"got {self.session_write_char_limit}"
            )
        if not (1 <= self.recall_k_max <= 100):
            raise ValueError(f"recall_k_max must be within [1, 100], got {self.recall_k_max}")
        if not (32 <= self.recall_token_limit <= 100_000):
            raise ValueError(f"recall_token_limit must be within [32, 100000], got {self.recall_token_limit}")
        if not (1 <= self.ttl_days <= 365):
            raise ValueError(f"ttl_days must be within [1, 365], got {self.ttl_days}")
        if not (1 <= self.embedding_dim <= 8192):
            raise ValueError(f"embedding_dim must be within [1, 8192], got {self.embedding_dim}")
        if not self.api_key_env:
            raise ValueError("api_key_env must be the NAME of an environment variable, not empty")
        if not (0.1 <= self.embedding_timeout_s <= 120):
            raise ValueError(f"embedding_timeout_s must be within [0.1, 120], got {self.embedding_timeout_s}")
        if self.evolve_llm_base_url and not self.evolve_llm_base_url.startswith(("http://", "https://")):
            raise ValueError(
                "MEMORY_EVOLVE_LLM_BASE_URL must start with http:// or https://, "
                f"got {self.evolve_llm_base_url!r}"
            )
        if not self.evolve_llm_api_key_env:
            raise ValueError("evolve_llm_api_key_env must be the NAME of an environment variable, not empty")
        if not (0.1 <= self.evolve_llm_timeout_s <= 120):
            raise ValueError(f"evolve_llm_timeout_s must be within [0.1, 120], got {self.evolve_llm_timeout_s}")

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "memory.db"

    @property
    def effective_model(self) -> str:
        """The name persisted in `meta.embedding_model`.

        An empty `memory.embedding_model` means the built-in local model, per
        C-18. If an OpenAI-compatible base URL is configured but no model name
        is given, the provider's usual small embedding model is the sane
        default; persisting the effective name is what lets a later model
        change fail loudly instead of silently re-embedding the database.
        """
        model = self.embedding_model.strip()
        if model:
            return model
        return "text-embedding-3-small" if self.embedding_base_url else "local-hash-v1"

    @property
    def effective_session_write_char_limit(self) -> int:
        return self.session_write_char_limit or self.session_inject_char_limit

    def effective_evolve_model(self, label: str = "cheap") -> str:
        """The `model` field sent to the chat provider.

        `evolve.model` in C-18 is a cost *label* ("cheap"), not a provider id —
        `skillgen.py` only copies it into `derived_from.model` for provenance.
        A real request needs a real id, so `MEMORY_EVOLVE_LLM_MODEL` wins and
        the label is the documented fallback for routers that map it.
        """
        model = self.evolve_llm_model.strip()
        return model or label

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "MemorySettings":
        env = dict(os.environ if env is None else env)
        data_dir = env.get("MEMORY_DATA_DIR") or env.get("VADE_DATA_DIR") or "data"
        model = (env.get("MEMORY_EMBEDDING_MODEL") or "").strip()
        api_key_env = (env.get("MEMORY_API_KEY_ENV") or "MEMORY_API_KEY").strip()
        if not api_key_env:
            raise ValueError("MEMORY_API_KEY_ENV must be the NAME of an environment variable")
        return cls(
            data_dir=Path(data_dir),
            global_capacity=_env_int(env, "MEMORY_GLOBAL_CAPACITY", 5000, 1, 100_000),
            session_capacity=_env_int(env, "MEMORY_SESSION_CAPACITY", 200, 1, 10_000),
            session_inject_char_limit=_env_int(
                env, "MEMORY_SESSION_INJECT_CHAR_LIMIT", 4200, 100, 100_000
            ),
            session_write_char_limit=_env_int(
                env, "EVOLVE_SESSION_CHAR_LIMIT", 0, 0, 100_000
            ),
            recall_k_max=_env_int(env, "MEMORY_RECALL_K_MAX", 20, 1, 100),
            recall_token_limit=_env_int(env, "MEMORY_RECALL_TOKEN_LIMIT", 800, 32, 100_000),
            ttl_days=_env_int(env, "MEMORY_TTL_DAYS", 7, 1, 365),
            embedding_model=model,
            embedding_dim=_env_int(env, "MEMORY_EMBEDDING_DIM", 1536, 1, 8192),
            embedding_base_url=(env.get("MEMORY_EMBEDDING_BASE_URL") or "").strip(),
            api_key_env=api_key_env,
            embedding_timeout_s=_env_float(env, "MEMORY_EMBEDDING_TIMEOUT_S", 10.0, 0.1, 120.0),
            evolve_llm_base_url=(env.get("MEMORY_EVOLVE_LLM_BASE_URL") or "").strip(),
            evolve_llm_api_key_env=(
                env.get("MEMORY_EVOLVE_LLM_API_KEY_ENV") or "MEMORY_API_KEY"
            ).strip(),
            evolve_llm_timeout_s=_env_float(env, "MEMORY_EVOLVE_LLM_TIMEOUT_S", 30.0, 0.1, 120.0),
            evolve_llm_model=(env.get("MEMORY_EVOLVE_LLM_MODEL") or "").strip(),
        )

    def with_overrides(self, **changes: object) -> "MemorySettings":
        """Small helper for tests and `--self-test`; keeps validation in one place."""
        return replace(self, **changes)  # type: ignore[arg-type]
