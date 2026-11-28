"""OpenAI-compatible chat client: the one piece `evolve/` was missing.

`llm.py` owns every *policy* decision — strict JSON, exactly one retry, the
30 s cap, `evolve_degraded` instead of a raise. This module owns exactly one
thing: turn a prompt into a string by POSTing `{base_url}/chat/completions`.
Anything that is not a 2xx carrying `choices[0].message.content` is raised, on
purpose, because `EvolveLLM` is the layer that decides what a failure means.
Duplicating retry/fallback here would give two policies that can disagree.

**Where the connection settings come from.** They are module-private
environment variables (`MEMORY_EVOLVE_LLM_*`), not `[evolve]` keys. That
follows `MemorySettings.embedding_base_url` exactly: the embedder has always
read `MEMORY_EMBEDDING_BASE_URL` / `MEMORY_API_KEY_ENV` from the environment,
and C-18's `[memory]`/`[evolve]` tables have no key for an upstream address
(`schema/config.schema.json`: 8 and 14 keys respectively). C-18 is the table
core hands the module; where the module's own outbound HTTP points is
deployment detail. Adding a key to `EvolveSettings` would make core reject the
config as unknown — see `docs/agents/46-M4b-evolve.md` §九.

The API key itself is never read from config, never stored and never logged:
only the *name* of an environment variable is configured, and the value is
re-read on every call so rotating the secret needs no restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Mapping

__all__ = ["LlmRequestError", "OpenAICompatLLM"]

#: Mirrors `evolve.llm.DEFAULT_TIMEOUT_S`. The outer `asyncio.wait_for` in
#: `EvolveLLM` is the authoritative cap; this one only bounds one socket.
DEFAULT_TIMEOUT_S = 30.0

_MAX_TIMEOUT_S = 120.0

_log = logging.getLogger("memory_module.evolve")


class LlmRequestError(RuntimeError):
    """The upstream did not answer with usable chat content.

    Deliberately *not* a `MemoryFailure`: this never reaches a tool caller, it
    only reaches `EvolveLLM`, which logs `evolve_degraded` and returns `None`.
    A bare `TimeoutError` is not wrapped — `EvolveLLM` has a dedicated branch
    for it and reporting `reason=timeout` beats `reason=LlmRequestError`.
    """


class OpenAICompatLLM:
    """`async def __call__(prompt: str) -> str` against `/chat/completions`.

    Non-streaming on purpose: `EvolveLLM` wants one complete JSON object, and
    a stream would only add fragment reassembly to a path whose whole job is
    "give me the text or fail".
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "MEMORY_API_KEY",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        environ: Mapping[str, str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError("evolve llm base_url must start with http:// or https://")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("evolve llm model must be a non-empty provider model id")
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise ValueError("evolve llm api_key_env must be the NAME of an environment variable")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError(f"evolve llm timeout_s must be a number, got {timeout_s!r}")
        if not (0.1 <= float(timeout_s) <= _MAX_TIMEOUT_S):
            raise ValueError(f"evolve llm timeout_s must be within [0.1, {_MAX_TIMEOUT_S}]")
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.api_key_env = api_key_env.strip()
        self.timeout_s = float(timeout_s)
        self._environ = environ
        self._log = logger or _log

    # ------------------------------------------------------------------ auth
    def _key(self) -> str:
        env = os.environ if self._environ is None else self._environ
        key = (env.get(self.api_key_env) or "").strip()
        if not key:
            raise LlmRequestError(
                f"environment variable {self.api_key_env} is not set; "
                "the memory module never accepts a literal API key"
            )
        return key

    @property
    def endpoint(self) -> str:
        """`base_url` already points at `/v1`, so this is the full chat URL."""
        return f"{self.base_url}/chat/completions"

    # --------------------------------------------------------------- request
    def _post(self, body: bytes) -> bytes:
        """One blocking POST. Runs in a worker thread via `asyncio.to_thread`."""
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key()}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 - base_url is operator config
                return response.read()
        except urllib.error.HTTPError as exc:
            raise LlmRequestError(f"chat provider returned HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            # `urllib` folds a socket timeout into URLError; unwrap it so
            # `EvolveLLM` reports `reason=timeout` rather than a generic error.
            if isinstance(reason, TimeoutError):
                raise TimeoutError("chat endpoint timed out") from None
            raise LlmRequestError(f"chat endpoint is unreachable: {reason}") from None
        except TimeoutError:
            # A bare `socket.timeout` is `TimeoutError` on 3.10+; pass it through.
            raise
        except OSError as exc:
            raise LlmRequestError(
                f"chat endpoint dropped the connection: {type(exc).__name__}"
            ) from None

    # ------------------------------------------------------------- callable
    async def __call__(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise LlmRequestError("prompt must be a non-empty string")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            # The evolver wants one deterministic JSON object; sampling noise
            # only makes `parse_strict_json` fail more often.
            "temperature": 0,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        raw = await asyncio.to_thread(self._post, body)
        return self._content(raw)

    def _content(self, raw: bytes) -> str:
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise LlmRequestError("chat provider returned non-JSON") from None
        choices = parsed.get("choices") if isinstance(parsed, Mapping) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise LlmRequestError("chat provider response has no choices[0]")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise LlmRequestError("chat provider response has no choices[0].message.content")
        return content
