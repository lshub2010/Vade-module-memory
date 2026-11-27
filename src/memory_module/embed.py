"""Embedding clients: one HTTP client, one deterministic local fallback.

The memory module owns its embedding call and never asks the `model` module to
do it (M4a §一). Two implementations:

* `OpenAIEmbeddingClient` — OpenAI-compatible `POST {base_url}/embeddings`,
  key resolved from an environment variable NAME (`MEMORY_API_KEY` by default).
* `LocalHashEmbedding` — a deterministic hashed bag-of-words vector. It exists
  so `--standalone`, the tests and an offline deployment work without a
  network or an API key. It is lexical, not semantic; the README says so.

No implementation logs text. Failures are `EmbeddingError("UNAVAILABLE")` with
a message that names the endpoint or the environment variable, never the text.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Callable, Mapping, Protocol, Sequence

__all__ = [
    "EmbeddingError",
    "EmbeddingClient",
    "LocalHashEmbedding",
    "OpenAIEmbeddingClient",
    "FakeEmbedding",
    "check_vector",
]

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_MAX_TIMEOUT_S = 120.0


class EmbeddingError(Exception):
    """The provider did not return a usable vector.

    Always `UNAVAILABLE` in the C-04 table: the caller must not write a half
    record. A provider returning the wrong dimension is a configuration error
    but still surfaced through this class so no row is created.
    """

    def __init__(self, message: str, code: str = "UNAVAILABLE") -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def check_vector(vec: Sequence[float], dim: int, *, where: str = "embedding") -> list[float]:
    """Validate a vector's shape and values; return it as a plain float list."""
    if not isinstance(vec, (list, tuple)):
        raise EmbeddingError(f"{where} must be a list of numbers, got {type(vec).__name__}")
    if len(vec) != dim:
        raise EmbeddingError(
            f"{where} has dimension {len(vec)} but memory.embedding_dim is {dim}; "
            "refusing to store a vector that cannot be compared with the existing index"
        )
    out: list[float] = []
    for index, value in enumerate(vec):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingError(f"{where}[{index}] is not a number ({type(value).__name__})")
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingError(f"{where}[{index}] is not finite")
        out.append(number)
    return out


class EmbeddingClient(Protocol):
    """The small surface `tools.py` needs; a test fake implements only this."""

    model: str
    dim: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...


class LocalHashEmbedding:
    """Deterministic, dependency-free hashed bag-of-words embedding.

    ASCII words are tokens; CJK runs contribute unigrams *and* bigrams so a
    Chinese query can overlap a longer sentence without a tokenizer. The vector
    is L2-normalized, so cosine similarity is just a dot product. Same text and
    same dim always produce the same vector — persistence/restart tests rely on
    that.
    """

    def __init__(self, dim: int = 1536, model: str = "local-hash-v1") -> None:
        if not (1 <= dim <= 8192):
            raise ValueError(f"dim must be within [1, 8192], got {dim}")
        self.dim = dim
        self.model = model

    def _tokens(self, text: str) -> list[str]:
        tokens: list[str] = []
        for word in _TOKEN_RE.findall(text.lower()):
            if _CJK_RE.search(word):
                chars = list(word)
                tokens.extend(chars)
                tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
            else:
                tokens.append(word)
        return tokens

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise EmbeddingError(f"text must be a string, got {type(text).__name__}")
        vector = [0.0] * self.dim
        counts = Counter(self._tokens(text))
        if not counts:
            counts = Counter([text.strip() or "\0"])
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:  # extremely unlikely with a signed hash; keep it total
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest[:4], "little") % self.dim] = 1.0
            norm = 1.0
        return [value / norm for value in vector]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def _default_transport(request: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - base_url is operator config
            return int(getattr(response, "status", 200)), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read() or b""
    except urllib.error.URLError as exc:
        raise EmbeddingError(f"embedding endpoint is unreachable: {exc.reason}") from None


class OpenAIEmbeddingClient:
    """OpenAI-compatible embeddings with one retry on 429/5xx.

    The key is read from `api_key_env` on every call, so rotating the secret in
    the process environment needs no code change and the secret never lives in
    the database, the manifest or a log line.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dim: int,
        api_key_env: str = "MEMORY_API_KEY",
        timeout_s: float = 10.0,
        transport: Callable[[urllib.request.Request, float], tuple[int, bytes]] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("embedding_base_url must start with http:// or https://")
        if not (0.1 <= timeout_s <= _MAX_TIMEOUT_S):
            raise ValueError(f"embedding_timeout_s must be within [0.1, {_MAX_TIMEOUT_S}]")
        if not api_key_env:
            raise ValueError("api_key_env must be the NAME of an environment variable")
        self.base_url = base_url.rstrip("/")
        self.model = model or "text-embedding-3-small"
        self.dim = dim
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self._transport = transport or _default_transport
        self._environ = environ

    def _key(self) -> str:
        env = os.environ if self._environ is None else self._environ
        key = (env.get(self.api_key_env) or "").strip()
        if not key:
            raise EmbeddingError(
                f"environment variable {self.api_key_env} is not set; "
                "the memory module never accepts a literal API key"
            )
        return key

    def _post(self, payload: Mapping[str, Any]) -> bytes:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key()}",
            },
        )
        last_status = 0
        for attempt in range(2):
            status, raw = self._transport(request, self.timeout_s)
            last_status = status
            if 200 <= status < 300:
                return raw
            if status == 429 or 500 <= status < 600:
                if attempt == 0:
                    continue
                raise EmbeddingError(
                    f"embedding provider returned HTTP {status} twice; giving up (retriable)"
                )
            raise EmbeddingError(f"embedding provider returned HTTP {status} (not retriable)")
        raise EmbeddingError(f"embedding provider returned HTTP {last_status}")  # pragma: no cover

    def embed(self, text: str) -> list[float]:
        raw = self._post({"model": self.model, "input": [text]})
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise EmbeddingError("embedding provider returned non-JSON") from None
        data = parsed.get("data") if isinstance(parsed, Mapping) else None
        if not isinstance(data, list) or not data or not isinstance(data[0], Mapping):
            raise EmbeddingError("embedding provider response has no data[0].embedding")
        embedding = data[0].get("embedding")
        if embedding is None:
            raise EmbeddingError("embedding provider response has no data[0].embedding")
        return check_vector(embedding, self.dim, where="provider embedding")

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        # `remember` is one text at a time (M4a §五); this exists for callers
        # that later need a batch and keeps the provider contract honest.
        return [self.embed(text) for text in texts]


class FakeEmbedding:
    """Injectable fake for tests and `--self-test`.

    `mapping` can pin exact vectors; anything else falls back to the same
    deterministic local hash so semantic-ish assertions are still possible.
    `fail_times` makes the next N calls raise `UNAVAILABLE`, which is how the
    "no half record" test drives the failure path.
    """

    def __init__(
        self,
        dim: int = 8,
        *,
        mapping: Mapping[str, Sequence[float]] | None = None,
        fail_times: int = 0,
        model: str = "fake-embedding",
    ) -> None:
        self.dim = dim
        self.model = model
        self.mapping = {key: list(value) for key, value in (mapping or {}).items()}
        self.fail_times = fail_times
        self.calls: list[str] = []
        self._local = LocalHashEmbedding(dim=dim, model=model)

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise EmbeddingError("fake embedding is unavailable")
        if text in self.mapping:
            return check_vector(self.mapping[text], self.dim)
        return self._local.embed(text)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
