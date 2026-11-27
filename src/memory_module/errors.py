"""Structured failures shared by the pure layers.

`tools.py` must not import the SDK (design §15.1), and `store.py` must not
import `tools.py` just to raise an error. This module is the one place both
sides agree on: a code from C-04 plus a human-readable message. `server.py`
translates `MemoryFailure` into `vade_sdk.ToolError` without changing the code.

Nothing here logs. In particular, callers must never put memory text in the
message: the same string ends up in core's log line.
"""

from __future__ import annotations

__all__ = ["MemoryFailure", "StartupError"]


class MemoryFailure(Exception):
    """A failure with a C-04 error code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class StartupError(RuntimeError):
    """The store cannot start safely (dimension/model mismatch, corrupt DB).

    Deliberately not a `MemoryFailure`: this happens before any tool call and
    must terminate the process with a readable reason instead of being turned
    into a per-call error envelope. Silently re-embedding would corrupt the
    existing index.
    """
