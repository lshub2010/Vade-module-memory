"""A real stdlib OpenAI-compatible upstream, run as its own process.

Used by the manual end-to-end check (and reusable by anyone verifying the
`python -m memory_module` path by hand). Prints the port it bound to on stdout,
then appends one JSON line per request to `--log`.

    python upstream_stub.py --port 0 --log /tmp/reqs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPLY = {
    "choices": [
        {
            "message": {
                "content": json.dumps(
                    {
                        "summary": "端到端检查：会话确认部署走本地回环。",
                        "facts": [
                            {
                                "text": "端到端检查确认部署走本地回环",
                                "namespace": "user:default",
                                "importance": 4,
                                "confidence": 0.9,
                            }
                        ],
                        "lessons": [],
                        "negative_feedback": False,
                        "skill_candidate": False,
                    },
                    ensure_ascii=False,
                )
            },
            "finish_reason": "stop",
        }
    ]
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--log", required=True)
    parser.add_argument("--status", type=int, default=200)
    args = parser.parse_args(argv)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            with open(args.log, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "path": self.path,
                            "authorization": self.headers.get("Authorization"),
                            "content_type": self.headers.get("Content-Type"),
                            "body": json.loads(raw.decode("utf-8")) if raw else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            payload = json.dumps(REPLY, ensure_ascii=False).encode("utf-8")
            self.send_response(args.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args) -> None:  # noqa: A002
            return None

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(server.server_address[1], flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
