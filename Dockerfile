# Build from the module repository root:
#   docker build -t vade-memory .
#
# No packaging step on purpose: the module is run as `python -m memory_module`.
# `vade-sdk` is not vendored here (this is a standalone module repo); mount it
# at /opt/vade-sdk for --core mode, exactly like the echo module. --self-test
# and --standalone work without it.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/opt/vade-sdk \
    MEMORY_DATA_DIR=/data

WORKDIR /app

RUN pip install --no-cache-dir "websockets>=13" "cryptography>=42"

COPY src /app/src

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin vade \
    && mkdir -p /data && chown -R 10001:10001 /data

USER 10001:10001
VOLUME ["/data"]

# data/ is the only state: memory.db + WAL. Mount a named volume or the
# container loses every memory on restart.
STOPSIGNAL SIGTERM
CMD ["python", "-m", "memory_module", "--core", "wss://core:8443"]
