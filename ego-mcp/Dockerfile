# ego-mcp served over Streamable HTTP. See docker-compose.yml for a ready setup.
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first so source edits don't invalidate the heavy layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src ./src
RUN uv sync --frozen

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    EGO_MCP_TRANSPORT=streamable-http \
    EGO_MCP_HTTP_HOST=0.0.0.0 \
    EGO_MCP_HTTP_PORT=8765 \
    EGO_MCP_DATA_DIR=/data \
    EGO_MCP_LOG_DIR=/logs

# Runs as an unprivileged user; compose overrides the uid/gid to match the
# owner of the bind-mounted data and log directories.
RUN mkdir -p /data /logs && chown 1000:1000 /data /logs
USER 1000:1000

EXPOSE 8765
VOLUME ["/data", "/logs"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['EGO_MCP_HTTP_PORT']}/healthz\", timeout=3)"]

CMD ["python", "-m", "ego_mcp"]
