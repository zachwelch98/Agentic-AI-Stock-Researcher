FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Dependencies first so this layer is cached across source-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# app/main.py's lifespan spawns both MCP servers (tools/mcp_finance_server and
# mcp_server_fetch) as subprocesses of this same venv via `sys.executable`.
COPY app/ app/
COPY agents/ agents/
COPY tools/ tools/
COPY data/ data/

ENV PYTHONUNBUFFERED=1
# Run the venv's own binaries directly rather than through `uv run`: `uv run`
# re-syncs the environment on every invocation, which at container start would
# silently pull pytest/respx (the `dev` group `--no-dev` excluded at build
# time) back in over the network.
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
