import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

# override=True: this project's .env should win over any stale/ambient
# environment variables of the same name (e.g. a leftover shell export).
load_dotenv(override=True)

# LangSmith tracing reads LANGCHAIN_* env vars directly (not through our Settings
# model), and fails loudly on every LLM call if tracing is enabled with no key —
# .env defaults LANGCHAIN_TRACING_V2=true so tracing "just works" once a key is
# added, so guard against that default causing noise before one is.
if os.environ.get("LANGCHAIN_TRACING_V2") == "true" and not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


class Settings(BaseModel):
    anthropic_api_key: str = ""
    finnhub_api_key: str = ""
    sec_edgar_user_agent: str = "agentic-ai-stock-researcher contact@example.com"

    # All Sonnet-tier nodes (planner, screener's LLM-judgment step, synthesizer)
    # share one model id; all 4 researcher nodes share the Haiku-tier one.
    sonnet_model: str = "claude-sonnet-5"
    haiku_model: str = "claude-haiku-4-5-20251001"

    # Deterministic screen thresholds
    min_market_cap: float = 300_000_000
    min_avg_dollar_volume: float = 1_000_000
    screen_top_n: int = 15

    # Researcher agent guardrails (agents/researchers/_shared.py ToolGuard)
    researcher_max_tool_calls: int = 5
    fetch_max_chars: int = 6000
    researcher_prompt_cache: bool = True
    blocked_fetch_hosts: tuple[str, ...] = ("sec.gov", "macrotrends.net")

    # Local run logging (app/local_trace.py). Empty local_trace_dir = disabled.
    # local_trace_db "" = don't build the SQLite index (app/trace_index.py).
    local_trace_dir: str = ""
    local_trace_db: str = ""
    local_trace_max_blob_bytes: int = 200_000
    local_trace_keep_runs: int = 50


def _local_trace_db(trace_dir: str) -> str:
    """LOCAL_TRACE_DB unset -> `<trace_dir>/../traces.db`; set to "" -> indexing off."""
    if "LOCAL_TRACE_DB" in os.environ:
        return os.environ["LOCAL_TRACE_DB"]
    return str(Path(trace_dir).parent / "traces.db") if trace_dir else ""


@lru_cache
def get_settings() -> Settings:
    local_trace_dir = os.environ.get("LOCAL_TRACE_DIR", "")
    return Settings(
        local_trace_dir=local_trace_dir,
        local_trace_db=_local_trace_db(local_trace_dir),
        local_trace_max_blob_bytes=int(os.environ.get("LOCAL_TRACE_MAX_BLOB_BYTES", "200000")),
        local_trace_keep_runs=int(os.environ.get("LOCAL_TRACE_KEEP_RUNS", "50")),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        finnhub_api_key=os.environ.get("FINNHUB_API_KEY", ""),
        sec_edgar_user_agent=os.environ.get(
            "SEC_EDGAR_USER_AGENT", "agentic-ai-stock-researcher contact@example.com"
        ),
        researcher_max_tool_calls=int(os.environ.get("RESEARCHER_MAX_TOOL_CALLS", "5")),
        fetch_max_chars=int(os.environ.get("FETCH_MAX_CHARS", "6000")),
        researcher_prompt_cache=os.environ.get("RESEARCHER_PROMPT_CACHE", "true").lower() != "false",
        blocked_fetch_hosts=tuple(
            h.strip().lower()
            for h in os.environ.get("BLOCKED_FETCH_HOSTS", "sec.gov,macrotrends.net").split(",")
            if h.strip()
        ),
        sonnet_model=os.environ.get("SONNET_MODEL", "claude-sonnet-5"),
        haiku_model=os.environ.get("HAIKU_MODEL", "claude-haiku-4-5-20251001"),
    )
