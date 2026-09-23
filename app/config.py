import os
from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    return Settings(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        finnhub_api_key=os.environ.get("FINNHUB_API_KEY", ""),
        sec_edgar_user_agent=os.environ.get(
            "SEC_EDGAR_USER_AGENT", "agentic-ai-stock-researcher contact@example.com"
        ),
        sonnet_model=os.environ.get("SONNET_MODEL", "claude-sonnet-5"),
        haiku_model=os.environ.get("HAIKU_MODEL", "claude-haiku-4-5-20251001"),
    )
