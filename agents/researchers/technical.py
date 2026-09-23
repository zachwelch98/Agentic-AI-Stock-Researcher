"""Technical-analysis researcher node (subgraph-internal): fixed tool call + one Haiku summary.

Unlike the other 3 researchers, this one doesn't go through the MCP
`get_price_history` tool directly — fetching a full bar series and computing
indicators is one logical operation, not something useful to split across two
separate tool calls (the agent would otherwise need to pass a large bars list
back as a second tool's input). `get_technical_indicators` below is a local
LangChain tool that does both steps, reusing the same Yahoo Finance bar-fetch
helper the MCP `get_price_history` tool uses.
"""

import json

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agents.researchers._shared import find_task, identity_line
from app.graph.state import TickerResearchState
from app.llm import get_haiku
from tools.mcp_finance_server import yahoo_client
from tools.technical_indicators import compute_technical_indicators

_SYSTEM_PROMPT = (
    "You are a technical-analysis researcher. You are given a stock's computed "
    "indicators, including a `signals` block with the trend, RSI state, MACD state, "
    "support/resistance levels and volume trend already interpreted. Write a 3-6 "
    "sentence summary that reports those signals as given — do not re-derive or "
    "contradict them — and includes the support and resistance levels and the volume "
    "window. If the focus asks about something the data doesn't contain, say so. "
    "Only make claims the numbers actually support."
)


class TechnicalSummary(BaseModel):
    summary: str = Field(description="3-6 sentence technical summary grounded in the indicator data")


@tool
async def get_technical_indicators(ticker: str, range: str = "6m") -> dict:
    """Fetch daily price history for `ticker` over `range` (1m/3m/6m/1y/2y)
    and compute SMA-20/50, EMA-12, RSI-14, MACD, and volume trend from it,
    plus a `signals` block with plain-language readings and support/resistance."""
    async with httpx.AsyncClient() as client:
        bars = await yahoo_client.get_daily_bars(client, ticker, range)
    return compute_technical_indicators(bars)


async def technical_researcher(state: TickerResearchState) -> dict:
    """One tool call + one structured LLM call (was a 3-call ReAct loop): fetching
    and interpreting indicators is a fixed pipeline, not something the model
    needs to decide step by step. Indicators are interpreted in code (`signals`)."""
    ticker = state["ticker"]
    profile = state.get("company_profile")
    task = find_task(state, "technical")
    focus_notes = task["focus_notes"] if task else ""

    try:
        indicators = await get_technical_indicators.ainvoke({"ticker": ticker})
    except httpx.HTTPError:
        indicators = None
    if not indicators or "error" in indicators:
        # No finding at all: validate_findings then counts the domain as missing.
        return {"findings": []}

    llm = get_haiku().with_structured_output(TechnicalSummary)
    instruction = identity_line(ticker, profile) + (f" Focus: {focus_notes}" if focus_notes else "")
    result: TechnicalSummary = await llm.ainvoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"{instruction}\n\nIndicators:\n{json.dumps(indicators, indent=2)}"),
        ]
    )
    company = profile["company_name"] if profile else ""
    finding = {
        "ticker": ticker,
        "domain": "technical",
        "summary": result.summary,
        "citations": [
            {
                "source_id": "get_technical_indicators",
                "identifier": ticker,
                "title": f"Technical indicators for {company or ticker} (Yahoo Finance daily bars)",
            }
        ],
        "raw_data": {},
        "identified_company": company,
        "identity_ok": True,
        "fetched_urls": [],
        "validation_issues": [],
    }
    return {"findings": [finding]}
