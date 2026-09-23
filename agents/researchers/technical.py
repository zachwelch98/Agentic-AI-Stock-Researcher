"""Technical-analysis researcher node (subgraph-internal): tool-calling Haiku agent.

Unlike the other 3 researchers, this one doesn't go through the MCP
`get_price_history` tool directly — fetching a full bar series and computing
indicators is one logical operation, not something useful to split across two
separate tool calls (the agent would otherwise need to pass a large bars list
back as a second tool's input). `get_technical_indicators` below is a local
LangChain tool that does both steps, reusing the same Yahoo Finance bar-fetch
helper the MCP `get_price_history` tool uses.
"""

import httpx
from langchain_core.tools import tool

from agents.researchers._shared import build_domain_agent, find_task, run_domain_research
from app.graph.state import TickerResearchState
from tools.mcp_finance_server import yahoo_client
from tools.technical_indicators import compute_technical_indicators

_SYSTEM_PROMPT = (
    "You are a technical-analysis researcher. Given a stock ticker, call "
    "get_technical_indicators (default range is fine unless the focus asks "
    "for a shorter/longer window) to get SMA-20/50, EMA-12, RSI-14, MACD, and "
    "recent volume trend. Interpret them: is the stock in an uptrend or "
    "downtrend relative to its moving averages, is RSI signaling overbought "
    "(>70) or oversold (<30), is MACD bullish or bearish, is volume "
    "confirming or diverging from the price trend. Summarize in 3-6 "
    "sentences and cite the indicator tool as your source. Only make claims "
    "the tool's numbers actually support."
)


@tool
async def get_technical_indicators(ticker: str, range: str = "6m") -> dict:
    """Fetch daily price history for `ticker` over `range` (1m/3m/6m/1y/2y)
    and compute SMA-20/50, EMA-12, RSI-14, MACD, and volume trend from it."""
    async with httpx.AsyncClient() as client:
        bars = await yahoo_client.get_daily_bars(client, ticker, range)
    return compute_technical_indicators(bars)


async def technical_researcher(state: TickerResearchState) -> dict:
    agent = build_domain_agent([get_technical_indicators], _SYSTEM_PROMPT)
    task = find_task(state, "technical")
    focus_notes = task["focus_notes"] if task else ""
    finding = await run_domain_research(agent, state["ticker"], "technical", focus_notes)
    return {"findings": [finding]}
