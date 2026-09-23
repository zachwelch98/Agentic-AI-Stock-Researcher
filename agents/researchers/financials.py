"""Financials researcher node (subgraph-internal): tool-calling Haiku agent."""

from agents.researchers._shared import build_domain_agent, find_task, run_domain_research
from app.graph.state import TickerResearchState
from tools.edgar import get_sec_filings
from tools.mcp_client import get_mcp_tool

_SYSTEM_PROMPT = (
    "You are a financial-statements researcher doing a red-flag skim. Given a "
    "stock ticker: (1) call get_fundamentals for current valuation/margin/debt "
    "metrics; (2) call get_sec_filings to find its most recent 10-K/10-Q; "
    "(3) use the fetch tool on the most recent 10-K or 10-Q's document_url to "
    "skim for red flags — deteriorating margins, rising debt, going-concern "
    "language, restatements, or unusual related-party transactions. Summarize "
    "the financial picture and any red flags found in 3-6 sentences, and cite "
    "every source you relied on. Only make claims the tools actually support."
)


async def financials_researcher(state: TickerResearchState) -> dict:
    tools = [get_mcp_tool("get_fundamentals"), get_sec_filings, get_mcp_tool("fetch")]
    agent = build_domain_agent(tools, _SYSTEM_PROMPT)
    task = find_task(state, "financials")
    focus_notes = task["focus_notes"] if task else ""
    finding = await run_domain_research(agent, state["ticker"], "financials", focus_notes)
    return {"findings": [finding]}
