"""Financials researcher node (subgraph-internal): tool-calling Haiku agent."""

from agents.researchers._shared import ToolGuard, build_domain_agent, find_task, run_domain_research
from app.graph.state import TickerResearchState
from tools.edgar import get_sec_filings, read_sec_filing
from tools.mcp_client import get_mcp_tool

_SYSTEM_PROMPT = (
    "You are a financial-statements researcher doing a red-flag skim. Given a "
    "stock ticker: (1) call get_fundamentals for current valuation/margin/debt "
    "metrics (this already gives revenue growth, net profit margin, debt/equity — "
    "don't re-fetch those from web pages); (2) call get_sec_filings with "
    "forms=[\"10-K\", \"10-Q\"] to find the most recent annual/quarterly report; "
    "(3) use read_sec_filing on its document_url (use its find= argument, e.g. "
    "\"Results of Operations\" or \"Total liabilities\", to jump to the financial statements/MD&A rather than paging through the cover page) to skim for red flags: "
    "deteriorating margins, rising debt, going-concern language, restatements, or "
    "unusual related-party transactions. Do NOT use the fetch tool on sec.gov (it is "
    "blocked). Distinguish net profit margin from operating margin, and only report "
    "a metric you actually saw with its correct name. Summarize the financial "
    "picture and any red flags in 3-6 sentences, and cite only pages you actually "
    "read (a filing you only saw listed is not a source). Only make claims the "
    "tools actually support."
)


async def financials_researcher(state: TickerResearchState) -> dict:
    ticker = state["ticker"]
    guard = ToolGuard.for_ticker(ticker)
    tools = guard.wrap_all([get_mcp_tool("get_fundamentals"), get_sec_filings, read_sec_filing, get_mcp_tool("fetch")])
    agent = build_domain_agent(tools, _SYSTEM_PROMPT)
    task = find_task(state, "financials")
    focus_notes = task["focus_notes"] if task else ""
    finding = await run_domain_research(
        agent, ticker, "financials", focus_notes, company_profile=state.get("company_profile"), guard=guard
    )
    return {"findings": [finding]}
