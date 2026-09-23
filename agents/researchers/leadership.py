"""Leadership researcher node (subgraph-internal): tool-calling Haiku agent."""

from agents.researchers._shared import ToolGuard, build_domain_agent, find_task, run_domain_research
from app.graph.state import TickerResearchState
from tools.edgar import get_sec_filings, read_sec_filing
from tools.mcp_client import get_mcp_tool

_SYSTEM_PROMPT = (
    "You are a leadership/governance researcher. Start from SEC filings, not the "
    "company website. Given a stock ticker: (1) call get_sec_filings with "
    "forms=[\"DEF 14A\"] and read_sec_filing on the latest proxy statement for "
    "executive officers, directors, tenure and backgrounds (use find=, e.g. "
    "\"Executive Officers\" or \"Directors\", to jump to the right section); (2) call get_sec_filings with forms=[\"4\"] and read up to 2 "
    "recent Form 4s for insider buying/selling, and with forms=[\"8-K\"] to spot "
    "leadership changes (Item 5.02). (3) Only if filings leave gaps, use the fetch "
    "tool on the company's homepage (from get_fundamentals) and follow links that "
    "page actually contains — never guess paths like /leadership or /about. Do NOT "
    "use fetch on sec.gov (blocked). Report insider activity only as far as a Form "
    "4 you read shows it, otherwise say it wasn't found. Summarize leadership "
    "stability and any recent changes or concerns in 3-6 sentences, and cite only "
    "pages you actually read. Only make claims the tools actually support."
)


async def leadership_researcher(state: TickerResearchState) -> dict:
    ticker = state["ticker"]
    guard = ToolGuard.for_ticker(ticker, max_calls=6)
    tools = guard.wrap_all([get_mcp_tool("get_fundamentals"), get_sec_filings, read_sec_filing, get_mcp_tool("fetch")])
    agent = build_domain_agent(tools, _SYSTEM_PROMPT)
    task = find_task(state, "leadership")
    focus_notes = task["focus_notes"] if task else ""
    finding = await run_domain_research(
        agent, ticker, "leadership", focus_notes, company_profile=state.get("company_profile"), guard=guard
    )
    return {"findings": [finding]}
