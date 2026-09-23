"""Leadership researcher node (subgraph-internal): tool-calling Haiku agent."""

from agents.researchers._shared import build_domain_agent, find_task, run_domain_research
from app.graph.state import TickerResearchState
from tools.mcp_client import get_mcp_tool

_SYSTEM_PROMPT = (
    "You are a leadership/governance researcher. Given a stock ticker: (1) call "
    "get_fundamentals to find the company's website; (2) use the fetch tool on "
    "that site (try paths like /leadership, /about/leadership, /investors, or "
    "/about-us if the homepage doesn't link directly to an executive/board "
    "page) to find information on the CEO, other key executives, and board "
    "composition — tenure, recent changes, notable background. Summarize "
    "leadership stability and any recent changes or concerns in 3-6 sentences, "
    "and cite every page you relied on. If you can't find a leadership page "
    "after a couple of tries, say so rather than guessing. Only make claims "
    "the tools actually support."
)


async def leadership_researcher(state: TickerResearchState) -> dict:
    tools = [get_mcp_tool("get_fundamentals"), get_mcp_tool("fetch")]
    agent = build_domain_agent(tools, _SYSTEM_PROMPT)
    task = find_task(state, "leadership")
    focus_notes = task["focus_notes"] if task else ""
    finding = await run_domain_research(agent, state["ticker"], "leadership", focus_notes)
    return {"findings": [finding]}
