"""News researcher node (subgraph-internal): tool-calling Haiku agent."""

from datetime import datetime, timedelta, timezone

from agents.researchers._shared import build_domain_agent, find_task, run_domain_research
from app.graph.state import TickerResearchState
from tools.mcp_client import get_mcp_tool


def _system_prompt() -> str:
    today = datetime.now(tz=timezone.utc).date()
    thirty_days_ago = today - timedelta(days=30)
    return (
        "You are a financial news researcher. Given a stock ticker, use the "
        "get_company_news tool to pull recent headlines and summaries — pass "
        f"from_date={thirty_days_ago.isoformat()} and to_date={today.isoformat()} "
        "(today's date) unless the focus given asks for a different window. If a "
        "headline looks especially relevant to the focus given, use the fetch "
        "tool on its URL to read more context before summarizing. Summarize "
        "recent news and overall sentiment in 3-6 sentences, and cite every "
        "article you relied on. Only make claims the tools actually support."
    )


async def news_researcher(state: TickerResearchState) -> dict:
    tools = [get_mcp_tool("get_company_news"), get_mcp_tool("fetch")]
    agent = build_domain_agent(tools, _system_prompt())
    task = find_task(state, "news")
    focus_notes = task["focus_notes"] if task else ""
    finding = await run_domain_research(agent, state["ticker"], "news", focus_notes)
    return {"findings": [finding]}
