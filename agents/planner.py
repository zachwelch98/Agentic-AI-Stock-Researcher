"""Planner nodes: Sonnet structured-output calls.

`plan_screening_task` proposes plausible tickers for the industry — NASDAQ no
longer exposes a fine-grained industry taxonomy to fuzzy-match against (see
CLAUDE_CODE_NOTES.md), so ticker sourcing for `screen_industry` happens here,
upstream of the deterministic fetch+score step in `agents/screener.py`.
`plan_research_tasks` builds the per-ticker research-task grid (one task per
ticker/domain pair) with a tailored focus note for each.
"""

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.graph.state import ResearchState
from app.llm import get_sonnet

DOMAINS = ("news", "financials", "leadership", "technical")


class ScreeningPlanOutput(BaseModel):
    # Flat list of ticker symbols, not a list of {ticker, company_name} objects: a nested
    # list-of-objects schema at this size (15-20 items) reliably made Sonnet emit a
    # malformed tool call (the whole payload stringified into one field, `reasoning`
    # dropped) — flattening to list[str] fixed it. Company names aren't needed downstream
    # anyway (the MCP tool re-fetches them from Finnhub).
    proposed_tickers: list[str] = Field(
        description="15-20 real, currently-trading US-listed ticker symbols plausibly in "
        "this industry, spanning a range of company sizes, not just the 2-3 largest"
    )
    reasoning: str


async def plan_screening_task(state: ResearchState) -> dict:
    llm = get_sonnet().with_structured_output(ScreeningPlanOutput)
    prompt = (
        f"Industry: {state['industry_query']}\n\n"
        "Propose 15-20 real, currently-trading NYSE- or NASDAQ-listed public "
        "companies in or closely adjacent to this industry. Only include ticker "
        "symbols you are confident actually exist and are currently listed. Favor "
        "a range of company sizes rather than only the 2-3 largest, since the "
        "screening step downstream applies its own market-cap/liquidity filter "
        "and composite scoring against real fundamentals data — your job here is "
        "just to propose a plausible candidate pool for that step to work with."
    )
    result: ScreeningPlanOutput = await llm.ainvoke([HumanMessage(content=prompt)])
    proposed_tickers = [t.strip().upper() for t in result.proposed_tickers]

    return {
        "screening_plan": {
            "industry_query": state["industry_query"],
            "proposed_tickers": proposed_tickers,
            "reasoning": result.reasoning,
        },
        "used_llm_fallback_tickers": True,
        "status": "screening",
    }


class ResearchTaskFocus(BaseModel):
    ticker: str
    domain: str
    focus_notes: str = Field(description="1-2 sentence, ticker-specific instruction for this domain's researcher")


class ResearchTasksOutput(BaseModel):
    tasks: list[ResearchTaskFocus]


def _default_focus_notes(ticker: str, domain: str) -> str:
    return f"Research {ticker}'s {domain} in general, with no specific angle emphasized."


async def plan_research_tasks(state: ResearchState) -> dict:
    tickers = state["final_candidates"]
    profiles = state.get("company_profiles") or {}
    identities = "; ".join(
        f"{t} = {profiles[t]['company_name']}" + (f" ({profiles[t]['website']})" if profiles[t].get("website") else "")
        for t in tickers
        if t in profiles
    )
    llm = get_sonnet().with_structured_output(ResearchTasksOutput)
    prompt = (
        f"Industry: {state['industry_query']}\n"
        f"Final candidates selected for deep-dive research: {tickers}\n"
        f"Resolved company identities (authoritative, from market data): {identities or 'unavailable'}\n"
        "Never guess or suggest a different company for a ticker, and never tell a "
        "researcher to figure out what the ticker refers to.\n"
        f"Screener's reasoning for selecting them: {state.get('screener_justification')}\n\n"
        f"For EACH of these {len(tickers)} tickers, write one specific focus_notes "
        f"instruction (1-2 sentences) for each of these 4 research domains: "
        f"{', '.join(DOMAINS)}. Tailor each instruction to what's most relevant for "
        "that specific company given the screener's reasoning above (e.g. if debt "
        "was flagged as a concern for a ticker, tell its financials researcher to "
        "focus on debt covenants/refinancing risk). Return exactly "
        f"{len(tickers) * len(DOMAINS)} tasks — one per (ticker, domain) pair."
    )
    result: ResearchTasksOutput = await llm.ainvoke([HumanMessage(content=prompt)])

    tasks_by_key = {(t.ticker.strip().upper(), t.domain.strip().lower()): t.focus_notes for t in result.tasks}
    tasks = [
        {
            "ticker": ticker,
            "domain": domain,
            "focus_notes": tasks_by_key.get((ticker, domain), _default_focus_notes(ticker, domain)),
        }
        for ticker in tickers
        for domain in DOMAINS
    ]
    return {"research_tasks": tasks, "status": "researching"}
