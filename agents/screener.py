"""Screener nodes and the deterministic composite-scoring formula.

`compute_composite_score` and `screen_candidates` are pure code, no network —
exercised directly by tests/test_screener.py, and also used inside the MCP
`screen_industry` tool itself (tools/mcp_finance_server/server.py) so the
scoring formula lives in exactly one place. `deterministic_screen` calls that
MCP tool with the ticker pool `plan_screening_task` proposed; `llm_judgment_screen`
is a real Sonnet structured-output call that picks the single most-undervalued
ticker (with a safety net against a hallucinated ticker not in the screened
pool). `accept_user_ticker` is the alternate entry point used when the caller
names a specific ticker directly, bypassing screening entirely.
"""

from statistics import median

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.state import CandidateFundamentals, CompanyProfile, ResearchState
from app.llm import get_sonnet
from tools.mcp_client import get_mcp_tool, parse_mcp_tool_result


def compute_composite_score(
    pe_ratio: float | None,
    pb_ratio: float | None,
    debt_to_equity: float | None,
    dividend_yield: float | None,
    pe_median: float | None,
    pb_median: float | None,
    de_median: float | None,
) -> float:
    """Industry-agnostic composite: +1 per fundamental better than the industry
    median (P/E positive-only, P/B, D/E), +0.5 bonus for any positive dividend
    yield (a bonus, not a filter, so REITs aren't unfairly penalized)."""
    score = 0.0
    if pe_ratio is not None and pe_ratio > 0 and pe_median is not None and pe_ratio < pe_median:
        score += 1.0
    if pb_ratio is not None and pb_median is not None and pb_ratio < pb_median:
        score += 1.0
    if debt_to_equity is not None and de_median is not None and debt_to_equity < de_median:
        score += 1.0
    if dividend_yield is not None and dividend_yield > 0:
        score += 0.5
    return score


def screen_candidates(raw_candidates: list[dict]) -> list[CandidateFundamentals]:
    """Filter by market cap / liquidity, score against industry medians, rank
    descending by score (tie-break: market cap), keep the top N."""
    settings = get_settings()
    filtered = [
        c
        for c in raw_candidates
        if c["market_cap"] >= settings.min_market_cap
        and c["avg_dollar_volume"] >= settings.min_avg_dollar_volume
    ]
    if not filtered:
        return []

    pe_values = [c["pe_ratio"] for c in filtered if c.get("pe_ratio") not in (None,) and c["pe_ratio"] > 0]
    pb_values = [c["pb_ratio"] for c in filtered if c.get("pb_ratio") is not None]
    de_values = [c["debt_to_equity"] for c in filtered if c.get("debt_to_equity") is not None]
    pe_median = median(pe_values) if pe_values else None
    pb_median = median(pb_values) if pb_values else None
    de_median = median(de_values) if de_values else None

    scored: list[CandidateFundamentals] = []
    for c in filtered:
        score = compute_composite_score(
            c.get("pe_ratio"),
            c.get("pb_ratio"),
            c.get("debt_to_equity"),
            c.get("dividend_yield"),
            pe_median,
            pb_median,
            de_median,
        )
        scored.append(
            CandidateFundamentals(
                ticker=c["ticker"],
                company_name=c["company_name"],
                market_cap=c["market_cap"],
                avg_dollar_volume=c["avg_dollar_volume"],
                pe_ratio=c.get("pe_ratio"),
                pb_ratio=c.get("pb_ratio"),
                debt_to_equity=c.get("debt_to_equity"),
                dividend_yield=c.get("dividend_yield"),
                composite_score=score,
                matched_industry_label=c.get("matched_industry_label"),
                match_score=c.get("match_score"),
            )
        )
    scored.sort(key=lambda c: (c["composite_score"], c["market_cap"]), reverse=True)
    return scored[: settings.screen_top_n]


async def deterministic_screen(state: ResearchState) -> dict:
    plan = state.get("screening_plan") or {}
    candidate_tickers = plan.get("proposed_tickers", [])

    tool = get_mcp_tool("screen_industry")
    raw_result = await tool.ainvoke({"industry": state["industry_query"], "candidate_tickers": candidate_tickers})
    result = parse_mcp_tool_result(raw_result)

    return {"screened_candidates": result["candidates"], "status": "screening"}


class FinalSelectionOutput(BaseModel):
    selected_ticker: str = Field(description="The single ticker chosen as the most undervalued pick")
    justification: str = Field(description="Why this ticker, grounded in the composite scores and fundamentals given")


async def llm_judgment_screen(state: ResearchState) -> dict:
    candidates = state["screened_candidates"]

    llm = get_sonnet().with_structured_output(FinalSelectionOutput)
    prompt = (
        f"Industry: {state['industry_query']}\n\n"
        "Here are the candidates that survived a deterministic fundamentals screen "
        "(each scored on P/E, P/B, and debt/equity vs. the industry median, plus a "
        "dividend-yield bonus — higher composite_score means more attractive on "
        f"fundamentals):\n{candidates}\n\n"
        "Select exactly 1 ticker as the single most undervalued pick for this "
        "industry. You may follow the deterministic ranking, or override it with "
        "qualitative judgment (e.g. avoiding a likely value trap), but you must "
        "choose from the candidates listed above and justify your choice."
    )
    result: FinalSelectionOutput = await llm.ainvoke([HumanMessage(content=prompt)])

    valid_tickers = {c["ticker"] for c in candidates}
    selected_ticker = result.selected_ticker.strip().upper()
    if selected_ticker not in valid_tickers:
        ranked = sorted(candidates, key=lambda c: c["composite_score"], reverse=True)
        selected_ticker = ranked[0]["ticker"]

    return {
        "final_candidates": [selected_ticker],
        "screener_justification": result.justification,
        "status": "researching",
    }


async def accept_user_ticker(state: ResearchState) -> dict:
    """Entry point for a user-supplied ticker: skips the industry screen entirely
    and hands the ticker straight to `plan_research_tasks`."""
    return {
        "final_candidates": [state["user_supplied_ticker"]],
        "used_llm_fallback_tickers": False,
        "screener_justification": "User-specified ticker; industry screening bypassed.",
        "status": "researching",
    }


async def insufficient_candidates(state: ResearchState) -> dict:
    return {
        "status": "insufficient_candidates",
        "final_report": {
            "industry_query": state["industry_query"],
            "outcome": "insufficient_candidates",
            "message": (
                f"Fewer than 3 candidates survived the deterministic screen for "
                f"'{state['industry_query']}'."
            ),
            "screened_candidates": state["screened_candidates"],
        },
    }


async def resolve_identity(state: ResearchState) -> dict:
    """Pin each final candidate to a company *before* any LLM sees the ticker.

    Deterministic (one `get_fundamentals` call per ticker, no LLM): the resulting
    `company_profiles` are the ground truth the planner and all 4 researchers are
    given, so an ambiguous symbol like "TE" (T1 Energy, not TE Connectivity/TEL)
    can't be silently swapped for a better-known company mid-run."""
    tool = get_mcp_tool("get_fundamentals")
    profiles: dict[str, CompanyProfile] = {}
    for ticker in state["final_candidates"]:
        try:
            result = parse_mcp_tool_result(await tool.ainvoke({"ticker": ticker}))
        except Exception as exc:
            return {"status": "failed", "error": f"Could not resolve ticker {ticker}: {exc}"}
        company_name = (result.get("company_name") or "").strip()
        if not company_name:
            return {"status": "failed", "error": f"Could not resolve ticker {ticker} to a company"}
        profiles[ticker] = CompanyProfile(
            ticker=ticker,
            company_name=company_name,
            website=result.get("website"),
            market_cap=result.get("market_cap"),
        )
    return {"company_profiles": profiles}
