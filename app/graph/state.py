import operator
from typing import Annotated, Literal, TypedDict


class ResearchTask(TypedDict):
    ticker: str
    domain: Literal["news", "financials", "leadership", "technical"]
    focus_notes: str


class CandidateFundamentals(TypedDict):
    ticker: str
    company_name: str
    market_cap: float
    avg_dollar_volume: float
    pe_ratio: float | None
    pb_ratio: float | None
    debt_to_equity: float | None
    dividend_yield: float | None
    composite_score: float
    matched_industry_label: str | None
    match_score: float | None


class DomainFinding(TypedDict):
    ticker: str
    domain: str
    summary: str
    citations: list[dict]
    raw_data: dict


class ResearchState(TypedDict):
    industry_query: str
    job_id: str
    user_supplied_ticker: str | None
    screening_plan: dict | None
    screened_candidates: list[CandidateFundamentals]
    used_llm_fallback_tickers: bool
    final_candidates: list[str]
    screener_justification: str | None
    research_tasks: list[ResearchTask]
    domain_findings: Annotated[list[DomainFinding], operator.add]
    final_report: dict | None
    status: Literal[
        "planning",
        "screening",
        "insufficient_candidates",
        "researching",
        "synthesizing",
        "done",
        "failed",
    ]
    error: str | None


class TickerResearchState(TypedDict):
    ticker: str
    research_tasks: list[ResearchTask]
    findings: Annotated[list[DomainFinding], operator.add]


def make_initial_state(
    industry_query: str, job_id: str, user_supplied_ticker: str | None = None
) -> ResearchState:
    return ResearchState(
        industry_query=industry_query,
        job_id=job_id,
        user_supplied_ticker=user_supplied_ticker,
        screening_plan=None,
        screened_candidates=[],
        used_llm_fallback_tickers=False,
        final_candidates=[],
        screener_justification=None,
        research_tasks=[],
        domain_findings=[],
        final_report=None,
        status="planning",
        error=None,
    )
