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


class CompanyProfile(TypedDict):
    ticker: str
    company_name: str
    website: str | None
    market_cap: float | None


class DomainFinding(TypedDict):
    ticker: str
    domain: str
    summary: str
    citations: list[dict]
    raw_data: dict
    # Written by run_domain_research / validate_findings. Optional in practice:
    # stub researchers in tests omit them, so readers use `.get(..., default)`.
    identified_company: str
    identity_ok: bool
    fetched_urls: list[str]
    validation_issues: list[str]


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
    company_profiles: dict[str, CompanyProfile]
    data_quality_warnings: list[str]
    # "TICKER:domain" entries dropped by agents/validation.py (identity mismatch etc.).
    excluded_domains: list[str]
    final_report: dict | None
    status: Literal[
        "planning",
        "screening",
        "insufficient_candidates",
        "researching",
        "validating",
        "synthesizing",
        "done",
        "failed",
    ]
    error: str | None


class TickerResearchState(TypedDict):
    ticker: str
    company_profile: CompanyProfile | None
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
        company_profiles={},
        data_quality_warnings=[],
        excluded_domains=[],
        final_report=None,
        status="planning",
        error=None,
    )
