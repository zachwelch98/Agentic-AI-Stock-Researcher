"""Synthesizer node: Sonnet structured-output call producing the final
citation-backed, single-ticker research report."""

import json

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.graph.state import ResearchState
from app.llm import get_sonnet

_FALLBACK_TICKERS_NOTE = (
    "Candidate tickers for this industry were proposed by an LLM rather than sourced "
    "from a structured market-data industry taxonomy (NASDAQ no longer exposes "
    "fine-grained per-company industry classifications). All fundamentals and research "
    "findings themselves come from live market data (Finnhub/Yahoo Finance/SEC EDGAR), not the LLM."
)


class TickerReportSection(BaseModel):
    ticker: str
    overall_take: str = Field(description="2-4 sentence synthesis of this ticker across all 4 research domains")
    strengths: list[str] = Field(description="2-4 grounded strengths")
    risks: list[str] = Field(description="2-4 grounded risks")


class FinalReportOutput(BaseModel):
    overall_recommendation: str = Field(
        description="3-6 sentence recommendation on this ticker as a potential undervalued "
        "pick, grounded in the research findings — strengths, risks, and an overall take"
    )
    tickers: list[TickerReportSection]


def _group_findings_by_ticker(state: ResearchState) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for finding in state["domain_findings"]:
        grouped.setdefault(finding["ticker"], []).append(finding)
    return grouped


async def synthesizer(state: ResearchState) -> dict:
    findings_by_ticker = _group_findings_by_ticker(state)

    ticker = state["final_candidates"][0]
    industry_line = f"Industry: {state['industry_query']}\n" if state.get("industry_query") else ""

    llm = get_sonnet().with_structured_output(FinalReportOutput)
    prompt = (
        f"{industry_line}"
        f"Ticker under research: {ticker}\n"
        f"Screener's justification for this pick: {state.get('screener_justification')}\n\n"
        f"Domain research findings:\n{json.dumps(findings_by_ticker, indent=2)}\n\n"
        f"Write a single-ticker research report for {ticker}. Give an overall take plus "
        "2-4 strengths and 2-4 risks grounded strictly in the findings above. Then write "
        "an overall_recommendation on whether this looks like a genuinely undervalued "
        "pick and why."
    )
    result: FinalReportOutput = await llm.ainvoke([HumanMessage(content=prompt)])

    report = {
        "industry_query": state["industry_query"],
        "candidates": state["final_candidates"],
        "used_llm_fallback_tickers": state["used_llm_fallback_tickers"],
        "screener_justification": state.get("screener_justification"),
        "overall_recommendation": result.overall_recommendation,
        "tickers": [
            {
                "ticker": t.ticker,
                "overall_take": t.overall_take,
                "strengths": t.strengths,
                "risks": t.risks,
                "domain_findings": findings_by_ticker.get(t.ticker, []),
            }
            for t in result.tickers
        ],
    }
    if state["used_llm_fallback_tickers"]:
        report["data_provenance_note"] = _FALLBACK_TICKERS_NOTE

    return {"final_report": report, "status": "done"}
