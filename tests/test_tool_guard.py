"""ToolGuard + identity checks (agents/researchers/_shared.py) — no LLM/network."""

import pytest
from langchain_core.tools import tool

from agents.researchers._shared import (
    DomainFindingOutput,
    ToolGuard,
    build_finding,
    company_names_match,
)

PROFILE = {"ticker": "TE", "company_name": "T1 Energy Inc", "website": None, "market_cap": None}


def _guard(**kw) -> ToolGuard:
    return ToolGuard(ticker="TE", max_calls=kw.pop("max_calls", 3), max_chars=kw.pop("max_chars", 100), blocked_hosts=("sec.gov",))


@tool
async def get_fundamentals(ticker: str) -> str:
    """fundamentals"""
    return f"data for {ticker}"


@tool
async def fetch(url: str, max_length: int = 5000) -> str:
    """fetch"""
    if "404" in url:
        return f"Failed to fetch {url} - status code 404"
    return f"Contents of {url}: " + "x" * 1000


@tool
async def get_company_news(ticker: str) -> str:
    """news"""
    return '{"articles": [{"url": "https://finnhub.io/api/news?id=1"}]}'


async def test_ticker_is_locked():
    g = _guard()
    t = g.wrap(get_fundamentals)
    assert (await t.ainvoke({"ticker": "TEL"})).startswith("refused: ticker is locked to TE")
    assert await t.ainvoke({"ticker": "te"}) == "data for te"
    assert g.calls == 1  # the refusal didn't spend budget


async def test_call_budget_is_enforced():
    g = _guard(max_calls=2)
    t = g.wrap(get_fundamentals)
    await t.ainvoke({"ticker": "TE"})
    await t.ainvoke({"ticker": "TE"})
    assert "budget exhausted" in await t.ainvoke({"ticker": "TE"})


async def test_blocked_host_and_duplicate_fetch_refused():
    g = _guard(max_chars=5000)
    f = g.wrap(fetch)
    assert "blocks automated access" in await f.ainvoke({"url": "https://www.sec.gov/Archives/x"})
    assert g.calls == 0
    await f.ainvoke({"url": "https://example.com/a"})
    assert "already fetched" in await f.ainvoke({"url": "https://example.com/a"})


async def test_results_are_truncated_and_failed_fetches_are_not_evidence():
    g = _guard(max_chars=100)
    f = g.wrap(fetch)
    out = await f.ainvoke({"url": "https://example.com/big", "max_length": 99999})
    assert len(out) < 200 and "[truncated" in out
    await f.ainvoke({"url": "https://example.com/404"})
    assert g.evidence_urls == {"https://example.com/big"}


async def test_news_urls_count_as_evidence_but_only_when_returned():
    g = _guard()
    await g.wrap(get_company_news).ainvoke({"ticker": "TE"})
    assert "https://finnhub.io/api/news?id=1" in g.evidence_urls


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("T1 Energy Inc.", "T1 Energy Inc", True),
        ("T1 Energy", "T1 Energy Inc", True),
        ("TE Connectivity PLC", "T1 Energy Inc", False),
        ("", "T1 Energy Inc", False),
    ],
)
def test_company_names_match(a, b, expected):
    assert company_names_match(a, b) is expected


def test_build_finding_flags_identity_mismatch_and_foreign_ticker():
    out = DomainFindingOutput(
        summary="s",
        identified_company="TE Connectivity Ltd",
        citations=[{"source_id": "get_fundamentals", "identifier": "TEL", "title": "TEL fundamentals"}],
    )
    finding = build_finding(out, "TE", "financials", PROFILE, _guard())
    assert finding["identity_ok"] is False
    assert len(finding["validation_issues"]) == 2


def test_build_finding_ok_when_company_matches():
    out = DomainFindingOutput(
        summary="s",
        identified_company="T1 Energy Inc.",
        citations=[{"source_id": "get_company_news", "identifier": "TE", "title": "news"}],
    )
    assert build_finding(out, "TE", "news", PROFILE, _guard())["identity_ok"] is True
