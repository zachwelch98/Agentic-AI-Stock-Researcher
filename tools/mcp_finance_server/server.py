"""Custom finance MCP server: FastMCP app + 4 tools, stdio entrypoint.

    uv run python -m tools.mcp_finance_server.server

Satisfies the "at least one MCP-backed tool" requirement. `screen_industry`
does the fuzzy industry match + fundamentals screen; `get_fundamentals`,
`get_price_history`, and `get_company_news` back the per-ticker researchers.
"""

from datetime import datetime, timezone

import httpx
from mcp.server.fastmcp import FastMCP

from agents.screener import screen_candidates
from tools.mcp_finance_server import nasdaq_universe, yahoo_client
from tools.mcp_finance_server.finnhub_client import FinnhubClient
from tools.mcp_finance_server.schemas import (
    CompanyNewsResult,
    FundamentalsResult,
    PriceHistoryResult,
    ScreenIndustryResult,
)

mcp = FastMCP("stock-researcher-finance")
_finnhub = FinnhubClient()

# Finnhub /stock/metric field names, per https://finnhub.io/docs/api/company-basic-financials.
# Some metrics have multiple near-equivalent keys depending on data availability; we take
# the first that's present.
_PE_KEYS = ("peExclExtraTTM", "peBasicExclExtraTTM", "peNormalizedAnnual", "peAnnual")
_PB_KEYS = ("pbAnnual", "pbQuarterly")
_DEBT_EQUITY_KEYS = ("totalDebt/totalEquityAnnual", "totalDebt/totalEquityQuarterly")
_DIVIDEND_YIELD_KEYS = ("dividendYieldIndicatedAnnual", "currentDividendYieldTTM")
_EPS_KEYS = ("epsBasicExclExtraItemsTTM", "epsInclExtraItemsTTM", "epsAnnual")
_REVENUE_GROWTH_KEYS = ("revenueGrowthTTMYoy", "revenueGrowthQuarterlyYoy")
_PROFIT_MARGIN_KEYS = ("netProfitMarginTTM", "netProfitMarginAnnual")
_AVG_VOLUME_KEYS = ("10DayAverageTradingVolume", "3MonthAverageTradingVolume")


def _first_present(metric: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = metric.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _fraction_from_percent(value: float | None) -> float | None:
    return None if value is None else value / 100.0


async def _fetch_candidate_raw(
    client: httpx.AsyncClient, symbol: str, matched_label: str | None, match_score: float | None
) -> dict | None:
    profile = await _finnhub.get_profile(client, symbol)
    financials = await _finnhub.get_basic_financials(client, symbol)
    metric = financials.get("metric", {}) if isinstance(financials, dict) else {}

    market_cap_millions = profile.get("marketCapitalization")
    if market_cap_millions is None:
        return None
    market_cap = float(market_cap_millions) * 1_000_000

    # 10DayAverageTradingVolume / 3MonthAverageTradingVolume are reported in millions of shares.
    avg_volume_shares_millions = _first_present(metric, _AVG_VOLUME_KEYS)
    quote = await _finnhub.get_quote(client, symbol)
    price = quote.get("c")
    avg_dollar_volume = (
        avg_volume_shares_millions * 1_000_000 * price
        if avg_volume_shares_millions is not None and price
        else 0.0
    )

    return {
        "ticker": symbol,
        "company_name": profile.get("name", symbol),
        "market_cap": market_cap,
        "avg_dollar_volume": avg_dollar_volume,
        "pe_ratio": _first_present(metric, _PE_KEYS),
        "pb_ratio": _first_present(metric, _PB_KEYS),
        "debt_to_equity": _first_present(metric, _DEBT_EQUITY_KEYS),
        "dividend_yield": _fraction_from_percent(_first_present(metric, _DIVIDEND_YIELD_KEYS)),
        "matched_industry_label": matched_label,
        "match_score": match_score,
    }


@mcp.tool()
async def screen_industry(industry: str, candidate_tickers: list[str]) -> dict:
    """Fetch Finnhub fundamentals for each of `candidate_tickers`, apply the
    market-cap/liquidity filter, and return the ranked top ~15 by composite
    score. `candidate_tickers` are proposed by the caller as plausible members
    of `industry` (NASDAQ's public screener no longer exposes fine-grained
    per-company industry labels, so ticker sourcing happens upstream, e.g. via
    an LLM planning step); tickers not found in the cached NASDAQ universe are
    dropped before spending a Finnhub call on them."""
    universe = await nasdaq_universe.get_universe()
    valid_symbols = nasdaq_universe.known_symbols(universe)

    raw_candidates: list[dict] = []
    async with httpx.AsyncClient() as client:
        for raw_symbol in candidate_tickers:
            symbol = raw_symbol.strip().upper()
            if symbol not in valid_symbols:
                continue
            try:
                raw = await _fetch_candidate_raw(client, symbol, industry, None)
            except httpx.HTTPError:
                continue
            if raw is not None:
                raw_candidates.append(raw)

    scored = screen_candidates(raw_candidates)
    return ScreenIndustryResult(
        candidates=scored, matched_industry_label=industry, match_score=None
    ).model_dump()


@mcp.tool()
async def get_fundamentals(ticker: str) -> dict:
    """Fetch current fundamentals for a single ticker from Finnhub."""
    async with httpx.AsyncClient() as client:
        profile = await _finnhub.get_profile(client, ticker)
        financials = await _finnhub.get_basic_financials(client, ticker)
    metric = financials.get("metric", {}) if isinstance(financials, dict) else {}

    market_cap_millions = profile.get("marketCapitalization")
    return FundamentalsResult(
        ticker=ticker,
        company_name=profile.get("name"),
        website=profile.get("weburl"),
        market_cap=float(market_cap_millions) * 1_000_000 if market_cap_millions is not None else None,
        pe_ratio=_first_present(metric, _PE_KEYS),
        pb_ratio=_first_present(metric, _PB_KEYS),
        debt_to_equity=_first_present(metric, _DEBT_EQUITY_KEYS),
        dividend_yield=_fraction_from_percent(_first_present(metric, _DIVIDEND_YIELD_KEYS)),
        eps=_first_present(metric, _EPS_KEYS),
        revenue_growth=_first_present(metric, _REVENUE_GROWTH_KEYS),
        profit_margin=_first_present(metric, _PROFIT_MARGIN_KEYS),
        fifty_two_week_high=metric.get("52WeekHigh"),
        fifty_two_week_low=metric.get("52WeekLow"),
    ).model_dump()


@mcp.tool()
async def get_price_history(ticker: str, range: str = "6m") -> dict:
    """Fetch daily OHLCV bars for `ticker` over `range` (one of: 1m, 3m, 6m, 1y, 2y)
    from Yahoo Finance's unofficial chart API — neither Finnhub's free tier
    (no historical candles) nor Financial Modeling Prep's free tier
    (historical prices restricted to a small symbol whitelist) work for
    arbitrary tickers."""
    async with httpx.AsyncClient() as client:
        bars = await yahoo_client.get_daily_bars(client, ticker, range)
    return PriceHistoryResult(ticker=ticker, range=range, bars=bars).model_dump()


@mcp.tool()
async def get_company_news(ticker: str, from_date: str, to_date: str) -> dict:
    """Fetch company news headlines for `ticker` between `from_date` and
    `to_date` (both YYYY-MM-DD) from Finnhub's company-news endpoint."""
    async with httpx.AsyncClient() as client:
        articles = await _finnhub.get_company_news(client, ticker, from_date, to_date)

    return CompanyNewsResult(
        ticker=ticker,
        from_date=from_date,
        to_date=to_date,
        articles=[
            {
                "headline": a.get("headline", ""),
                "summary": a.get("summary", ""),
                "source": a.get("source", ""),
                "url": a.get("url", ""),
                "datetime": datetime.fromtimestamp(a["datetime"], tz=timezone.utc).isoformat()
                if a.get("datetime")
                else "",
            }
            for a in articles
        ],
    ).model_dump()


if __name__ == "__main__":
    mcp.run(transport="stdio")
