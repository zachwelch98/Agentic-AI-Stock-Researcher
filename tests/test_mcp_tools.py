import json

import httpx
import pytest
import respx

from tools.mcp_finance_server import nasdaq_universe, server

NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"
FINNHUB_BASE = "https://finnhub.io/api/v1"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"


def _nasdaq_payload(rows):
    return {"data": {"table": {"rows": rows}}, "totalrecords": len(rows)}


def _finnhub_profile(name: str, market_cap_millions: float):
    return {"name": name, "marketCapitalization": market_cap_millions, "finnhubIndustry": "Banking"}


def _finnhub_metric(**overrides):
    base = {
        "peExclExtraTTM": 10.0,
        "pbAnnual": 1.0,
        "totalDebt/totalEquityAnnual": 1.0,
        "dividendYieldIndicatedAnnual": 2.0,
        "epsBasicExclExtraItemsTTM": 5.0,
        "revenueGrowthTTMYoy": 3.0,
        "netProfitMarginTTM": 20.0,
        "52WeekHigh": 100.0,
        "52WeekLow": 50.0,
        "10DayAverageTradingVolume": 5.0,
    }
    base.update(overrides)
    return {"metric": base}


@pytest.fixture(autouse=True)
def _isolate_nasdaq_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(nasdaq_universe, "CACHE_PATH", tmp_path / "nasdaq_universe_cache.json")


@pytest.fixture(autouse=True)
def _fake_api_keys(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-finnhub-key")
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
@respx.mock
async def test_screen_industry_scores_known_tickers_and_drops_unknown():
    respx.get(NASDAQ_SCREENER_URL).mock(
        return_value=httpx.Response(
            200,
            json=_nasdaq_payload(
                [
                    {"symbol": "RB1", "name": "Regional Bank One", "marketCap": "5,000,000,000"},
                    {"symbol": "RB2", "name": "Regional Bank Two", "marketCap": "3,000,000,000"},
                ]
            ),
        )
    )
    respx.get(f"{FINNHUB_BASE}/stock/profile2", params={"symbol": "RB1"}).mock(
        return_value=httpx.Response(200, json=_finnhub_profile("Regional Bank One", 5000.0))
    )
    respx.get(f"{FINNHUB_BASE}/stock/metric", params={"symbol": "RB1", "metric": "all"}).mock(
        return_value=httpx.Response(200, json=_finnhub_metric(peExclExtraTTM=8.0))
    )
    respx.get(f"{FINNHUB_BASE}/quote", params={"symbol": "RB1"}).mock(
        return_value=httpx.Response(200, json={"c": 50.0})
    )
    respx.get(f"{FINNHUB_BASE}/stock/profile2", params={"symbol": "RB2"}).mock(
        return_value=httpx.Response(200, json=_finnhub_profile("Regional Bank Two", 3000.0))
    )
    respx.get(f"{FINNHUB_BASE}/stock/metric", params={"symbol": "RB2", "metric": "all"}).mock(
        return_value=httpx.Response(200, json=_finnhub_metric(peExclExtraTTM=20.0))
    )
    respx.get(f"{FINNHUB_BASE}/quote", params={"symbol": "RB2"}).mock(
        return_value=httpx.Response(200, json={"c": 30.0})
    )

    result = await server.screen_industry(
        "Regional Banks", ["RB1", "RB2", "NOT_A_REAL_TICKER"]
    )

    assert len(result["candidates"]) == 2
    tickers = {c["ticker"] for c in result["candidates"]}
    assert tickers == {"RB1", "RB2"}
    assert result["matched_industry_label"] == "Regional Banks"
    # RB1 has the lower P/E (8.0 < median) so it should score higher than RB2 (20.0).
    assert result["candidates"][0]["ticker"] == "RB1"
    assert result["candidates"][0]["composite_score"] > result["candidates"][1]["composite_score"]


@pytest.mark.asyncio
@respx.mock
async def test_screen_industry_empty_candidate_list_returns_empty():
    respx.get(NASDAQ_SCREENER_URL).mock(return_value=httpx.Response(200, json=_nasdaq_payload([])))
    result = await server.screen_industry("Nonexistent Industry", [])
    assert result["candidates"] == []


@pytest.mark.asyncio
@respx.mock
async def test_get_fundamentals_maps_finnhub_fields():
    respx.get(f"{FINNHUB_BASE}/stock/profile2", params={"symbol": "AAPL"}).mock(
        return_value=httpx.Response(200, json=_finnhub_profile("Apple Inc", 3_000_000.0))
    )
    respx.get(f"{FINNHUB_BASE}/stock/metric", params={"symbol": "AAPL", "metric": "all"}).mock(
        return_value=httpx.Response(200, json=_finnhub_metric(peExclExtraTTM=30.0, pbAnnual=45.0))
    )

    result = await server.get_fundamentals("AAPL")

    assert result["ticker"] == "AAPL"
    assert result["company_name"] == "Apple Inc"
    assert result["market_cap"] == 3_000_000.0 * 1_000_000
    assert result["pe_ratio"] == 30.0
    assert result["pb_ratio"] == 45.0
    assert result["dividend_yield"] == pytest.approx(0.02)


@pytest.mark.asyncio
@respx.mock
async def test_get_price_history_maps_yahoo_bars_sorted_by_date():
    # Timestamps intentionally out of order to verify get_daily_bars sorts by date.
    respx.get(YAHOO_CHART_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "chart": {
                    "result": [
                        {
                            "timestamp": [1790087400, 1790001000],  # 2026-09-22, 2026-09-21
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [2, 1],
                                        "high": [3, 2],
                                        "low": [1, 0.5],
                                        "close": [2.5, 1.5],
                                        "volume": [200, 100],
                                    }
                                ]
                            },
                        }
                    ]
                }
            },
        )
    )

    result = await server.get_price_history("AAPL", "1m")

    assert [bar["date"] for bar in result["bars"]] == ["2026-09-21", "2026-09-22"]
    assert result["bars"][0]["close"] == 1.5


@pytest.mark.asyncio
@respx.mock
async def test_get_company_news_maps_finnhub_articles():
    respx.get(f"{FINNHUB_BASE}/company-news").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "headline": "Some headline",
                    "summary": "Some summary",
                    "source": "Reuters",
                    "url": "https://example.com/a",
                    "datetime": 1758518400,
                }
            ],
        )
    )

    result = await server.get_company_news("AAPL", "2026-09-01", "2026-09-22")

    assert len(result["articles"]) == 1
    assert result["articles"][0]["headline"] == "Some headline"
    assert result["articles"][0]["datetime"].startswith("2025-09-22")


@respx.mock
def test_nasdaq_universe_refresh_serves_stale_cache_on_failure(tmp_path):
    path = tmp_path / "cache.json"
    stale_payload = {"fetched_at": 0.0, "rows": [{"symbol": "OLD", "name": "Old Co", "market_cap": 1.0}]}
    path.write_text(json.dumps(stale_payload))

    respx.get(NASDAQ_SCREENER_URL).mock(return_value=httpx.Response(500))

    import asyncio

    cache = asyncio.run(nasdaq_universe.refresh_cache(path=path))
    assert cache["rows"] == stale_payload["rows"]


@respx.mock
def test_nasdaq_universe_trims_rows_and_parses_market_cap(tmp_path):
    path = tmp_path / "cache.json"
    respx.get(NASDAQ_SCREENER_URL).mock(
        return_value=httpx.Response(
            200,
            json=_nasdaq_payload(
                [
                    {"symbol": "AAA", "name": "AAA Corp", "marketCap": "1,234,500"},
                    {"symbol": None, "name": "Bad Row", "marketCap": "1"},
                ]
            ),
        )
    )

    import asyncio

    cache = asyncio.run(nasdaq_universe.refresh_cache(path=path))
    assert cache["rows"] == [{"symbol": "AAA", "name": "AAA Corp", "market_cap": 1234500.0}]
