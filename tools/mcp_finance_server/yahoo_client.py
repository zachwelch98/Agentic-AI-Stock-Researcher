"""Yahoo Finance's unofficial chart API — used for price history only.

Neither Finnhub's free tier (`/stock/candle` returns 403) nor Financial
Modeling Prep's free tier (historical prices are restricted to a small
whitelist of symbols — confirmed by testing arbitrary regional-bank tickers,
which all returned 402) work for arbitrary tickers. Yahoo's undocumented
chart endpoint has no such restriction and needs no API key — same
"unofficial, needs a browser User-Agent" caveat as the NASDAQ screener
endpoint already used elsewhere in this codebase.
"""

from datetime import datetime, timezone

import httpx
from aiolimiter import AsyncLimiter

CHART_URL_TEMPLATE = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

RANGE_TO_YAHOO_RANGE = {"1m": "1mo", "3m": "3mo", "6m": "6mo", "1y": "1y", "2y": "2y"}

# Being polite to an unofficial/undocumented endpoint; no published rate limit.
_yahoo_limiter = AsyncLimiter(20, 60)


async def get_daily_bars(client: httpx.AsyncClient, symbol: str, range_: str) -> list[dict]:
    """Daily OHLCV bars for `symbol` over `range_` (1m/3m/6m/1y/2y), oldest -> newest."""
    yahoo_range = RANGE_TO_YAHOO_RANGE.get(range_, "6mo")
    async with _yahoo_limiter:
        resp = await client.get(
            CHART_URL_TEMPLATE.format(symbol=symbol),
            params={"range": yahoo_range, "interval": "1d"},
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=15.0,
        )
    resp.raise_for_status()
    payload = resp.json()
    results = payload.get("chart", {}).get("result") or []
    if not results:
        return []

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]

    bars = []
    for i, ts in enumerate(timestamps):
        values = {key: (quote.get(key) or [None] * len(timestamps))[i] for key in ("open", "high", "low", "close", "volume")}
        if any(v is None for v in values.values()):
            continue
        bars.append(
            {
                "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                "open": values["open"],
                "high": values["high"],
                "low": values["low"],
                "close": values["close"],
                "volume": values["volume"],
            }
        )
    bars.sort(key=lambda bar: bar["date"])
    return bars
