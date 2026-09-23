"""Rate-limited async Finnhub client.

Shares one `AsyncLimiter(55, 60)` across all Finnhub calls (free tier: 60
calls/min) regardless of how many parallel researcher branches are calling
concurrently — LangGraph's `Send` fan-out means up to 12 researcher calls can
be in flight at once.
"""

import httpx
from aiolimiter import AsyncLimiter

from app.config import get_settings

BASE_URL = "https://finnhub.io/api/v1"

_finnhub_limiter = AsyncLimiter(55, 60)


class FinnhubClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or get_settings().finnhub_api_key

    async def _get(self, client: httpx.AsyncClient, path: str, params: dict) -> dict | list:
        request_params = {**params, "token": self.api_key}
        async with _finnhub_limiter:
            resp = await client.get(f"{BASE_URL}{path}", params=request_params, timeout=15.0)
        resp.raise_for_status()
        return resp.json()

    async def get_profile(self, client: httpx.AsyncClient, symbol: str) -> dict:
        return await self._get(client, "/stock/profile2", {"symbol": symbol})

    async def get_basic_financials(self, client: httpx.AsyncClient, symbol: str) -> dict:
        return await self._get(client, "/stock/metric", {"symbol": symbol, "metric": "all"})

    async def get_quote(self, client: httpx.AsyncClient, symbol: str) -> dict:
        return await self._get(client, "/quote", {"symbol": symbol})

    async def get_company_news(
        self, client: httpx.AsyncClient, symbol: str, from_date: str, to_date: str
    ) -> list[dict]:
        result = await self._get(client, "/company-news", {"symbol": symbol, "from": from_date, "to": to_date})
        return result if isinstance(result, list) else []
