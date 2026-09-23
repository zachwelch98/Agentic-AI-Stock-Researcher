"""NASDAQ ticker-universe cache: fetch/refresh + ticker existence lookup.

Fetches `api.nasdaq.com/api/screener/stocks` (undocumented/unofficial, no API
key, needs a browser-like User-Agent), trims each row to
{symbol, name, market_cap}, and writes atomically to a JSON cache with a 24h
TTL. If a refresh fetch fails but a stale cache exists on disk, the stale data
is served with a warning rather than hard-failing.

Note: this endpoint used to also return a per-row `industry` field, which is
what the original design relied on for fuzzy industry matching. NASDAQ has
since dropped that field from the response entirely (only a coarse ~12-bucket
`sector` remains, and only as a server-side filter, not a per-row value) — so
industry -> ticker sourcing now happens via an LLM proposal step upstream
(see agents/planner.py's plan_screening_task), and this cache's job is
narrower: confirming a proposed ticker is real before spending a Finnhub call
on it, and providing an authoritative market cap.
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

CACHE_PATH = Path("data/nasdaq_universe_cache.json")
CACHE_TTL_HOURS = 24.0


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _read_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _is_stale(cache: dict, ttl_hours: float) -> bool:
    fetched_at = cache.get("fetched_at")
    if fetched_at is None:
        return True
    return (time.time() - fetched_at) > ttl_hours * 3600


def _trim_row(row: dict) -> dict | None:
    symbol = row.get("symbol")
    name = row.get("name")
    if not symbol or not name:
        return None
    market_cap_raw = row.get("marketCap") or "0"
    try:
        market_cap = float(str(market_cap_raw).replace(",", "").replace("$", "") or 0)
    except ValueError:
        market_cap = 0.0
    return {"symbol": symbol, "name": name, "market_cap": market_cap}


async def fetch_nasdaq_universe(client: httpx.AsyncClient) -> list[dict]:
    resp = await client.get(NASDAQ_SCREENER_URL, headers={"User-Agent": BROWSER_USER_AGENT}, timeout=30.0)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("data", {}).get("table", {}).get("rows", []) or []
    trimmed = [_trim_row(r) for r in rows]
    return [r for r in trimmed if r is not None]


async def refresh_cache(path: Path | None = None, ttl_hours: float = CACHE_TTL_HOURS) -> dict:
    if path is None:
        path = CACHE_PATH
    try:
        async with httpx.AsyncClient() as client:
            rows = await fetch_nasdaq_universe(client)
        cache = {"fetched_at": time.time(), "rows": rows}
        _atomic_write_json(path, cache)
        return cache
    except (httpx.HTTPError, ValueError) as exc:
        existing = _read_cache(path)
        if existing is not None:
            logger.warning("NASDAQ universe refresh failed (%s); serving stale cache from disk.", exc)
            return existing
        raise


async def get_universe(path: Path | None = None, ttl_hours: float = CACHE_TTL_HOURS) -> list[dict]:
    """Return the cached universe, refreshing first if the cache is missing or stale."""
    if path is None:
        path = CACHE_PATH
    cache = _read_cache(path)
    if cache is None or _is_stale(cache, ttl_hours):
        cache = await refresh_cache(path, ttl_hours)
    return cache["rows"]


def known_symbols(universe: list[dict]) -> set[str]:
    return {row["symbol"] for row in universe}
