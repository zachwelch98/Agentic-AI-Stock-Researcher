"""Direct HTTP client for SEC EDGAR: ticker -> CIK cache + submissions JSON.

Called directly (not through Fetch MCP) since these are already structured
JSON endpoints, not pages needing text extraction. SEC requires a descriptive
`User-Agent` identifying the requester and enforces a strict rate limit
(<= 10 req/s); both are respected here.
"""

import html
import json
import os
import re
import tempfile
import time
from pathlib import Path

import httpx
from aiolimiter import AsyncLimiter
from langchain_core.tools import tool

from app.config import get_settings

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"

CIK_CACHE_PATH = Path("data/sec_ticker_cik_cache.json")
CIK_CACHE_TTL_HOURS = 24.0

_edgar_limiter = AsyncLimiter(9, 1)


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


def _edgar_headers() -> dict[str, str]:
    return {"User-Agent": get_settings().sec_edgar_user_agent}


async def _fetch_ticker_cik_map(client: httpx.AsyncClient) -> dict[str, str]:
    async with _edgar_limiter:
        resp = await client.get(TICKER_CIK_URL, headers=_edgar_headers(), timeout=30.0)
    resp.raise_for_status()
    raw = resp.json()
    return {
        str(entry["ticker"]).upper(): str(entry["cik_str"]).zfill(10)
        for entry in raw.values()
    }


async def get_ticker_cik_map(
    path: Path | None = None, ttl_hours: float = CIK_CACHE_TTL_HOURS
) -> dict[str, str]:
    """Ticker (upper-cased) -> zero-padded 10-digit CIK, refreshing the cache if stale."""
    if path is None:
        path = CIK_CACHE_PATH
    cache = _read_cache(path)
    if cache is not None and not _is_stale(cache, ttl_hours):
        return cache["ticker_to_cik"]

    async with httpx.AsyncClient() as client:
        try:
            mapping = await _fetch_ticker_cik_map(client)
        except httpx.HTTPError:
            if cache is not None:
                return cache["ticker_to_cik"]
            raise

    _atomic_write_json(path, {"fetched_at": time.time(), "ticker_to_cik": mapping})
    return mapping


async def get_cik_for_ticker(ticker: str) -> str | None:
    mapping = await get_ticker_cik_map()
    return mapping.get(ticker.upper())


async def get_submissions(cik: str, client: httpx.AsyncClient | None = None) -> dict:
    """Fetch the SEC EDGAR submissions JSON for a given (10-digit, zero-padded) CIK."""
    url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik)
    owns_client = client is None
    client = client or httpx.AsyncClient()
    try:
        async with _edgar_limiter:
            resp = await client.get(url, headers=_edgar_headers(), timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    finally:
        if owns_client:
            await client.aclose()


async def get_submissions_for_ticker(ticker: str) -> dict | None:
    cik = await get_cik_for_ticker(ticker)
    if cik is None:
        return None
    return await get_submissions(cik)


def select_filings(recent: dict, forms: list[str] | None = None, limit: int = 15) -> list[dict]:
    """Turn EDGAR's parallel-array `filings.recent` into filing dicts, newest first,
    optionally restricted to `forms` (applied *before* the limit — the most recent
    filings are dominated by Form 4s, which would otherwise hide proxy statements)."""
    wanted = {f.strip().upper() for f in forms} if forms else None
    filings = []
    for form, filing_date, accession_number, primary_document, cik in zip(
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
        recent.get("_cik", []),
    ):
        if wanted and form.upper() not in wanted:
            continue
        accession_no_dashes = accession_number.replace("-", "")
        filings.append(
            {
                "form": form,
                "filing_date": filing_date,
                "document_url": (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_document}"
                ),
            }
        )
        if len(filings) >= limit:
            break
    return filings


@tool
async def get_sec_filings(ticker: str, forms: list[str] | None = None, limit: int = 15) -> dict:
    """Fetch recent SEC EDGAR filings for a US-listed ticker. Optionally pass
    `forms` (e.g. ["10-K", "10-Q"], ["DEF 14A"], ["4"], ["8-K"]) to list only
    those form types — do this to find proxy statements, insider (Form 4)
    filings or 8-Ks, which are buried in the unfiltered list. Each filing has a
    `document_url` to pass to read_sec_filing to read the actual filing text."""
    cik = await get_cik_for_ticker(ticker)
    if cik is None:
        return {"error": f"No SEC EDGAR CIK found for ticker {ticker}"}

    submissions = await get_submissions(cik)
    recent = dict(submissions.get("filings", {}).get("recent", {}))
    cik_no_leading_zeros = str(int(cik))
    recent["_cik"] = [cik_no_leading_zeros] * len(recent.get("form", []))

    return {
        "ticker": ticker,
        "company_name": submissions.get("name"),
        "recent_filings": select_filings(recent, forms, limit),
    }


_SEC_ARCHIVE_PREFIX = "https://www.sec.gov/Archives/"


def html_to_text(raw: str) -> str:
    # Inline-XBRL filings open with a huge hidden <ix:header> of context/unit metadata.
    raw = re.sub(r"(?is)<(script|style|ix:header)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


@tool
async def read_sec_filing(url: str, max_chars: int = 6000, start_index: int = 0, find: str = "") -> str:
    """Read the text of an SEC filing at a `document_url` returned by
    get_sec_filings (only https://www.sec.gov/Archives/... URLs). Returns up to
    `max_chars` characters starting at `start_index`. To jump straight to a
    section of a long filing (10-K/10-Q/proxy), pass `find` — a phrase such as
    "Results of Operations", "Total liabilities" or "Executive Officers" — and
    reading starts at its first occurrence; otherwise page with start_index."""
    if not url.startswith(_SEC_ARCHIVE_PREFIX):
        return "refused: read_sec_filing only reads https://www.sec.gov/Archives/ document URLs."
    try:
        async with _edgar_limiter:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(url, headers=_edgar_headers(), timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        return f"Failed to fetch {url}: {exc}"
    text = html_to_text(resp.text)
    if find:
        hit = text.lower().find(find.lower())
        if hit < 0:
            return f"Phrase {find!r} not found in {url} ({len(text)} chars); try another phrase or start_index."
        start_index = max(hit - 200, 0)
    chunk = text[start_index : start_index + max_chars]
    more = f" [{len(text) - start_index - len(chunk)} more chars; call again with start_index={start_index + len(chunk)}]" if start_index + len(chunk) < len(text) else ""
    return f"Contents of {url} (chars {start_index}-{start_index + len(chunk)} of {len(text)}): {chunk}{more}"
