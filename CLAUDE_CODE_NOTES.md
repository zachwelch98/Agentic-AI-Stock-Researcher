# Claude Code Workflow Notes

This file documents how Claude Code was used to build this project, as required by the
grading checkpoint.

## Process

- `PLAN.md` captures the full design, decided in an interview with the user before any
  code was written (orchestration pattern, screening formula, data sources, MCP
  boundary, model tiering).
- The build follows the 6-phase plan in `PLAN.md` in order: scaffold → graph skeleton
  with stubbed tools → MCP integration → FastAPI layer → tracing/eval → containerize.
- Each phase's tests are written and run before moving to the next phase, per the
  Verification plan in `PLAN.md`.

## Notable decisions made during implementation

### Phase 3 (MCP integration): two data-source deviations from PLAN.md

PLAN.md's data-source choices were made in the original design interview but hadn't
been tested against live APIs yet. Live smoke-testing during Phase 3 surfaced two
places where the real APIs no longer behave as documented, requiring in-flight design
changes (confirmed with the user before implementing):

1. **NASDAQ's public screener no longer returns per-row `industry`.** The plan's fuzzy
   industry-matching design assumed `api.nasdaq.com/api/screener/stocks` still returned
   a fine-grained `industry` field per ticker (as it apparently once did). Live testing
   showed the field is always `null` now — the API only supports a coarse ~12-bucket
   `sector` as a server-side filter, not a queryable per-row industry. Fix: ticker
   sourcing for `screen_industry` moved upstream to an LLM proposal step (fits the
   plan's already-designed `used_llm_fallback_tickers` path — see PLAN.md's NASDAQ
   cache section — which is now the *standard* path rather than a rare fallback, since
   there's no structured taxonomy left to match against). The MCP tool itself changed
   from `screen_industry(industry: str)` to `screen_industry(industry: str,
   candidate_tickers: list[str])`: it now does real, load-bearing, deterministic work
   (Finnhub fundamentals fetch + composite scoring for a given ticker list, cross-checked
   against the NASDAQ universe cache to drop hallucinated tickers before spending a
   Finnhub call), while the "which tickers plausibly belong to this industry" judgment
   — which is inherently fuzzy — is an LLM's job, not a deterministic tool's.
   `nasdaq_universe.py`'s fuzzy-matching helpers (`rapidfuzz`-based) were removed as
   dead code along with the now-unused `rapidfuzz` dependency.

2. **Finnhub's free tier no longer includes historical OHLCV candles**, and this took
   three attempts to work around, in order:
   - `/stock/candle` returns 403 ("You don't have access to this resource.") on the free
     tier — needed for `get_price_history` and, downstream, the technical-analysis
     researcher's SMA/EMA/RSI/MACD.
   - The plan's other free/no-key fallback candidate, Stooq's CSV endpoint, is now
     behind a JS proof-of-work bot check that a plain HTTP client can't clear.
   - Financial Modeling Prep's `historical-price-eod` endpoint looked like a working
     replacement in an initial spot-check (AAPL/JPM both returned clean data), so it
     was wired in and used through most of Phase 4. A full live pipeline run later
     revealed the actual behavior: FMP's free tier restricts historical prices to a
     small whitelist of well-known symbols — AAPL works, but *every* regional-bank
     ticker tested, including large ones like USB/PNC/TFC, returned 402 "This value
     set for 'symbol' is not available under your current subscription." Since this
     project needs price history for arbitrary LLM-proposed tickers, not a fixed
     whitelist, FMP wasn't actually viable once real (non-mega-cap) tickers were
     exercised. This is a good example of why the plan's phased approach (smoke-test
     against a couple of real symbols, but also do a full live end-to-end run before
     considering a data source "verified") matters — a narrow spot-check passed but
     didn't reveal the whitelist restriction.
   - Final fix: `get_price_history` (and the technical researcher's local indicator
     tool) use Yahoo Finance's unofficial chart API
     (`query1.finance.yahoo.com/v8/finance/chart/{symbol}`) instead — no API key, no
     whitelist, verified against arbitrary regional-bank tickers. Same
     "unofficial/undocumented, needs a browser User-Agent" caveat as the NASDAQ
     screener endpoint elsewhere in this codebase. Finnhub remains the source for
     profile/fundamentals/news/quote; FMP is no longer used anywhere in this project.

Both changes were verified against live data (not just respx mocks) before locking
them in: real Finnhub `/stock/metric` field names, real FMP OHLCV bars, and a full
protocol round-trip through `MultiServerMCPClient` (both the custom finance server and
the official Fetch server booting together over stdio) were exercised directly.

### Phase 4 (real LLM + MCP wiring, FastAPI layer): four issues found by live testing

Phase 2's stub node functions let the graph-routing tests stay fast and offline, but
they also meant nothing had exercised a real Anthropic call, a real MCP tool call from
inside a graph node, or the FastAPI process lifecycle end-to-end. Standing up a real
`uv run uvicorn app.main:app` and driving it with `curl` (rather than trusting the code
looked right) surfaced four real bugs, in order:

1. **`MultiServerMCPClient.get_tools()` spawns a fresh stdio subprocess session per tool
   call** — its own docstring says so ("a new session will be created for each tool
   call"). That silently violates the "both servers spawned once, not per-request"
   requirement even though the code reads as if it reuses one client. Fix:
   `tools/mcp_client.py`'s `open_persistent_mcp_sessions()` instead opens one long-lived
   session per server via `client.session(name)`, held open for the app's lifetime by an
   `AsyncExitStack` in `app/main.py`'s lifespan, and loads tools bound to those sessions.
   Verified by timing repeated calls (subprocess spawn is ~1s; reused-session calls were
   ~0.15-0.4s, dominated by real Finnhub latency, not process startup).

2. **A stale, corrupted `ANTHROPIC_API_KEY` in the shell profile shadowed the correct
   `.env` value.** `python-dotenv`'s `load_dotenv()` does not override existing
   environment variables by default, so an ambient shell export (ending, bizarrely, in
   the literal text `"export"` — looks like two `export FOO=...` lines got merged
   without a newline at some point) silently won over the correct project `.env` value,
   producing a 401 that looked like a bad key when the `.env` key was actually fine.
   Fixed with `load_dotenv(override=True)` in `app/config.py`, so this project's `.env`
   is always authoritative regardless of what's ambient in the shell.

3. **`claude-sonnet-5` rejects the `temperature` parameter** ("`temperature` is
   deprecated for this model") — `app/llm.py` originally passed `temperature=0` on both
   `ChatAnthropic` instances (a very standard thing to do for deterministic-ish
   structured-output calls on older model generations). Removed entirely; the Claude 5
   family doesn't accept it.

4. **A large nested list-of-objects structured-output schema made Sonnet emit a
   malformed tool call.** `plan_screening_task`'s original `ScreeningPlanOutput` schema
   asked for `proposed_tickers: list[{ticker, company_name}]` at 15-20 items alongside a
   separate `reasoning: str` field. Live testing showed Sonnet reliably (not
   intermittently) stringified and duplicated the whole payload into the
   `proposed_tickers` field and dropped `reasoning` entirely — a real pydantic
   `ValidationError`, not a flaky one-off. Isolated by re-running the same call with
   `include_raw=True` to inspect the raw tool-call arguments directly. Fix: flattened
   the schema to `proposed_tickers: list[str]` (company names weren't used downstream
   anyway — the MCP tool re-fetches them from Finnhub). Notably, two *other* nested
   list-of-objects schemas in this codebase — `plan_research_tasks`'s 12-item
   `{ticker, domain, focus_notes}` list, and the synthesizer's 3-item
   `{ticker, overall_take, strengths: [...], risks: [...]}` list (doubly nested) — were
   tested at realistic size and worked fine, so this isn't "nested schemas are broken
   in general," specifically this one shape+size combination.

After these four fixes, a full live run (`POST /research` for "Regional Banks" through
a real `uvicorn` process, no mocks) was driven end-to-end: LLM ticker proposal → MCP
`screen_industry` fetch+score → LLM judgment picking the final 3 → LLM research-task
planning → 3 parallel tickers × 4 sequential tool-calling researcher agents → LLM
synthesis into the comparative report — see the job result captured during that run for
the first fully real pipeline output.
