# Agentic AI Stock Researcher — Implementation Plan

## Context

This is a graded weekend checkpoint (build + technical exam + Loom walkthrough) for an AI
engineering course, and simultaneously the user's first hands-on project in building an
agentic AI system with LangGraph. The grading PDF locks the stack (Python, LangGraph,
LangChain, FastAPI, MCP, Claude Code, LangSmith) and a required architecture shape (Planner +
Researcher(s) + Synthesizer graph, ≥1 MCP-backed tool, FastAPI service, LangSmith tracing,
a lightweight eval harness, Docker, and Claude-Code-built with documented workflow notes).

On top of that scaffold, an extensive requirements interview with the user (documented in
this conversation) landed on a specific product: the user either names an industry or a
specific ticker. If given an industry, the system screens it down to the single most
undervalued stock using a deterministic fundamentals-based composite score plus an
LLM-judgment step; if given a ticker directly, that screening is skipped entirely. Either
way, the system runs a 4-domain deep-dive (news, financials, leadership, technical
analysis) on that one ticker and returns one citation-backed research report and
recommendation. Every architectural choice below (orchestration pattern, screening
formula, data sources, MCP boundary, model tiering, etc.) was explicitly decided in that
interview — this plan turns those decisions into a concrete, executable build, not a set of
open questions.

(The build originally targeted the top 3 undervalued stocks with one comparative report
across them; it was narrowed to a single ticker — either screened or user-supplied — after
the initial build was complete. This plan reflects the current, single-ticker design.)

The repo now has a full working implementation matching the design below — `agents/`,
`app/`, `tools/`, `tests/`, `eval/` are real, built code, not just this design doc.

## Repo layout

```
agentic-ai-stock-researcher/
├── README.md                       # problem statement, Mermaid graph diagram, setup, tradeoffs, eval results
├── CLAUDE_CODE_NOTES.md
├── Dockerfile
├── docker-compose.yml              # optional
├── .env.example
├── .dockerignore / .gitignore      # excludes data/*.json caches, .env, __pycache__, .venv
├── pyproject.toml / uv.lock
├── app/
│   ├── main.py                     # FastAPI app factory + lifespan (spawns MCP subprocesses, primes NASDAQ cache)
│   ├── config.py                   # env-driven settings, model IDs, rate-limit constants
│   ├── api/routes.py               # POST /research, GET /jobs/{job_id}, GET /healthz
│   ├── api/schemas.py              # Pydantic request/response models, JobStatus enum
│   ├── jobs.py                     # in-memory job store + async background runner
│   ├── graph/state.py              # ResearchState, TickerResearchState, typed sub-shapes
│   ├── graph/build.py              # StateGraph + ticker subgraph construction/compile
│   ├── graph/routing.py            # conditional edge fns: route_from_start, route_after_screen, fan_out_to_ticker_research
│   └── static/index.html, app.js   # minimal polling frontend, industry-or-ticker input
├── agents/
│   ├── planner.py                  # plan_screening_task, plan_research_tasks
│   ├── screener.py                 # deterministic_screen, llm_judgment_screen, accept_user_ticker, insufficient_candidates, scoring
│   ├── researchers/news.py, financials.py, leadership.py, technical.py
│   └── synthesizer.py
├── tools/
│   ├── mcp_client.py                # MultiServerMCPClient wiring for both MCP servers
│   ├── edgar.py                     # direct HTTP: ticker→CIK cache, submissions JSON
│   ├── technical_indicators.py      # `ta`-library wrapper (SMA/EMA/RSI/MACD/volume trend)
│   └── mcp_finance_server/          # the CUSTOM MCP server (own runnable package)
│       ├── server.py                # FastMCP app + 4 tools, stdio entrypoint
│       ├── finnhub_client.py        # rate-limited async Finnhub client
│       ├── nasdaq_universe.py       # cache fetch/refresh/query + fuzzy industry matching
│       └── schemas.py               # tool I/O Pydantic models (source of README schema docs)
├── eval/golden_set.json, run_eval.py
├── traces/README.md, regional_banks_run.png   # LangSmith screenshot added in Phase 4
├── data/.gitkeep, nasdaq_universe_cache.json, sec_ticker_cik_cache.json   # generated, gitignored
├── tests/test_screener.py, test_graph_routing.py, test_mcp_tools.py, test_api.py
└── scripts/refresh_nasdaq_cache.py  # standalone manual/cron cache refresh
```

## LangGraph design

### State (`app/graph/state.py`)

```python
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

class DomainFinding(TypedDict):
    ticker: str
    domain: str
    summary: str
    citations: list[dict]      # {source_id, identifier, title}
    raw_data: dict

class ResearchState(TypedDict):
    industry_query: str
    job_id: str
    user_supplied_ticker: str | None   # set when the caller names a ticker directly, bypassing screening
    screening_plan: dict | None
    screened_candidates: list[CandidateFundamentals]
    used_llm_fallback_tickers: bool
    final_candidates: list[str]        # always exactly 1 ticker once populated
    screener_justification: str | None
    research_tasks: list[ResearchTask]
    domain_findings: Annotated[list[DomainFinding], operator.add]   # reducer: Send-fanned branches write here
    final_report: dict | None
    status: Literal["planning", "screening", "insufficient_candidates",
                     "researching", "synthesizing", "done", "failed"]
    error: str | None

class TickerResearchState(TypedDict):
    ticker: str
    research_tasks: list[ResearchTask]
    findings: Annotated[list[DomainFinding], operator.add]
```

`domain_findings` **must** carry the `operator.add` reducer — even with a single ticker,
the 4 domain researchers running downstream of `Send` write to the same key, and without
the reducer this raises `InvalidUpdateError`. The fan-out itself (`Send` per ticker in
`final_candidates`) is written generically over the list, so it happens to also support
N>1 tickers, but the pipeline only ever populates `final_candidates` with one.

### Nodes

| Node | Type | Model | File |
|---|---|---|---|
| `plan_screening_task` | LLM, structured output | Sonnet | `agents/planner.py` |
| `deterministic_screen` | pure code, calls MCP `screen_industry` | — | `agents/screener.py` |
| `llm_judgment_screen` | LLM, structured output, picks 1 ticker | Sonnet | `agents/screener.py` |
| `accept_user_ticker` | pure code, bypasses screening entirely | — | `agents/screener.py` |
| `insufficient_candidates` | pure code, terminal | — | `agents/screener.py` |
| `plan_research_tasks` | LLM, structured output, builds a 1×4 grid | Sonnet | `agents/planner.py` |
| `run_ticker_research` | async wrapper invoking a compiled subgraph | — | `app/graph/build.py` |
| `news_researcher` / `financials_researcher` / `leadership_researcher` / `technical_researcher` | LLM, tool-calling (subgraph-internal nodes) | Haiku | `agents/researchers/*.py` |
| `synthesizer` | LLM, structured output | Sonnet | `agents/synthesizer.py` |

### Edges (supervisor pattern, deterministic routing — no LLM spent on orchestration itself)

```
START --[conditional: route_from_start]--> {
    accept_user_ticker     (caller supplied a specific ticker — screening skipped)
    plan_screening_task    (caller supplied an industry)
}
plan_screening_task → deterministic_screen
deterministic_screen --[conditional: route_after_screen]--> {
    insufficient_candidates   (if fewer than 3 candidates survive the deterministic screen)
    llm_judgment_screen       (otherwise)
}
llm_judgment_screen → plan_research_tasks
accept_user_ticker → plan_research_tasks
plan_research_tasks --[Send fan-out: fan_out_to_ticker_research]--> run_ticker_research  (one Send, for the single final candidate)
run_ticker_research → synthesizer   (LangGraph joins the Send branch before firing this edge)
synthesizer → END
insufficient_candidates → END
```

```python
def route_from_start(state: ResearchState) -> str:
    return "accept_user_ticker" if state.get("user_supplied_ticker") else "plan_screening_task"

def route_after_screen(state: ResearchState) -> str:
    return "insufficient_candidates" if len(state["screened_candidates"]) < 3 else "llm_judgment_screen"

graph.add_conditional_edges(START, route_from_start,
    {"accept_user_ticker": "accept_user_ticker", "plan_screening_task": "plan_screening_task"})
graph.add_conditional_edges("deterministic_screen", route_after_screen,
    {"insufficient_candidates": "insufficient_candidates", "llm_judgment_screen": "llm_judgment_screen"})
```

The `< 3` minimum-pool-size gate in `route_after_screen` is intentionally left unchanged
even though only 1 ticker is ultimately selected: it's a screen-quality bar (a real pool
of alternatives should exist before trusting a single pick), not a proxy for how many
tickers get selected.

### The single-ticker fan-out (reused across both entry paths)

Build the 4 researcher nodes once, as an inner parallel fan-out/join
`StateGraph(TickerResearchState)`, compiled a single time and invoked once per run via
`Send` — this is what keeps it 4 pieces of node code, reusable regardless of which entry
path populated `final_candidates`. The 4 researchers are fully independent (none reads
another's output, each only writes to the shared `findings` reducer), so they're wired to
run concurrently rather than as a chain:

```python
ticker_subgraph = StateGraph(TickerResearchState)
researcher_names = ["news_researcher", "financials_researcher", "leadership_researcher", "technical_researcher"]
for name in researcher_names:
    ticker_subgraph.add_node(name, ...)
for name in researcher_names:
    ticker_subgraph.add_edge(START, name)
    ticker_subgraph.add_edge(name, END)
compiled_ticker_graph = ticker_subgraph.compile()

async def run_ticker_research(input_state: TickerResearchState) -> dict:
    result = await compiled_ticker_graph.ainvoke(input_state)
    return {"domain_findings": result["findings"]}

graph.add_node("run_ticker_research", run_ticker_research)
```

```python
from langgraph.types import Send

def fan_out_to_ticker_research(state: ResearchState) -> list[Send]:
    return [
        Send("run_ticker_research", {
            "ticker": ticker,
            "research_tasks": [t for t in state["research_tasks"] if t["ticker"] == ticker],
        })
        for ticker in state["final_candidates"]
    ]

graph.add_conditional_edges("plan_research_tasks", fan_out_to_ticker_research, ["run_ticker_research"])
```

### Deterministic screening score (`agents/screener.py`)

Industry-agnostic composite: +1 if P/E (positive only) below industry median, +1 if P/B below
median, +1 if Debt/Equity below median, +0.5 bonus if dividend yield > 0 (not a filter — this
is intentional so REITs, which lean on yield rather than P/E, aren't unfairly penalized).
Candidates are pre-filtered to market cap ≥ $300M and avg daily dollar volume ≥ $1M. Rank
descending by score (tie-break: market cap), keep top ~10-15 for the LLM judgment step.

## MCP layer

**Custom MCP server** (`tools/mcp_finance_server/`), built with the official `mcp` SDK's
`FastMCP` (decorator-based; auto-generates the JSON tool schema that goes straight into the
README):

- `screen_industry(industry: str) -> list[dict]` — fuzzy-matches `industry` against the
  cached NASDAQ taxonomy, applies the cap/volume filter, pulls Finnhub fundamentals, returns
  the ranked top ~15. Returns `[]` on no taxonomy match (signals the LLM-fallback path).
- `get_fundamentals(ticker: str) -> dict`
- `get_price_history(ticker: str, range: str) -> list[dict]` (OHLCV from Finnhub)
- `get_company_news(ticker: str, from_date: str, to_date: str) -> list[dict]` (Finnhub
  company-news; needed for the News researcher, so it's the natural 4th tool alongside the 3
  originally scoped)

**Official Fetch MCP server** (`mcp-server-fetch`, pip-installed, launched via
`python -m mcp_server_fetch` rather than `uvx` — keeps the Docker image from needing a
separate installer at runtime) — used for genuinely unstructured content: news article
bodies, IR leadership pages, and 10-K/10-Q excerpts for the financials researcher's red-flag
skim.

**SEC EDGAR submissions JSON** is called directly via `tools/edgar.py` (plain async `httpx`,
own rate limiter, required descriptive `User-Agent` header) rather than through Fetch MCP,
since it's already structured JSON, not a page needing text extraction. The custom finance
MCP server unambiguously satisfies the "≥1 MCP-backed tool" requirement; Fetch MCP still does
real, load-bearing work for unstructured sources.

**Runtime wiring**: both servers are stdio subprocesses spawned **once**, in FastAPI's
`lifespan` context manager, not per-request. Tools are fetched once via
`langchain-mcp-adapters`' `MultiServerMCPClient` and cached on `app.state.mcp_tools`; agent
node functions pull them from there.

## NASDAQ ticker-universe cache

`tools/mcp_finance_server/nasdaq_universe.py`: fetches
`api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000` (no key, needs a
browser-like `User-Agent`), trims to `{symbol, name, sector, industry, marketCap}`, writes
atomically to `data/nasdaq_universe_cache.json` with a 24h TTL. Refreshed eagerly at FastAPI
startup, checked lazily inside `screen_industry`, and available standalone via
`scripts/refresh_nasdaq_cache.py`. If a refresh fetch fails but a stale cache exists, serve
the stale data with a warning rather than hard-failing (the endpoint is undocumented/
unofficial). Industry-name matching uses `rapidfuzz.process.extractOne` against the cached
taxonomy labels (accept threshold: score ≥ 70); below that, `screen_industry` returns `[]`
and `deterministic_screen` falls back to asking the LLM for plausible tickers, setting
`used_llm_fallback_tickers=True` so the Synthesizer must flag this in the final report as
"LLM-generated candidate list, not sourced from market data." SEC's ticker→CIK mapping is
cached the same way in `data/sec_ticker_cik_cache.json`.

## FastAPI layer

Async, in-memory job store (plain dict keyed by `job_id` — fine for v1, doesn't survive
restarts, must run uvicorn with a single worker since the dict isn't shared across
processes). `POST /research` accepts a Pydantic-validated payload with exactly one of
`industry` or `ticker` (both-set or neither-set is 422, matching the existing
empty-string-is-422 behavior) — 202 + `job_id` on success. Supplying `ticker` bypasses
industry screening entirely and routes straight into the research subgraph. `GET
/jobs/{job_id}` returns status + result, 404 for unknown ids. `GET /healthz` is a trivial
sync 200 (defensible: zero I/O liveness probe). Every request logs route/status/duration/
job_id/tool errors. Static frontend (`app/static/`) polls `/jobs/{job_id}` every ~1.5s and
renders per-agent status live; it exposes both an industry field and a ticker field so
either entry path is reachable from the UI.

## Golden eval set (`eval/golden_set.json`)

Fixed, author-written cases chosen to stress different parts of the pipeline:

1. **Regional Banks** — baseline, abundant clean data
2. **Direct ticker (JPM)** — exercises the ticker-bypass entry path directly, skipping
   screening
3. **Semiconductor Equipment** — cyclical, tests screen vs. cyclical-decline mispricing
4. **Specialty Retail** — tests avoiding "value trap" picks
5. **Genomic/Biotech Research** — deliberately hostile (many negative-earnings names, thin
   leadership data) — exercises the `insufficient_candidates` conditional-edge path
6. **REITs** — different valuation convention (yield/FFO over P/E) — tests the scoring
   formula isn't overfit to case #1

`eval/run_eval.py` POSTs each case to `/research` (by `industry` or `ticker`, whichever the
case specifies), polls to completion, runs a structural check (exactly 1 final candidate,
≥3 citations across its findings, all 4 domains present, or a valid
`insufficient_candidates` outcome for the biotech case), prints a pass-rate summary.

## Key library choices

- Dependency manager: **uv** (`pyproject.toml` + `uv.lock`)
- **fastapi**, **uvicorn[standard]**
- **langgraph** (`StateGraph`, `Send` from `langgraph.types`, subgraph-as-node)
- **langchain-anthropic** (`ChatAnthropic`, `.with_structured_output(...)` on every LLM node)
- **mcp** (official SDK, `FastMCP`) + **langchain-mcp-adapters** (`MultiServerMCPClient`)
- **mcp-server-fetch** (official reference Fetch server, pip-installed)
- **httpx** (async) for Finnhub/NASDAQ/SEC EDGAR
- **rapidfuzz** for industry-name fuzzy matching
- **ta** for technical indicators (not `pandas-ta` — effectively unmaintained, breaks on
  numpy ≥1.24)
- **aiolimiter** — shared `AsyncLimiter(55, 60)` for Finnhub across all 12 parallel
  researcher calls, `AsyncLimiter(9, 1)` for SEC EDGAR
- **langsmith** (via `langchain-core`, env-var configured)
- **pytest**, **pytest-asyncio**, **respx** (httpx mocking) for tests
- **pydantic** v2 throughout

Models: **Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) for the 4 researcher roles;
**Claude Sonnet 5** (`claude-sonnet-5`) for Planner, Screener's LLM-judgment step, and
Synthesizer.

## Build phasing

1. **Scope + scaffold** — `uv init`, pinned deps, directory skeleton, `.env.example`,
   README/CLAUDE_CODE_NOTES stubs.
2. **State + graph skeleton with stubbed tools** — build the full graph (including the
   conditional edge and `Send` fan-out) end-to-end with fake tool returns; lock in routing
   behavior with `tests/test_graph_routing.py` before wiring real tools. Highest-risk piece —
   validate in isolation first.
3. **MCP integration** — build `tools/mcp_finance_server/`, sign up for a free Finnhub key,
   wire the Fetch server, swap stubbed tool calls for real MCP invocations. Smoke-test the
   custom server standalone with the MCP Inspector before touching the graph.
4. **FastAPI layer** — job store, schemas, routes, lifespan (spawns MCP subprocesses, primes
   NASDAQ cache), static polling frontend.
5. **Tracing + eval** — enable LangSmith env vars, run one real Regional Banks query,
   screenshot the trace to `traces/`, write and run `eval/golden_set.json` +
   `eval/run_eval.py`, capture results in the README.
6. **Containerize + ship** — Dockerfile, `docker build`/`docker run` smoke test, finalize
   README (Mermaid diagram, setup, tradeoffs, eval results) and CLAUDE_CODE_NOTES.md.

## Verification plan

**Bootstrap**
```
uv sync
cp .env.example .env   # ANTHROPIC_API_KEY, FINNHUB_API_KEY, LANGCHAIN_API_KEY, SEC_EDGAR_USER_AGENT
uv run python scripts/refresh_nasdaq_cache.py   # confirm cache file + row count
```

**MCP server standalone**: `npx @modelcontextprotocol/inspector uv run python -m tools.mcp_finance_server.server`, manually invoke `screen_industry("Regional Banks")` before it's touched by the graph.

**Unit tests**
```
uv run pytest tests/test_screener.py        # scoring math + llm_judgment_screen/accept_user_ticker, no network
uv run pytest tests/test_graph_routing.py   # route_from_start (industry vs. ticker), <3 candidates -> insufficient_candidates,
                                             # >=3 -> final_candidates has exactly 1 entry, domain_findings has exactly 4
uv run pytest tests/test_mcp_tools.py       # 4 tools against respx-mocked Finnhub/NASDAQ responses
```

**End-to-end local**
```
uv run uvicorn app.main:app --reload
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/research -H "Content-Type: application/json" -d '{"industry":"Regional Banks"}'
curl -s -X POST localhost:8000/research -H "Content-Type: application/json" -d '{"ticker":"JPM"}'
curl -s -X POST localhost:8000/research -H "Content-Type: application/json" -d '{"industry":"Regional Banks","ticker":"JPM"}'   # expect 422
curl -s localhost:8000/jobs/<job_id>                      # poll both jobs above until "done"; confirm final_candidates has exactly 1 ticker
curl -s localhost:8000/jobs/does-not-exist                # expect 404
curl -s -X POST localhost:8000/research -d '{}'           # expect 422
```
Also run the Genomic/Biotech Research case to confirm the `insufficient_candidates` path
fires and `run_ticker_research` never runs. Open `http://localhost:8000/` and confirm the
polling UI shows live per-agent status for both the industry and direct-ticker inputs.

**LangSmith**: with tracing enabled, re-run Regional Banks; confirm the nested span tree
(planner → deterministic_screen with the `screen_industry` MCP tool-call span visible →
llm_judgment_screen → plan_research_tasks → one `run_ticker_research` span, with 4
concurrent researcher spans (started together, not chained) and their own tool-call
children → synthesizer). Then re-run
with a direct ticker (e.g. `JPM`) and confirm the trace starts at `accept_user_ticker`,
skipping `plan_screening_task`/`deterministic_screen`/`llm_judgment_screen` entirely.
Screenshot the industry-path trace to `traces/regional_banks_run.png`.

**Eval harness**
```
uv run uvicorn app.main:app &
uv run python eval/run_eval.py --base-url http://localhost:8000
```

**Docker**
```
docker build -t stock-researcher .
docker run --rm -p 8000:8000 --env-file .env stock-researcher
```
Re-run the `/healthz` and `/research` checks against the container; confirm clean startup of
both MCP stdio subprocesses in the logs.
