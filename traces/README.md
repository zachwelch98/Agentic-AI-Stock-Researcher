# LangSmith Tracing

## Setup

Set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` in `.env`
(see `.env.example`). `app/config.py` auto-disables tracing if `LANGCHAIN_TRACING_V2=true`
but no key is set, so local dev without a LangSmith account doesn't spam auth errors on
every LLM call.

Tracing is otherwise "free" here — LangChain reads the `LANGCHAIN_*` env vars directly and
instruments every `ChatAnthropic` call and MCP tool call automatically; nothing in this
codebase calls the LangSmith SDK directly except this eval/verification tooling.

## What the trace tree looks like

For one `/research` request, LangSmith should show a single root run (`LangGraph`)
containing, in order:

```
LangGraph (root)
├── plan_screening_task          (Sonnet, structured output)       [industry path only]
├── deterministic_screen         (screen_industry MCP tool-call span)
├── llm_judgment_screen          (Sonnet, structured output)
├── accept_user_ticker                                             [ticker path only]
├── resolve_identity             (get_fundamentals tool call, no LLM — pins ticker -> company)
├── plan_research_tasks          (Sonnet, structured output)
├── run_ticker_research × 1      (one per final candidate)
│   └── 4 researcher spans, run in parallel (app/graph/build.py wires
│       START -> each researcher -> END; none reads another's output):
│       news / financials / leadership: ReAct loops (Haiku) whose tools are wrapped by
│         ToolGuard (ticker lock, tool-call budget, blocked hosts, result truncation);
│         tools: get_company_news / get_fundamentals / get_sec_filings / read_sec_filing / fetch
│       technical_researcher: one get_technical_indicators call + one Haiku summary
├── validate_findings            (deterministic: identity + citation checks, no LLM)
└── synthesizer                  (Sonnet, structured output)
```

Runs can end early with `status: failed` at `resolve_identity` (ticker doesn't resolve) or at
`validate_findings` (fewer than 3 domains with usable data -> `outcome: insufficient_data`).

Measured effect of the identity/validation/efficiency changes on the same `TE` ticker run
(before -> after): 33 -> 18 LLM calls, 149.7K -> 76.2K input tokens (17.7K of it read from
cache), 68s -> 49s, 9 failed tool calls -> 0 failures (14 cheap budget/duplicate refusals),
and a report about T1 Energy only instead of a T1 Energy / TE Connectivity mix.

Verified against a real run on 2026-09-23 (project `agentic-ai-stock-researcher`,
industry "Regional Banks") via the LangSmith SDK — `client.list_runs(project_name=...,
run_type="llm")` showed exactly this shape: 3 Sonnet spans (the run didn't reach the
synthesizer before running out of Anthropic credit) and 106 Haiku spans across the
researcher agents' ReAct loops.

## Cost, measured (not estimated)

Pulled from that same real trace via the LangSmith SDK — see `CLAUDE_CODE_NOTES.md` for
the full breakdown. Headline numbers:

- A full run costs roughly **$1.75-$2.50** (Regional Banks; other industries should be
  similar in magnitude), and **the 4 Sonnet-backed nodes account for well under 5%
  of that** — the 12 tool-calling Haiku researcher agents dominate cost.
- **98.7% of all tokens in that trace were input, not output.** Each researcher agent's
  ReAct loop resends its entire growing conversation — including full fetched
  web-page/article text — on every turn, with no prompt caching in place. That's the
  actual cost driver, not model "thinking." Worth revisiting if this graduates past a
  course checkpoint (caching the stable system prompt + tool schemas per researcher
  would cut a meaningful share of this).

## Capturing `regional_banks_run.png`

1. `uv run uvicorn app.main:app`
2. `curl -X POST localhost:8000/research -H "Content-Type: application/json" -d '{"industry":"Regional Banks"}'`
3. Open the run in the LangSmith UI (`smith.langchain.com` → your project) and confirm
   the span tree above.
4. Screenshot the expanded trace tree to `traces/regional_banks_run.png`.

This screenshot hasn't been captured yet as of this commit — the run used to verify the
trace shape above failed partway through (insufficient Anthropic credit), so the
screenshot should be taken from a fresh, fully-completed run once credit is restored.

## Local logging (offline, no LangSmith needed)

LangSmith is hosted, so nothing under `traces/` is a trace. As a complement, a passive
callback handler (`app/local_trace.py`) can write every run to disk. It only *observes*
events the graph already emits — no extra LLM or tool calls, no graph/prompt changes.

Enable it by setting `LOCAL_TRACE_DIR=traces/runs` in `.env` (empty/unset = off). Options
(see `.env.example`): `LOCAL_TRACE_DB`, `LOCAL_TRACE_MAX_BLOB_BYTES` (200000),
`LOCAL_TRACE_KEEP_RUNS` (50; older runs are pruned along with their DB rows).

Per run, `traces/runs/<utc-timestamp>_<job_id>/`:

```
events.jsonl          one JSON line per event: start / end / error for graph nodes ("chain"),
                      LLM calls and tool calls, with ts, latency_ms, token usage, model, the
                      owning node (`node`, `node_path`), previews, and blob references
blobs/<sha256>.json   full inputs/outputs, content-addressed: identical content is stored
                      once (a ReAct loop resends its whole conversation every turn, but each
                      distinct message is one blob). Over-size payloads are truncated
                      (`truncated: true`); configured API keys are redacted.
summary.json          status, error, tags (git commit, model ids), totals, per-node totals
```

Events from parallel branches interleave; use `ts` and `parent_uuid` (LangChain run ids),
not file order. LLM/tool events under a ReAct agent's internal `agent`/`tools` steps are
attributed to the enclosing researcher node.

**Not captured:** HTTP calls made *inside* the MCP finance-server subprocess (Finnhub,
Yahoo, NASDAQ) — only the tool-call boundary is visible to the callback.

### SQLite index

At job end each run is upserted into `traces/traces.db` (`runs`, `events`, and a
`node_totals` view). The run directories are the source of truth; the DB is derived and
rebuildable, and holds hashes + previews, not payloads.

```
uv run python scripts/index_traces.py --rebuild --report   # rebuild from files + print summary tables
sqlite3 traces/traces.db "SELECT node, SUM(input_tokens) FROM node_totals GROUP BY node"
```

Both `traces/runs/` and `traces/traces.db*` are gitignored. Traces can contain fetched
article/filing text, so treat them as local data.
