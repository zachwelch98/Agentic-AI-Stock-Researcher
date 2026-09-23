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
├── plan_screening_task          (Sonnet, structured output)
├── deterministic_screen         (screen_industry MCP tool-call span)
├── llm_judgment_screen          (Sonnet, structured output)
├── plan_research_tasks          (Sonnet, structured output)
├── run_ticker_research × 3      (parallel — one per final candidate)
│   └── each contains 4 sequential researcher spans:
│       news_researcher → financials_researcher → leadership_researcher → technical_researcher
│       each researcher span is itself a ReAct loop with its own tool-call children
│       (get_company_news / fetch / get_fundamentals / get_sec_filings / get_technical_indicators)
└── synthesizer                  (Sonnet, structured output)
```

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
