# Agentic AI Stock Researcher

> Status: scaffold in progress. This README is filled in during Phase 6 of the build
> (see `PLAN.md`) with the problem statement, architecture diagram, setup instructions,
> tradeoffs, and eval results.

## What this is

Given either an industry name or a specific ticker, this system:

1. If given an industry: screens it down to the single most undervalued stock using a
   deterministic fundamentals-based composite score plus an LLM-judgment step. If given
   a ticker directly, screening is skipped entirely.
2. Runs a 4-domain deep-dive (news, financials, leadership, technical analysis) on that
   one ticker, using LangGraph.
3. Returns one citation-backed research report and recommendation.

Built with Python, LangGraph, LangChain, FastAPI, MCP, and Claude Code, with LangSmith
tracing and a lightweight eval harness. See `PLAN.md` for the full design.

## Setup

```bash
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY, FINNHUB_API_KEY, LANGCHAIN_API_KEY, SEC_EDGAR_USER_AGENT
uv run python scripts/refresh_nasdaq_cache.py
uv run uvicorn app.main:app --reload
```

Then open `http://localhost:8000/`.

## Architecture

_Mermaid diagram + tradeoffs + eval results to be added in Phase 6._
