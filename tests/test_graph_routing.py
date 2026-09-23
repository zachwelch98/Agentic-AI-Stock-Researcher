"""Graph-structure tests: conditional edge + Send fan-out + reducer, all with
stub node functions (no LLM/MCP calls) injected via build_research_graph's
node_overrides — this is deliberately independent of the real, LLM/MCP-backed
node implementations in agents/*.py so it stays fast and offline."""

import pytest

from agents.screener import screen_candidates
from app.graph.build import build_research_graph, build_ticker_subgraph
from app.graph.routing import fan_out_to_ticker_research, route_after_screen, route_from_start
from app.graph.state import make_initial_state

DOMAINS = ("news", "financials", "leadership", "technical")

_STUB_RAW_CANDIDATES: dict[str, list[dict]] = {
    "regional banks": [
        {"ticker": "RB1", "company_name": "Stub Regional Bank 1", "market_cap": 5e9, "avg_dollar_volume": 2e7, "pe_ratio": 8.0, "pb_ratio": 0.9, "debt_to_equity": 0.8, "dividend_yield": 0.03},
        {"ticker": "RB2", "company_name": "Stub Regional Bank 2", "market_cap": 3e9, "avg_dollar_volume": 1.5e7, "pe_ratio": 12.0, "pb_ratio": 1.2, "debt_to_equity": 1.1, "dividend_yield": 0.0},
        {"ticker": "RB3", "company_name": "Stub Regional Bank 3", "market_cap": 4e9, "avg_dollar_volume": 1.8e7, "pe_ratio": 9.5, "pb_ratio": 1.0, "debt_to_equity": 0.9, "dividend_yield": 0.02},
        {"ticker": "RB4", "company_name": "Stub Regional Bank 4", "market_cap": 2e9, "avg_dollar_volume": 1.2e7, "pe_ratio": 15.0, "pb_ratio": 1.4, "debt_to_equity": 1.3, "dividend_yield": 0.0},
        {"ticker": "RB5", "company_name": "Stub Regional Bank 5", "market_cap": 6e9, "avg_dollar_volume": 2.5e7, "pe_ratio": 7.0, "pb_ratio": 0.8, "debt_to_equity": 0.7, "dividend_yield": 0.04},
    ],
    "biotech stub insufficient": [
        {"ticker": "BIO1", "company_name": "Stub Biotech 1", "market_cap": 5e8, "avg_dollar_volume": 2e6, "pe_ratio": None, "pb_ratio": 3.0, "debt_to_equity": 0.2, "dividend_yield": None},
        {"ticker": "BIO2", "company_name": "Stub Biotech 2", "market_cap": 4e8, "avg_dollar_volume": 1.5e6, "pe_ratio": None, "pb_ratio": 4.0, "debt_to_equity": 0.1, "dividend_yield": None},
    ],
}


async def _stub_plan_screening_task(state):
    return {"screening_plan": {"industry_query": state["industry_query"]}, "status": "screening"}


async def _stub_deterministic_screen(state):
    raw = _STUB_RAW_CANDIDATES.get(state["industry_query"].strip().lower(), [])
    return {"screened_candidates": screen_candidates(raw), "status": "screening"}


async def _stub_llm_judgment_screen(state):
    ranked = sorted(state["screened_candidates"], key=lambda c: c["composite_score"], reverse=True)
    return {
        "final_candidates": [ranked[0]["ticker"]],
        "screener_justification": "stub justification",
        "status": "researching",
    }


async def _stub_accept_user_ticker(state):
    return {
        "final_candidates": [state["user_supplied_ticker"]],
        "used_llm_fallback_tickers": False,
        "screener_justification": "stub bypass justification",
        "status": "researching",
    }


async def _stub_resolve_identity(state):
    return {
        "company_profiles": {
            t: {"ticker": t, "company_name": f"Stub {t} Inc", "website": None, "market_cap": None}
            for t in state["final_candidates"]
        }
    }


async def _stub_plan_research_tasks(state):
    tasks = [
        {"ticker": ticker, "domain": domain, "focus_notes": "stub"}
        for ticker in state["final_candidates"]
        for domain in DOMAINS
    ]
    return {"research_tasks": tasks, "status": "researching"}


async def _stub_synthesizer(state):
    report = {
        "industry_query": state["industry_query"],
        "candidates": state["final_candidates"],
        "domain_findings": state["domain_findings"],
    }
    return {"final_report": report, "status": "done"}


def _stub_researcher(domain: str):
    async def researcher(state):
        return {
            "findings": [
                {"ticker": state["ticker"], "domain": domain, "summary": "stub", "citations": [], "raw_data": {}}
            ]
        }

    return researcher


def _build_stub_graph():
    return build_research_graph(
        node_overrides={
            "plan_screening_task": _stub_plan_screening_task,
            "deterministic_screen": _stub_deterministic_screen,
            "llm_judgment_screen": _stub_llm_judgment_screen,
            "accept_user_ticker": _stub_accept_user_ticker,
            "resolve_identity": _stub_resolve_identity,
            "plan_research_tasks": _stub_plan_research_tasks,
            "synthesizer": _stub_synthesizer,
        },
        ticker_node_overrides={f"{domain}_researcher": _stub_researcher(domain) for domain in DOMAINS},
    )


def test_route_after_screen_insufficient():
    state = make_initial_state("x", "job")
    state["screened_candidates"] = [{"ticker": "A"}, {"ticker": "B"}]
    assert route_after_screen(state) == "insufficient_candidates"


def test_route_after_screen_sufficient():
    state = make_initial_state("x", "job")
    state["screened_candidates"] = [{"ticker": "A"}, {"ticker": "B"}, {"ticker": "C"}]
    assert route_after_screen(state) == "llm_judgment_screen"


def test_route_from_start_industry_path():
    state = make_initial_state("Regional Banks", "job")
    assert route_from_start(state) == "plan_screening_task"


def test_route_from_start_user_ticker_path():
    state = make_initial_state("", "job", user_supplied_ticker="JPM")
    assert route_from_start(state) == "accept_user_ticker"


def test_ticker_subgraph_runs_domain_researchers_in_parallel_not_a_chain():
    subgraph = build_ticker_subgraph(
        node_overrides={f"{domain}_researcher": _stub_researcher(domain) for domain in DOMAINS}
    )
    edges = {(e.source, e.target) for e in subgraph.get_graph().edges}
    researcher_nodes = {f"{domain}_researcher" for domain in DOMAINS}

    for node in researcher_nodes:
        assert ("__start__", node) in edges
        assert (node, "__end__") in edges

    # No researcher should feed into another — that would make it a chain again.
    for source, target in edges:
        if source in researcher_nodes:
            assert target == "__end__"


def test_fan_out_produces_one_send_per_final_candidate():
    state = make_initial_state("x", "job")
    state["final_candidates"] = ["AAA", "BBB", "CCC"]
    state["research_tasks"] = [
        {"ticker": ticker, "domain": domain, "focus_notes": ""}
        for ticker in state["final_candidates"]
        for domain in DOMAINS
    ]
    sends = fan_out_to_ticker_research(state)
    assert len(sends) == 3
    assert {s.arg["ticker"] for s in sends} == {"AAA", "BBB", "CCC"}
    for s in sends:
        assert s.node == "run_ticker_research"
        assert len(s.arg["research_tasks"]) == 4


@pytest.mark.asyncio
async def test_sufficient_candidates_path_produces_four_findings():
    graph = _build_stub_graph()
    result = await graph.ainvoke(make_initial_state("Regional Banks", "job-sufficient"))

    assert result["status"] == "done"
    assert len(result["final_candidates"]) == 1
    assert len(result["domain_findings"]) == 4
    domains_by_ticker: dict[str, set[str]] = {}
    for finding in result["domain_findings"]:
        domains_by_ticker.setdefault(finding["ticker"], set()).add(finding["domain"])
    assert set(domains_by_ticker.keys()) == set(result["final_candidates"])
    for domains in domains_by_ticker.values():
        assert domains == set(DOMAINS)


@pytest.mark.asyncio
async def test_user_ticker_path_bypasses_screening():
    graph = _build_stub_graph()
    initial_state = make_initial_state("", "job-direct-ticker", user_supplied_ticker="JPM")
    result = await graph.ainvoke(initial_state)

    assert result["status"] == "done"
    assert result["final_candidates"] == ["JPM"]
    assert result["screening_plan"] is None
    assert result["screened_candidates"] == []
    assert len(result["domain_findings"]) == 4
    assert {f["domain"] for f in result["domain_findings"]} == set(DOMAINS)
    assert all(f["ticker"] == "JPM" for f in result["domain_findings"])


@pytest.mark.asyncio
async def test_insufficient_candidates_path_skips_research():
    graph = _build_stub_graph()
    result = await graph.ainvoke(make_initial_state("Biotech Stub Insufficient", "job-insufficient"))

    assert result["status"] == "insufficient_candidates"
    assert result["final_candidates"] == []
    assert result["domain_findings"] == []
    assert result["final_report"]["outcome"] == "insufficient_candidates"


@pytest.mark.asyncio
async def test_unknown_industry_is_insufficient():
    graph = _build_stub_graph()
    result = await graph.ainvoke(make_initial_state("Nonexistent Industry XYZ", "job-unknown"))

    assert result["status"] == "insufficient_candidates"
    assert result["domain_findings"] == []
