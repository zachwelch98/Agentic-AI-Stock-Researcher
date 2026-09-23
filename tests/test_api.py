"""FastAPI layer tests: HTTP contract only (status codes, schemas, 404/422),
using a stub lifespan with a trivial graph — no real MCP subprocesses or LLM
calls, matching how tests/test_graph_routing.py keeps the graph-structure
tests offline."""

import time
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph

from app.graph.state import ResearchState
from app.main import create_app


async def _stub_node(state: ResearchState) -> dict:
    return {
        "status": "done",
        "final_report": {"industry_query": state["industry_query"], "outcome": "ok"},
    }


def _build_stub_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("stub", _stub_node)
    graph.add_edge(START, "stub")
    graph.add_edge("stub", END)
    return graph.compile()


def _build_test_app() -> FastAPI:
    compiled = _build_stub_graph()

    @asynccontextmanager
    async def stub_lifespan(app: FastAPI):
        app.state.research_graph = compiled
        app.state.background_tasks = set()
        yield

    return create_app(lifespan_override=stub_lifespan)


@pytest.fixture
def client():
    with TestClient(_build_test_app()) as c:
        yield c


def test_healthz(client):
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_research_empty_industry_is_422(client):
    res = client.post("/research", json={"industry": ""})
    assert res.status_code == 422


def test_research_missing_field_is_422(client):
    res = client.post("/research", json={})
    assert res.status_code == 422


def test_research_both_industry_and_ticker_is_422(client):
    res = client.post("/research", json={"industry": "Regional Banks", "ticker": "JPM"})
    assert res.status_code == 422


def test_research_ticker_only_is_accepted(client):
    res = client.post("/research", json={"ticker": "JPM"})
    assert res.status_code == 202


def test_unknown_job_is_404(client):
    res = client.get("/jobs/does-not-exist")
    assert res.status_code == 404


def test_research_happy_path_completes(client):
    res = client.post("/research", json={"industry": "Regional Banks"})
    assert res.status_code == 202
    body = res.json()
    assert body["status"] == "pending"
    job_id = body["job_id"]

    job_body = None
    for _ in range(50):
        job_res = client.get(f"/jobs/{job_id}")
        assert job_res.status_code == 200
        job_body = job_res.json()
        if job_body["status"] == "done":
            break
        time.sleep(0.05)

    assert job_body is not None
    assert job_body["status"] == "done"
    assert job_body["result"]["industry_query"] == "Regional Banks"
    assert job_body["result"]["outcome"] == "ok"
