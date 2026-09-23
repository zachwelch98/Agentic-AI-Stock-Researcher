"""Local run logging (app/local_trace.py): drives real graphs through `run_job`
with a fake chat model / stub nodes — no live LLM, MCP or network — and
inspects the files (and SQLite index) the passive handler writes."""

import json
import sqlite3
from uuid import uuid4

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from app import jobs, local_trace
from app.config import Settings
from app.graph.state import ResearchState
from app.jobs import create_job, run_job
from app.local_trace import LocalTraceHandler, prune_old_runs
from tests.test_graph_routing import _build_stub_graph

SECRET = "sk-ant-supersecret-value"


class ScriptedChat(BaseChatModel):
    """Plays back scripted AIMessages; counts how many times it was called."""

    script: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.script[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def _usage(inp: int, out: int) -> dict:
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}


def _scripted_model() -> ScriptedChat:
    return ScriptedChat(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "get_price", "args": {"ticker": "ZION"}, "id": "call_1"}],
                usage_metadata=_usage(100, 10),
            ),
            AIMessage(content="ZION is at 42.", usage_metadata=_usage(160, 12)),
        ]
    )


@tool
def get_price(ticker: str) -> str:
    """Return a price."""
    return f"{ticker}: 42.0"


def _react_graph(model: ScriptedChat):
    """A graph whose node calls a nested ReAct agent WITHOUT forwarding config —
    the same shape as agents/researchers/_shared.py:run_domain_research, so it
    guards that callbacks reach nested runs via contextvars."""
    agent = create_react_agent(model, [get_price], prompt="You are a researcher.")

    async def news_researcher(state: ResearchState) -> dict:
        result = await agent.ainvoke({"messages": [HumanMessage(content="Research ZION")]})
        return {"status": "done", "final_report": {"answer": result["messages"][-1].content}}

    graph = StateGraph(ResearchState)
    graph.add_node("news_researcher", news_researcher)
    graph.add_edge(START, "news_researcher")
    graph.add_edge("news_researcher", END)
    return graph.compile()


def _failing_graph():
    async def boom(state: ResearchState) -> dict:
        raise RuntimeError("upstream exploded")

    graph = StateGraph(ResearchState)
    graph.add_node("boom", boom)
    graph.add_edge(START, "boom")
    graph.add_edge("boom", END)
    return graph.compile()


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    s = Settings(
        local_trace_dir=str(tmp_path / "runs"),
        local_trace_db=str(tmp_path / "traces.db"),
        anthropic_api_key=SECRET,
    )
    monkeypatch.setattr(jobs, "get_settings", lambda: s)
    return s


async def _run(graph, *, industry=None, ticker=None):
    job = create_job(industry, ticker)
    await run_job(job.job_id, graph)
    return job


def _events(job) -> list[dict]:
    lines = (jobs_dir(job) / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def jobs_dir(job):
    from pathlib import Path

    return Path(job.trace_dir)


def _summary(job) -> dict:
    return json.loads((jobs_dir(job) / "summary.json").read_text())


# ---- coverage of every interaction kind -------------------------------------


async def test_react_agent_llm_and_tool_events_attributed_to_researcher_node(settings):
    job = await _run(_react_graph(_scripted_model()), ticker="ZION")

    assert job.status.value == "done"
    events = _events(job)

    llm_ends = [e for e in events if e["kind"] == "llm" and e["event"] == "end"]
    assert [e["usage"]["input"] for e in llm_ends] == [100, 160]
    tool_ends = [e for e in events if e["kind"] == "tool" and e["event"] == "end"]
    assert [e["name"] for e in tool_ends] == ["get_price"]

    # Nested prebuilt "agent"/"tools" nodes roll up to the enclosing researcher.
    for e in llm_ends + tool_ends:
        assert e["node"] == "news_researcher"
        assert e["node_path"].startswith("news_researcher")

    summary = _summary(job)
    assert summary["status"] == "done"
    assert summary["totals"]["llm_calls"] == 2
    assert summary["totals"]["tool_calls"] == 1
    assert summary["totals"]["input_tokens"] == 260
    assert summary["by_node"]["news_researcher"]["llm_calls"] == 2


async def test_react_history_is_deduplicated_across_turns(settings):
    job = await _run(_react_graph(_scripted_model()), ticker="ZION")

    starts = [e for e in _events(job) if e["kind"] == "llm" and e["event"] == "start"]
    first, second = starts[0]["input_refs"], starts[1]["input_refs"]
    assert set(first) < set(second)  # the second turn re-sends turn 1's messages...
    blob_dir = jobs_dir(job) / "blobs"
    assert len(list(blob_dir.glob("*.json"))) == _summary(job)["unique_blobs"]
    # ...but each distinct message is stored once: 6 references, 4 distinct blobs.
    distinct = set(first) | set(second)
    assert len(distinct) < len(first) + len(second)
    assert all((blob_dir / f"{ref}.json").exists() for ref in distinct)
    # Turn 2 is 4 messages but only the tool result is new: the AI tool-call message
    # was already stored as turn 1's output blob (same content hash).
    assert starts[1]["input_preview"].startswith("(4 messages; 1 new)")


async def test_direct_tool_call_inside_a_node_is_traced(settings):
    """Mirrors agents/screener.py:deterministic_screen — a node calling an MCP
    tool directly via `tool.ainvoke(...)` (no ReAct loop, config not forwarded)."""

    async def deterministic_screen(state: ResearchState) -> dict:
        await get_price.ainvoke({"ticker": "ZION"})
        return {"status": "done", "final_report": {}}

    graph = StateGraph(ResearchState)
    graph.add_node("deterministic_screen", deterministic_screen)
    graph.add_edge(START, "deterministic_screen")
    graph.add_edge("deterministic_screen", END)

    job = await _run(graph.compile(), ticker="ZION")

    tool_ends = [e for e in _events(job) if e["kind"] == "tool" and e["event"] == "end"]
    assert [(e["name"], e["node"]) for e in tool_ends] == [("get_price", "deterministic_screen")]
    assert _summary(job)["by_node"]["deterministic_screen"]["tool_calls"] == 1


@pytest.mark.parametrize(
    "kwargs, expected_nodes",
    [
        (
            {"industry": "Regional Banks"},
            {
                "plan_screening_task",
                "deterministic_screen",
                "llm_judgment_screen",
                "resolve_identity",
                "plan_research_tasks",
                "run_ticker_research",
                "news_researcher",
                "financials_researcher",
                "leadership_researcher",
                "technical_researcher",
                "validate_findings",
                "synthesizer",
            },
        ),
        (
            {"ticker": "JPM"},
            {
                "accept_user_ticker",
                "resolve_identity",
                "plan_research_tasks",
                "run_ticker_research",
                "news_researcher",
                "validate_findings",
                "synthesizer",
            },
        ),
        (
            {"industry": "Biotech Stub Insufficient"},
            {"plan_screening_task", "deterministic_screen", "insufficient_candidates"},
        ),
    ],
    ids=["industry-path", "user-ticker-path", "insufficient-path"],
)
async def test_every_graph_node_is_traced_including_fan_out_and_subgraph(settings, kwargs, expected_nodes):
    job = await _run(_build_stub_graph(), **kwargs)

    assert job.status.value == "done"
    chain_nodes = {e["name"] for e in _events(job) if e["kind"] == "chain" and e["event"] == "end"}
    assert expected_nodes <= chain_nodes
    assert "LangGraph" in chain_nodes  # root run


async def test_ticker_path_bypasses_screening_nodes_in_trace(settings):
    job = await _run(_build_stub_graph(), ticker="JPM")
    names = {e["name"] for e in _events(job) if e["kind"] == "chain"}
    assert "plan_screening_task" not in names and "deterministic_screen" not in names


# ---- errors, safety, redaction ----------------------------------------------


async def test_failed_job_writes_error_event_and_failed_summary(settings):
    job = await _run(_failing_graph(), ticker="ZION")

    assert job.status.value == "failed"
    errors = [e for e in _events(job) if e["event"] == "error"]
    assert any("upstream exploded" in e["error"] for e in errors)
    summary = _summary(job)
    assert summary["status"] == "failed"
    assert "upstream exploded" in summary["error"]
    assert summary["totals"]["errors"] >= 1


async def test_tool_error_event(tmp_path):
    handler = LocalTraceHandler(tmp_path / "run", job_id="j")
    run_id = uuid4()
    await handler.on_tool_start({"name": "fetch"}, "x", run_id=run_id, inputs={"url": "u"})
    await handler.on_tool_error(TimeoutError("boom"), run_id=run_id)

    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["start", "error"]
    assert events[1]["name"] == "fetch" and events[1]["error"] == "TimeoutError: boom"


async def test_secrets_are_redacted_and_large_payloads_truncated(tmp_path):
    handler = LocalTraceHandler(tmp_path / "run", job_id="j", max_blob_bytes=200, secrets=[SECRET])
    run_id = uuid4()
    await handler.on_tool_start(
        {"name": "fetch"}, "", run_id=run_id, inputs={"note": f"key={SECRET}", "blob": "x" * 5000}
    )

    events_text = (tmp_path / "run" / "events.jsonl").read_text()
    assert SECRET not in events_text
    for blob in (tmp_path / "run" / "blobs").glob("*.json"):
        assert SECRET not in blob.read_text()
    event = json.loads(events_text.splitlines()[0])
    assert event["truncated"] is True
    stored = json.loads((tmp_path / "run" / "blobs" / f"{event['input_refs'][0]}.json").read_text())
    assert stored["_truncated"] is True and stored["original_bytes"] > 200


async def test_handler_io_failure_does_not_fail_the_job(settings, monkeypatch):
    def broken_emit(self, record):
        raise OSError("disk full")

    monkeypatch.setattr(LocalTraceHandler, "_emit", broken_emit)
    job = await _run(_build_stub_graph(), ticker="JPM")
    assert job.status.value == "done"
    assert job.result is not None


async def test_index_failure_does_not_fail_the_job(settings, monkeypatch):
    def broken_index(db_path, run_dir):
        raise sqlite3.DatabaseError("database is locked")

    monkeypatch.setattr(local_trace.trace_index, "index_run", broken_index)
    job = await _run(_build_stub_graph(), ticker="JPM")
    assert job.status.value == "done"
    assert _summary(job)["status"] == "done"  # summary.json still written


# ---- off switch, no extra LLM calls -----------------------------------------


async def test_disabled_by_default_writes_nothing_and_changes_nothing(tmp_path, monkeypatch):
    off = Settings(local_trace_dir="", local_trace_db="")
    monkeypatch.setattr(jobs, "get_settings", lambda: off)
    monkeypatch.chdir(tmp_path)

    job = await _run(_build_stub_graph(), ticker="JPM")

    assert job.status.value == "done"
    assert job.trace_dir is None
    assert list(tmp_path.iterdir()) == []


async def test_tracing_adds_no_llm_calls(settings, monkeypatch):
    traced_model = _scripted_model()
    await _run(_react_graph(traced_model), ticker="ZION")

    monkeypatch.setattr(jobs, "get_settings", lambda: Settings(local_trace_dir="", local_trace_db=""))
    untraced_model = _scripted_model()
    await _run(_react_graph(untraced_model), ticker="ZION")

    assert traced_model.calls == untraced_model.calls == 2


# ---- retention ---------------------------------------------------------------


async def test_pruning_keeps_newest_runs_and_deletes_their_index_rows(tmp_path):
    from app import trace_index

    trace_dir = tmp_path / "runs"
    db = str(tmp_path / "traces.db")
    for i in range(5):
        run_dir = trace_dir / f"2026092{i}T000000Z_job{i}"
        run_dir.mkdir(parents=True)
        (run_dir / "events.jsonl").write_text("")
        trace_index.index_run(db, run_dir)

    removed = prune_old_runs(trace_dir, keep=2, db_path=db)

    assert removed == ["20260920T000000Z_job0", "20260921T000000Z_job1", "20260922T000000Z_job2"]
    assert sorted(p.name for p in trace_dir.iterdir()) == ["20260923T000000Z_job3", "20260924T000000Z_job4"]
    conn = sqlite3.connect(db)
    assert sorted(r[0] for r in conn.execute("SELECT run_id FROM runs")) == [
        "20260923T000000Z_job3",
        "20260924T000000Z_job4",
    ]
    conn.close()
