from typing import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.planner import plan_research_tasks, plan_screening_task
from agents.researchers.financials import financials_researcher
from agents.researchers.leadership import leadership_researcher
from agents.researchers.news import news_researcher
from agents.researchers.technical import technical_researcher
from agents.screener import (
    accept_user_ticker,
    deterministic_screen,
    insufficient_candidates,
    llm_judgment_screen,
)
from agents.synthesizer import synthesizer
from app.graph.routing import fan_out_to_ticker_research, route_after_screen, route_from_start
from app.graph.state import ResearchState, TickerResearchState


def build_ticker_subgraph(node_overrides: dict[str, Callable] | None = None) -> CompiledStateGraph:
    """The 4 domain researchers as a parallel fan-out/join subgraph, compiled
    once and invoked once per run (via `Send`) with the final ticker's input —
    this is what keeps it 4 pieces of node code reusable across both entry
    paths. The 4 researchers are fully independent (none reads another's
    output; each only writes to the shared `findings` reducer), so they run
    concurrently rather than as a chain.

    `node_overrides` lets tests swap in stub researcher functions to validate
    graph structure without live LLM/tool calls."""
    nodes: dict[str, Callable] = {
        "news_researcher": news_researcher,
        "financials_researcher": financials_researcher,
        "leadership_researcher": leadership_researcher,
        "technical_researcher": technical_researcher,
    }
    if node_overrides:
        nodes.update(node_overrides)

    graph = StateGraph(TickerResearchState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    for name in nodes:
        graph.add_edge(START, name)
        graph.add_edge(name, END)

    return graph.compile()


def _make_run_ticker_research(compiled_ticker_graph: CompiledStateGraph) -> Callable:
    async def run_ticker_research(input_state: TickerResearchState) -> dict:
        result = await compiled_ticker_graph.ainvoke(input_state)
        return {"domain_findings": result["findings"]}

    return run_ticker_research


def build_research_graph(
    node_overrides: dict[str, Callable] | None = None,
    ticker_node_overrides: dict[str, Callable] | None = None,
) -> CompiledStateGraph:
    """`node_overrides` / `ticker_node_overrides` let tests swap in stub node
    functions (no LLM/MCP calls) to validate the conditional edge and `Send`
    fan-out in isolation. Production (app/main.py) calls this with no
    overrides, using the real LLM- and MCP-backed nodes."""
    compiled_ticker_graph = build_ticker_subgraph(ticker_node_overrides)
    run_ticker_research = _make_run_ticker_research(compiled_ticker_graph)

    nodes: dict[str, Callable] = {
        "plan_screening_task": plan_screening_task,
        "deterministic_screen": deterministic_screen,
        "llm_judgment_screen": llm_judgment_screen,
        "accept_user_ticker": accept_user_ticker,
        "insufficient_candidates": insufficient_candidates,
        "plan_research_tasks": plan_research_tasks,
        "run_ticker_research": run_ticker_research,
        "synthesizer": synthesizer,
    }
    if node_overrides:
        nodes.update(node_overrides)

    graph = StateGraph(ResearchState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_conditional_edges(
        START,
        route_from_start,
        {"accept_user_ticker": "accept_user_ticker", "plan_screening_task": "plan_screening_task"},
    )
    graph.add_edge("plan_screening_task", "deterministic_screen")
    graph.add_conditional_edges(
        "deterministic_screen",
        route_after_screen,
        {
            "insufficient_candidates": "insufficient_candidates",
            "llm_judgment_screen": "llm_judgment_screen",
        },
    )
    graph.add_edge("llm_judgment_screen", "plan_research_tasks")
    graph.add_edge("accept_user_ticker", "plan_research_tasks")
    graph.add_conditional_edges("plan_research_tasks", fan_out_to_ticker_research, ["run_ticker_research"])
    graph.add_edge("run_ticker_research", "synthesizer")
    graph.add_edge("synthesizer", END)
    graph.add_edge("insufficient_candidates", END)

    return graph.compile()
