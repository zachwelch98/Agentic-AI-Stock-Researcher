from langgraph.types import Send

from app.graph.state import ResearchState


def route_from_start(state: ResearchState) -> str:
    return "accept_user_ticker" if state.get("user_supplied_ticker") else "plan_screening_task"


def route_after_screen(state: ResearchState) -> str:
    return "insufficient_candidates" if len(state["screened_candidates"]) < 3 else "llm_judgment_screen"


def fan_out_to_ticker_research(state: ResearchState) -> list[Send]:
    return [
        Send(
            "run_ticker_research",
            {
                "ticker": ticker,
                "research_tasks": [t for t in state["research_tasks"] if t["ticker"] == ticker],
                "findings": [],
            },
        )
        for ticker in state["final_candidates"]
    ]
