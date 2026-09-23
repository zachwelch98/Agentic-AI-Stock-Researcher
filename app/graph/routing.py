from langgraph.graph import END
from langgraph.types import Send

from app.graph.state import ResearchState

def route_from_start(state: ResearchState) -> str:
    return "accept_user_ticker" if state.get("user_supplied_ticker") else "plan_screening_task"


def route_after_screen(state: ResearchState) -> str:
    return "insufficient_candidates" if len(state["screened_candidates"]) < 3 else "llm_judgment_screen"


def route_after_resolve(state: ResearchState) -> str:
    return END if state.get("status") == "failed" else "plan_research_tasks"


def route_after_validate(state: ResearchState) -> str:
    return END if state.get("status") == "failed" else "synthesizer"


def fan_out_to_ticker_research(state: ResearchState) -> list[Send]:
    profiles = state.get("company_profiles") or {}
    return [
        Send(
            "run_ticker_research",
            {
                "ticker": ticker,
                "company_profile": profiles.get(ticker),
                "research_tasks": [t for t in state["research_tasks"] if t["ticker"] == ticker],
                "findings": [],
            },
        )
        for ticker in state["final_candidates"]
    ]
