"""Shared infrastructure for the 4 domain researcher nodes.

Each researcher is a tool-calling Haiku agent built with
`langgraph.prebuilt.create_react_agent`, using `response_format` so the final
answer comes back as structured (summary, citations) instead of needing to be
parsed out of free text.
"""

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from app.graph.state import ResearchTask, TickerResearchState
from app.llm import get_haiku


class Citation(BaseModel):
    source_id: str = Field(description="Which tool/source this came from, e.g. 'get_company_news', 'fetch'")
    identifier: str = Field(description="URL, ticker, or other identifier for the specific source")
    title: str = Field(description="Short human-readable title/description of the source")


class DomainFindingOutput(BaseModel):
    summary: str = Field(description="3-6 sentence summary of findings, grounded only in tool results")
    citations: list[Citation] = Field(default_factory=list)


def build_domain_agent(tools, system_prompt: str):
    return create_react_agent(get_haiku(), tools, prompt=system_prompt, response_format=DomainFindingOutput)


def find_task(state: TickerResearchState, domain: str) -> ResearchTask | None:
    return next((t for t in state["research_tasks"] if t["domain"] == domain), None)


async def run_domain_research(agent, ticker: str, domain: str, focus_notes: str) -> dict:
    instruction = f"Research {ticker}. Focus: {focus_notes}" if focus_notes else f"Research {ticker}."
    result = await agent.ainvoke({"messages": [HumanMessage(content=instruction)]})
    structured: DomainFindingOutput = result["structured_response"]
    return {
        "ticker": ticker,
        "domain": domain,
        "summary": structured.summary,
        "citations": [c.model_dump() for c in structured.citations],
        "raw_data": {},
    }
