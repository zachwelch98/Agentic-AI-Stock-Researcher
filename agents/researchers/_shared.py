"""Shared infrastructure for the 4 domain researcher nodes.

Each researcher is a tool-calling Haiku agent built with
`langgraph.prebuilt.create_react_agent`, using `response_format` so the final
answer comes back as structured (summary, citations) instead of needing to be
parsed out of free text.

`ToolGuard` wraps an agent's tools with the run's guardrails: the ticker is
locked to the resolved company, tool calls are budgeted, hosts the generic
`fetch` tool can't read are refused up front, and oversized results are
truncated. It also records which URLs the agent actually got content from, so
`agents/validation.py` can reject citations to pages that were never read.
"""

import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.state import CompanyProfile, ResearchTask, TickerResearchState
from app.llm import get_haiku

IDENTITY_RULE = (
    "\n\nIdentity rule: the user message names the exact company for this ticker. "
    "Research only that company. If a tool returns data for a different company "
    "(different name or ticker), do not switch tickers or substitute a better-known "
    "company — stop and say the data did not match. Set identified_company to the "
    "company name your tool data actually describes."
)

CITATION_RULE = (
    "\n\nCitation rule: each citation's identifier must be the exact URL of a page you "
    "actually read (or, for get_company_news, the article's URL from the tool output), or "
    "the ticker itself for structured tool data such as get_fundamentals. Never use a "
    "publisher name (\"Yahoo\") or a description as an identifier."
)

_URL_RE = re.compile(r"https?://[^\s\"'<>)\]\\]+")
_FETCH_TOOLS = {"fetch", "read_sec_filing"}
# Listing tools whose own output is content (headline + summary), so their URLs
# are legitimate to cite. `get_sec_filings` is deliberately excluded: it only
# lists filings, and citing a filing you never read is exactly the failure to catch.
_CONTENT_LISTING_TOOLS = {"get_company_news"}
_FETCH_FAILURE_MARKERS = ("Failed to fetch", "robots.txt", "status code 4", "status code 5", "refused:")


class Citation(BaseModel):
    source_id: str = Field(description="Which tool/source this came from, e.g. 'get_company_news', 'fetch'")
    identifier: str = Field(description="URL, ticker, or other identifier for the specific source")
    title: str = Field(description="Short human-readable title/description of the source")


class DomainFindingOutput(BaseModel):
    summary: str = Field(description="3-6 sentence summary of findings, grounded only in tool results")
    citations: list[Citation] = Field(default_factory=list)
    identified_company: str = Field(
        default="", description="Name of the company your tool data actually describes"
    )


def _result_text(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in result)
    return str(result)


def _truncate_result(result: object, limit: int) -> object:
    marker = f"\n[truncated to {limit} chars]"
    if isinstance(result, str):
        return result[:limit] + marker if len(result) > limit else result
    if isinstance(result, list):
        out, remaining = [], limit
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if len(text) > remaining:
                    block = {**block, "text": text[: max(remaining, 0)] + marker}
                remaining -= len(text)
            out.append(block)
        return out
    return result


@dataclass
class ToolGuard:
    ticker: str
    max_calls: int
    max_chars: int
    blocked_hosts: tuple[str, ...]
    calls: int = 0
    evidence_urls: set[str] = field(default_factory=set)

    @classmethod
    def for_ticker(cls, ticker: str, max_calls: int | None = None) -> "ToolGuard":
        settings = get_settings()
        return cls(
            ticker=ticker.upper(),
            max_calls=max_calls or settings.researcher_max_tool_calls,
            max_chars=settings.fetch_max_chars,
            blocked_hosts=settings.blocked_fetch_hosts,
        )

    @property
    def recursion_limit(self) -> int:
        # One agent step + one tools step per call, plus the final structured-output pass.
        return self.max_calls * 2 + 6

    def wrap_all(self, tools: list[BaseTool]) -> list[BaseTool]:
        return [self.wrap(t) for t in tools]

    def wrap(self, tool: BaseTool) -> BaseTool:
        async def guarded(**kwargs):
            refusal = self._refuse(tool.name, kwargs)
            if refusal:
                return refusal
            self.calls += 1
            if tool.name == "fetch" and self.max_chars:
                kwargs["max_length"] = min(int(kwargs.get("max_length") or self.max_chars), self.max_chars)
            if tool.name == "read_sec_filing":
                kwargs["max_chars"] = min(int(kwargs.get("max_chars") or self.max_chars), self.max_chars)
            result = await tool.ainvoke(kwargs)
            self._record(tool.name, kwargs, result)
            # read_sec_filing already self-limits (and appends a paging hint that truncation would cut).
            return result if tool.name == "read_sec_filing" or not self.max_chars else _truncate_result(result, self.max_chars)

        return StructuredTool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            coroutine=guarded,
        )

    def _refuse(self, name: str, kwargs: dict) -> str | None:
        ticker = kwargs.get("ticker")
        if ticker is not None and str(ticker).strip().upper() != self.ticker:
            return (
                f"refused: ticker is locked to {self.ticker}; do not research {ticker}. "
                "If the data doesn't match the named company, report the mismatch instead."
            )
        if self.calls >= self.max_calls:
            return "refused: tool-call budget exhausted. Write your final answer now from what you have."
        url = kwargs.get("url")
        if name == "fetch" and url:
            host = (re.sub(r"^https?://", "", str(url)).split("/")[0]).lower()
            if any(host == h or host.endswith("." + h) for h in self.blocked_hosts):
                return (
                    f"refused: {host} can't be read by the fetch tool (blocks automated access). "
                    "Use read_sec_filing for SEC filings, or a different source."
                )
            if str(url) in self.evidence_urls:
                return "refused: that URL was already fetched; use what you have."
        return None

    def _record(self, name: str, kwargs: dict, result: object) -> None:
        text = _result_text(result)
        if name in _FETCH_TOOLS and kwargs.get("url"):
            if not any(marker in text[:300] for marker in _FETCH_FAILURE_MARKERS):
                self.evidence_urls.add(str(kwargs["url"]))
        elif name in _CONTENT_LISTING_TOOLS:
            self.evidence_urls.update(_URL_RE.findall(text))


_CORP_SUFFIXES = {"inc", "corp", "corporation", "plc", "ltd", "limited", "co", "company", "holdings", "group", "the", "sa", "nv", "ag", "llc", "lp"}


def _normalize_company(name: str) -> str:
    tokens = re.sub(r"[^a-z0-9 ]", " ", name.lower()).split()
    return " ".join(t for t in tokens if t not in _CORP_SUFFIXES)


def company_names_match(a: str, b: str) -> bool:
    na, nb = _normalize_company(a), _normalize_company(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    ta, tb = set(na.split()), set(nb.split())
    return len(ta & tb) / len(ta | tb) >= 0.5


def _foreign_ticker_citations(citations: list[dict], ticker: str) -> list[str]:
    """Citations whose identifier is a bare ticker symbol other than the locked one."""
    return [
        c["identifier"]
        for c in citations
        if re.fullmatch(r"[A-Z]{1,5}([.-][A-Z])?", c["identifier"] or "") and c["identifier"] != ticker
    ]


def _cache_last_message(state: dict) -> dict:
    """Mark the newest message as a cache breakpoint. A ReAct loop resends its whole
    growing history every turn; with the breakpoint on the tail, the next turn reads
    that prefix from cache instead of paying full input price for it again. (Only
    prefixes above the model's minimum cacheable size are cached, so short early
    turns are unaffected.)"""
    messages = list(state["messages"])
    last = messages[-1]
    blocks = (
        [{"type": "text", "text": last.content}]
        if isinstance(last.content, str)
        else [dict(b) if isinstance(b, dict) else {"type": "text", "text": str(b)} for b in last.content]
    )
    if blocks and blocks[-1].get("text", "x"):
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
        messages[-1] = last.model_copy(update={"content": blocks})
    return {"llm_input_messages": messages}


def build_domain_agent(tools, system_prompt: str):
    return create_react_agent(
        get_haiku(),
        tools,
        prompt=system_prompt + IDENTITY_RULE + CITATION_RULE,
        response_format=DomainFindingOutput,
        pre_model_hook=_cache_last_message if get_settings().researcher_prompt_cache else None,
    )


def find_task(state: TickerResearchState, domain: str) -> ResearchTask | None:
    return next((t for t in state["research_tasks"] if t["domain"] == domain), None)


def identity_line(ticker: str, profile: CompanyProfile | None) -> str:
    if not profile:
        return f"Research {ticker}."
    site = f", website {profile['website']}" if profile.get("website") else ""
    return f"Research {ticker} — {profile['company_name']}{site}."


def build_finding(
    structured: DomainFindingOutput,
    ticker: str,
    domain: str,
    company_profile: CompanyProfile | None,
    guard: ToolGuard | None,
) -> dict:
    citations = [c.model_dump() for c in structured.citations]
    issues: list[str] = []
    identity_ok = True
    if company_profile:
        if not company_names_match(structured.identified_company, company_profile["company_name"]):
            identity_ok = False
            issues.append(
                f"identified company {structured.identified_company!r} != "
                f"{company_profile['company_name']!r}"
            )
        foreign = _foreign_ticker_citations(citations, ticker)
        if foreign:
            identity_ok = False
            issues.append(f"cites other tickers: {sorted(set(foreign))}")
    return {
        "ticker": ticker,
        "domain": domain,
        "summary": structured.summary,
        "citations": citations,
        "raw_data": {},
        "identified_company": structured.identified_company,
        "identity_ok": identity_ok,
        "fetched_urls": sorted(guard.evidence_urls) if guard else [],
        "validation_issues": issues,
    }


async def run_domain_research(
    agent,
    ticker: str,
    domain: str,
    focus_notes: str,
    *,
    company_profile: CompanyProfile | None = None,
    guard: ToolGuard | None = None,
) -> dict:
    instruction = identity_line(ticker, company_profile)
    if focus_notes:
        instruction += f" Focus: {focus_notes}"
    config = {"recursion_limit": guard.recursion_limit} if guard else None
    last_state: dict = {}
    try:
        async for last_state in agent.astream(
            {"messages": [HumanMessage(content=instruction)]}, config=config, stream_mode="values"
        ):
            pass
        structured = last_state["structured_response"]
    except GraphRecursionError:
        # The model kept calling tools past its budget (refusals just make it retry).
        # Force a final answer from what was gathered rather than failing the whole run.
        structured = await _force_final_answer(instruction, last_state.get("messages", []))
    return build_finding(structured, ticker, domain, company_profile, guard)


async def _force_final_answer(instruction: str, messages: list) -> DomainFindingOutput:
    gathered = [
        f"[{m.name or 'tool'}] {_result_text(m.content)[:3000]}"
        for m in messages
        if isinstance(m, ToolMessage) and not _result_text(m.content).startswith("refused:")
    ]
    notes = [_result_text(m.content) for m in messages if isinstance(m, AIMessage) and _result_text(m.content).strip()]
    llm = get_haiku().with_structured_output(DomainFindingOutput)
    return await llm.ainvoke(
        [
            SystemMessage(content="Write the final research finding using ONLY the tool results below." + IDENTITY_RULE),
            HumanMessage(
                content=f"{instruction}\n\nTool results gathered:\n" + "\n\n".join(gathered or ["(none)"])
                + ("\n\nYour notes so far:\n" + "\n".join(notes[-2:]) if notes else "")
            ),
        ]
    )
