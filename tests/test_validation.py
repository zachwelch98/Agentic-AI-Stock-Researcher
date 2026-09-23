from agents.validation import usable_findings, validate_findings
from app.graph.state import make_initial_state

DOMAINS = ("news", "financials", "leadership", "technical")


def _finding(domain, identity_ok=True, citations=(), fetched=()):
    return {
        "ticker": "TE",
        "domain": domain,
        "summary": "s",
        "citations": list(citations),
        "raw_data": {},
        "identity_ok": identity_ok,
        "fetched_urls": list(fetched),
        "validation_issues": [] if identity_ok else ["identified company mismatch"],
    }


def _state(findings):
    state = make_initial_state("", "job", user_supplied_ticker="TE")
    state["final_candidates"] = ["TE"]
    state["domain_findings"] = findings
    return state


async def test_mixed_company_domain_is_excluded_but_run_continues():
    state = _state([_finding(d, identity_ok=(d != "financials")) for d in DOMAINS])

    result = await validate_findings(state)

    assert result["excluded_domains"] == ["TE:financials"]
    assert result.get("status") != "failed"
    assert any("TE:financials excluded" in w for w in result["data_quality_warnings"])
    assert {f["domain"] for f in usable_findings({**state, **result})} == {"news", "leadership", "technical"}


async def test_too_few_valid_domains_ends_with_insufficient_data():
    state = _state([_finding(d, identity_ok=(d in ("news", "technical"))) for d in DOMAINS])

    result = await validate_findings(state)

    assert result["status"] == "failed"
    assert result["final_report"]["outcome"] == "insufficient_data"


async def test_missing_domain_counts_against_threshold():
    state = _state([_finding("news"), _finding("technical")])

    assert (await validate_findings(state))["status"] == "failed"


async def test_unread_url_citation_is_dropped_and_warned():
    cite_read = {"source_id": "fetch", "identifier": "https://a.com/read", "title": "r"}
    cite_unread = {"source_id": "get_sec_filings", "identifier": "https://sec.gov/10q", "title": "u"}
    cite_ticker = {"source_id": "get_technical_indicators", "identifier": "TE", "title": "t"}
    cite_vague = {"source_id": "get_sec_filings", "identifier": "Form 4 filings (July-August 2026)", "title": "v"}
    findings = [
        _finding("financials", citations=[cite_read, cite_unread, cite_ticker, cite_vague], fetched=["https://a.com/read"]),
        *[_finding(d) for d in ("news", "leadership", "technical")],
    ]
    state = _state(findings)

    result = await validate_findings(state)

    assert any("https://sec.gov/10q" in w for w in result["data_quality_warnings"])
    assert any("Form 4 filings" in w for w in result["data_quality_warnings"])
    fin = next(f for f in usable_findings({**state, **result}) if f["domain"] == "financials")
    assert [c["identifier"] for c in fin["citations"]] == ["https://a.com/read", "TE"]


async def test_stub_findings_without_bookkeeping_keys_pass():
    stub = [{"ticker": "TE", "domain": d, "summary": "stub", "citations": [{"identifier": "https://x", "source_id": "s", "title": "t"}], "raw_data": {}} for d in DOMAINS]

    result = await validate_findings(_state(stub))

    assert result.get("status") != "failed" and result["data_quality_warnings"] == []
