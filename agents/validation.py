"""Deterministic gate between the researchers and the synthesizer.

No LLM. It rejects what the synthesizer must not see: findings about a
different company than the one resolved by `resolve_identity`, and citations
to pages the agent never actually read. If too few domains survive, the run
ends with an explicit `insufficient_data` report instead of a confident-looking
synthesis over mixed or missing data.
"""

from app.graph.state import DomainFinding, ResearchState

DOMAINS = ("news", "financials", "leadership", "technical")
MIN_VALID_DOMAINS = 3


def _unsupported_citations(finding: DomainFinding) -> list[str]:
    """Citations the agent has no evidence for: URLs it never got content from, and
    free-text "identifiers" that aren't a URL or the ticker. Findings without
    `fetched_urls` (e.g. stub researchers) are given the benefit of the doubt."""
    if "fetched_urls" not in finding:
        return []
    evidence = set(finding["fetched_urls"])
    ticker = finding["ticker"]
    unsupported = []
    for c in finding.get("citations", []):
        ident = c.get("identifier", "")
        if ident.startswith("http"):
            if ident not in evidence:
                unsupported.append(ident)
        elif ident != ticker:
            # Neither a page nor the ticker's own tool data, e.g. "Form 4 filings
            # (July-August 2026)": a description of a source, not a source.
            unsupported.append(ident)
    return unsupported


def usable_findings(state: ResearchState) -> list[DomainFinding]:
    """Findings that passed validation, with unsupported citations removed —
    what the synthesizer and the final report are allowed to use."""
    excluded = set(state.get("excluded_domains") or [])
    usable = []
    for finding in state["domain_findings"]:
        if f"{finding['ticker']}:{finding['domain']}" in excluded:
            continue
        unsupported = set(_unsupported_citations(finding))
        if unsupported:
            finding = {
                **finding,
                "citations": [c for c in finding["citations"] if c.get("identifier") not in unsupported],
            }
        usable.append(finding)
    return usable


async def validate_findings(state: ResearchState) -> dict:
    excluded: list[str] = []
    warnings: list[str] = []
    for finding in state["domain_findings"]:
        key = f"{finding['ticker']}:{finding['domain']}"
        if not finding.get("identity_ok", True):
            excluded.append(key)
            warnings.append(
                f"{key} excluded: findings did not match the resolved company "
                f"({'; '.join(finding.get('validation_issues') or ['identity mismatch'])})"
            )
        for url in _unsupported_citations(finding):
            warnings.append(f"{key}: citation dropped, page was never successfully read: {url}")

    usable_domains: dict[str, set[str]] = {}
    for finding in state["domain_findings"]:
        if f"{finding['ticker']}:{finding['domain']}" not in excluded:
            usable_domains.setdefault(finding["ticker"], set()).add(finding["domain"])

    for ticker in state["final_candidates"]:
        got = usable_domains.get(ticker, set())
        missing = [d for d in DOMAINS if d not in got]
        if missing and len(got) >= MIN_VALID_DOMAINS:
            warnings.append(f"{ticker}: no findings for {', '.join(missing)}")
        if len(got) < MIN_VALID_DOMAINS:
            message = (
                f"Only {len(got)} of {len(DOMAINS)} research domains produced usable data for {ticker}; "
                "refusing to synthesize a report."
            )
            return {
                "status": "failed",
                "error": message,
                "excluded_domains": excluded,
                "data_quality_warnings": warnings,
                "final_report": {
                    "industry_query": state["industry_query"],
                    "candidates": state["final_candidates"],
                    "outcome": "insufficient_data",
                    "message": message,
                    "data_quality_warnings": warnings,
                },
            }
    return {"excluded_domains": excluded, "data_quality_warnings": warnings, "status": "synthesizing"}
