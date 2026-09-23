"""Golden-set eval harness.

    uv run uvicorn app.main:app &
    uv run python eval/run_eval.py --base-url http://localhost:8000

POSTs each case in golden_set.json (by industry, or directly by ticker) to
/research, polls to completion, and runs a structural check: exactly 1 final
candidate, >=3 citations across its findings, all 4 domains present, or a
valid insufficient_candidates outcome (the deliberately hostile biotech case
is expected to take this path). Prints a pass-rate summary and writes
eval/last_run_results.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
RESULTS_PATH = Path(__file__).parent / "last_run_results.json"
POLL_INTERVAL_SECONDS = 5
MAX_POLL_ATTEMPTS = 180  # 15 minutes per case


def check_result(job: dict) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if job["status"] != "done":
        problems.append(f"job did not complete (status={job['status']}, error={job.get('error')})")
        return False, problems

    result = job.get("result")
    if result is None:
        problems.append("no result payload")
        return False, problems

    if result.get("outcome") == "insufficient_candidates":
        return True, problems  # a valid terminal outcome, not a failure

    candidates = result.get("candidates", [])
    if len(candidates) != 1:
        problems.append(f"expected 1 candidate, got {len(candidates)}")

    for t in result.get("tickers", []):
        findings = t.get("domain_findings", [])
        domains = {f["domain"] for f in findings}
        if domains != {"news", "financials", "leadership", "technical"}:
            problems.append(f"{t['ticker']}: missing domains, has {sorted(domains)}")
        total_citations = sum(len(f.get("citations", [])) for f in findings)
        if total_citations < 3:
            problems.append(f"{t['ticker']}: only {total_citations} citations across all domains (expected >= 3)")

    return len(problems) == 0, problems


def run_case(client: httpx.Client, base_url: str, case: dict) -> dict:
    payload = {"ticker": case["ticker"]} if "ticker" in case else {"industry": case["industry"]}
    resp = client.post(f"{base_url}/research", json=payload)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    job: dict = {}
    for _ in range(MAX_POLL_ATTEMPTS):
        job_resp = client.get(f"{base_url}/jobs/{job_id}")
        job_resp.raise_for_status()
        job = job_resp.json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    passed, problems = check_result(job)
    return {"id": case["id"], **payload, "passed": passed, "problems": problems, "job": job}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    golden_set = json.loads(GOLDEN_SET_PATH.read_text())
    results = []
    with httpx.Client(timeout=30.0) as client:
        for case in golden_set:
            label = case.get("ticker") or case.get("industry")
            print(f"Running {case['id']} ({label})...")
            result = run_case(client, args.base_url, case)
            results.append(result)
            print(f"  {'PASS' if result['passed'] else 'FAIL'}")
            for p in result["problems"]:
                print(f"    - {p}")

    passed_count = sum(1 for r in results if r["passed"])
    print(f"\n{passed_count}/{len(results)} cases passed")
    for r in results:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['id']}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    sys.exit(0 if passed_count == len(results) else 1)


if __name__ == "__main__":
    main()
