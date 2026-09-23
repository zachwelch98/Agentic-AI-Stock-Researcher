"""In-memory job store + async background runner.

Plain dict keyed by job_id — fine for v1, doesn't survive restarts, and only
works with a single uvicorn worker (the dict isn't shared across processes).
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langgraph.graph.state import CompiledStateGraph

from app.api.schemas import JobStatus
from app.config import get_settings
from app.graph.state import make_initial_state
from app.local_trace import start_run_trace

logger = logging.getLogger(__name__)


@dataclass
class Job:
    job_id: str
    industry_query: str | None = None
    ticker: str | None = None
    status: JobStatus = JobStatus.PENDING
    graph_status: str | None = None
    result: dict | None = None
    error: str | None = None
    trace_dir: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


_jobs: dict[str, Job] = {}


def create_job(industry_query: str | None = None, ticker: str | None = None) -> Job:
    job = Job(job_id=str(uuid.uuid4()), industry_query=industry_query, ticker=ticker)
    _jobs[job.job_id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


async def run_job(job_id: str, graph: CompiledStateGraph) -> None:
    job = _jobs[job_id]
    job.status = JobStatus.RUNNING
    # Local run logging (LOCAL_TRACE_DIR): a passive callback that only observes
    # events the graph already emits — no extra LLM/tool calls. None when off.
    tracer = start_run_trace(job_id, job.industry_query, job.ticker, get_settings())
    config = None
    if tracer is not None:
        job.trace_dir = str(tracer.run_dir)
        config = {"callbacks": [tracer], "metadata": {"job_id": job_id}}
    try:
        initial_state = make_initial_state(job.industry_query or "", job_id, user_supplied_ticker=job.ticker)
        last_state: dict = dict(initial_state)
        # stream_mode="values" yields the full, already-reduced state after each
        # super-step, so job.graph_status tracks live progress (planning ->
        # screening -> researching -> done) without having to hand-merge the
        # Send-fanned-out `domain_findings` reducer ourselves.
        async for state_snapshot in graph.astream(initial_state, config=config, stream_mode="values"):
            last_state = state_snapshot
            job.graph_status = state_snapshot.get("status")
        job.result = last_state.get("final_report")
        if last_state.get("status") == "failed":
            # Graph-level failure (unresolvable ticker, too little usable research
            # data): not an exception, but not a successful report either.
            job.status = JobStatus.FAILED
            job.error = last_state.get("error")
        else:
            job.status = JobStatus.DONE
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        job.status = JobStatus.FAILED
        job.error = str(exc)
    finally:
        if tracer is not None:
            tracer.finish(job.status.value, job.error)
