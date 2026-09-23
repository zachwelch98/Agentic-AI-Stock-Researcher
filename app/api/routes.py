import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException, Request, status

from app.api.schemas import JobResponse, ResearchAcceptedResponse, ResearchRequest
from app.jobs import create_job, get_job, run_job

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@router.post("/research", response_model=ResearchAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_research(payload: ResearchRequest, request: Request) -> ResearchAcceptedResponse:
    started_at = time.monotonic()
    job = create_job(payload.industry, payload.ticker)

    graph = request.app.state.research_graph
    task = asyncio.create_task(run_job(job.job_id, graph))
    request.app.state.background_tasks.add(task)
    task.add_done_callback(request.app.state.background_tasks.discard)

    logger.info(
        "POST /research route=research status=202 duration_ms=%.1f job_id=%s industry=%r ticker=%r",
        (time.monotonic() - started_at) * 1000,
        job.job_id,
        payload.industry,
        payload.ticker,
    )
    return ResearchAcceptedResponse(job_id=job.job_id, status=job.status)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str) -> JobResponse:
    started_at = time.monotonic()
    job = get_job(job_id)
    if job is None:
        logger.info(
            "GET /jobs/%s route=jobs status=404 duration_ms=%.1f",
            job_id,
            (time.monotonic() - started_at) * 1000,
        )
        raise HTTPException(status_code=404, detail="Job not found")

    logger.info(
        "GET /jobs/%s route=jobs status=200 duration_ms=%.1f job_status=%s",
        job_id,
        (time.monotonic() - started_at) * 1000,
        job.status,
    )
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        industry_query=job.industry_query,
        ticker=job.ticker,
        graph_status=job.graph_status,
        result=job.result,
        error=job.error,
    )
