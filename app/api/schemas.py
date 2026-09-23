from enum import Enum

from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ResearchRequest(BaseModel):
    industry: str | None = Field(
        default=None, min_length=1, description="Industry name to screen, e.g. 'Regional Banks'"
    )
    ticker: str | None = Field(
        default=None, min_length=1, description="A specific ticker to research directly, skipping screening"
    )

    @model_validator(mode="after")
    def exactly_one_of_industry_or_ticker(self) -> "ResearchRequest":
        if bool(self.industry) == bool(self.ticker):
            raise ValueError("Provide exactly one of 'industry' or 'ticker'.")
        return self


class ResearchAcceptedResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    industry_query: str | None
    ticker: str | None = None
    graph_status: str | None = Field(
        default=None, description="Current graph node status, e.g. 'screening', 'researching', 'done'"
    )
    result: dict | None = None
    error: str | None = None
