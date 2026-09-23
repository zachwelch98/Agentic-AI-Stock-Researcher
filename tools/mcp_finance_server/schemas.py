"""Tool I/O Pydantic models for the custom finance MCP server.

These models are the source of truth for the tool schemas documented in the
README — they're also what FastMCP uses to auto-generate the JSON schema
exposed to MCP clients.
"""

from pydantic import BaseModel, Field


class ScreenedCandidate(BaseModel):
    ticker: str
    company_name: str
    market_cap: float
    avg_dollar_volume: float
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    debt_to_equity: float | None = None
    dividend_yield: float | None = None
    composite_score: float
    matched_industry_label: str | None = None
    match_score: float | None = None


class ScreenIndustryResult(BaseModel):
    candidates: list[ScreenedCandidate] = Field(default_factory=list)
    matched_industry_label: str | None = None
    match_score: float | None = None


class FundamentalsResult(BaseModel):
    ticker: str
    company_name: str | None = None
    website: str | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    debt_to_equity: float | None = None
    dividend_yield: float | None = None
    eps: float | None = None
    revenue_growth: float | None = None
    profit_margin: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None


class PriceBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PriceHistoryResult(BaseModel):
    ticker: str
    range: str
    bars: list[PriceBar] = Field(default_factory=list)


class CompanyNewsItem(BaseModel):
    headline: str
    summary: str
    source: str
    url: str
    datetime: str


class CompanyNewsResult(BaseModel):
    ticker: str
    from_date: str
    to_date: str
    articles: list[CompanyNewsItem] = Field(default_factory=list)
