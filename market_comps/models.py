# market_comps/models.py
"""
Pydantic data models for the Market Comps Finder.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input / Filter models
# ---------------------------------------------------------------------------


class ScanFilters(BaseModel):
    """Optional filters applied during the market scanning step."""

    countries: list[str] = Field(
        default_factory=list,
        description="ISO country names/codes to restrict results to, e.g. ['United States', 'Canada'].",
    )
    exchanges: list[str] = Field(
        default_factory=list,
        description="Exchange names to restrict results to, e.g. ['NYSE', 'NASDAQ'].",
    )
    sectors: list[str] = Field(
        default_factory=list,
        description="GICS sectors to restrict results to, e.g. ['Technology', 'Healthcare'].",
    )
    industries: list[str] = Field(
        default_factory=list,
        description="Industries / sub-industries to restrict results to.",
    )
    min_market_cap_usd: Optional[float] = Field(
        None,
        description="Minimum market cap in USD (e.g. 1e9 for $1B).",
    )
    max_market_cap_usd: Optional[float] = Field(
        None,
        description="Maximum market cap in USD.",
    )


# ---------------------------------------------------------------------------
# Intermediate / internal models
# ---------------------------------------------------------------------------


class CompanyCandidate(BaseModel):
    """A candidate comparable company identified by the scanner."""

    name: str
    ticker: str = Field(description="Primary exchange ticker, e.g. 'CRM'.")
    exchange: str = Field(default="", description="Exchange, e.g. 'NYSE'.")


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class CompanyMetrics(BaseModel):
    """Full metric set for one comparable company."""

    # Identification
    name: str
    ticker: str
    exchange: str = ""
    country: str = ""
    sector: str = ""
    industry: str = ""
    description: str = ""

    # Market data
    market_cap_usd: Optional[float] = None
    ev_usd: Optional[float] = None  # Enterprise Value

    # TTM (Trailing Twelve Months) financials
    revenue_ttm_usd: Optional[float] = None
    ev_to_revenue_ttm: Optional[float] = None  # EV / TTM Revenue

    # NTM (Next Twelve Months) estimates — from analyst consensus where available
    revenue_ntm_usd: Optional[float] = None
    ev_to_revenue_ntm: Optional[float] = None  # EV / NTM Revenue

    # Supplemental
    gross_margin_pct: Optional[float] = None
    ebitda_margin_pct: Optional[float] = None
    revenue_growth_yoy_pct: Optional[float] = None

    # Data quality flag — True if yfinance returned valid data
    data_available: bool = True
    data_notes: str = ""


class LLMUsage(BaseModel):
    """Tracks token consumption and estimated costs across all LLM calls."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    call_count: int = 0

    def add(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        input_price_per_m: float,
        output_price_per_m: float,
    ) -> None:
        """Accumulate tokens and cost from one LLM call."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.estimated_cost_usd += (
            prompt_tokens / 1_000_000 * input_price_per_m
            + completion_tokens / 1_000_000 * output_price_per_m
        )
        self.call_count += 1


class CompsResult(BaseModel):
    """Final output from CompsEngine.run()."""

    query: str
    filters: ScanFilters = Field(default_factory=ScanFilters)
    candidates_found: int = 0
    comps: list[CompanyMetrics] = Field(default_factory=list)
    llm_usage: LLMUsage = Field(default_factory=LLMUsage)
    model_used: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    errors: list[str] = Field(default_factory=list)
