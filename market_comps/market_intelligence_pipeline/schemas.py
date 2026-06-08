from typing import List, Optional
from pydantic import BaseModel, Field

class MAEvent(BaseModel):
    company: str = Field(description="The target company")
    acquirer: str = Field(description="The acquiring company")
    announcement_date: str = Field(description="Date the deal was announced (YYYY-MM or YYYY-MM-DD)")
    deal_value: Optional[float] = Field(description="The numeric deal value, if known")
    deal_value_currency: Optional[str] = Field(description="Currency of the deal value, e.g. USD")
    deal_status: str = Field(description="Status of the deal, e.g. Announced, Completed, Rumored")
    strategic_rationale: str = Field(description="Brief explanation of why the deal happened")
    source_url: Optional[str] = Field(description="Source URL, if available")
    confidence: str = Field(description="Confidence in this data: HIGH, MEDIUM, LOW")

class FundraisingEvent(BaseModel):
    company: str = Field(description="The company raising funds")
    round_type: str = Field(description="Round type (e.g., Series A, Seed, Growth)")
    amount: Optional[float] = Field(description="Numeric amount raised")
    currency: Optional[str] = Field(description="Currency, e.g. USD")
    date: str = Field(description="Date of the round (YYYY-MM or YYYY-MM-DD)")
    lead_investors: List[str] = Field(description="List of lead investors")
    other_investors: List[str] = Field(description="List of other participating investors")
    post_money_valuation: Optional[float] = Field(description="Post-money valuation if known")
    source_url: Optional[str] = Field(description="Source URL, if available")
    confidence: str = Field(description="Confidence in this data: HIGH, MEDIUM, LOW")

class IPOEvent(BaseModel):
    company: str = Field(description="The company that went public")
    ticker: str = Field(description="The stock ticker symbol")
    exchange: str = Field(description="The stock exchange, e.g. NYSE, NASDAQ")
    ipo_date: str = Field(description="Date of the IPO (YYYY-MM or YYYY-MM-DD)")
    amount_raised: Optional[float] = Field(description="Amount raised in the IPO")
    valuation_at_ipo: Optional[float] = Field(description="Company valuation at IPO")
    source_url: Optional[str] = Field(description="Source URL, if available")
    confidence: str = Field(description="Confidence in this data: HIGH, MEDIUM, LOW")

class PublicComp(BaseModel):
    company: str = Field(description="The comparable public company name")
    ticker: str = Field(description="The stock ticker symbol")
    exchange: str = Field(description="The stock exchange")
    rationale: str = Field(description="Why this company is a good comparable")
    confidence: str = Field(description="Confidence in this data: HIGH, MEDIUM, LOW")

class Competitor(BaseModel):
    company: str = Field(description="The competitor company name")
    sub_section: str = Field(description="Sub-section of competition, e.g. Incumbent, Direct Competitor, Indirect Competitor, Start-up, Adjacent Space")
    rationale: str = Field(description="Why this company is a competitor and their market positioning")
    source_url: Optional[str] = Field(description="Source URL, if available")
    confidence: str = Field(description="Confidence in this data: HIGH, MEDIUM, LOW")

class MarketIntelligenceExtraction(BaseModel):
    industry_classification: str = Field(description="The primary industry and sub-industries identified")
    ma_events: List[MAEvent] = Field(default_factory=list, description="Recent M&A events (last 36 months)")
    fundraising_events: List[FundraisingEvent] = Field(default_factory=list, description="Recent fundraising rounds (last 24 months)")
    ipo_events: List[IPOEvent] = Field(default_factory=list, description="Recent IPOs (last 5 years)")
    public_comps: List[PublicComp] = Field(default_factory=list, description="Current public comparable companies")
    competitors: List[Competitor] = Field(default_factory=list, description="Competitors in the market space")

class MAExtractionResponse(BaseModel):
    ma_events: List[MAEvent] = Field(default_factory=list, description="Recent M&A events (last 36 months)")

class FundraisingExtractionResponse(BaseModel):
    fundraising_events: List[FundraisingEvent] = Field(default_factory=list, description="Recent fundraising rounds (last 24 months)")

class IPOExtractionResponse(BaseModel):
    ipo_events: List[IPOEvent] = Field(default_factory=list, description="Recent IPOs (last 5 years)")

class PublicCompsExtractionResponse(BaseModel):
    public_comps: List[PublicComp] = Field(default_factory=list, description="Current public comparable companies")

class CompetitorExtractionResponse(BaseModel):
    competitors: List[Competitor] = Field(default_factory=list, description="Competitors in the market space")
