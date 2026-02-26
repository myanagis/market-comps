# market_comps/competition/competition_finder.py
"""
CompetitionFinder — uses the LLMChorus to discover both public and private
competitors for a given company, then runs a structured extraction pass.

Flow:
  Phase 1: LLMChorus asks 5 models for competitors in parallel.
  Phase 2: Summarizer LLM deduplicates + extracts structured JSON list.
  Returns: CompetitionResult with a clean list of Competitor objects + free-text landscape.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Callable

from market_comps.llm_client import LLMClient
from market_comps.models import LLMUsage
from market_comps.cross_checker import LLMChorus, ChorusResult, ModelResponse
from market_comps.cross_checker.cross_checker import DEFAULT_MODELS, DEFAULT_SUMMARY_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_COMPETITION_QUESTION_TEMPLATE = """\
Identify the key competitors for the following company:

Company: {company}
Description: {description}
{competitors_to_include}

List BOTH public AND private competitors. For each competitor provide:
- Company name
- Whether it is public (traded) or private
- Brief description (1-2 sentences) of what they do
- Key differentiator vs the subject company
- If PUBLIC: stock ticker, market cap (USD), last reported annual revenue (USD), EV/Revenue multiple
- If PRIVATE: latest funding round type (Seed/Series A/.../Unknown), amount raised (USD), year of latest round, notable investors, and details if they have been acquired (acquirer, exit amount USD, exit year)

Be specific and factual. Cite your sources with full URLs.
Only include companies you are confident are real competitors with verifiable information.
"""

_EXTRACTION_SYSTEM = """\
You are a financial data extraction engine. Extract structured competitor data from the provided research.

Rules:
- Only include companies explicitly mentioned in the research below.
- Do NOT invent companies, tickers, or financial figures.
- If a data point is missing or uncertain, use null.
- Tickers must be real exchange-listed symbols (e.g. "CRM", "ADBE"); use null for private companies.
- Round monetary values to reasonable precision (e.g. market cap in billions).
- Deduplicate: if the same company appears multiple times, keep a single merged record.

Return ONLY valid JSON in this exact structure (no markdown, no prose):
{
  "landscape": "<2-4 sentence summary of the competitive landscape>",
  "competitors": [
    {
      "name": "<company name>",
      "type": "<public|private>",
      "ticker": "<TICKER or null>",
      "country": "<headquarters country, e.g. United States, or null>",
      "description": "<what they do, 1-2 sentences>",
      "differentiation": "<key differentiator vs subject company, 1 sentence>",
      "market_cap_usd": <number in USD or null>,
      "revenue_usd": <annual revenue in USD or null>,
      "ev_to_revenue": <EV/Revenue multiple as float or null>,
      "latest_round": "<round type e.g. Series C, or null>",
      "amount_raised_usd": <total amount raised in USD or null>,
      "funding_year": <year as integer or null>,
      "investors": ["<investor 1>", "<investor 2>"],
      "exit_acquirer": "<name of acquiring company or null>",
      "exit_amount_usd": <exit valuation in USD or null>,
      "exit_date": "<year or YYYY-MM or null>",
      "source_urls": ["<url1>", "<url2>"]
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Competitor:
    name: str
    type: str                        # "public" or "private"
    ticker: Optional[str] = None
    description: str = ""
    differentiation: str = ""
    country: Optional[str] = None    # headquartered country
    # Public fields
    market_cap_usd: Optional[float] = None
    revenue_usd: Optional[float] = None
    ev_to_revenue: Optional[float] = None
    # Private fields
    latest_round: Optional[str] = None
    amount_raised_usd: Optional[float] = None
    funding_year: Optional[int] = None
    investors: list[str] = field(default_factory=list)
    exit_acquirer: Optional[str] = None
    exit_amount_usd: Optional[float] = None
    exit_date: Optional[str] = None
    # Sources
    source_urls: list[str] = field(default_factory=list)


@dataclass
class CompetitionResult:
    company: str
    description: str
    competitors: list[Competitor] = field(default_factory=list)
    landscape: str = ""
    chorus_result: Optional[ChorusResult] = None
    extraction_usage: LLMUsage = field(default_factory=LLMUsage)
    errors: list[str] = field(default_factory=list)

    @property
    def public_competitors(self) -> list[Competitor]:
        return [c for c in self.competitors if c.type == "public"]

    @property
    def private_competitors(self) -> list[Competitor]:
        return [c for c in self.competitors if c.type == "private"]

    @property
    def total_llm_usage(self) -> LLMUsage:
        agg = LLMUsage()
        if self.chorus_result:
            t = self.chorus_result.total_usage
            agg.total_prompt_tokens += t.total_prompt_tokens
            agg.total_completion_tokens += t.total_completion_tokens
            agg.total_tokens += t.total_tokens
            agg.estimated_cost_usd += t.estimated_cost_usd
            agg.call_count += t.call_count
        agg.total_prompt_tokens += self.extraction_usage.total_prompt_tokens
        agg.total_completion_tokens += self.extraction_usage.total_completion_tokens
        agg.total_tokens += self.extraction_usage.total_tokens
        agg.estimated_cost_usd += self.extraction_usage.estimated_cost_usd
        agg.call_count += self.extraction_usage.call_count
        return agg


# ---------------------------------------------------------------------------
# CompetitionFinder
# ---------------------------------------------------------------------------

class CompetitionFinder:
    """
    Discover and structure the competitive landscape for a company.

    1. Queries the LLMChorus (5 models) for competitors.
    2. Runs a single extraction LLM pass to deduplicate + structure the responses as JSON.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chorus_models: Optional[list[str]] = None,
        summary_model: str = DEFAULT_SUMMARY_MODEL,
    ) -> None:
        self._chorus = LLMChorus(api_key=api_key)
        self._llm = LLMClient(api_key=api_key)
        self._chorus_models = chorus_models or DEFAULT_MODELS
        self._summary_model = summary_model

    def run(self, company: str, description: str = "", competitors_to_include: str = "", **kwargs) -> CompetitionResult:
        """
        Run the research.
        Kwargs:
            on_model_complete: Optional callback for each model result.
        """
        on_model_complete = kwargs.get("on_model_complete")
        result = CompetitionResult(company=company, description=description)

        # ── Phase 1: Chorus research ──────────────────────────────────────
        logger.info("CompetitionFinder: Phase 1 — querying chorus for %s", company)
        
        comp_include_str = ""
        if competitors_to_include.strip():
            comp_include_str = f"Make sure to evaluate and include these specific competitors if relevant: {competitors_to_include.strip()}"

        question = _COMPETITION_QUESTION_TEMPLATE.format(
            company=company,
            description=description or "Not specified",
            competitors_to_include=comp_include_str,
        )
        try:
            chorus_result = self._chorus.run(
                question=question,
                models=self._chorus_models,
                summary_model=self._summary_model,
                temperature=0.3,
                on_model_complete=on_model_complete,
            )
            result.chorus_result = chorus_result
        except Exception as exc:
            logger.error("Chorus phase failed: %s", exc)
            result.errors.append(f"Chorus error: {exc}")
            return result

        # Collect all raw model responses for extraction
        all_responses = "\n\n---\n\n".join(
            f"### {r.model}\n{r.content}"
            for r in chorus_result.responses
            if r.success
        )
        if chorus_result.summary:
            all_responses += f"\n\n---\n\n### Synthesized Summary\n{chorus_result.summary}"

        if not all_responses.strip():
            result.errors.append("No successful model responses received.")
            return result

        # ── Phase 2: Structured extraction ────────────────────────────────
        logger.info("CompetitionFinder: Phase 2 — extracting structured data")
        extraction_prompt = (
            f"Company being analyzed: **{company}**\n"
            f"Description: {description or 'Not specified'}\n\n"
            f"Research from multiple AI models:\n\n{all_responses}"
        )
        try:
            raw, usage = self._llm.simple_text(
                prompt=extraction_prompt,
                system_prompt=_EXTRACTION_SYSTEM,
                model=self._summary_model,
                temperature=0.1,
            )
            result.extraction_usage = usage
            parsed = _parse_extraction(raw)
            result.landscape = parsed.get("landscape", "")
            result.competitors = _parse_competitors(parsed.get("competitors", []))
        except Exception as exc:
            logger.error("Extraction failed: %s", exc)
            result.errors.append(f"Extraction error: {exc}")

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_extraction(raw: str) -> dict:
    """Strip markdown code fences and parse JSON."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 1)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```", 1)[0].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse failed: %s | raw: %s", exc, s[:500])
        return {}


def _parse_competitors(raw: list[dict]) -> list[Competitor]:
    out = []
    for item in raw:
        try:
            out.append(Competitor(
                name=item.get("name", "Unknown"),
                type=item.get("type", "private").lower(),
                ticker=item.get("ticker") or None,
                description=item.get("description", ""),
                differentiation=item.get("differentiation", ""),
                country=item.get("country") or None,
                market_cap_usd=_to_float(item.get("market_cap_usd")),
                revenue_usd=_to_float(item.get("revenue_usd")),
                ev_to_revenue=_to_float(item.get("ev_to_revenue")),
                latest_round=item.get("latest_round") or None,
                amount_raised_usd=_to_float(item.get("amount_raised_usd")),
                funding_year=_to_int(item.get("funding_year")),
                investors=item.get("investors") or [],
                exit_acquirer=item.get("exit_acquirer") or None,
                exit_amount_usd=_to_float(item.get("exit_amount_usd")),
                exit_date=item.get("exit_date") or None,
                source_urls=item.get("source_urls") or [],
            ))
        except Exception as exc:
            logger.warning("Skipping competitor row: %s | %s", exc, item)
    return out


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
