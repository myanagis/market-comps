# market_comps/scanner.py
"""
MarketScanner: Uses the LLM to identify publicly-traded comparable companies
for a given query (company name, industry, or sub-industry).

This is Step 1 of the two-step pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from market_comps.llm_client import LLMClient
from market_comps.models import CompanyCandidate, LLMUsage, ScanFilters

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert equity research analyst specializing in identifying \
publicly traded comparable companies. You have deep knowledge of global equity markets, \
sectors, industries, and company fundamentals. Always respond with valid JSON only."""

_SCAN_PROMPT_TEMPLATE = """\
I need to identify publicly traded comparable companies for the following query:

QUERY: {query}

{filter_block}

Please identify exactly {n_candidates} publicly traded companies that are the \
best public market comparables for this query. For each company return:
  - "name": full company name
  - "ticker": primary trading ticker symbol (use the main US ticker if dual-listed; \
for purely international companies use their primary exchange ticker)
  - "exchange": exchange where the ticker trades (e.g., NYSE, NASDAQ, LSE, TSX, ASX)

Return a JSON object with a single key "companies" whose value is an array of objects \
with the fields above. Only include real, currently publicly traded companies.

Example format:
{{
  "companies": [
    {{"name": "Salesforce Inc.", "ticker": "CRM", "exchange": "NYSE"}},
    {{"name": "ServiceNow Inc.", "ticker": "NOW", "exchange": "NYSE"}}
  ]
}}
"""


def _build_filter_block(filters: ScanFilters) -> str:
    """Build a natural-language filter description to inject into the prompt."""
    parts: list[str] = []

    if filters.countries:
        parts.append(f"- Restrict to companies headquartered in: {', '.join(filters.countries)}")
    if filters.exchanges:
        parts.append(f"- Restrict to companies listed on: {', '.join(filters.exchanges)}")
    if filters.sectors:
        parts.append(f"- Restrict to GICS sectors: {', '.join(filters.sectors)}")
    if filters.industries:
        parts.append(f"- Restrict to industries / sub-industries: {', '.join(filters.industries)}")
    if filters.min_market_cap_usd is not None:
        mc = filters.min_market_cap_usd
        parts.append(f"- Minimum market cap: ${mc/1e9:.1f}B USD")
    if filters.max_market_cap_usd is not None:
        mc = filters.max_market_cap_usd
        parts.append(f"- Maximum market cap: ${mc/1e9:.1f}B USD")

    if not parts:
        return ""
    return "FILTERS (apply all of the following):\n" + "\n".join(parts) + "\n"


class MarketScanner:
    """
    Identifies a list of publicly-traded comparable companies using an LLM.

    Parameters
    ----------
    llm_client : LLMClient, optional
        Provide your own client (useful for testing / dependency injection).
        If None, a default client is created from settings.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm = llm_client or LLMClient()

    def scan(
        self,
        query: str,
        n_candidates: int = 10,
        filters: Optional[ScanFilters] = None,
    ) -> tuple[list[CompanyCandidate], LLMUsage]:
        """
        Scan for comparable companies.

        Parameters
        ----------
        query : str
            Free-text description of the target company, industry, or sub-industry.
        n_candidates : int
            How many comparable companies to return.
        filters : ScanFilters, optional
            Optional filters to narrow the search.

        Returns
        -------
        (candidates, usage)
            candidates – list of CompanyCandidate
            usage      – LLMUsage from this call
        """
        filters = filters or ScanFilters()
        filter_block = _build_filter_block(filters)

        prompt = _SCAN_PROMPT_TEMPLATE.format(
            query=query,
            n_candidates=n_candidates,
            filter_block=filter_block,
        )

        logger.info("MarketScanner: scanning for %d comps for query=%r", n_candidates, query)
        data, usage = self._llm.structured_output(
            prompt=prompt,
            json_schema={
                "type": "object",
                "properties": {
                    "companies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "ticker": {"type": "string"},
                                "exchange": {"type": "string"},
                            },
                            "required": ["name", "ticker", "exchange"],
                        },
                    }
                },
                "required": ["companies"],
            },
            system_prompt=_SYSTEM_PROMPT,
        )

        raw_companies = data.get("companies", [])
        candidates: list[CompanyCandidate] = []
        for item in raw_companies:
            try:
                candidates.append(
                    CompanyCandidate(
                        name=item.get("name", ""),
                        ticker=str(item.get("ticker", "")).upper().strip(),
                        exchange=item.get("exchange", ""),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed candidate %s: %s", item, exc)

        logger.info("MarketScanner: found %d candidates", len(candidates))
        return candidates, usage
