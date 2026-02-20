# market_comps/metrics_fetcher.py
"""
MetricsFetcher: Enriches CompanyCandidate objects with live financial data
from yfinance (TTM and NTM metrics) plus an LLM-generated description.

This is Step 2 of the two-step pipeline.

Data mapping (yfinance info keys):
  - market cap    → info['marketCap']
  - EV            → info['enterpriseValue']
  - TTM revenue   → info['totalRevenue']
  - NTM revenue   → info['revenueEstimates'] quarterly forward estimates (summed)
                    or derived from info['revenueGrowth'] applied to TTM if estimates unavailable
  - Gross margin  → info['grossMargins']
  - EBITDA margin → info['ebitdaMargins']
  - Rev growth    → info['revenueGrowth']
  - Country       → info['country']
  - Sector        → info['sector']
  - Industry      → info['industry']
  - Exchange      → info['exchange']
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import yfinance as yf

from market_comps.llm_client import LLMClient
from market_comps.models import CompanyCandidate, CompanyMetrics, LLMUsage

logger = logging.getLogger(__name__)

_DESC_SYSTEM = (
    "You are a financial analyst. Write a concise 1-2 sentence business description "
    "for the company provided. Focus on what the company does, its main products/services, "
    "and its primary market. Do not include financial figures."
)

_DESC_PROMPT_TEMPLATE = (
    "Write a 1-2 sentence business description for {name} (ticker: {ticker}), "
    "a {industry} company in the {sector} sector. "
    "The company is headquartered in {country}."
)

_BATCH_DESC_PROMPT_TEMPLATE = """\
Write a 1-2 sentence business description for each of the following companies.
Return a JSON object with a key "descriptions" that is an array of objects, \
each with "ticker" and "description" fields.

Companies:
{company_list}

Return only valid JSON, no markdown fences.
"""


def _safe_float(value) -> Optional[float]:
    """Convert a value to float, returning None if not possible."""
    try:
        f = float(value)
        return f if f == f else None  # filter NaN
    except (TypeError, ValueError):
        return None


def _fetch_yfinance_info(ticker: str) -> dict:
    """Fetch yfinance info dict for a ticker. Returns empty dict on failure."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        # yfinance returns a minimal stub for unknown tickers
        if not info or info.get("quoteType") is None:
            logger.warning("yfinance returned empty info for %s", ticker)
            return {}
        return info
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return {}


def _estimate_ntm_revenue(info: dict) -> Optional[float]:
    """
    Attempt to derive NTM (Next Twelve Months) revenue from yfinance data.

    Strategy:
      1. Use analyst forward revenue estimate if available (revenueEstimate key).
      2. Fall back to TTM * (1 + revenueGrowth) where revenueGrowth is yfinance's
         forward 12-month consensus growth rate.
      3. Return None if neither is available.
    """
    # Strategy 1: direct analyst estimate
    ntm = _safe_float(info.get("revenueEstimate"))
    if ntm and ntm > 0:
        return ntm

    # Strategy 2: apply forward growth rate to TTM
    ttm = _safe_float(info.get("totalRevenue"))
    growth = _safe_float(info.get("revenueGrowth"))  # e.g. 0.15 for 15%
    if ttm and growth is not None:
        return ttm * (1.0 + growth)

    return None


class MetricsFetcher:
    """
    Fetches financial metrics for a list of candidate companies.

    Parameters
    ----------
    llm_client : LLMClient, optional
        Used for generating company descriptions.
    max_workers : int
        Thread pool size for concurrent yfinance fetches.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_workers: int = 8,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._max_workers = max_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        candidates: list[CompanyCandidate],
    ) -> tuple[list[CompanyMetrics], LLMUsage]:
        """
        Enrich candidates with financial data and LLM descriptions.

        Returns
        -------
        (metrics_list, usage)
            metrics_list – one CompanyMetrics per successful candidate
            usage        – accumulated LLMUsage across all description calls
        """
        cumulative_usage = LLMUsage()
        partial_metrics: list[CompanyMetrics] = []

        # Step A: fetch yfinance data concurrently
        logger.info("MetricsFetcher: fetching yfinance data for %d tickers", len(candidates))
        yf_results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_to_ticker = {
                pool.submit(_fetch_yfinance_info, c.ticker): c for c in candidates
            }
            for future in as_completed(future_to_ticker):
                candidate = future_to_ticker[future]
                try:
                    yf_results[candidate.ticker] = future.result()
                except Exception as exc:
                    logger.warning("Unexpected error for %s: %s", candidate.ticker, exc)
                    yf_results[candidate.ticker] = {}

        # Step B: build CompanyMetrics from yfinance data
        for candidate in candidates:
            info = yf_results.get(candidate.ticker, {})
            metrics = self._build_metrics(candidate, info)
            partial_metrics.append(metrics)

        # Step C: batch-generate descriptions via LLM
        logger.info("MetricsFetcher: generating descriptions via LLM")
        desc_map, usage = self._batch_descriptions(partial_metrics)
        cumulative_usage.add(
            usage.total_prompt_tokens,
            usage.total_completion_tokens,
            0, 0,  # pricing already baked in
        )
        # Manual cost carry-over
        cumulative_usage.estimated_cost_usd += usage.estimated_cost_usd
        cumulative_usage.call_count += usage.call_count

        for m in partial_metrics:
            desc = desc_map.get(m.ticker)
            if desc:
                m.description = desc

        return partial_metrics, cumulative_usage

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_metrics(
        self,
        candidate: CompanyCandidate,
        info: dict,
    ) -> CompanyMetrics:
        """Build a CompanyMetrics object from yfinance info."""
        if not info:
            return CompanyMetrics(
                name=candidate.name,
                ticker=candidate.ticker,
                exchange=candidate.exchange,
                data_available=False,
                data_notes="yfinance returned no data",
            )

        market_cap = _safe_float(info.get("marketCap"))
        ev = _safe_float(info.get("enterpriseValue"))
        revenue_ttm = _safe_float(info.get("totalRevenue"))
        revenue_ntm = _estimate_ntm_revenue(info)

        # EV / Revenue multiples
        ev_to_rev_ttm: Optional[float] = None
        if ev and revenue_ttm and revenue_ttm > 0:
            ev_to_rev_ttm = ev / revenue_ttm

        ev_to_rev_ntm: Optional[float] = None
        if ev and revenue_ntm and revenue_ntm > 0:
            ev_to_rev_ntm = ev / revenue_ntm

        gross_margin = _safe_float(info.get("grossMargins"))
        ebitda_margin = _safe_float(info.get("ebitdaMargins"))
        rev_growth = _safe_float(info.get("revenueGrowth"))

        # Exchange: prefer yfinance's value over LLM-provided
        exchange = info.get("exchange") or candidate.exchange

        return CompanyMetrics(
            name=info.get("longName") or candidate.name,
            ticker=candidate.ticker.upper(),
            exchange=exchange,
            country=info.get("country") or "",
            sector=info.get("sector") or "",
            industry=info.get("industry") or "",
            market_cap_usd=market_cap,
            ev_usd=ev,
            revenue_ttm_usd=revenue_ttm,
            revenue_ntm_usd=revenue_ntm,
            ev_to_revenue_ttm=ev_to_rev_ttm,
            ev_to_revenue_ntm=ev_to_rev_ntm,
            gross_margin_pct=gross_margin * 100 if gross_margin is not None else None,
            ebitda_margin_pct=ebitda_margin * 100 if ebitda_margin is not None else None,
            revenue_growth_yoy_pct=rev_growth * 100 if rev_growth is not None else None,
            data_available=True,
        )

    def _batch_descriptions(
        self,
        metrics_list: list[CompanyMetrics],
    ) -> tuple[dict[str, str], LLMUsage]:
        """
        Generate 1-2 sentence descriptions for all companies in one LLM call.

        Returns (ticker → description dict, LLMUsage).
        """
        if not metrics_list:
            return {}, LLMUsage()

        company_lines = "\n".join(
            f"- {m.ticker}: {m.name} | Sector: {m.sector or 'N/A'} | "
            f"Industry: {m.industry or 'N/A'} | Country: {m.country or 'N/A'}"
            for m in metrics_list
        )
        prompt = _BATCH_DESC_PROMPT_TEMPLATE.format(company_list=company_lines)

        try:
            data, usage = self._llm.structured_output(
                prompt=prompt,
                json_schema={
                    "type": "object",
                    "properties": {
                        "descriptions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "ticker": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                            },
                        }
                    },
                },
            )
            desc_map = {
                item["ticker"].upper(): item["description"]
                for item in data.get("descriptions", [])
                if "ticker" in item and "description" in item
            }
            return desc_map, usage
        except Exception as exc:
            logger.warning("Batch description generation failed: %s", exc)
            return {}, LLMUsage()
