# market_comps/comps_engine.py
"""
CompsEngine: Top-level orchestrator that combines MarketScanner and MetricsFetcher
into a single cohesive pipeline.

Usage::

    engine = CompsEngine()
    result = engine.run(query="Salesforce", n_comps=10)
    print(result.comps)
"""

from __future__ import annotations

import logging
from typing import Optional

from market_comps.llm_client import LLMClient
from market_comps.metrics_fetcher import MetricsFetcher
from market_comps.models import CompsResult, LLMUsage, ScanFilters
from market_comps.scanner import MarketScanner

logger = logging.getLogger(__name__)


class CompsEngine:
    """
    Orchestrates the two-step market comps pipeline:

    1. **Scan**  – MarketScanner identifies candidate comparable companies.
    2. **Enrich** – MetricsFetcher fetches live metrics and descriptions.

    Parameters
    ----------
    api_key : str, optional
        OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
    model : str, optional
        LLM model ID. Defaults to DEFAULT_MODEL env var.
    max_fetch_workers : int
        Concurrency for yfinance fetches (default 8).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_fetch_workers: int = 8,
    ) -> None:
        self._llm = LLMClient(api_key=api_key, model=model)
        self._scanner = MarketScanner(llm_client=self._llm)
        self._fetcher = MetricsFetcher(
            llm_client=self._llm,
            max_workers=max_fetch_workers,
        )

    @property
    def model(self) -> str:
        return self._llm.model

    @model.setter
    def model(self, value: str) -> None:
        self._llm.model = value

    def run(
        self,
        query: str,
        n_comps: int = 10,
        filters: Optional[ScanFilters] = None,
        n_scan_candidates: Optional[int] = None,
    ) -> CompsResult:
        """
        Run the full comps pipeline.

        Parameters
        ----------
        query : str
            Company name, industry, or sub-industry to find comps for.
        n_comps : int
            Desired number of final comparable companies (default 10).
        filters : ScanFilters, optional
            Optional filters applied during the scanning step.
        n_scan_candidates : int, optional
            How many candidates to ask the LLM to generate. Defaults to
            n_comps + 3 to allow for some fallout during metric fetching.

        Returns
        -------
        CompsResult
        """
        filters = filters or ScanFilters()
        n_scan = n_scan_candidates or min(n_comps + 3, 30)

        cumulative_usage = LLMUsage()
        errors: list[str] = []

        # ── Step 1: Scan ──────────────────────────────────────────────────
        logger.info("CompsEngine: STEP 1 — scanning, query=%r, n=%d", query, n_scan)
        try:
            candidates, scan_usage = self._scanner.scan(
                query=query,
                n_candidates=n_scan,
                filters=filters,
            )
            _merge_usage(cumulative_usage, scan_usage)
        except Exception as exc:
            logger.error("Scanner failed: %s", exc)
            errors.append(f"Scanner error: {exc}")
            return CompsResult(
                query=query,
                filters=filters,
                comps=[],
                llm_usage=cumulative_usage,
                model_used=self._llm.model,
                errors=errors,
            )

        # ── Step 2: Enrich ────────────────────────────────────────────────
        logger.info("CompsEngine: STEP 2 — fetching metrics for %d candidates", len(candidates))
        try:
            metrics_list, fetch_usage = self._fetcher.fetch(candidates)
            _merge_usage(cumulative_usage, fetch_usage)
        except Exception as exc:
            logger.error("MetricsFetcher failed: %s", exc)
            errors.append(f"Metrics fetch error: {exc}")
            metrics_list = []

        # Trim to requested n_comps
        final_comps = metrics_list[:n_comps]

        return CompsResult(
            query=query,
            filters=filters,
            candidates_found=len(candidates),
            comps=final_comps,
            llm_usage=cumulative_usage,
            model_used=self._llm.model,
            errors=errors,
        )


def _merge_usage(target: LLMUsage, source: LLMUsage) -> None:
    """Add source usage into target in-place."""
    target.total_prompt_tokens += source.total_prompt_tokens
    target.total_completion_tokens += source.total_completion_tokens
    target.total_tokens += source.total_tokens
    target.estimated_cost_usd += source.estimated_cost_usd
    target.call_count += source.call_count
