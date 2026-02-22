# market_comps/chorus_comps_engine.py
"""
ChorusCompsEngine — uses LLMChorus (5 models) to identify public-comp tickers,
deduplicates them with a summarizer, then pipes to the existing MetricsFetcher.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from market_comps.cross_checker import LLMChorus
from market_comps.cross_checker.cross_checker import DEFAULT_MODELS, DEFAULT_SUMMARY_MODEL
from market_comps.llm_client import LLMClient
from market_comps.metrics_fetcher import MetricsFetcher
from market_comps.models import CompsResult, LLMUsage, ScanFilters

logger = logging.getLogger(__name__)

_COMPS_QUESTION_TEMPLATE = """\
List the {n} best publicly traded comparable companies for:

"{query}"

For each company provide:
- Company name
- Stock ticker symbol (e.g. CRM, ADBE, MSFT)
- Primary exchange (NYSE, NASDAQ, etc.)
- 1-sentence rationale for why it's a good comparable

Cite sources with full URLs. Only include real, currently listed public companies.
"""

_DEDUP_SYSTEM = """\
You are a financial research deduplication engine.
Given ticker suggestions from multiple AI models, return a deduplicated, ranked list of tickers.

Rules:
- Only include real, exchange-listed stock tickers you are confident about.
- Remove duplicates (same company suggested by multiple models).
- Remove any clearly incorrect or non-public tickers.
- Rank by relevance (most comparable first).

Return ONLY valid JSON: {"tickers": ["TICK1", "TICK2", ...]}
"""


class ChorusCompsEngine:
    """
    Chorus-powered public comps: queries 5 LLMs in parallel for tickers,
    deduplicates, then enriches with live market data via MetricsFetcher.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chorus_models: Optional[list[str]] = None,
        summary_model: str = DEFAULT_SUMMARY_MODEL,
        max_fetch_workers: int = 8,
    ) -> None:
        self._chorus = LLMChorus(api_key=api_key)
        self._dedup_llm = LLMClient(api_key=api_key)
        self._fetcher = MetricsFetcher(
            llm_client=self._dedup_llm,
            max_workers=max_fetch_workers,
        )
        self._chorus_models = chorus_models or DEFAULT_MODELS
        self._summary_model = summary_model

    def run(
        self,
        query: str,
        n_comps: int = 10,
        filters: Optional[ScanFilters] = None,
    ) -> CompsResult:
        filters = filters or ScanFilters()
        cumulative_usage = LLMUsage()
        errors: list[str] = []

        # ── Phase 1: Chorus ticker research ──────────────────────────────
        logger.info("ChorusCompsEngine: Phase 1 — querying chorus for comps on %r", query)
        question = _COMPS_QUESTION_TEMPLATE.format(query=query, n=n_comps + 5)
        try:
            chorus_result = self._chorus.run(
                question=question,
                models=self._chorus_models,
                summary_model=self._summary_model,
                temperature=0.2,
            )
            # Accumulate chorus usage
            t = chorus_result.total_usage
            cumulative_usage.total_prompt_tokens += t.total_prompt_tokens
            cumulative_usage.total_completion_tokens += t.total_completion_tokens
            cumulative_usage.total_tokens += t.total_tokens
            cumulative_usage.estimated_cost_usd += t.estimated_cost_usd
            cumulative_usage.call_count += t.call_count
        except Exception as exc:
            logger.error("Chorus phase failed: %s", exc)
            errors.append(f"Chorus error: {exc}")
            return CompsResult(
                query=query, filters=filters, comps=[],
                llm_usage=cumulative_usage, model_used="chorus", errors=errors,
            )

        # ── Phase 2: Dedup + extract tickers ─────────────────────────────
        logger.info("ChorusCompsEngine: Phase 2 — deduplicating tickers")
        all_text = "\n\n---\n\n".join(
            f"Model: {r.model}\n{r.content}"
            for r in chorus_result.responses if r.success
        )
        if chorus_result.summary:
            all_text += f"\n\n---\n\nSynthesized:\n{chorus_result.summary}"

        tickers: list[str] = []
        try:
            raw, dedup_usage = self._dedup_llm.simple_text(
                prompt=f"Query: {query}\n\nModel responses:\n{all_text}",
                system_prompt=_DEDUP_SYSTEM,
                model=self._summary_model,
                temperature=0.1,
            )
            _merge(cumulative_usage, dedup_usage)
            data = _parse_json(raw)
            tickers = data.get("tickers", [])[:n_comps + 3]
            logger.info("Dedup returned %d tickers: %s", len(tickers), tickers)
        except Exception as exc:
            logger.error("Dedup failed: %s", exc)
            errors.append(f"Dedup error: {exc}")

        if not tickers:
            return CompsResult(
                query=query, filters=filters, comps=[],
                candidates_found=0, llm_usage=cumulative_usage,
                model_used="chorus", errors=errors + ["No tickers extracted."],
            )

        # ── Phase 3: Enrich with live metrics ────────────────────────────
        logger.info("ChorusCompsEngine: Phase 3 — fetching metrics for %s", tickers)
        from market_comps.models import CompanyCandidate
        candidates = [CompanyCandidate(name=t, ticker=t, exchange="") for t in tickers]
        try:
            metrics_list, fetch_usage = self._fetcher.fetch(candidates)
            _merge(cumulative_usage, fetch_usage)
        except Exception as exc:
            logger.error("MetricsFetcher failed: %s", exc)
            errors.append(f"Metrics fetch error: {exc}")
            metrics_list = []

        return CompsResult(
            query=query,
            filters=filters,
            candidates_found=len(tickers),
            comps=metrics_list[:n_comps],
            llm_usage=cumulative_usage,
            model_used=f"chorus({len(self._chorus_models)} models)",
            errors=errors,
        )


def _merge(target: LLMUsage, source: LLMUsage) -> None:
    target.total_prompt_tokens += source.total_prompt_tokens
    target.total_completion_tokens += source.total_completion_tokens
    target.total_tokens += source.total_tokens
    target.estimated_cost_usd += source.estimated_cost_usd
    target.call_count += source.call_count


def _parse_json(raw: str) -> dict:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 1)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```", 1)[0].strip()
    return json.loads(s)
