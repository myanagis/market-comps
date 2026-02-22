# market_comps/company_primer/primer_finder.py
"""
CompanyPrimerFinder — research one or more companies using the LLMChorus,
then extract structured profile data via a summarizer pass.

Per-company output:
  - description      : what the company does (2-4 sentences)
  - industry         : primary industry / sector
  - target_customer  : who they sell to
  - location         : HQ location
  - key_facts        : 3-5 notable facts (funding, revenue, notable clients, etc.)
  - sources          : list of cited URLs with reliability tier
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from market_comps.cross_checker import LLMChorus
from market_comps.cross_checker.cross_checker import (
    DEFAULT_MODELS, DEFAULT_SUMMARY_MODEL,
)
from market_comps.llm_client import LLMClient
from market_comps.models import LLMUsage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 3 smallest / fastest default models for primer (3 is enough)
# ---------------------------------------------------------------------------
PRIMER_DEFAULT_MODELS: list[str] = [
    "deepseek/deepseek-v3.2",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_RESEARCH_QUESTION_TEMPLATE = """\
Research the following company and provide a detailed profile:

Company name: {name}
{context_line}

Please provide:
1. **Description** — what the company does (products, services, business model), 2-4 sentences
2. **Industry / Sector** — primary industry and sub-sector
3. **Target Customer** — who they primarily sell to (enterprise, SMB, consumer, government, etc.)
4. **Headquarters Location** — city, country
5. **Key Facts** — 3-5 notable facts such as: founding year, employee count, funding raised, \
notable customers or partners, revenue if known, recent news

Accuracy is paramount. Only state facts you are confident about. Cite every claim with a full URL.
"""

_EXTRACTION_SYSTEM = """\
You are a structured data extraction engine. Given research about a company from multiple AI models,
extract a single clean profile.

Rules:
- Only include facts that appear in the research below — do NOT invent anything.
- If a field is unknown or uncertain, use null or an empty list.
- Deduplicate: if multiple models say the same thing, include it once.
- For sources, include every URL cited across all models; mark each with a reliability tier:
  "authoritative" (official company site, .gov, major news orgs, Wikipedia) or
  "secondary" (blogs, VC trackers, niche sites, newsletters).

Return ONLY valid JSON, no markdown fences, no prose:
{
  "description": "<2-4 sentence description of what the company does>",
  "industry": "<primary industry / sector>",
  "target_customer": "<who they primarily sell to>",
  "location": "<city, country>",
  "key_facts": ["<fact 1>", "<fact 2>", "<fact 3>"],
  "sources": [
    {"url": "<url>", "label": "<short label>", "tier": "<authoritative|secondary>"}
  ]
}
"""

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SourceRef:
    url: str
    label: str
    tier: str  # "authoritative" or "secondary"


@dataclass
class CompanyProfile:
    name: str
    context: str = ""          # user-supplied URL or description
    description: str = ""
    industry: str = ""
    target_customer: str = ""
    location: str = ""
    key_facts: list[str] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)
    chorus_usage: LLMUsage = field(default_factory=LLMUsage)
    extraction_usage: LLMUsage = field(default_factory=LLMUsage)
    error: Optional[str] = None

    @property
    def total_usage(self) -> LLMUsage:
        agg = LLMUsage()
        for u in (self.chorus_usage, self.extraction_usage):
            agg.total_prompt_tokens += u.total_prompt_tokens
            agg.total_completion_tokens += u.total_completion_tokens
            agg.total_tokens += u.total_tokens
            agg.estimated_cost_usd += u.estimated_cost_usd
            agg.call_count += u.call_count
        return agg


@dataclass
class PrimerResult:
    profiles: list[CompanyProfile] = field(default_factory=list)

    @property
    def total_usage(self) -> LLMUsage:
        agg = LLMUsage()
        for p in self.profiles:
            t = p.total_usage
            agg.total_prompt_tokens += t.total_prompt_tokens
            agg.total_completion_tokens += t.total_completion_tokens
            agg.total_tokens += t.total_tokens
            agg.estimated_cost_usd += t.estimated_cost_usd
            agg.call_count += t.call_count
        return agg


# ---------------------------------------------------------------------------
# CompanyPrimerFinder
# ---------------------------------------------------------------------------

class CompanyPrimerFinder:
    """
    Research one or more companies with the LLMChorus and extract
    structured profiles.

    Parameters
    ----------
    chorus_models : list[str], optional
        Models to query in parallel. Defaults to PRIMER_DEFAULT_MODELS.
    summary_model : str, optional
        Model used for the extraction/summarization step.
    api_key : str, optional
        OpenRouter API key.
    """

    def __init__(
        self,
        chorus_models: Optional[list[str]] = None,
        summary_model: str = DEFAULT_SUMMARY_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        self._chorus = LLMChorus(api_key=api_key)
        self._extractor = LLMClient(api_key=api_key)
        self._chorus_models = chorus_models or PRIMER_DEFAULT_MODELS
        self._summary_model = summary_model

    def run_one(
        self,
        name: str,
        context: str = "",
        on_model_complete: Optional[Callable] = None,
    ) -> CompanyProfile:
        """Research a single company. Returns a CompanyProfile."""
        profile = CompanyProfile(name=name, context=context)

        context_line = f"Additional context: {context}" if context.strip() else ""
        question = _RESEARCH_QUESTION_TEMPLATE.format(
            name=name, context_line=context_line
        )

        # Phase 1: Chorus
        logger.info("CompanyPrimerFinder: Phase 1 — chorus for %s", name)
        try:
            chorus_result = self._chorus.run(
                question=question,
                models=self._chorus_models,
                summary_model=self._summary_model,
                temperature=0.2,
                on_model_complete=on_model_complete,
            )
            t = chorus_result.total_usage
            profile.chorus_usage = t
        except Exception as exc:
            logger.error("Chorus failed for %s: %s", name, exc)
            profile.error = f"Chorus error: {exc}"
            return profile

        # Collect all text
        all_text = "\n\n---\n\n".join(
            f"### {r.model}\n{r.content}"
            for r in chorus_result.responses
            if r.success
        )
        if chorus_result.summary:
            all_text += f"\n\n---\n\n### Synthesis\n{chorus_result.summary}"

        if not all_text.strip():
            profile.error = "No model responses received."
            return profile

        # Phase 2: Structured extraction
        logger.info("CompanyPrimerFinder: Phase 2 — extraction for %s", name)
        extraction_prompt = (
            f"Company: **{name}**\n"
            f"{('Context: ' + context) if context else ''}\n\n"
            f"Research:\n\n{all_text}"
        )
        try:
            raw, usage = self._extractor.simple_text(
                prompt=extraction_prompt,
                system_prompt=_EXTRACTION_SYSTEM,
                model=self._summary_model,
                temperature=0.1,
            )
            profile.extraction_usage = usage
            _populate_profile(profile, raw)
        except Exception as exc:
            logger.error("Extraction failed for %s: %s", name, exc)
            profile.error = f"Extraction error: {exc}"

        return profile

    def run(
        self,
        companies: list[dict],   # [{"name": str, "context": str}, ...]
        on_company_start: Optional[Callable[[str], None]] = None,
        on_model_complete: Optional[Callable] = None,
    ) -> PrimerResult:
        """
        Research a list of companies sequentially.
        Calls on_company_start(name) before each company's chorus run.
        """
        result = PrimerResult()
        for entry in companies:
            name = entry.get("name", "").strip()
            context = entry.get("context", "").strip()
            if not name:
                continue
            if on_company_start:
                on_company_start(name)
            profile = self.run_one(
                name=name, context=context, on_model_complete=on_model_complete
            )
            result.profiles.append(profile)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _populate_profile(profile: CompanyProfile, raw: str) -> None:
    """Parse JSON extraction response into profile fields."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 1)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse failed: %s | raw[:400]: %s", exc, s[:400])
        return

    profile.description = data.get("description") or ""
    profile.industry = data.get("industry") or ""
    profile.target_customer = data.get("target_customer") or ""
    profile.location = data.get("location") or ""
    profile.key_facts = data.get("key_facts") or []
    raw_sources = data.get("sources") or []
    profile.sources = [
        SourceRef(
            url=s.get("url", ""),
            label=s.get("label", ""),
            tier=s.get("tier", "secondary"),
        )
        for s in raw_sources
        if s.get("url")
    ]
