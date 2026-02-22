# market_comps/cross_checker/cross_checker.py
"""
Cross Checker — query multiple LLMs in parallel and synthesize results.

Steps:
  1. Send the same question (+ injected instructions) to N models concurrently.
  2. Collect all responses and feed them to a summarizer model.
  3. Return structured CrossCheckResult with per-model info + summary.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

from market_comps.llm_client import LLMClient
from market_comps.models import LLMUsage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default models and configuration
# ---------------------------------------------------------------------------

DEFAULT_MODELS: list[str] = [
    "deepseek/deepseek-v3.2",           # DeepSeek V3.2
    "x-ai/grok-4-fast",                 # xAI Grok 4 Fast
    "openai/gpt-4o-mini",               # GPT-4o mini
    "meta-llama/llama-3.3-70b-instruct",  # Llama 3.3 70B
    "minimax/minimax-m2.5",             # MiniMax M2.5
]

DEFAULT_SUMMARY_MODEL = "openai/gpt-4o-mini"

# Injected at the end of every user message — not editable by the user,
# but shown in the UI for transparency.
SYSTEM_INSTRUCTIONS = """\


---
**IMPORTANT — follow these rules strictly in your response:**
1. **Cite ALL sources** with full URLs in markdown format: [Source Name](https://...)
2. **Include every relevant source you find**, regardless of how authoritative it is. \
Do NOT silently omit a source because it comes from a niche site, newsletter, or industry tracker.
3. **Mark each source by reliability tier:**
   - ✅ **Authoritative**: government (.gov), academic journals, Wikipedia, major news orgs (Reuters, AP, BBC, FT, WSJ, Bloomberg, etc.), official company pages
   - ⚠️ **Secondary**: industry trackers, niche news sites, newsletters, blogs, VC databases (e.g. vcnewsdaily.com, Crunchbase, PitchBook, etc.) — include these but flag them with ⚠️
4. **No hallucinations** — only state facts you are confident about; if uncertain, say so explicitly.
5. **Do not invent URLs, statistics, or facts.** If you cannot find a reliable source, say so.
"""

SUMMARY_INSTRUCTIONS = """\
You are synthesizing responses from multiple AI models to produce a single, authoritative answer.

Rules:
1. **Preserve ALL source URLs** — every URL cited in any model response must appear in your answer. \
Write each citation as a full clickable markdown link: [Source Name](https://full-url-here.com). \
Never truncate, shorten, or paraphrase a URL. If a URL was provided, include it in full.
2. **No hallucinations** — do not add new facts or URLs not present in the individual responses.
3. **Resolve disagreements** — if models disagree, say so explicitly rather than silently picking one side.
4. **Use tables for structured data** — if the answer contains comparative, numerical, or multi-attribute \
information (e.g. statistics, company comparisons, timelines, pricing), present it as a markdown table.
5. **Deduplicate** — remove redundant explanations but keep all unique facts and sources.
6. **Sources section** — end your response with a "## Sources" section that lists every unique URL \
as a numbered markdown hyperlink, e.g.: 1. [Reuters](https://reuters.com/article/...).
"""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse:
    """Result from one model."""
    model: str
    content: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    error: Optional[str] = None
    elapsed_seconds: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.content)


@dataclass
class ChorusResult:
    """Aggregated result from LLMChorus.run()."""
    question: str
    models: list[str]
    summary_model: str

    responses: list[ModelResponse] = field(default_factory=list)

    summary: str = ""
    summary_usage: LLMUsage = field(default_factory=LLMUsage)
    summary_error: Optional[str] = None
    total_elapsed_seconds: float = 0.0

    @property
    def total_usage(self) -> LLMUsage:
        """Aggregate usage across all model calls + summary."""
        agg = LLMUsage()
        for r in self.responses:
            agg.total_prompt_tokens += r.usage.total_prompt_tokens
            agg.total_completion_tokens += r.usage.total_completion_tokens
            agg.total_tokens += r.usage.total_tokens
            agg.estimated_cost_usd += r.usage.estimated_cost_usd
            agg.call_count += r.usage.call_count
        agg.total_prompt_tokens += self.summary_usage.total_prompt_tokens
        agg.total_completion_tokens += self.summary_usage.total_completion_tokens
        agg.total_tokens += self.summary_usage.total_tokens
        agg.estimated_cost_usd += self.summary_usage.estimated_cost_usd
        agg.call_count += self.summary_usage.call_count
        return agg


# ---------------------------------------------------------------------------
# CrossChecker
# ---------------------------------------------------------------------------

class LLMChorus:
    """
    Queries multiple LLMs in parallel, then synthesizes results.
    (Formerly CrossChecker — renamed to match "Chorus of LLMs" branding.)

    Usage::

        chorus = LLMChorus()
        result = chorus.run("What caused the 2008 financial crisis?")
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        # One shared LLMClient — each call can override the model
        self._client = LLMClient(api_key=api_key)

    def run(
        self,
        question: str,
        models: Optional[list[str]] = None,
        summary_model: str = DEFAULT_SUMMARY_MODEL,
        max_workers: int = 8,
        temperature: float = 0.3,
        on_model_complete: Optional[Callable[["ModelResponse"], None]] = None,
    ) -> ChorusResult:
        """
        Phase 1: query all models concurrently.
        Phase 2: summarize all responses with the summary model.

        Parameters
        ----------
        on_model_complete : callable, optional
            Called in the main thread each time a model finishes.
            Receives the completed ``ModelResponse``. Use this to push
            progress updates to a UI (e.g. ``st.status``).
        """
        models = models or DEFAULT_MODELS
        result = ChorusResult(
            question=question,
            models=models,
            summary_model=summary_model,
        )

        # ── Phase 1: parallel queries ─────────────────────────────────────
        augmented_question = question + SYSTEM_INSTRUCTIONS
        logger.info("CrossChecker: querying %d models in parallel", len(models))
        t0 = time.time()

        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for model in models:
                future = pool.submit(
                    self._query_model,
                    model=model,
                    question=augmented_question,
                    temperature=temperature,
                )
                futures[future] = model

            for future in as_completed(futures):
                model = futures[future]
                try:
                    response = future.result()
                except Exception as exc:
                    logger.error("Unexpected error querying %s: %s", model, exc)
                    response = ModelResponse(model=model, error=str(exc))
                result.responses.append(response)
                if on_model_complete is not None:
                    try:
                        on_model_complete(response)
                    except Exception as cb_exc:
                        logger.warning("on_model_complete callback raised: %s", cb_exc)

        # Sort back to original order
        order = {m: i for i, m in enumerate(models)}
        result.responses.sort(key=lambda r: order.get(r.model, 999))

        # ── Phase 2: summarize ────────────────────────────────────────────
        if any(r.success for r in result.responses):
            logger.info("CrossChecker: summarizing with %s", summary_model)
            result.summary, result.summary_usage, result.summary_error = (
                self._summarize(result, summary_model)
            )

        result.total_elapsed_seconds = time.time() - t0
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _query_model(
        self, model: str, question: str, temperature: float
    ) -> ModelResponse:
        logger.debug("Querying model: %s", model)
        t0 = time.time()
        try:
            content, usage = self._client.simple_text(
                prompt=question,
                system_prompt=(
                    "You are a knowledgeable, helpful, and truthful assistant. "
                    "Always follow the instructions embedded in the user message."
                ),
                model=model,
                temperature=temperature,
            )
            return ModelResponse(model=model, content=content, usage=usage,
                                 elapsed_seconds=time.time() - t0)
        except Exception as exc:
            logger.error("Error querying %s: %s", model, exc)
            return ModelResponse(model=model, error=str(exc),
                                 elapsed_seconds=time.time() - t0)

    def _summarize(
        self, result: CrossCheckResult, summary_model: str
    ) -> tuple[str, LLMUsage, Optional[str]]:
        responses_text = "\n\n".join(
            f"### {r.model}\n{r.content}"
            for r in result.responses
            if r.success
        )
        prompt = (
            f"**Question:** {result.question}\n\n"
            f"**Individual model responses:**\n\n{responses_text}"
        )
        try:
            summary, usage = self._client.simple_text(
                prompt=prompt,
                system_prompt=SUMMARY_INSTRUCTIONS,
                model=summary_model,
                temperature=0.2,
            )
            return summary, usage, None
        except Exception as exc:
            logger.error("Summarization error: %s", exc)
            return "", LLMUsage(), str(exc)
