from prefect import flow
import logging
from market_comps.models import LLMUsage
from market_comps.market_intelligence_pipeline.tasks_discovery import (
    classify_market_task, generate_search_queries_task, retrieve_and_extract_events_task
)
from market_comps.market_intelligence_pipeline.tasks_processing import (
    normalize_and_dedupe_task, verify_evidence_task
)
from market_comps.market_intelligence_pipeline.tasks_enrichment import enrich_market_data_task

logger = logging.getLogger(__name__)

class IntelligenceResult:
    def __init__(self):
        self.data = {}
        self.raw_extractions: dict[str, dict] = {}
        self.usage = LLMUsage()
        self.errors = []

from typing import Callable, Optional

@flow(name="market_intelligence_flow")
def run_market_intelligence_pipeline(
    query: str, 
    description: str,
    discovery_models: list[str],
    processing_model: str,
    verification_model: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> IntelligenceResult:
    logger.info(f"Starting Market Intelligence Flow for: {query}")
    result = IntelligenceResult()
    
    def update_progress(msg: str):
        if progress_callback:
            progress_callback(msg)
    
    try:
        # Step 1: Classify Market
        update_progress("Step 1: Classifying market and identifying keywords...")
        classification, class_usage = classify_market_task(query, description, discovery_models[0])
        result.usage.add(
            prompt_tokens=class_usage.total_prompt_tokens,
            completion_tokens=class_usage.total_completion_tokens,
            input_price_per_m=0, output_price_per_m=0, model=discovery_models[0], step_name="classify"
        )
        result.usage.estimated_cost_usd += class_usage.estimated_cost_usd
        
        # Step 2: Generate Queries
        update_progress("Step 2: Generating optimal search queries for M&A, Fundraising, and IPOs...")
        search_queries, q_usage = generate_search_queries_task(classification, discovery_models[0])
        result.usage.add(
            prompt_tokens=q_usage.total_prompt_tokens,
            completion_tokens=q_usage.total_completion_tokens,
            input_price_per_m=0, output_price_per_m=0, model=discovery_models[0], step_name="queries"
        )
        result.usage.estimated_cost_usd += q_usage.estimated_cost_usd

        # Step 3 & 4: Retrieve and Extract
        update_progress("Step 3: Extracting deals from 3 parallel intelligence agents...")
        raw_extractions = []
        for model in discovery_models:
            # We call the task. In a true async/dask runner these would run parallel.
            ext_data, ext_usage = retrieve_and_extract_events_task(query, search_queries, model)
            raw_extractions.append(ext_data)
            result.raw_extractions[model] = ext_data
            result.usage.add(
                prompt_tokens=ext_usage.total_prompt_tokens,
                completion_tokens=ext_usage.total_completion_tokens,
                input_price_per_m=0, output_price_per_m=0, model=model, step_name="extract"
            )
            result.usage.estimated_cost_usd += ext_usage.estimated_cost_usd

        # Step 5: Normalize and Dedupe
        update_progress("Step 4: Merging and deduplicating records across all agents...")
        deduped_data, dedup_usage = normalize_and_dedupe_task(raw_extractions, processing_model)
        result.usage.add(
            prompt_tokens=dedup_usage.total_prompt_tokens,
            completion_tokens=dedup_usage.total_completion_tokens,
            input_price_per_m=0, output_price_per_m=0, model=processing_model, step_name="dedupe"
        )
        result.usage.estimated_cost_usd += dedup_usage.estimated_cost_usd

        # Step 6: Verify Evidence
        update_progress("Step 5: Verifying evidence and filtering out AI hallucinations...")
        verified_data, ver_usage = verify_evidence_task(deduped_data, verification_model)
        result.usage.add(
            prompt_tokens=ver_usage.total_prompt_tokens,
            completion_tokens=ver_usage.total_completion_tokens,
            input_price_per_m=0, output_price_per_m=0, model=verification_model, step_name="verify"
        )
        result.usage.estimated_cost_usd += ver_usage.estimated_cost_usd

        # Step 7: Enrich Market Data
        update_progress("Step 6: Enriching live metrics via Yahoo Finance API...")
        final_data, enrich_usage = enrich_market_data_task(verified_data, processing_model)
        result.usage.add(
            prompt_tokens=enrich_usage.total_prompt_tokens,
            completion_tokens=enrich_usage.total_completion_tokens,
            input_price_per_m=0, output_price_per_m=0, model=processing_model, step_name="enrich"
        )
        result.usage.estimated_cost_usd += enrich_usage.estimated_cost_usd

        result.data = final_data
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        result.errors.append(str(e))
        
    return result
