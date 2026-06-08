from prefect import flow
import logging
from market_comps.models import LLMUsage
from market_comps.market_intelligence_pipeline.tasks_discovery import (
    classify_market_task, generate_search_queries_task, 
    extract_ma_events_task, extract_fundraising_events_task, 
    extract_ipo_events_task, extract_public_comps_task,
    extract_competitors_task
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

    def merge_usage(task_usage: LLMUsage):
        for trace in task_usage.traces:
            result.usage.traces.append(trace)
        result.usage.total_prompt_tokens += task_usage.total_prompt_tokens
        result.usage.total_completion_tokens += task_usage.total_completion_tokens
        result.usage.total_tokens += task_usage.total_tokens
        result.usage.estimated_cost_usd += task_usage.estimated_cost_usd
        result.usage.call_count += task_usage.call_count
    
    try:
        # Step 1: Classify Market
        update_progress("Step 1: Classifying market and identifying keywords...")
        classification, class_usage = classify_market_task(query, description, discovery_models[0])
        merge_usage(class_usage)
        
        # Step 2: Generate Queries
        update_progress("Step 2: Generating optimal search queries for M&A, Fundraising, and IPOs...")
        search_queries, q_usage = generate_search_queries_task(classification, discovery_models[0])
        merge_usage(q_usage)

        # Step 3 & 4: Retrieve and Extract
        update_progress("Step 3: Extracting specific categories across 3 parallel intelligence agents...")
        raw_extractions = []
        for model in discovery_models:
            # We call the individual category tasks for each model
            ma_data, ma_usage = extract_ma_events_task(query, search_queries, model)
            fundraising_data, fund_usage = extract_fundraising_events_task(query, search_queries, model)
            ipo_data, ipo_usage = extract_ipo_events_task(query, search_queries, model)
            comps_data, comps_usage = extract_public_comps_task(query, search_queries, model)
            competitor_data, competitor_usage = extract_competitors_task(query, search_queries, model)
            
            merge_usage(ma_usage)
            merge_usage(fund_usage)
            merge_usage(ipo_usage)
            merge_usage(comps_usage)
            merge_usage(competitor_usage)
            
            # Reconstruct the monolithic schema format for the rest of the pipeline
            ext_data = {
                "industry_classification": classification.get("primary_industry", ""),
                "ma_events": ma_data.get("ma_events", []),
                "fundraising_events": fundraising_data.get("fundraising_events", []),
                "ipo_events": ipo_data.get("ipo_events", []),
                "public_comps": comps_data.get("public_comps", []),
                "competitors": competitor_data.get("competitors", [])
            }
            raw_extractions.append(ext_data)
            result.raw_extractions[model] = ext_data

        # Step 5: Normalize and Dedupe
        update_progress("Step 4: Merging and deduplicating records across all agents...")
        deduped_data, dedup_usage = normalize_and_dedupe_task(raw_extractions, processing_model)
        merge_usage(dedup_usage)

        # Step 6: Verify Evidence
        update_progress("Step 5: Verifying evidence and filtering out AI hallucinations...")
        verified_data, ver_usage = verify_evidence_task(deduped_data, verification_model)
        merge_usage(ver_usage)

        # Step 7: Enrich Market Data
        update_progress("Step 6: Enriching live metrics via Yahoo Finance API...")
        final_data, enrich_usage = enrich_market_data_task(verified_data, processing_model)
        merge_usage(enrich_usage)

        result.data = final_data
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        result.errors.append(str(e))
        
    return result
