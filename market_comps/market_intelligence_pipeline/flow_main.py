from prefect import flow
import logging
from market_comps.models import LLMUsage
from market_comps.market_intelligence_pipeline.tasks_discovery import (
    generate_segment_context_and_queries_task, extract_segment_task
)
from market_comps.market_intelligence_pipeline.config import SEGMENTS
from market_comps.market_intelligence_pipeline.tasks_processing import (
    normalize_and_dedupe_segment_task, verify_segment_task
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
        # Step 1: Generate Context & Queries per segment
        update_progress("Step 1: Generating optimal search queries and market context for each segment...")
        segment_contexts = {}
        for seg_id, config in SEGMENTS.items():
            context, q_usage = generate_segment_context_and_queries_task(query, description, config, discovery_models[0])
            merge_usage(q_usage)
            segment_contexts[seg_id] = context

        # Step 2 & 3: Retrieve and Extract
        update_progress("Step 2: Extracting specific categories across parallel intelligence agents...")
        
        # Accumulators for segment-specific results
        segment_extractions = {seg_id: [] for seg_id in SEGMENTS}

        for model in discovery_models:
            ext_data = {}
            
            for seg_id, config in SEGMENTS.items():
                data, usage = extract_segment_task(query, segment_contexts[seg_id], model, seg_id, config)
                merge_usage(usage)
                segment_extractions[seg_id].append(data)
                ext_data[seg_id] = data.get(seg_id, [])
                
            result.raw_extractions[model] = ext_data

        # Step 4 & 5: Normalize, Dedupe, and Verify by Segment
        update_progress("Step 3 & 4: Deduping and Verifying each segment individually...")
        
        all_subindustries = set()
        for ctx in segment_contexts.values():
            all_subindustries.update(ctx.get("subindustries", []))
            
        verified_data = {
            "industry_classification": ", ".join(all_subindustries)
        }
        
        for seg_id, config in SEGMENTS.items():
            deduped_data, d_usage = normalize_and_dedupe_segment_task(segment_extractions[seg_id], config["schema"].model_json_schema(), config["name"], processing_model)
            merge_usage(d_usage)
            verified_seg, v_usage = verify_segment_task(deduped_data, config["schema"].model_json_schema(), config["name"], verification_model)
            merge_usage(v_usage)
            verified_data[seg_id] = verified_seg.get(seg_id, [])

        # Step 7: Enrich Market Data
        update_progress("Step 6: Enriching live metrics via Yahoo Finance API...")
        final_data, enrich_usage = enrich_market_data_task(verified_data, processing_model)
        merge_usage(enrich_usage)

        result.data = final_data
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        result.errors.append(str(e))
        
    return result
