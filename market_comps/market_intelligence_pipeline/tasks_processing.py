from prefect import task
import logging
from market_comps.llm_client import LLMClient
from market_comps.models import LLMUsage
from market_comps.market_intelligence_pipeline.schemas import MarketIntelligenceExtraction

logger = logging.getLogger(__name__)

@task(name="normalize_and_dedupe")
def normalize_and_dedupe_task(extractions: list[dict], model: str) -> tuple[dict, LLMUsage]:
    logger.info("Normalizing and deduping events.")
    client = LLMClient(model=model)
    
    system = """You are a financial data normalization and deduplication engine.
    You will receive multiple raw JSON extractions from different AI agents.
    Your task is to merge them into a single, comprehensive output matching the schema.
    Rules:
    1. Resolve duplicate events (e.g. if two agents find the same M&A deal, merge the info into one record).
    2. Normalize company names (e.g. "Stripe Inc." and "Stripe" -> "Stripe").
    3. Retain the highest confidence available for an event.
    """
    
    prompt = f"Raw extractions from multiple agents:\n{extractions}\nPlease deduplicate and return the canonical JSON."
    json_schema = MarketIntelligenceExtraction.model_json_schema()
    
    data, usage = client.structured_output(
        prompt=prompt, 
        json_schema=json_schema, 
        system_prompt=system,
        step_name="normalize_and_dedupe"
    )
    return data, usage

@task(name="verify_evidence")
def verify_evidence_task(deduped_data: dict, model: str) -> tuple[dict, LLMUsage]:
    logger.info("Verifying evidence and filtering hallucinations.")
    client = LLMClient(model=model)
    
    system = """You are an expert fact-checker and hallucination-detection engine.
    Review the provided Market Intelligence JSON.
    Rules:
    1. Filter out any claims that seem highly likely to be AI hallucinations (e.g., non-existent companies, bizarre deal values).
    2. Flag questionable items by lowering their 'confidence' to LOW.
    3. Ensure lookback windows strictly applied logically: M&A (36m), Fundraising (24m), IPOs (5yr), Comps (current).
    4. REMOVE any M&A deals or fundraising rounds where the company and the acquirer/investor are the exact same entity (e.g. self-acquisitions).
    Return the filtered, validated JSON matching the exact schema.
    """
    
    prompt = f"Data to verify:\n{deduped_data}\nPlease verify and return the clean JSON."
    json_schema = MarketIntelligenceExtraction.model_json_schema()
    
    data, usage = client.structured_output(
        prompt=prompt, 
        json_schema=json_schema, 
        system_prompt=system,
        step_name="verify_evidence"
    )
    
    # Python backup strict filter
    if "ma_events" in data:
        data["ma_events"] = [
            e for e in data["ma_events"] 
            if e.get("company", "").strip().lower() != e.get("acquirer", "").strip().lower()
        ]
        
    if "fundraising_events" in data:
        filtered_funds = []
        for e in data["fundraising_events"]:
            company = e.get("company", "").strip().lower()
            leads = [l.strip().lower() for l in e.get("lead_investors", [])]
            # If the ONLY lead investor is the company itself, drop it as an obvious hallucination
            if len(leads) == 1 and leads[0] == company:
                continue
            filtered_funds.append(e)
        data["fundraising_events"] = filtered_funds
        
    return data, usage
