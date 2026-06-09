from prefect import task
import logging
from market_comps.llm_client import LLMClient
from market_comps.models import LLMUsage

logger = logging.getLogger(__name__)

@task(name="normalize_and_dedupe_segment", retries=2, retry_delay_seconds=2)
def normalize_and_dedupe_segment_task(extractions: list[dict], json_schema: dict, segment_name: str, model: str) -> tuple[dict, LLMUsage]:
    logger.info(f"Normalizing and deduping {segment_name}.")
    client = LLMClient(model=model)
    
    system = f"""You are a financial data normalization and deduplication engine specializing in {segment_name}.
    You will receive multiple raw JSON extractions from different AI agents.
    Your task is to merge them into a single, comprehensive output matching the schema.
    Rules:
    1. Resolve duplicate events (e.g. if two agents find the same event, merge the info into one record).
    2. Normalize company names (e.g. "Stripe Inc." and "Stripe" -> "Stripe").
    3. Retain the highest confidence available for an event.
    """
    
    prompt = f"Raw {segment_name} extractions from multiple agents:\n{extractions}\nPlease deduplicate and return the canonical JSON."
    
    data, usage = client.structured_output(
        prompt=prompt, 
        json_schema=json_schema, 
        system_prompt=system,
        step_name=f"normalize_and_dedupe_{segment_name}"
    )
    return data, usage

@task(name="verify_segment", retries=2, retry_delay_seconds=2)
def verify_segment_task(deduped_data: dict, json_schema: dict, segment_name: str, model: str) -> tuple[dict, LLMUsage]:
    logger.info(f"Verifying evidence for {segment_name}.")
    client = LLMClient(model=model)
    
    system = f"""You are an expert fact-checker and hallucination-detection engine specializing in {segment_name}.
    Review the provided JSON.
    Rules:
    1. Filter out any claims that seem highly likely to be AI hallucinations.
    2. Flag questionable items by lowering their 'confidence' to LOW.
    3. Ensure standard lookback windows are applied: M&A (36m), Fundraising (24m), IPOs (5yr), Comps/Competitors (current).
    4. REMOVE any deals or rounds where the company and the acquirer/investor are the exact same entity.
    Return the filtered, validated JSON matching the exact schema.
    """
    
    prompt = f"Data to verify:\n{deduped_data}\nPlease verify and return the clean JSON."
    
    data, usage = client.structured_output(
        prompt=prompt, 
        json_schema=json_schema, 
        system_prompt=system,
        step_name=f"verify_{segment_name}"
    )
    
    # Python backup strict filter for M&A and Fundraising
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
            if len(leads) == 1 and leads[0] == company:
                continue
            filtered_funds.append(e)
        data["fundraising_events"] = filtered_funds
        
    return data, usage
