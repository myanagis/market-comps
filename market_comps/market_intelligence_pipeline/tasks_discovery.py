from prefect import task
import logging
from market_comps.llm_client import LLMClient
from market_comps.models import LLMUsage
from market_comps.market_intelligence_pipeline.schemas import (
    MarketIntelligenceExtraction,
)

logger = logging.getLogger(__name__)

@task(name="generate_segment_context_and_queries", retries=2, retry_delay_seconds=2)
def generate_segment_context_and_queries_task(query: str, description: str, config: dict, model: str) -> tuple[dict, LLMUsage]:
    segment_name = config["name"]
    logger.info(f"Generating context and search queries for {segment_name}.")
    client = LLMClient(model=model)
    system = f"You are an expert investment analyst finding relevant markets and generating search queries specifically for {segment_name}. You are highly accurate and concise."
    prompt = f"Company/Market Query: {query}\nDescription: {description}\nGenerate market context (primary market, product category, subindustries, adjacent markets, keywords), an unverified initial hypothesis list of likely companies, and specific search queries optimized to find {segment_name}."
    schema = {
        "type": "object",
        "properties": {
            "primary_market": {"type": "string"},
            "product_category": {"type": "string"},
            "subindustries": {"type": "array", "items": {"type": "string"}},
            "adjacent_markets": {"type": "array", "items": {"type": "string"}},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "unverified_company_list": {"type": "array", "items": {"type": "string"}},
            "queries": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["primary_market", "product_category", "subindustries", "adjacent_markets", "keywords", "unverified_company_list", "queries"]
    }
    return client.structured_output(
        prompt=prompt, 
        json_schema=schema, 
        system_prompt=system,
        step_name=f"generate_context_{segment_name.lower().replace(' ', '_').replace('&', 'and')}"
    )

@task(name="extract_segment", retries=2, retry_delay_seconds=2)
def extract_segment_task(query: str, segment_context: dict, model: str, segment_id: str, config: dict) -> tuple[dict, LLMUsage]:
    logger.info(f"Extracting {config['name']} using model: {model}")
    client = LLMClient(model=model)
    system = config["system"]
    query_str = segment_context.get("queries", [])
    prompt = config["prompt_template"].format(query=query, search_queries=query_str)
    
    context_str = f"\nPrimary Market: {segment_context.get('primary_market', '')}\nProduct Category: {segment_context.get('product_category', '')}\nSubindustries: {segment_context.get('subindustries', [])}\nAdjacent Markets: {segment_context.get('adjacent_markets', [])}\nKeywords: {segment_context.get('keywords', [])}\nUnverified Company Seed List: {segment_context.get('unverified_company_list', [])}"
    prompt += context_str
    
    return client.structured_output(
        prompt=prompt, 
        json_schema=config["schema"].model_json_schema(), 
        system_prompt=system,
        step_name=f"extract_{segment_id}"
    )
