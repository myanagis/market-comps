from prefect import task
import logging
from market_comps.llm_client import LLMClient
from market_comps.models import LLMUsage
from market_comps.market_intelligence_pipeline.schemas import (
    MarketIntelligenceExtraction,
)

logger = logging.getLogger(__name__)

@task(name="classify_market", retries=2, retry_delay_seconds=2)
def classify_market_task(query: str, description: str, model: str) -> tuple[dict, LLMUsage]:
    logger.info(f"Classifying market for: {query}")
    client = LLMClient(model=model)
    system = "You are an expert investment analyst finding relevant markets/submarkets for the given company. You are highly accurate and concise."
    prompt = f"Company/Market Query: {query}\nDescription: {description}\nClassify the market industry, list subindustries, adjacent markets, and key search terms."
    schema = {
        "type": "object",
        "properties": {
            "primary_industry": {"type": "string"},
            "subindustries": {"type": "array", "items": {"type": "string"}},
            "adjacent_markets": {"type": "array", "items": {"type": "string"}},
            "keywords": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["primary_industry", "subindustries", "adjacent_markets", "keywords"]
    }
    return client.structured_output(
        prompt=prompt, 
        json_schema=schema, 
        system_prompt=system,
        step_name="classify_market"
    )

@task(name="generate_search_queries_for_segment", retries=2, retry_delay_seconds=2)
def generate_search_queries_for_segment_task(classification: dict, config: dict, model: str) -> tuple[list[str], LLMUsage]:
    segment_name = config["name"]
    logger.info(f"Generating search queries for {segment_name}.")
    client = LLMClient(model=model)
    system = f"You are an expert investment analyst. Generate precise web search queries to find {segment_name} for the given market classification. These queries will soon query real live sources via search engines, so optimize for high-quality results."
    prompt = f"Market Classification: {classification}\nGenerate search queries for: {segment_name}."
    schema = {
        "type": "object",
        "properties": {
            "queries": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["queries"]
    }
    data, usage = client.structured_output(
        prompt=prompt, 
        json_schema=schema, 
        system_prompt=system,
        step_name=f"generate_queries_{segment_name.lower().replace(' ', '_').replace('&', 'and')}"
    )
    return data.get("queries", []), usage

@task(name="extract_segment", retries=2, retry_delay_seconds=2)
def extract_segment_task(query: str, search_queries: dict, model: str, segment_id: str, config: dict) -> tuple[dict, LLMUsage]:
    logger.info(f"Extracting {config['name']} using model: {model}")
    client = LLMClient(model=model)
    system = config["system"]
    query_str = search_queries.get(config["query_key"], [])
    prompt = config["prompt_template"].format(query=query, search_queries=query_str)
    
    return client.structured_output(
        prompt=prompt, 
        json_schema=config["schema"].model_json_schema(), 
        system_prompt=system,
        step_name=f"extract_{segment_id}"
    )
