from prefect import task
import logging
from market_comps.llm_client import LLMClient
from market_comps.models import LLMUsage
from market_comps.market_intelligence_pipeline.schemas import MarketIntelligenceExtraction

logger = logging.getLogger(__name__)

@task(name="classify_market", retries=2, retry_delay_seconds=2)
def classify_market_task(query: str, description: str, model: str) -> tuple[dict, LLMUsage]:
    logger.info(f"Classifying market for: {query}")
    client = LLMClient(model=model)
    prompt = f"Query: {query}\nDescription: {description}\nClassify the market industry, list subindustries, adjacent markets, and key search terms."
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
    return client.structured_output(prompt=prompt, json_schema=schema, step_name="classify_market")

@task(name="generate_search_queries", retries=2, retry_delay_seconds=2)
def generate_search_queries_task(classification: dict, model: str) -> tuple[dict, LLMUsage]:
    logger.info("Generating search queries based on classification.")
    client = LLMClient(model=model)
    prompt = f"Given this market classification: {classification}, generate specific web search queries to find:\n1. M&A deals\n2. Fundraising rounds\n3. IPOs\n4. Public comps"
    schema = {
        "type": "object",
        "properties": {
            "ma_queries": {"type": "array", "items": {"type": "string"}},
            "fundraising_queries": {"type": "array", "items": {"type": "string"}},
            "ipo_queries": {"type": "array", "items": {"type": "string"}},
            "comps_queries": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["ma_queries", "fundraising_queries", "ipo_queries", "comps_queries"]
    }
    return client.structured_output(prompt=prompt, json_schema=schema, step_name="generate_search_queries")

@task(name="retrieve_and_extract_events", retries=2, retry_delay_seconds=2)
def retrieve_and_extract_events_task(query: str, search_queries: dict, model: str) -> tuple[dict, LLMUsage]:
    logger.info(f"Retrieving and extracting events using model: {model}")
    client = LLMClient(model=model)
    
    system = """You are a top-tier market research extraction engine.
    You will use your vast training data to perform exhaustive retrieval of Market Intelligence events.
    Follow these strict lookback windows:
    - M&A: last 36 months
    - Fundraising: last 24 months
    - IPOs: last 5 years
    - Public comps: current active tickers
    
    Be extremely exhaustive. Ensure data matches the required schema fields. Use HIGH, MEDIUM, LOW for confidence.
    """
    prompt = f"Target Company/Market: {query}\nSuggested Search Queries: {search_queries}\nExtract all relevant M&A, Fundraising, IPOs, and Public Comps matching the lookbacks."
    
    # We use MarketIntelligenceExtraction schema dynamically.
    # To keep it simple, we just pass the pydantic json schema.
    json_schema = MarketIntelligenceExtraction.model_json_schema()
    
    data, usage = client.structured_output(
        prompt=prompt, 
        json_schema=json_schema, 
        system_prompt=system,
        step_name="retrieve_and_extract"
    )
    return data, usage
