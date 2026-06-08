from prefect import task
import logging
from market_comps.llm_client import LLMClient
from market_comps.models import LLMUsage
from market_comps.market_intelligence_pipeline.schemas import (
    MarketIntelligenceExtraction,
    MAExtractionResponse,
    FundraisingExtractionResponse,
    IPOExtractionResponse,
    PublicCompsExtractionResponse,
    CompetitorExtractionResponse,
)

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
    prompt = f"Given this market classification: {classification}, generate specific web search queries to find:\n1. M&A deals\n2. Fundraising rounds\n3. IPOs\n4. Public comps\n5. Direct and indirect competitors"
    schema = {
        "type": "object",
        "properties": {
            "ma_queries": {"type": "array", "items": {"type": "string"}},
            "fundraising_queries": {"type": "array", "items": {"type": "string"}},
            "ipo_queries": {"type": "array", "items": {"type": "string"}},
            "comps_queries": {"type": "array", "items": {"type": "string"}},
            "competitor_queries": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["ma_queries", "fundraising_queries", "ipo_queries", "comps_queries", "competitor_queries"]
    }
    return client.structured_output(prompt=prompt, json_schema=schema, step_name="generate_search_queries")

@task(name="extract_ma_events", retries=2, retry_delay_seconds=2)
def extract_ma_events_task(query: str, search_queries: dict, model: str) -> tuple[dict, LLMUsage]:
    logger.info(f"Extracting M&A events using model: {model}")
    client = LLMClient(model=model)
    system = """You are a top-tier market research extraction engine specializing in M&A deals.
    You will use your vast training data to perform exhaustive retrieval of Mergers & Acquisitions.
    Strict Lookback window: last 36 months ONLY.
    Be extremely exhaustive. Ensure data matches the required schema fields. Use HIGH, MEDIUM, LOW for confidence.
    CRITICAL: Do NOT guess or hallucinate. If you cannot find evidence of an M&A deal, return an empty list."""
    prompt = f"Target Company/Market: {query}\nSuggested M&A Search Queries: {search_queries.get('ma_queries', [])}\nExtract all relevant M&A deals matching the 36-month lookback."
    return client.structured_output(
        prompt=prompt, 
        json_schema=MAExtractionResponse.model_json_schema(), 
        system_prompt=system,
        step_name="extract_ma_events"
    )

@task(name="extract_fundraising_events", retries=2, retry_delay_seconds=2)
def extract_fundraising_events_task(query: str, search_queries: dict, model: str) -> tuple[dict, LLMUsage]:
    logger.info(f"Extracting Fundraising events using model: {model}")
    client = LLMClient(model=model)
    system = """You are a top-tier market research extraction engine specializing in Venture Capital and Private Equity Fundraising.
    You will use your vast training data to perform exhaustive retrieval of Fundraising rounds.
    Strict Lookback window: last 24 months ONLY.
    Be extremely exhaustive. Ensure data matches the required schema fields. Use HIGH, MEDIUM, LOW for confidence.
    CRITICAL: Do NOT guess or hallucinate. If you cannot find evidence of a Fundraising round, return an empty list."""
    prompt = f"Target Company/Market: {query}\nSuggested Fundraising Search Queries: {search_queries.get('fundraising_queries', [])}\nExtract all relevant Fundraising rounds matching the 24-month lookback."
    return client.structured_output(
        prompt=prompt, 
        json_schema=FundraisingExtractionResponse.model_json_schema(), 
        system_prompt=system,
        step_name="extract_fundraising_events"
    )

@task(name="extract_ipo_events", retries=2, retry_delay_seconds=2)
def extract_ipo_events_task(query: str, search_queries: dict, model: str) -> tuple[dict, LLMUsage]:
    logger.info(f"Extracting IPO events using model: {model}")
    client = LLMClient(model=model)
    system = """You are a top-tier market research extraction engine specializing in Initial Public Offerings (IPOs).
    You will use your vast training data to perform exhaustive retrieval of IPOs.
    Strict Lookback window: last 5 years ONLY.
    Be extremely exhaustive. Ensure data matches the required schema fields. Use HIGH, MEDIUM, LOW for confidence.
    CRITICAL: Do NOT guess or hallucinate. If you cannot find evidence of an IPO, return an empty list."""
    prompt = f"Target Company/Market: {query}\nSuggested IPO Search Queries: {search_queries.get('ipo_queries', [])}\nExtract all relevant IPOs matching the 5-year lookback."
    return client.structured_output(
        prompt=prompt, 
        json_schema=IPOExtractionResponse.model_json_schema(), 
        system_prompt=system,
        step_name="extract_ipo_events"
    )

@task(name="extract_public_comps", retries=2, retry_delay_seconds=2)
def extract_public_comps_task(query: str, search_queries: dict, model: str) -> tuple[dict, LLMUsage]:
    logger.info(f"Extracting Public Comps using model: {model}")
    client = LLMClient(model=model)
    system = """You are a top-tier market research extraction engine specializing in Financial Comparable Analysis.
    You will use your vast training data to identify current active publicly traded comparable companies.
    Be extremely exhaustive. Ensure data matches the required schema fields. Use HIGH, MEDIUM, LOW for confidence.
    CRITICAL: Do NOT guess or hallucinate. If a company is not publicly traded, do NOT include it."""
    prompt = f"Target Company/Market: {query}\nSuggested Public Comps Search Queries: {search_queries.get('comps_queries', [])}\nExtract all relevant active public comparable companies."
    return client.structured_output(
        prompt=prompt, 
        json_schema=PublicCompsExtractionResponse.model_json_schema(), 
        system_prompt=system,
        step_name="extract_public_comps"
    )

@task(name="extract_competitors", retries=2, retry_delay_seconds=2)
def extract_competitors_task(query: str, search_queries: dict, model: str) -> tuple[dict, LLMUsage]:
    logger.info(f"Extracting Competitors using model: {model}")
    client = LLMClient(model=model)
    system = """You are a top-tier market research extraction engine specializing in Competitor Analysis.
    You will use your vast training data to identify current direct and indirect competitors in this market.
    Include incumbents, start-ups, and companies in adjacent spaces.
    Be extremely exhaustive. Ensure data matches the required schema fields. Use HIGH, MEDIUM, LOW for confidence.
    CRITICAL: Do NOT guess or hallucinate."""
    prompt = f"Target Company/Market: {query}\nSuggested Competitor Search Queries: {search_queries.get('competitor_queries', [])}\nExtract all relevant competitors."
    return client.structured_output(
        prompt=prompt, 
        json_schema=CompetitorExtractionResponse.model_json_schema(), 
        system_prompt=system,
        step_name="extract_competitors"
    )
