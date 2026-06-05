from prefect import task
import logging
from market_comps.metrics_fetcher import MetricsFetcher
from market_comps.llm_client import LLMClient
from market_comps.models import CompanyCandidate, LLMUsage

logger = logging.getLogger(__name__)

@task(name="enrich_market_data", retries=2, retry_delay_seconds=2)
def enrich_market_data_task(verified_data: dict, model: str) -> tuple[dict, LLMUsage]:
    logger.info("Enriching market data with Yahoo Finance API.")
    
    # We need a dedicated LLMClient for the metrics fetcher deduplication step internally
    fetcher_llm = LLMClient(model=model)
    fetcher = MetricsFetcher(llm_client=fetcher_llm, max_workers=1)
    
    candidates = []
    
    # Collect public comps and IPOs that have tickers
    public_comps = verified_data.get("public_comps", [])
    ipo_events = verified_data.get("ipo_events", [])
    
    for comp in public_comps:
        ticker = comp.get("ticker")
        if ticker:
            candidates.append(CompanyCandidate(name=comp.get("company", ""), ticker=ticker, exchange=comp.get("exchange", "")))
            
    # Remove duplicates from candidates
    unique_candidates = {c.ticker: c for c in candidates}.values()
    
    if not unique_candidates:
        return verified_data, LLMUsage()
        
    metrics_list, fetch_usage = fetcher.fetch(list(unique_candidates))
    
    # Map metrics back to the verified data
    metrics_by_ticker = {m.ticker: m for m in metrics_list}
    
    for comp in public_comps:
        ticker = comp.get("ticker")
        if ticker in metrics_by_ticker:
            m = metrics_by_ticker[ticker]
            comp["live_metrics"] = {
                "market_cap_usd": m.market_cap_usd,
                "ev_usd": m.ev_usd,
                "revenue_ttm_usd": m.revenue_ttm_usd,
                "ev_to_revenue_ttm": m.ev_to_revenue_ttm,
                "ebitda_margin_pct": m.ebitda_margin_pct,
                "data_available": m.data_available
            }
            
    verified_data["public_comps"] = public_comps
    
    return verified_data, fetch_usage
