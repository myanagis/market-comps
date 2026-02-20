"""Smoke test for market_comps package imports and models."""

from market_comps.config import settings
from market_comps.models import ScanFilters, CompanyCandidate, CompanyMetrics, LLMUsage, CompsResult
from market_comps.llm_client import LLMClient
from market_comps.scanner import MarketScanner
from market_comps.metrics_fetcher import MetricsFetcher
from market_comps.comps_engine import CompsEngine

print("All imports OK")
print(f"API key present: {bool(settings.openrouter_api_key)}")
print(f"Default model: {settings.default_model}")

# Quick usage model test
u = LLMUsage()
u.add(1000, 200, 0.075, 0.30)
print(f"LLMUsage test: tokens={u.total_tokens}, cost=${u.estimated_cost_usd:.6f}")

# Quick ScanFilters test
f = ScanFilters(countries=["United States"], exchanges=["NASDAQ"])
print(f"ScanFilters OK: {f}")

# LLMClient instantiation (no API call)
client = LLMClient()
print(f"LLMClient OK: model={client.model}")

print("\n=== All smoke tests passed ===")
