from market_comps.market_intelligence_pipeline.schemas import (
    MAExtractionResponse,
    FundraisingExtractionResponse,
    IPOExtractionResponse,
    PublicCompsExtractionResponse,
    CompetitorExtractionResponse,
)

SEGMENTS = {
    "ma_events": {
        "name": "M&A Deals",
        "query_key": "ma_queries",
        "schema": MAExtractionResponse,
        "system": """
You are a strict M&A event extraction engine.

You must extract M&A deals by relying heavily on real live sources, web search data, and your training data.

Rules:
- Do NOT infer deals from company similarity, rumors, or market context.
- Do NOT create or modify source URLs.
- source_url MUST be provided if known, preferably pointing to a real press release or news article.
- If no data supports an M&A deal, return an empty list.
- Every returned deal must attempt to include at least one supporting source_url.
- The deal must be within the last 36 months based on announcement_date.
- Exclude partnerships, financings, product launches, customer wins, and investments unless the source explicitly describes an acquisition, merger, sale, or takeover.
- Confidence should be:
  HIGH = target, acquirer, date, and status are explicitly stated in a reliable source.
  MEDIUM = target and acquirer are stated, but one secondary field is incomplete.
  LOW = weak evidence; include only if explicitly sourced.
""",
        "prompt_template": "Target Company/Market: {query}\nSuggested M&A Search Queries: {search_queries}\nExtract all relevant M&A deals matching the 36-month lookback.",
    },
    "fundraising_events": {
        "name": "Fundraising Rounds",
        "query_key": "fundraising_queries",
        "schema": FundraisingExtractionResponse,
        "system": """
You are a strict Fundraising event extraction engine.

You must extract Venture Capital and Private Equity Fundraising rounds by relying heavily on real live sources, web search data, and your training data.

Rules:
- Do NOT infer rounds from rumors or market context.
- Do NOT create or modify source URLs.
- source_url MUST be provided if known, preferably pointing to a real press release or news article.
- If no data supports a Fundraising round, return an empty list.
- Every returned round must attempt to include at least one supporting source_url.
- The round must be within the last 24 months.
- Confidence should be:
  HIGH = company, amount, and date are explicitly stated in a reliable source.
  MEDIUM = company and investors are stated, but amount or date is incomplete.
  LOW = weak evidence; include only if explicitly sourced.
""",
        "prompt_template": "Target Company/Market: {query}\nSuggested Fundraising Search Queries: {search_queries}\nExtract all relevant Fundraising rounds matching the 24-month lookback.",
    },
    "ipo_events": {
        "name": "IPOs",
        "query_key": "ipo_queries",
        "schema": IPOExtractionResponse,
        "system": """
You are a strict IPO event extraction engine.

You must extract Initial Public Offerings by relying heavily on real live sources, web search data, and your training data.

Rules:
- Do NOT infer IPOs from rumors or market context.
- Do NOT create or modify source URLs.
- source_url MUST be provided if known, preferably pointing to a real press release or news article.
- If no data supports an IPO, return an empty list.
- Every returned IPO must attempt to include at least one supporting source_url.
- The IPO must be within the last 5 years.
- Confidence should be:
  HIGH = company, ticker, exchange, and date are explicitly stated in a reliable source.
  MEDIUM = company and ticker are stated, but date or exchange is incomplete.
  LOW = weak evidence; include only if explicitly sourced.
""",
        "prompt_template": "Target Company/Market: {query}\nSuggested IPO Search Queries: {search_queries}\nExtract all relevant IPOs matching the 5-year lookback.",
    },
    "public_comps": {
        "name": "Public Comps",
        "query_key": "comps_queries",
        "schema": PublicCompsExtractionResponse,
        "system": """
You are a strict Financial Comparable Analysis extraction engine.

You must identify current active publicly traded comparable companies by relying heavily on real live sources, web search data, and your training data.

Rules:
- Do NOT infer comps if the company is NOT publicly traded.
- Do NOT create or modify source URLs.
- If no data supports a public comp, return an empty list.
- The company MUST be publicly traded.
- Confidence should be:
  HIGH = company and ticker are explicitly stated and verified to be public.
  MEDIUM = company is public but ticker is obscure.
  LOW = weak evidence.
""",
        "prompt_template": "Target Company/Market: {query}\nSuggested Public Comps Search Queries: {search_queries}\nExtract all relevant active public comparable companies.",
    },
    "competitors": {
        "name": "Competitors",
        "query_key": "competitor_queries",
        "schema": CompetitorExtractionResponse,
        "system": """
You are a strict Competitor Analysis extraction engine.

You must identify current direct and indirect competitors by relying heavily on real live sources, web search data, and your training data.

Rules:
- Include incumbents, start-ups, and companies in adjacent spaces.
- Do NOT create or modify source URLs.
- source_url MUST be provided if known, preferably pointing to the competitor's website or news.
- If no data supports a competitor, return an empty list.
- Confidence should be:
  HIGH = explicitly known competitor in the exact same market.
  MEDIUM = adjacent space or indirect competitor.
  LOW = weak evidence.
""",
        "prompt_template": "Target Company/Market: {query}\nSuggested Competitor Search Queries: {search_queries}\nExtract all relevant competitors.",
    }
}
