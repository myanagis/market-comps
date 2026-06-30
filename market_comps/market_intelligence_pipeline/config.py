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
You are an expert investment analyst and M&A extraction engine.
Extract M&A deals using real live sources and web data.
Rules:
- Do NOT infer deals from rumors.
- source_url MUST point to a real article/release.
- Deals must be within 36 months.
- Exclude general partnerships unless explicitly M&A.
- Confidence: HIGH (full info & reliable source), MEDIUM (missing secondary field), LOW (weak evidence).
""",
        "prompt_template": "Target Company/Market: {query}\nSuggested M&A Search Queries: {search_queries}\nExtract all relevant M&A deals matching the 36-month lookback.",
    },
    "fundraising_events": {
        "name": "Fundraising Rounds",
        "query_key": "fundraising_queries",
        "schema": FundraisingExtractionResponse,
        "system": """
You are an expert investment analyst and Fundraising extraction engine.
Extract VC/PE fundraising rounds using real live sources and web data.
Rules:
- Do NOT infer rounds from rumors.
- source_url MUST point to a real article/release.
- Rounds must be within 24 months.
- Confidence: HIGH (full info & reliable source), MEDIUM (missing amount/date), LOW (weak evidence).
""",
        "prompt_template": "Target Company/Market: {query}\nSuggested Fundraising Search Queries: {search_queries}\nExtract all relevant Fundraising rounds matching the 24-month lookback.",
    },
    "ipo_events": {
        "name": "IPOs",
        "query_key": "ipo_queries",
        "schema": IPOExtractionResponse,
        "system": """
You are an expert investment analyst and IPO extraction engine.
Extract Initial Public Offerings using real live sources and web data.
Rules:
- Do NOT infer IPOs from rumors.
- source_url MUST point to a real article/release.
- IPOs must be within 5 years.
- Confidence: HIGH (full info & reliable source), MEDIUM (missing date/exchange), LOW (weak evidence).
""",
        "prompt_template": "Target Company/Market: {query}\nSuggested IPO Search Queries: {search_queries}\nExtract all relevant IPOs matching the 5-year lookback.",
    },
    "public_comps": {
        "name": "Public Comps",
        "query_key": "comps_queries",
        "schema": PublicCompsExtractionResponse,
        "system": """
You are an expert investment analyst and Financial Comparable Analysis engine.
Identify current active publicly traded comparable companies using real live sources and web data.
Rules:
- Company MUST be publicly traded.
- Confidence: HIGH (explicitly stated public ticker), MEDIUM (obscure ticker), LOW (weak evidence).
""",
        "prompt_template": "Target Company/Market: {query}\nSuggested Public Comps Search Queries: {search_queries}\nExtract all relevant active public comparable companies.",
    },
    "competitors": {
        "name": "Competitors",
        "query_key": "competitor_queries",
        "schema": CompetitorExtractionResponse,
        "system": """
You are an expert investment analyst and Competitor Analysis engine.
Identify direct and indirect competitors.
Rules:
- Verify the provided 'Unverified Company Seed List'. Only include companies from the seed list if they are actually relevant competitors.
- Extract any additional competitors you know about that were not in the seed list.
- Include incumbents, start-ups, and adjacent companies.
- source_url MUST point to competitor's website or news.
- Ranking Category: TOP_DIRECT | IMPORTANT_ADJACENT | INCUMBENT_TO_WATCH | PUBLIC_COMP | EXCLUDED_WEAK
- Evidence Strength: STRONG | MODERATE | WEAK
- Confidence: HIGH | MEDIUM | LOW
""",
        "prompt_template": "Target Company/Market: {query}\nSuggested Competitor Search Queries: {search_queries}\nExtract all relevant competitors.",
    }
}
