# ==============================================================================
# UNIFIED EXTRACTION SCHEMAS
# ==============================================================================

# A shared core for company-related documents
_COMPANY_FIELDS = {
    "company_name": {"type": "string", "description": "The name of the company or startup"},
    "description": {"type": "string", "description": "Brief overview of what the company does"},
    "industry": {"type": "string", "description": "Primary industry or sector"},
    "founded_year": {"type": "integer"},
    "website": {"type": "string"},
    "linkedin_url": {"type": "string"},
    "headquarters": {"type": "string", "description": "City or country of headquarters"},
    "business_model": {"type": "string", "description": "e.g., B2B SaaS, Marketplace, D2C"},
    "target_audience": {"type": "string", "description": "Who the product or service is sold to"},
    "competitive_advantage": {"type": "string", "description": "Moat or unique differentiator"},
    "competitors": {"type": "string", "description": "Comma-separated list of competitors"},
    
    "total_funding": {"type": "number", "description": "Total capital raised to date"},
    "latest_valuation": {"type": "number", "description": "Most recent post-money valuation"},
    "latest_funding_round": {"type": "string", "description": "e.g., Series A, Seed"},
    
    "annual_revenue": {"type": "number", "description": "Most recent ARR or annual revenue figure"},
    "revenue_growth_percent": {"type": "number", "description": "Year over year revenue growth percentage"},
    "gross_margin_percent": {"type": "number", "description": "Gross margin percentage"},
    "ebitda": {"type": "number", "description": "EBITDA figure"},
    
    "key_risks": {"type": "string", "description": "Primary risks facing the business"},
    "notable_customers": {"type": "string", "description": "Comma-separated list of notable customers"},
    "customer_quotes": {"type": "string", "description": "Quotes or testimonials from customers"},
    
    "ip_and_patents": {"type": "string", "description": "Details on intellectual property or patents"},
    "regulatory_risks": {"type": "string", "description": "Compliance or regulatory risks"},
    "mna_activity": {"type": "string", "description": "Past acquisitions or M&A strategy"},
    
    "people": {
        "type": "array",
        "description": "Founders, executives, and key management",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full name"},
                "title": {"type": "string", "description": "Job title or role"},
                "linkedin_url": {"type": "string"},
                "email": {"type": "string"}
            },
            "required": ["name", "title"]
        }
    }
}

STARTUP_PITCH_DECK_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": _COMPANY_FIELDS,
                "required": ["company_name"]
            }
        }
    },
    "required": ["entities"]
}

COMPANY_DUE_DILIGENCE_SCHEMA = STARTUP_PITCH_DECK_SCHEMA
COMPANY_FINANCIAL_DOCUMENT_SCHEMA = STARTUP_PITCH_DECK_SCHEMA
MARKET_RESEARCH_REPORT_SCHEMA = STARTUP_PITCH_DECK_SCHEMA
INVESTOR_PORTFOLIO_PAGE_SCHEMA = STARTUP_PITCH_DECK_SCHEMA

LEGAL_CONTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "contract_title": {"type": "string"},
                    "effective_date": {"type": "string"},
                    "parties": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity_name": {"type": "string"},
                                "role": {"type": "string"}
                            },
                            "required": ["entity_name"]
                        }
                    },
                    "payment_terms": {"type": "string"},
                    "obligations": {"type": "string"}
                },
                "required": ["contract_title"]
            }
        }
    },
    "required": ["entities"]
}

SCHEMA_BY_CLASS = {
    "startup_pitch_deck": STARTUP_PITCH_DECK_SCHEMA,
    "company_due_diligence": COMPANY_DUE_DILIGENCE_SCHEMA,
    "company_financial_document": COMPANY_FINANCIAL_DOCUMENT_SCHEMA,
    "market_research_report": MARKET_RESEARCH_REPORT_SCHEMA,
    "investor_portfolio_page": INVESTOR_PORTFOLIO_PAGE_SCHEMA,
    "legal_contract": LEGAL_CONTRACT_SCHEMA,
}
