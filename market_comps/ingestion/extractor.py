"""
Extractor — LLM Entity Extraction
===================================
Uses OpenRouter LLM to extract structured entities and relationships from raw text.
Writes to extracted_entities and extracted_relationships tables.
No CRM reconciliation — that's the reconciler's job.

Each pipeline_type has its own schema and prompt tuned for the expected page format.
"""

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from market_comps.db.models import (
    IngestionRun, DocumentText, ExtractionJob, ExtractedEntity, ExtractedRelationship
)
from market_comps.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ==============================================================================
# LLM SCHEMAS — one per pipeline type
# ==============================================================================

PROGRAM_COMPANY_SCHEMA = {
    "type": "object",
    "properties": {
        "companies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "url": {"type": "string", "description": "Company's own website URL"},
                    "description": {"type": "string"},
                    "industry": {"type": "string"},
                    "founded_year": {"type": "integer"},
                    "linkedin_url": {"type": "string"},
                    "profile_path": {"type": "string", "description": "Relative URL path to the company's detail page, e.g. /companies/acme"},
                    "founders": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Full name if first/last not separated"},
                                "first_name": {"type": "string"},
                                "last_name": {"type": "string"},
                                "title": {"type": "string"},
                                "linkedin_url": {"type": "string"},
                                "email": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["name"]
            }
        }
    },
    "required": ["companies"]
}

PROFILE_DETAIL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "url": {"type": "string", "description": "The company's own website URL"},
        "description": {"type": "string"},
        "industry": {"type": "string"},
        "founded_year": {"type": "integer"},
        "linkedin_url": {"type": "string"},
        "founders": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name if first/last not separated"},
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "title": {"type": "string"},
                    "linkedin_url": {"type": "string"},
                    "email": {"type": "string"}
                }
            }
        }
    },
    "required": ["name"]
}

SCHEMA_BY_TYPE = {
    "PROGRAM_COMPANY_PAGE": PROGRAM_COMPANY_SCHEMA,
    "INVESTOR_PORTFOLIO_PAGE": PROGRAM_COMPANY_SCHEMA,  # Same structure, different prompt
    "API_COMPANY_SEARCH": PROGRAM_COMPANY_SCHEMA,
    "DOCUMENT_ENTITIES": PROGRAM_COMPANY_SCHEMA,
}


# ==============================================================================
# EXTRACTION FUNCTIONS
# ==============================================================================

def extract_entities_from_text(
    db: Session,
    run: IngestionRun,
    job: ExtractionJob,
    doc_text: DocumentText,
    pipeline_type: str,
    config: dict
) -> dict:
    """Use the LLM to extract structured entities from raw text content.

    Writes ExtractedEntity and ExtractedRelationship records to the DB.

    Returns a dict with extraction stats and the raw LLM response.
    """
    text = doc_text.raw_content or ""
    custom_instruction = config.get("llm_instruction", "")
    
    from market_comps.ingestion.schema_builder import load_toml_schema_as_json, get_schema_description
    
    # Check if it's a hardcoded schema or dynamic TOML
    is_dynamic = False
    schema = SCHEMA_BY_TYPE.get(pipeline_type)
    if not schema:
        schema = load_toml_schema_as_json(pipeline_type)
        if schema:
            is_dynamic = True
            
    if not schema:
        schema = PROGRAM_COMPANY_SCHEMA
        pipeline_type = "PROGRAM_COMPANY_PAGE"

    # Build the prompt
    if is_dynamic:
        desc = get_schema_description(pipeline_type)
        prompt = f"Extract all entities from this document based on the provided schema. {desc}"
        if custom_instruction:
            prompt += f"\n\nADDITIONAL INSTRUCTIONS: {custom_instruction}"
    else:
        prompt = _build_extraction_prompt(pipeline_type, custom_instruction)
        
    prompt += f"\n\nTEXT:\n{text[:50000]}"

    # Call the LLM
    from market_comps.config import DEFAULT_LLM_MODEL
    llm = LLMClient()
    parsed, usage = llm.structured_output(
        prompt=prompt,
        json_schema=schema,
        system_prompt="You are a data extraction assistant. Follow the user's instructions and schema strictly.",
        model=DEFAULT_LLM_MODEL,
        step_name="entity_extraction"
    )

    entity_count = 0
    relationship_count = 0
    
    if is_dynamic:
        # Dynamic schema handling (Generic)
        entities_list = parsed.get("entities", [])
        if not isinstance(entities_list, list):
            entities_list = [parsed]
            
        for entity_data in entities_list:
            if not isinstance(entity_data, dict):
                continue
                
            # Default to the schema name as entity type (e.g. COMPANY_REVENUE_METRICS)
            entity_type = pipeline_type.upper()
            raw_name = entity_data.get("company_name", "Unknown")
            
            # If the schema defines a company, we treat it as an organization payload
            if "company_name" in entity_data:
                entity_type = "ORGANIZATION"
                
            generic_entity = ExtractedEntity(
                extraction_job_id=job.id,
                entity_type=entity_type,
                raw_name=raw_name,
                normalized_name=raw_name.lower(),
                extracted_payload_json=entity_data,
            )
            db.add(generic_entity)
            db.flush()
            entity_count += 1
            
        companies = entities_list
        
    else:
        # Legacy hardcoded schema handling (companies array)
        companies = _normalize_companies(parsed)

        for c in companies:
            if not isinstance(c, dict):
                continue

            # Create ORGANIZATION entity
            org_entity = ExtractedEntity(
                extraction_job_id=job.id,
                entity_type="ORGANIZATION",
                raw_name=c.get("name", "Unknown"),
                normalized_name=(c.get("name") or "unknown").lower(),
                extracted_payload_json=c,
            )
            db.add(org_entity)
            db.flush()
            entity_count += 1

            # Create PERSON entities + FOUNDER_OF relationships
            founders = c.get("founders", [])
            for f in founders:
                if not isinstance(f, dict):
                    continue
                fname = f.get("first_name", "")
                lname = f.get("last_name", "")
                if not fname and not lname:
                    full = f.get("name", "").strip()
                    if not full:
                        continue
                    parts = full.split(None, 1)
                    fname = parts[0]
                    lname = parts[1] if len(parts) > 1 else ""
                    f["first_name"] = fname
                    f["last_name"] = lname
                if not fname:
                    continue

                person_entity = ExtractedEntity(
                    extraction_job_id=job.id,
                    entity_type="PERSON",
                    raw_name=f"{fname} {lname}",
                    normalized_name=f"{fname} {lname}".lower(),
                    extracted_payload_json=f,
                )
                db.add(person_entity)
                db.flush()
                entity_count += 1

                # Create FOUNDER_OF relationship (person → org)
                rel = ExtractedRelationship(
                    extraction_job_id=job.id,
                    relationship_type="FOUNDER_OF",
                    source_extracted_entity_id=person_entity.id,
                    source_entity_type="PERSON",
                    target_extracted_entity_id=org_entity.id,
                    target_entity_type="ORGANIZATION",
                    relationship_payload_json={"title": f.get("title", "Founder")},
                )
                db.add(rel)
                relationship_count += 1

    db.flush()

    return {
        "entities_extracted": entity_count,
        "relationships_extracted": relationship_count,
        "llm_usage": usage.model_dump(mode="json"),
        "companies_raw": companies
    }


def extract_profile_detail(
    db: Session,
    run: IngestionRun,
    job: ExtractionJob,
    doc_text: DocumentText,
    company_name: str
) -> dict:
    """Deep scrape Pass 2: extract rich firmographics from a single profile page.

    Returns the extracted detail dict and LLM usage.
    """
    text = doc_text.raw_content or ""

    prompt = (
        f"Extract detailed company information for '{company_name}' from this profile page. "
        "Include the company's own website URL, industry, founded year, LinkedIn URL, "
        "and all founders with their names, titles, LinkedIn URLs, and emails.\n"
        f"\nTEXT:\n{text[:30000]}"
    )

    llm = LLMClient()
    from market_comps.config import DEFAULT_LLM_MODEL
    parsed, usage = llm.structured_output(
        prompt=prompt,
        json_schema=PROFILE_DETAIL_SCHEMA,
        system_prompt="You are a data extraction assistant. Follow the user's instructions and schema strictly.",
        model=DEFAULT_LLM_MODEL,
        step_name="profile_extraction",
    )

    detail = parsed if isinstance(parsed, dict) else {}
    return {"detail": detail, "llm_usage": usage.model_dump(mode="json")}


# ==============================================================================
# HELPERS
# ==============================================================================

def _build_extraction_prompt(pipeline_type: str, custom_instruction: str) -> str:
    """Build a pipeline-type-specific extraction prompt."""
    base_prompts = {
        "PROGRAM_COMPANY_PAGE": (
            "Extract all companies from this accelerator/program directory page. "
            "For each company, extract its name, website URL, LinkedIn URL, description, "
            "industry, founded year, and the relative link path to its detail page if available. "
            "Also extract all founders/team members with their names, titles, LinkedIn URLs, and emails."
        ),
        "INVESTOR_PORTFOLIO_PAGE": (
            "Extract all portfolio companies from this investor's portfolio page. "
            "For each company, extract its name, website URL, LinkedIn URL, description, "
            "industry, founded year, and any detail page link. "
            "Also extract founders/team members with their names, titles, LinkedIn URLs, and emails."
        ),
        "API_COMPANY_SEARCH": (
            "Extract all companies from the following data. "
            "For each company, extract its name, website URL, LinkedIn URL, description, "
            "industry, and founded year."
        ),
        "DOCUMENT_ENTITIES": (
            "Extract all companies and people mentioned in this document. "
            "For each company, extract its name, website URL, description, industry, and founded year. "
            "For any people mentioned, extract their names, titles, LinkedIn URLs, and link them to the appropriate company as founders or team members."
        ),
    }

    prompt = base_prompts.get(pipeline_type, base_prompts["PROGRAM_COMPANY_PAGE"])
    if custom_instruction:
        prompt += f"\n\nADDITIONAL INSTRUCTIONS: {custom_instruction}"
    return prompt


def _normalize_companies(parsed) -> list[dict]:
    """Safely extract a list of company dicts from LLM output."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return parsed.get("companies", [])
    return []
