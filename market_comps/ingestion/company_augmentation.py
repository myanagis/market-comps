import json
import logging
from typing import Any, List, Dict
import hashlib
import requests
from datetime import datetime

from market_comps.config import settings
from market_comps.llm_client import LLMClient
from market_comps.db.session import SessionLocal
from market_comps.db.models import (
    SourceDocument, DocumentText, CompanyAugmentationReport,
    PipelineRun, Organization, Person, PersonOrganizationRole
)

logger = logging.getLogger(__name__)

AUGMENTATION_SCHEMA = """
1. Market
2. Product & Differentiation
3. Team
4. Traction
5. Thesis / Mandate Fit (Note: CT presence/connection is highly important to our thesis)
"""

def generate_search_queries(company_name: str, company_domain: str, company_description: str) -> List[str]:
    """Use a fast LLM to generate 3 targeted Exa search queries."""
    client = LLMClient()
    prompt = f"""
    We need to research a company named '{company_name}' (domain: {company_domain}).
    Description: {company_description or 'None provided'}
    
    Please generate exactly 3 highly targeted search queries to find information on this company to fill out an investment memo.
    Queries should target:
    1. General overview, product, and differentiation
    2. Team, founders, and key personnel
    3. Recent traction, news, or fundraises
    
    Return a JSON array of strings containing ONLY the 3 queries.
    """
    try:
        queries, _ = client.structured_output(
            prompt=prompt,
            json_schema={"type": "array", "items": {"type": "string"}},
            model=settings.default_model
        )
        if isinstance(queries, list) and len(queries) > 0:
            return queries[:3]
    except Exception as e:
        logger.error(f"Failed to generate search queries: {e}")
    
    # Fallback
    return [
        f"{company_name} company overview product",
        f"{company_name} team founders",
        f"{company_name} news funding traction"
    ]

def fetch_exa_results(queries: List[str]) -> List[Dict]:
    """Fetch results from Exa API."""
    if not settings.exa_api_key:
        raise ValueError("EXA_API_KEY is not set.")
        
    all_results = []
    seen_urls = set()
    
    for query in queries:
        url = "https://api.exa.ai/search"
        headers = {
            "x-api-key": settings.exa_api_key,
            "Content-Type": "application/json"
        }
        data = {
            "query": query,
            "numResults": 2,
            "contents": {"text": True}
        }
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            res_json = response.json()
            
            for item in res_json.get("results", []):
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    all_results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "text": item.get("text", ""),
                        "date": item.get("publishedDate", "")
                    })
        except Exception as e:
            logger.error(f"Exa search failed for query '{query}': {e}")
            
    return all_results

def process_and_score_evidence(documents: List[Dict]) -> Dict:
    """Process documents into the required schema and score them."""
    if not documents:
        default_section = {
            "summary": "No web documents found to analyze.",
            "evidenced_data": [],
            "score_1_to_10": 0,
            "confidence": "Low",
            "reasoning": "No search results returned from Exa API."
        }
        return {
            "executive_summary": "No web presence could be found for this company during augmentation.",
            "market": default_section,
            "product_and_differentiation": default_section,
            "team": default_section,
            "traction": default_section,
            "thesis_mandate_fit": default_section
        }
        
    client = LLMClient()
    
    doc_text_block = ""
    for i, doc in enumerate(documents):
        doc_text_block += f"\\n\\n--- Document {i+1}: {doc['title']} ({doc['url']}) ---\\n{doc['text'][:5000]}"
        
    prompt = f"""
    You are an expert venture capital analyst. Analyze the following documents and extract key evidence into the following schema:
    {AUGMENTATION_SCHEMA}
    
    For each section, provide:
    1. "summary": A brief 1-3 sentence summary/synthesis of the evidence found.
    2. "evidenced_data": A list of direct, verbatim quotes or atomic facts extracted from the text that fit this section. Do NOT paraphrase, extract verbatim.
    3. "score_1_to_10": An integer score (1-10) evaluating the strength/quality of this section based ONLY on the evidence provided. If there is no evidence, score it a 0. Calibrate your ratings harshly: 5 should be average, 10 should be exceptional/world's best.
    4. "confidence": High, Medium, or Low. Only use 'High' if we have complete, comprehensive data. Most should be 'Low' or 'Medium' confidence.
    5. "reasoning": A brief 1-sentence justification for the score.
    
    Additionally, provide a top-level "executive_summary" (1-2 paragraphs) summarizing the company as a whole.
    
    DOCUMENTS:
    {doc_text_block}
    """
    
    schema = {
        "type": "object",
        "properties": {
            "executive_summary": {"type": "string"},
            "market": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "evidenced_data": {"type": "array", "items": {"type": "string"}},
                    "score_1_to_10": {"type": "integer"},
                    "confidence": {"type": "string"},
                    "reasoning": {"type": "string"}
                }
            },
            "product_and_differentiation": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "evidenced_data": {"type": "array", "items": {"type": "string"}},
                    "score_1_to_10": {"type": "integer"},
                    "confidence": {"type": "string"},
                    "reasoning": {"type": "string"}
                }
            },
            "team": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "evidenced_data": {"type": "array", "items": {"type": "string"}},
                    "score_1_to_10": {"type": "integer"},
                    "confidence": {"type": "string"},
                    "reasoning": {"type": "string"}
                }
            },
            "traction": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "evidenced_data": {"type": "array", "items": {"type": "string"}},
                    "score_1_to_10": {"type": "integer"},
                    "confidence": {"type": "string"},
                    "reasoning": {"type": "string"}
                }
            },
            "thesis_mandate_fit": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "evidenced_data": {"type": "array", "items": {"type": "string"}},
                    "score_1_to_10": {"type": "integer"},
                    "confidence": {"type": "string"},
                    "reasoning": {"type": "string"}
                }
            }
        },
        "required": ["executive_summary", "market", "product_and_differentiation", "team", "traction", "thesis_mandate_fit"]
    }
    
    result, _ = client.structured_output(
        prompt=prompt,
        json_schema=schema,
        model="google/gemini-2.5-pro" # Use a stronger model for complex processing
    )
    return result

def extract_entities(documents: List[Dict]) -> List[Dict]:
    """Extract people and roles from the text."""
    if not documents:
        return []
        
    client = LLMClient()
    doc_text_block = "\\n".join([d["text"][:3000] for d in documents])
    
    prompt = f"""
    Extract any team members, founders, or executives mentioned in the text.
    Return a JSON array of objects with:
    - first_name
    - last_name
    - title
    - city (if mentioned)
    - linkedin_url (if mentioned)
    
    DOCUMENTS:
    {doc_text_block}
    """
    
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "title": {"type": "string"},
                "city": {"type": "string"},
                "linkedin_url": {"type": "string"}
            }
        }
    }
    
    try:
        result, _ = client.structured_output(prompt=prompt, json_schema=schema, model=settings.default_model)
        return result
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return []

def run_augmentation_pipeline(org_id: int):
    """Main entrypoint for the augmentation pipeline."""
    db = SessionLocal()
    try:
        org = db.query(Organization).filter_by(id=org_id).first()
        if not org:
            raise ValueError("Organization not found")
            
        # Create Report
        report = CompanyAugmentationReport(
            organization_id=org_id,
            schema_version="1.0",
            status="RUNNING"
        )
        db.add(report)
        db.commit()
        
        # Create Pipeline Run
        run = PipelineRun(
            pipeline_id=None,
            run_status="IN_PROGRESS"
        )
        db.add(run)
        db.commit()
        
        report.pipeline_run_id = run.id
        db.commit()
        
        # 1. Generate Queries
        queries = generate_search_queries(org.name, org.primary_domain or "", org.description or "")
        
        # 2. Fetch Data
        docs_data = fetch_exa_results(queries)
        
        for d in docs_data:
            content_hash = hashlib.sha256((d["text"] or "").encode()).hexdigest()
            src_doc = SourceDocument(
                pipeline_run_id=run.id,
                document_type="WEB_PAGE",
                document_class="augmentation",
                title=d["title"][:255] if d["title"] else "Web Page",
                source_url=d["url"],
                source_name=d["url"].split('/')[2] if '//' in d["url"] else d["url"][:255],
                document_date=d["date"] if d["date"] else None,
                content_hash=content_hash
            )
            db.add(src_doc)
            db.flush()
            
            doc_text = DocumentText(
                source_document_id=src_doc.id,
                data_type="PAGE_TEXT",
                raw_content=d["text"],
                content_hash=content_hash
            )
            db.add(doc_text)
            
        db.commit()
        
        # 3 & 4. Process and Score
        extracted_data = process_and_score_evidence(docs_data)
        
        # Extract scoring into separate field for ease of use
        scoring = {}
        for k, v in extracted_data.items():
            if k == "executive_summary":
                continue
            scoring[k] = {
                "score": v.get("score_1_to_10"),
                "confidence": v.get("confidence"),
                "reasoning": v.get("reasoning")
            }
            
        report.extracted_data_json = extracted_data
        report.scoring_json = scoring
        report.status = "SUCCESS"
        
        # 5. Extract Entities and Upsert
        people = extract_entities(docs_data)
        from market_comps.db.models import AuditTrail
        for p in people:
            first = p.get("first_name")
            last = p.get("last_name")
            if not first or not last: continue
            
            full_name = f"{first} {last}"
            person = db.query(Person).filter(Person.full_name.ilike(full_name)).first()
            if not person:
                person = Person(
                    first_name=first,
                    last_name=last,
                    full_name=full_name,
                    city=p.get("city"),
                    linkedin_url=p.get("linkedin_url"),
                    country="US"
                )
                db.add(person)
                db.flush()
                
                db.add(AuditTrail(
                    canonical_entity_type="PERSON",
                    canonical_entity_id=str(person.id),
                    mutation_type="CREATE",
                    source="WEB_AUGMENTATION",
                    created_by="SYSTEM"
                ))
            
            role = db.query(PersonOrganizationRole).filter_by(person_id=person.id, organization_id=org.id).first()
            if not role:
                role = PersonOrganizationRole(
                    person_id=person.id,
                    organization_id=org.id,
                    title=p.get("title") or "Executive",
                    source="WEB_AUGMENTATION"
                )
                db.add(role)
                db.flush()
                
                db.add(AuditTrail(
                    canonical_entity_type="PERSON_ROLE",
                    canonical_entity_id=str(role.id),
                    mutation_type="CREATE",
                    source="WEB_AUGMENTATION",
                    created_by="SYSTEM"
                ))
                
        db.commit()
        run.run_status = "SUCCESS"
        run.completed_at = datetime.utcnow()
        db.commit()
        
    except Exception as e:
        logger.error(f"Augmentation pipeline failed: {e}")
        if 'report' in locals():
            report.status = "FAILED"
            report.error_message = str(e)
            db.commit()
        raise e
    finally:
        db.close()

def clear_augmentation_data(org_id: int):
    """Clear all augmentation data for a company to allow a fresh run."""
    db = SessionLocal()
    from market_comps.db.models import AuditTrail
    try:
        # Delete PersonRoles linked to WEB_AUGMENTATION
        roles = db.query(PersonOrganizationRole).filter_by(organization_id=org_id, source="WEB_AUGMENTATION").all()
        for role in roles:
            db.query(AuditTrail).filter_by(canonical_entity_type="PERSON_ROLE", canonical_entity_id=str(role.id)).delete(synchronize_session=False)
            db.delete(role)
            
        reports = db.query(CompanyAugmentationReport).filter_by(organization_id=org_id).all()
        pipeline_run_ids = [r.pipeline_run_id for r in reports if r.pipeline_run_id]
        
        # Delete Reports
        db.query(CompanyAugmentationReport).filter_by(organization_id=org_id).delete(synchronize_session=False)
        
        if pipeline_run_ids:
            # SourceDocuments
            docs = db.query(SourceDocument).filter(SourceDocument.pipeline_run_id.in_(pipeline_run_ids)).all()
            doc_ids = [d.id for d in docs]
            
            if doc_ids:
                db.query(DocumentText).filter(DocumentText.source_document_id.in_(doc_ids)).delete(synchronize_session=False)
                db.query(SourceDocument).filter(SourceDocument.id.in_(doc_ids)).delete(synchronize_session=False)
                
            db.query(PipelineRun).filter(PipelineRun.id.in_(pipeline_run_ids)).delete(synchronize_session=False)
            
        db.commit()
    except Exception as e:
        logger.error(f"Failed to clear augmentation data: {e}")
        db.rollback()
        raise e
    finally:
        db.close()
