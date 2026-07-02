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
    PipelineRun, Organization, Person, PersonOrganizationRole,
    CompanyProfile, PersonEmail, Investment, AuditTrail
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
    
    Please generate exactly 4 highly targeted search queries to find information on this company to fill out an investment memo.
    Queries should target:
    1. General overview, product, and differentiation
    2. Team, founders, and key personnel
    3. Recent traction, news, or press releases
    4. Specific funding rounds, investment amounts, and lead investors
    
    Return a JSON object with a 'queries' array containing ONLY the 4 strings.
    """
    try:
        schema = {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        result, _ = client.structured_output(
            prompt=prompt,
            json_schema=schema,
            model=settings.default_model
        )
        if isinstance(result, dict) and "queries" in result:
            return result["queries"][:4]
        elif isinstance(result, list):
            return result[:4]
    except Exception as e:
        logger.error(f"Failed to generate search queries: {e}")
    
    # Fallback
    return [
        f"{company_name} company overview product",
        f"{company_name} team founders",
        f"{company_name} news traction",
        f"{company_name} funding rounds investors amounts"
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
            "numResults": 3,
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

def extract_company_basics(documents: List[Dict]) -> Dict:
    if not documents:
        return {}
    client = LLMClient()
    doc_text_block = "\\n".join([d["text"] for d in documents])
    
    prompt = f"""
    Extract the following company basics from the text.
    IMPORTANT: Only extract a field if you are highly certain it is correct based explicitly on the text. If you are unsure or the information is not present, return null for that field. Do not hallucinate or guess.
    
    Fields:
    - website: The company's primary website URL
    - description_short: A concise description of what the company does (under 20 words)
    - hq_location: The headquarters location (City, State, Country)
    - founded_year: The year the company was founded (integer)
    - sector: The broad industry category
    - subsector: The more specific market or niche
    
    DOCUMENTS:
    {doc_text_block}
    """
    
    schema = {
        "type": "object",
        "properties": {
            "website": {"type": ["string", "null"]},
            "description_short": {"type": ["string", "null"]},
            "hq_location": {"type": ["string", "null"]},
            "founded_year": {"type": ["integer", "null"]},
            "sector": {"type": ["string", "null"]},
            "subsector": {"type": ["string", "null"]}
        }
    }
    
    try:
        result, _ = client.structured_output(prompt=prompt, json_schema=schema, model=settings.default_model)
        return result
    except Exception as e:
        logger.error(f"Company basics extraction failed: {e}")
        return {}

def extract_entities(documents: List[Dict]) -> List[Dict]:
    """Extract people and roles from the text."""
    if not documents:
        return []
        
    client = LLMClient()
    doc_text_block = "\\n".join([d["text"] for d in documents])
    
    prompt = f"""
    Extract any team members, founders, or executives mentioned in the text.
    Return a JSON object with a 'people' array containing objects with:
    - first_name
    - last_name
    - title
    - city (if mentioned)
    - linkedin_url (if mentioned)
    - email (if mentioned)
    - is_founder (boolean, true if explicitly mentioned as a founder)
    
    DOCUMENTS:
    {doc_text_block}
    """
    
    schema = {
        "type": "object",
        "properties": {
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string"},
                        "last_name": {"type": "string"},
                        "title": {"type": "string"},
                        "city": {"type": "string"},
                        "linkedin_url": {"type": "string"},
                        "email": {"type": "string"},
                        "is_founder": {"type": "boolean"}
                    }
                }
            }
        }
    }
    
    try:
        result, _ = client.structured_output(prompt=prompt, json_schema=schema, model=settings.default_model)
        if isinstance(result, dict) and "people" in result:
            return result["people"]
        if isinstance(result, list):
            return result
        return []
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return []

def extract_investments(documents: List[Dict]) -> List[Dict]:
    if not documents:
        return []
    client = LLMClient()
    
    # We want to know which document the investment came from for traceability
    doc_text_block = ""
    for i, doc in enumerate(documents):
        doc_text_block += f"\n\n--- Document {i} (URL: {doc['url']}) ---\n{doc['text']}"
        
    prompt = f"""
    Extract any specific funding rounds or investments mentioned in the text.
    Return a JSON object with an 'investments' array containing objects with:
    - investor_name: The name of the firm or person who invested
    - round_type: e.g. "Series A", "Seed", "Venture Round"
    - total_round_amount: The total amount raised in the round (e.g. "$10M")
    - firm_investment_amount: The specific amount this investor contributed, if mentioned (e.g. "$2M")
    - investment_date: When the round happened (YYYY-MM-DD or similar)
    - is_lead: boolean, true if this investor lead the round
    - source_doc_index: the integer index of the document (0-based) this was found in
    
    Only extract clear investments.
    
    DOCUMENTS:
    {doc_text_block}
    """
    
    schema = {
        "type": "object",
        "properties": {
            "investments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "investor_name": {"type": "string"},
                        "round_type": {"type": "string"},
                        "total_round_amount": {"type": ["string", "null"]},
                        "firm_investment_amount": {"type": ["string", "null"]},
                        "investment_date": {"type": ["string", "null"]},
                        "is_lead": {"type": "boolean"},
                        "source_doc_index": {"type": "integer"}
                    },
                    "required": ["investor_name", "source_doc_index"]
                }
            }
        }
    }
    
    try:
        result, _ = client.structured_output(prompt=prompt, json_schema=schema, model=settings.default_model)
        if isinstance(result, dict) and "investments" in result:
            return result["investments"]
        if isinstance(result, list):
            return result
        return []
    except Exception as e:
        logger.error(f"Investment extraction failed: {e}")
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
            d["db_id"] = src_doc.id
            
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
        
        # 5. Extract and Upsert Basics
        basics = extract_company_basics(docs_data)
        if basics.get("website"): org.website_url = basics["website"]
        if basics.get("description_short"): org.description = basics["description_short"]
        
        loc = basics.get("hq_location")
        if loc:
            parts = [p.strip() for p in loc.split(",")]
            if len(parts) >= 1: org.city = parts[0]
            if len(parts) >= 2: org.state = parts[1]
            if len(parts) >= 3: org.country = parts[2]
            
        profile = db.query(CompanyProfile).filter_by(organization_id=org.id).first()
        if not profile:
            profile = CompanyProfile(organization_id=org.id)
            db.add(profile)
        
        if basics.get("founded_year"): profile.founded_year = basics["founded_year"]
        if basics.get("sector"): profile.industry = basics["sector"]
        if basics.get("subsector"): profile.subindustry = basics["subsector"]
        
        db.flush()

        # 6. Extract and Upsert Entities (Founders / Team)
        people = extract_entities(docs_data)
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
            
            if p.get("email"):
                existing_email = db.query(PersonEmail).filter_by(person_id=person.id, email=p["email"]).first()
                if not existing_email:
                    email_record = PersonEmail(
                        person_id=person.id,
                        email=p["email"],
                        organization_id=org.id,
                        is_primary=True
                    )
                    db.add(email_record)
            
            title = p.get("title") or ("Founder" if p.get("is_founder") else "Executive")
            role = db.query(PersonOrganizationRole).filter_by(person_id=person.id, organization_id=org.id).first()
            if not role:
                role = PersonOrganizationRole(
                    person_id=person.id,
                    organization_id=org.id,
                    title=title,
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
            elif p.get("is_founder") and "founder" not in (role.title or "").lower():
                role.title = "Founder & " + (role.title or "Executive")
        
        # 7. Extract and Upsert Investments
        investments_data = extract_investments(docs_data)
        for inv in investments_data:
            investor_name = inv.get("investor_name")
            if not investor_name: continue
            
            investor_org = db.query(Organization).filter(Organization.name.ilike(investor_name)).first()
            if not investor_org:
                investor_org = Organization(
                    name=investor_name,
                    organization_type="INVESTOR",
                    status="ACTIVE"
                )
                db.add(investor_org)
                db.flush()
                
                db.add(AuditTrail(
                    canonical_entity_type="ORGANIZATION",
                    canonical_entity_id=str(investor_org.id),
                    mutation_type="CREATE",
                    source="WEB_AUGMENTATION",
                    created_by="SYSTEM"
                ))
            
            doc_idx = inv.get("source_doc_index")
            source_doc_id = None
            if doc_idx is not None and 0 <= doc_idx < len(docs_data):
                source_doc_id = docs_data[doc_idx].get("db_id")
                
            investment = Investment(
                investor_organization_id=investor_org.id,
                company_organization_id=org.id,
                round_type=inv.get("round_type"),
                total_round_amount=inv.get("total_round_amount"),
                firm_investment_amount=inv.get("firm_investment_amount"),
                is_lead=inv.get("is_lead", False),
                source_document_id=source_doc_id
            )
            
            # Simple date parsing attempt
            date_str = inv.get("investment_date")
            if date_str:
                try:
                    investment.investment_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    pass
            
            db.add(investment)
            db.flush()
            
            db.add(AuditTrail(
                canonical_entity_type="INVESTMENT",
                canonical_entity_id=str(investment.id),
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


def fetch_jina_content(url: str) -> str:
    import requests
    try:
        response = requests.get(f"https://r.jina.ai/{url}", timeout=15)
        if response.status_code == 200:
            return response.text
    except Exception:
        pass
    return ""

def run_manual_url_augmentation(org_id: int, url: str):
    """Augment a company's profile incrementally by reading a single manually provided URL."""
    db = SessionLocal()
    from market_comps.db.models import Organization, CompanyAugmentationReport, PipelineRun, SourceDocument, DocumentText, Person, PersonOrganizationRole, Investment, CompanyProfile, AuditTrail
    import hashlib
    from datetime import datetime
    
    try:
        org = db.query(Organization).filter(Organization.id == org_id).first()
        if not org:
            raise ValueError("Organization not found")
            
        # Link to the existing pipeline run or create a new one
        report = db.query(CompanyAugmentationReport).filter_by(organization_id=org_id).order_by(CompanyAugmentationReport.created_at.desc()).first()
        if report and report.pipeline_run_id:
            run_id = report.pipeline_run_id
        else:
            run = PipelineRun(pipeline_id=None, run_status="IN_PROGRESS")
            db.add(run)
            db.commit()
            run_id = run.id
            if report:
                report.pipeline_run_id = run_id
                db.commit()
                
        # Fetch Data
        text = fetch_jina_content(url)
        if not text:
            raise ValueError(f"Could not extract meaningful content from {url}")
            
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        
        src_doc = SourceDocument(
            pipeline_run_id=run_id,
            document_type="WEB_PAGE",
            document_class="manual_augmentation",
            title="Manual Upload",
            source_url=url,
            source_name=url.split('/')[2] if '//' in url else url[:255],
            source_type="MANUAL_UPLOAD",
            content_hash=content_hash
        )
        db.add(src_doc)
        db.flush()
        
        doc_text = DocumentText(
            source_document_id=src_doc.id,
            data_type="PAGE_TEXT",
            raw_content=text,
            content_hash=content_hash
        )
        db.add(doc_text)
        db.flush()
        
        docs_data = [{"index": 0, "url": url, "text": text[:15000], "title": "Manual Upload", "db_id": src_doc.id, "date": ""}]
        
        # 1. Basics (Upsert only missing fields)
        basics = extract_company_basics(docs_data)
        if basics:
            if basics.get("website") and not org.website_url: org.website_url = basics["website"]
            if basics.get("description_short") and not org.description: org.description = basics["description_short"]
            
            loc = basics.get("hq_location")
            if loc and not org.city:
                parts = [p.strip() for p in loc.split(",")]
                if len(parts) >= 1: org.city = parts[0]
                if len(parts) >= 2: org.state = parts[1]
                if len(parts) >= 3: org.country = parts[2]
                
            profile = db.query(CompanyProfile).filter_by(organization_id=org.id).first()
            if not profile:
                profile = CompanyProfile(organization_id=org.id)
                db.add(profile)
            
            if basics.get("founded_year") and not profile.founded_year: profile.founded_year = basics["founded_year"]
            if basics.get("sector") and not profile.industry: profile.industry = basics["sector"]
            if basics.get("subsector") and not profile.subindustry: profile.subindustry = basics["subsector"]
            
        db.flush()

        # 2. People (Incremental Upsert)
        people = extract_entities(docs_data)
        for p in people:
            first = p.get("first_name")
            last = p.get("last_name")
            if not first or not last: continue
            
            existing_person = db.query(Person).filter(Person.first_name.ilike(first), Person.last_name.ilike(last)).first()
            if existing_person:
                existing_role = db.query(PersonOrganizationRole).filter_by(person_id=existing_person.id, organization_id=org.id).first()
                if not existing_role:
                    role = PersonOrganizationRole(person_id=existing_person.id, organization_id=org.id, title=p.get("title"))
                    db.add(role)
            else:
                person = Person(first_name=first, last_name=last, city=p.get("city"), linkedin_url=p.get("linkedin_url"))
                db.add(person)
                db.flush()
                role = PersonOrganizationRole(person_id=person.id, organization_id=org.id, title=p.get("title"))
                db.add(role)
        db.flush()
        
        # 3. Investments (Incremental Upsert)
        investments_data = extract_investments(docs_data)
        for inv in investments_data:
            investor_name = inv.get("investor_name")
            if not investor_name: continue
            
            investor_org = db.query(Organization).filter(Organization.name.ilike(investor_name)).first()
            if not investor_org:
                investor_org = Organization(name=investor_name, organization_type="INVESTOR", status="ACTIVE")
                db.add(investor_org)
                db.flush()
            
            r_type = inv.get("round_type")
            existing_inv = db.query(Investment).filter_by(investor_organization_id=investor_org.id, company_organization_id=org.id, round_type=r_type).first()
            
            if not existing_inv:
                investment = Investment(
                    investor_organization_id=investor_org.id,
                    company_organization_id=org.id,
                    round_type=r_type,
                    total_round_amount=inv.get("total_round_amount"),
                    firm_investment_amount=inv.get("firm_investment_amount"),
                    is_lead=inv.get("is_lead", False),
                    source_document_id=src_doc.id
                )
                date_str = inv.get("investment_date")
                if date_str:
                    try:
                        investment.investment_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except:
                        pass
                db.add(investment)
                
        db.commit()
        return True
        
    except Exception as e:
        logger.error(f"Manual augmentation failed: {e}")
        db.rollback()
        raise e
    finally:
        db.close()
