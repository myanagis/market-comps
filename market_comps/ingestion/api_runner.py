import json
import logging
from datetime import datetime
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from sqlalchemy.orm import Session
from market_comps.db.models import (
    IngestionConfig, IngestionJob, RawEntity, EntityUpdate,
    Organization, CompanyProfile, Person, PersonOrganizationRole, 
    ProgramProfile, ProgramMembership
)
from market_comps.llm_client import LLMClient

logger = logging.getLogger(__name__)

def extract_and_reconcile_entities(db: Session, job: IngestionJob, config: IngestionConfig, text_content: str, source_url: str) -> dict:
    """Uses the LLM to extract entities from text, and maps them to the CRM."""
    ds = config.data_source
    llm = LLMClient()
    
    schema = {
        "type": "object",
        "properties": {
            "companies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "url": {"type": "string"},
                        "description": {"type": "string"},
                        "industry": {"type": "string"},
                        "founded_year": {"type": "integer"},
                        "linkedin_url": {"type": "string"},
                        "founders": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "first_name": {"type": "string"},
                                    "last_name": {"type": "string"},
                                    "linkedin_url": {"type": "string"},
                                    "title": {"type": "string"},
                                    "email": {"type": "string"}
                                },
                                "required": ["first_name", "last_name"]
                            }
                        },
                        "program_tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["name", "description"]
                }
            }
        },
        "required": ["companies"]
    }
    
    custom_instruction = ""
    if config.metadata_json and isinstance(config.metadata_json, dict):
        custom_instruction = config.metadata_json.get("llm_instruction", "")
        
    prompt = f"Extract companies from the following text.\n"
    if custom_instruction:
        prompt += f"INSTRUCTIONS: {custom_instruction}\n"
    prompt += f"\nTEXT:\n{text_content[:50000]}" # limit to fit in context just in case
    
    parsed, usage = llm.structured_output(
        prompt=prompt,
        json_schema=schema,
        system_prompt="You are a data extraction assistant. Follow the user's instructions and schema strictly.",
        model="google/gemini-2.5-flash"
    )
    
    if isinstance(parsed, list):
        companies = parsed
    elif isinstance(parsed, dict):
        companies = parsed.get("companies", [])
    else:
        companies = []
        
    records_created = 0
    for c in companies:
        if not isinstance(c, dict):
            continue
            
        name = c.get("name") or "Unknown"
        company_url = c.get("url", "")
        
        # RECONCILIATION 1: Find or Create Organization & CompanyProfile
        domain = ""
        if company_url:
            parsed_url = urlparse(company_url)
            domain = parsed_url.netloc.replace("www.", "")
        
        org = None
        if domain:
            org = db.query(Organization).filter_by(primary_domain=domain).first()
        if not org:
            org = db.query(Organization).filter_by(normalized_name=name.lower()).first()
        
        is_new_org = False
        if org:
            if c.get("description") and org.description != c.get("description"):
                org.description = c.get("description")
            if c.get("linkedin_url") and not org.linkedin_url:
                org.linkedin_url = c.get("linkedin_url")
        else:
            is_new_org = True
            org = Organization(
                name=name,
                normalized_name=name.lower(),
                primary_domain=domain if domain else None,
                website_url=company_url,
                description=c.get("description"),
                linkedin_url=c.get("linkedin_url"),
                organization_type="COMPANY"
            )
            db.add(org)
            
        db.flush() # Ensure org.id is available
        
        profile = db.query(CompanyProfile).filter_by(organization_id=org.id).first()
        if not profile:
            profile = CompanyProfile(organization_id=org.id)
            db.add(profile)
            
        if c.get("industry") and not profile.industry:
            profile.industry = c.get("industry")
        if c.get("founded_year") and not profile.founded_year:
            profile.founded_year = c.get("founded_year")
        
        # RECONCILIATION 2: RawEntity mapping
        entity = RawEntity(
            ingestion_job_id=job.id,
            entity_type="ORGANIZATION",
            matched_organization_id=org.id,
            raw_name=name,
            normalized_name=name.lower(),
            source_url=source_url, 
            raw_payload_json=c,
            detected_at=datetime.utcnow()
        )
        db.add(entity)
        db.flush() 
        
        # RECONCILIATION 2.5: EntityUpdate logging
        if is_new_org:
            db.add(EntityUpdate(
                organization_id=org.id,
                raw_entity_id=entity.id,
                ingestion_job_id=job.id,
                update_reason="AUTO_CREATE",
                update_action="CREATE",
                source=ds.source_name
            ))
        else:
            db.add(EntityUpdate(
                organization_id=org.id,
                raw_entity_id=entity.id,
                ingestion_job_id=job.id,
                update_reason="SOURCE_PRIORITY",
                update_action="UPDATE",
                source=ds.source_name
            ))
        
        # RECONCILIATION 3: Founders -> Person + PersonOrganizationRole
        founders = c.get("founders", [])
        for f in founders:
            fname = f.get("first_name", "")
            lname = f.get("last_name", "")
            if not fname or not lname:
                continue
                
            person = db.query(Person).filter_by(first_name=fname, last_name=lname).first()
            full_name = f"{fname} {lname}"
            
            is_new_person = False
            if not person:
                is_new_person = True
                person = Person(
                    first_name=fname,
                    last_name=lname,
                    full_name=full_name,
                    linkedin_url=f.get("linkedin_url")
                )
                db.add(person)
            else:
                if f.get("linkedin_url") and not person.linkedin_url:
                    person.linkedin_url = f.get("linkedin_url")
                    
            db.flush()
            
            if is_new_person:
                db.add(EntityUpdate(
                    person_id=person.id,
                    raw_entity_id=entity.id,
                    ingestion_job_id=job.id,
                    update_reason="AUTO_CREATE",
                    update_action="CREATE",
                    source=ds.source_name
                ))
                
            # Link founder to company
            role = db.query(PersonOrganizationRole).filter_by(person_id=person.id, organization_id=org.id).first()
            title = f.get("title") or "Founder"
            if not role:
                role = PersonOrganizationRole(
                    person_id=person.id,
                    organization_id=org.id,
                    title=title,
                    is_current=True
                )
                db.add(role)
            else:
                role.title = title
                
            # Save Email if found
            email_val = f.get("email")
            if email_val:
                from market_comps.db.models import PersonEmail
                existing_email = db.query(PersonEmail).filter_by(email=email_val).first()
                if not existing_email:
                    em = PersonEmail(
                        person_id=person.id,
                        email=email_val,
                        organization_id=org.id,
                        is_primary=True
                    )
                    db.add(em)
                
        # RECONCILIATION 4: Program Tags -> ProgramMembership
        program_tags = c.get("program_tags", [])
        for tag in program_tags:
            prog = db.query(ProgramProfile).filter_by(program_name=tag).first()
            if prog:
                membership = db.query(ProgramMembership).filter_by(company_organization_id=org.id, program_id=prog.id).first()
                if not membership:
                    membership = ProgramMembership(
                        company_organization_id=org.id,
                        program_id=prog.id,
                        is_active=True
                    )
                    db.add(membership)
                    
        # Track what happened for UI
        c["__reconciliation_status__"] = "CREATED_ORG" if is_new_org else "UPDATED_ORG"
        c["__organization_id__"] = org.id
                    
        records_created += 1
        
    return {
        "records_created": records_created,
        "job_logs_json": {"llm_usage": usage.model_dump(), "extracted_companies": companies}
    }


def run_ingestion_config(db: Session, config_id: int, triggered_by: str = "MANUAL") -> IngestionJob:
    """Executes an ingestion config and logs the result."""
    
    config = db.query(IngestionConfig).filter(IngestionConfig.id == config_id).first()
    if not config:
        raise ValueError(f"IngestionConfig {config_id} not found")
        
    ds = config.data_source
    
    # Create the job record
    job = IngestionJob(
        ingestion_config_id=config.id,
        job_status="RUNNING",
        triggered_by=triggered_by,
        started_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Construct URL
    base = (ds.base_url or "").rstrip("/")
    endpoint = (config.endpoint_url or "").lstrip("/")
    url = f"{base}/{endpoint}" if base else endpoint
    
    method = config.http_method or "GET"
    headers = config.headers_json or {}
    params = config.query_params_json or {}
    
    try:
        # Fetch data
        response = requests.request(method, url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        records_created = 0
        
        if config.ingestion_type == "SCRAPE":
            from playwright.sync_api import sync_playwright
            import os
            
            with sync_playwright() as p:
                try:
                    browser = p.chromium.launch(headless=True)
                except Exception as e:
                    if "Executable doesn't exist" in str(e) or "playwright install" in str(e):
                        logger.warning("Playwright browser not found. Installing chromium...")
                        os.system("playwright install chromium")
                        browser = p.chromium.launch(headless=True)
                    else:
                        raise e
                        
                page = browser.new_page()
                page.goto(url, wait_until="networkidle")
                
                # Scroll to bottom to trigger lazy loading
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(2000)
                
                html_content = page.content()
                browser.close()
                
            # Strip HTML
            soup = BeautifulSoup(html_content, "html.parser")
            for script in soup(["script", "style"]):
                script.extract()
            text_content = soup.get_text(separator="\n", strip=True)
            
            # Save source content directly to the job BEFORE LLM processing
            job.source_content = text_content
            db.commit()
            
            # Run the extracted LLM logic
            result = extract_and_reconcile_entities(db, job, config, text_content, url)
            records_created = result["records_created"]
            job.job_logs_json = result["job_logs_json"]
            
        elif config.ingestion_type == "API":
            data = response.json()
            
            job.source_content = json.dumps(data)
            db.commit()
            
            # If it's a list, save each as an entity
            if isinstance(data, list):
                for item in data:
                    name = item.get("name") or item.get("title") or "Unknown"
                    entity = RawEntity(
                        ingestion_job_id=job.id,
                        entity_type="ORGANIZATION",
                        raw_name=name,
                        normalized_name=name.lower(),
                        source_url=url,
                        raw_payload_json=item,
                        detected_at=datetime.utcnow()
                    )
                    db.add(entity)
                    records_created += 1
                job.job_logs_json = {"status": "success", "data": data[:10]} # Save first 10 for logs
            else:
                job.job_logs_json = {"status": "success", "data": data}
        
        # Complete job
        job.job_status = "SUCCESS"
        job.completed_at = datetime.utcnow()
        job.records_processed = records_created
        job.records_created = records_created
        db.commit()
        
    except Exception as e:
        logger.error(f"Ingestion job failed: {e}")
        db.rollback() # Rollback the session so we don't save broken entities
        
        # But we still want to save the job failure status!
        # Re-fetch the job in the new transaction context since rollback clears it
        failed_job = db.query(IngestionJob).filter(IngestionJob.id == job.id).first()
        if failed_job:
            failed_job.job_status = "FAILED"
            failed_job.error_message = str(e)
            failed_job.completed_at = datetime.utcnow()
            db.commit()
        
    return job
