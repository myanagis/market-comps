import json
import logging
from datetime import datetime
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from sqlalchemy.orm import Session
from market_comps.db.models import (
    IngestionConfig, IngestionJob, RawEntity,
    Organization, CompanyProfile, Person, PersonOrganizationRole, 
    ProgramProfile, ProgramMembership
)
from market_comps.llm_client import LLMClient

logger = logging.getLogger(__name__)

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
            # Strip HTML
            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style"]):
                script.extract()
            text_content = soup.get_text(separator="\n", strip=True)
            
            # Send to LLM
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
                                "founders": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "first_name": {"type": "string"},
                                            "last_name": {"type": "string"},
                                            "linkedin_url": {"type": "string"}
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
            
            # Check for custom instructions in metadata
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
                    # Fallback to normalized name
                    org = db.query(Organization).filter_by(normalized_name=name.lower()).first()
                
                if org:
                    if c.get("description"): org.description = c.get("description")
                else:
                    org = Organization(
                        name=name,
                        normalized_name=name.lower(),
                        primary_domain=domain if domain else None,
                        website_url=company_url,
                        description=c.get("description"),
                        organization_type="COMPANY"
                    )
                    db.add(org)
                    
                db.flush() # Ensure org.id is available
                
                profile = db.query(CompanyProfile).filter_by(organization_id=org.id).first()
                if not profile:
                    profile = CompanyProfile(organization_id=org.id)
                    db.add(profile)
                
                # RECONCILIATION 2: RawEntity mapping
                entity = RawEntity(
                    ingestion_job_id=job.id,
                    entity_type="ORGANIZATION",
                    matched_organization_id=org.id,
                    raw_name=name,
                    normalized_name=name.lower(),
                    source_url=url, # Use the URL we scraped as the source
                    raw_payload_json=c,
                    detected_at=datetime.utcnow()
                )
                db.add(entity)
                
                # RECONCILIATION 3: Founders -> Person + PersonOrganizationRole
                founders = c.get("founders", [])
                for f in founders:
                    fname = f.get("first_name", "")
                    lname = f.get("last_name", "")
                    if not fname or not lname:
                        continue
                        
                    person = db.query(Person).filter_by(first_name=fname, last_name=lname).first()
                    full_name = f"{fname} {lname}"
                    
                    if not person:
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
                    
                    # Link founder to company
                    role = db.query(PersonOrganizationRole).filter_by(person_id=person.id, organization_id=org.id).first()
                    if not role:
                        role = PersonOrganizationRole(
                            person_id=person.id,
                            organization_id=org.id,
                            title="Founder",
                            is_current=True
                        )
                        db.add(role)
                        
                # RECONCILIATION 4: Program Tags -> ProgramMembership
                program_tags = c.get("program_tags", [])
                for tag in program_tags:
                    # Find matching program by exact name
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
                            
                records_created += 1
                
            job.job_logs_json = {"llm_usage": usage.model_dump(), "extracted_companies": companies}
            
        elif config.ingestion_type == "API":
            data = response.json()
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
