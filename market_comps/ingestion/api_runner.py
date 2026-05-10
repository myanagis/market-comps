"""
Ingestion Pipeline
==================

Architecture:
    1. EXTRACTION LAYER  — Pure data retrieval. No database operations.
       - fetch_page_text()           → Playwright browser rendering + HTML stripping
       - llm_extract_companies()     → Single-pass LLM: extract companies from directory text
       - llm_extract_directory()     → Deep scrape Pass 1: extract company names + profile URLs
       - llm_extract_profile()       → Deep scrape Pass 2: extract rich firmographics from one profile page

    2. BUSINESS LOGIC LAYER  — Database reconciliation. No LLM calls.
       - reconcile_company()         → Upsert one Organization + CompanyProfile + RawEntity + EntityUpdate
       - reconcile_person()          → Upsert one Person + PersonOrganizationRole + PersonEmail + EntityUpdate
       - reconcile_program_tags()    → Link Organization to ProgramProfiles via ProgramMembership

    3. ORCHESTRATOR  — Ties extraction and business logic together.
       - run_ingestion_config()      → Main entry point: creates job, dispatches to scrape/API path,
                                       calls extraction, then reconciliation, and logs results.
"""

import json
import logging
import os
from datetime import datetime
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from market_comps.db.models import (
    IngestionConfig, IngestionJob, RawEntity, EntityUpdate,
    Organization, CompanyProfile, Person, PersonOrganizationRole,
    PersonEmail, ProgramProfile, ProgramMembership
)
from market_comps.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. EXTRACTION LAYER  —  Pure data retrieval, no database operations
# ==============================================================================

def fetch_page_text(url: str) -> str:
    """Render a URL with Playwright (headless Chromium) and return stripped text.

    Handles:
        - Auto-installing Chromium if missing (Streamlit Cloud)
        - Scrolling to trigger lazy-loaded content
        - Stripping <script> and <style> tags
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as e:
            if "Executable doesn't exist" in str(e) or "playwright install" in str(e):
                logger.warning("Playwright browser not found — installing chromium…")
                os.system("playwright install chromium")
                browser = p.chromium.launch(headless=True)
            else:
                raise

        page = browser.new_page()
        page.goto(url, wait_until="networkidle")

        # Scroll to bottom to trigger lazy loading
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(2000)

        html_content = page.content()
        browser.close()

    return strip_html(html_content)


def strip_html(html: str) -> str:
    """Remove script/style tags and return clean text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.extract()
    return soup.get_text(separator="\n", strip=True)


# --- LLM Schemas ---------------------------------------------------------------

COMPANY_SCHEMA = {
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

DIRECTORY_LINK_SCHEMA = {
    "type": "object",
    "properties": {
        "companies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "profile_path": {"type": "string", "description": "Relative URL path to the company's profile page, e.g. /companies/acme"},
                    "description": {"type": "string"},
                    "program_tags": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name"]
            }
        }
    },
    "required": ["companies"]
}

PROFILE_SCHEMA = {
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
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "linkedin_url": {"type": "string"},
                    "title": {"type": "string"},
                    "email": {"type": "string"}
                },
                "required": ["first_name", "last_name"]
            }
        }
    },
    "required": ["name"]
}


def llm_extract_companies(text: str, custom_instruction: str = "") -> tuple[list[dict], dict]:
    """Single-pass extraction: pull all company data from directory text.

    Returns (list_of_company_dicts, llm_usage_dict).
    """
    llm = LLMClient()
    schema = {
        "type": "object",
        "properties": {"companies": {"type": "array", "items": COMPANY_SCHEMA}},
        "required": ["companies"]
    }

    prompt = (
        "Extract all companies from the following text. "
        "For each company, extract the company's own website URL and LinkedIn URL if available. "
        "Also extract industry, founded year, founders (with their names, titles, LinkedIn URLs, and emails), "
        "and any program or cohort tags mentioned.\n"
    )
    if custom_instruction:
        prompt += f"INSTRUCTIONS: {custom_instruction}\n"
    prompt += f"\nTEXT:\n{text[:50000]}"

    parsed, usage = llm.structured_output(
        prompt=prompt,
        json_schema=schema,
        system_prompt="You are a data extraction assistant. Follow the user's instructions and schema strictly.",
        model="google/gemini-2.5-flash"
    )

    companies = _normalize_parsed_companies(parsed)
    return companies, usage.model_dump()


def llm_extract_directory(text: str, custom_instruction: str = "") -> tuple[list[dict], dict]:
    """Deep scrape Pass 1: extract company names and their profile link paths.

    Returns (list_of_stub_dicts, llm_usage_dict).
    Each stub has at minimum: name, profile_path, description.
    """
    llm = LLMClient()

    prompt = (
        "Extract every company listed on this directory page. "
        "For each company, extract its name and the relative URL path to its detail/profile page "
        "(e.g. '/companies/acme'). Also extract a brief description and any program tags if visible.\n"
    )
    if custom_instruction:
        prompt += f"INSTRUCTIONS: {custom_instruction}\n"
    prompt += f"\nTEXT:\n{text[:50000]}"

    parsed, usage = llm.structured_output(
        prompt=prompt,
        json_schema=DIRECTORY_LINK_SCHEMA,
        system_prompt="You are a data extraction assistant. Follow the user's instructions and schema strictly.",
        model="google/gemini-2.5-flash"
    )

    companies = _normalize_parsed_companies(parsed)
    return companies, usage.model_dump()


def llm_extract_profile(text: str, company_name: str = "") -> tuple[dict, dict]:
    """Deep scrape Pass 2: extract rich firmographics from a single company profile page.

    Returns (company_detail_dict, llm_usage_dict).
    """
    llm = LLMClient()

    prompt = (
        f"Extract detailed company information for '{company_name}' from this profile page. "
        "Include the company's own website URL, industry, founded year, LinkedIn URL, "
        "and all founders with their names, titles, LinkedIn URLs, and emails.\n"
        f"\nTEXT:\n{text[:30000]}"
    )

    parsed, usage = llm.structured_output(
        prompt=prompt,
        json_schema=PROFILE_SCHEMA,
        system_prompt="You are a data extraction assistant. Follow the user's instructions and schema strictly.",
        model="google/gemini-2.5-flash"
    )

    detail = parsed if isinstance(parsed, dict) else {}
    return detail, usage.model_dump()


def _normalize_parsed_companies(parsed) -> list[dict]:
    """Safely extract a list of company dicts from LLM output."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return parsed.get("companies", [])
    return []


# ==============================================================================
# 2. BUSINESS LOGIC LAYER  —  Database reconciliation, no LLM calls
# ==============================================================================

def reconcile_company(
    db: Session,
    company_data: dict,
    job: IngestionJob,
    source_name: str,
    source_url: str
) -> tuple[Organization, bool]:
    """Find or create an Organization + CompanyProfile from extracted data.

    Also creates a RawEntity mapping and an EntityUpdate audit log entry.

    Returns (organization, is_new).
    """
    name = company_data.get("name") or "Unknown"
    company_url = company_data.get("url", "")

    # --- Resolve domain for matching ---
    domain = ""
    if company_url:
        parsed_url = urlparse(company_url)
        domain = parsed_url.netloc.replace("www.", "")

    # --- Find existing org by domain, then by name ---
    org = None
    if domain:
        org = db.query(Organization).filter_by(primary_domain=domain).first()
    if not org:
        org = db.query(Organization).filter_by(normalized_name=name.lower()).first()

    # --- Upsert Organization ---
    is_new_org = False
    if org:
        if company_data.get("description") and org.description != company_data.get("description"):
            org.description = company_data["description"]
        if company_data.get("linkedin_url") and not org.linkedin_url:
            org.linkedin_url = company_data["linkedin_url"]
    else:
        is_new_org = True
        org = Organization(
            name=name,
            normalized_name=name.lower(),
            primary_domain=domain or None,
            website_url=company_url,
            description=company_data.get("description"),
            linkedin_url=company_data.get("linkedin_url"),
            organization_type="COMPANY"
        )
        db.add(org)

    db.flush()

    # --- Upsert CompanyProfile ---
    profile = db.query(CompanyProfile).filter_by(organization_id=org.id).first()
    if not profile:
        profile = CompanyProfile(organization_id=org.id)
        db.add(profile)
    if company_data.get("industry") and not profile.industry:
        profile.industry = company_data["industry"]
    if company_data.get("founded_year") and not profile.founded_year:
        profile.founded_year = company_data["founded_year"]

    # --- RawEntity audit record ---
    entity = RawEntity(
        ingestion_job_id=job.id,
        entity_type="ORGANIZATION",
        matched_organization_id=org.id,
        raw_name=name,
        normalized_name=name.lower(),
        source_url=source_url,
        raw_payload_json=company_data,
        detected_at=datetime.utcnow()
    )
    db.add(entity)
    db.flush()

    # --- EntityUpdate audit log ---
    db.add(EntityUpdate(
        organization_id=org.id,
        raw_entity_id=entity.id,
        ingestion_job_id=job.id,
        update_reason="AUTO_CREATE" if is_new_org else "SOURCE_PRIORITY",
        update_action="CREATE" if is_new_org else "UPDATE",
        source=source_name
    ))

    # --- Reconcile child entities ---
    reconcile_people(db, company_data.get("founders", []), org, entity, job, source_name)
    reconcile_program_tags(db, company_data.get("program_tags", []), org)

    # --- UI tracking ---
    company_data["__reconciliation_status__"] = "CREATED_ORG" if is_new_org else "UPDATED_ORG"
    company_data["__organization_id__"] = org.id

    return org, is_new_org


def reconcile_people(
    db: Session,
    founders: list[dict],
    org: Organization,
    raw_entity: RawEntity,
    job: IngestionJob,
    source_name: str
):
    """Upsert Person records, link them to the Organization, and save emails."""
    for f in founders:
        fname = f.get("first_name", "")
        lname = f.get("last_name", "")
        if not fname or not lname:
            continue

        full_name = f"{fname} {lname}"
        person = db.query(Person).filter_by(first_name=fname, last_name=lname).first()

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
                person.linkedin_url = f["linkedin_url"]

        db.flush()

        if is_new_person:
            db.add(EntityUpdate(
                person_id=person.id,
                raw_entity_id=raw_entity.id,
                ingestion_job_id=job.id,
                update_reason="AUTO_CREATE",
                update_action="CREATE",
                source=source_name
            ))

        # Link founder to company via role
        role = db.query(PersonOrganizationRole).filter_by(
            person_id=person.id, organization_id=org.id
        ).first()
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

        # Save email
        email_val = f.get("email")
        if email_val:
            existing = db.query(PersonEmail).filter_by(email=email_val).first()
            if not existing:
                db.add(PersonEmail(
                    person_id=person.id,
                    email=email_val,
                    organization_id=org.id,
                    is_primary=True
                ))


def reconcile_program_tags(db: Session, program_tags: list[str], org: Organization):
    """Link an Organization to matching ProgramProfiles via ProgramMembership."""
    for tag in program_tags:
        prog = db.query(ProgramProfile).filter_by(program_name=tag).first()
        if prog:
            membership = db.query(ProgramMembership).filter_by(
                company_organization_id=org.id, program_id=prog.id
            ).first()
            if not membership:
                db.add(ProgramMembership(
                    company_organization_id=org.id,
                    program_id=prog.id,
                    is_active=True
                ))


# ==============================================================================
# 3. ORCHESTRATOR  —  Ties extraction and business logic together
# ==============================================================================

def run_ingestion_config(db: Session, config_id: int, triggered_by: str = "MANUAL") -> IngestionJob:
    """Main entry point: runs a full ingestion pipeline for a given config.

    Steps:
        1. Load config and create a job record.
        2. Fetch raw data (Playwright for SCRAPE, requests for API).
        3. Save source content to the job for auditability.
        4. Extract entities via LLM (single-pass or deep scrape).
        5. Reconcile each extracted entity against the CRM database.
        6. Log results and mark job as complete.
    """
    # --- STEP 1: Load config, create job ---
    config = db.query(IngestionConfig).filter(IngestionConfig.id == config_id).first()
    if not config:
        raise ValueError(f"IngestionConfig {config_id} not found")

    ds = config.data_source
    meta = config.metadata_json or {}

    job = IngestionJob(
        ingestion_config_id=config.id,
        job_status="RUNNING",
        triggered_by=triggered_by,
        started_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # --- Construct URL ---
    base = (ds.base_url or "").rstrip("/")
    endpoint = (config.endpoint_url or "").lstrip("/")
    url = f"{base}/{endpoint}" if base else endpoint

    method = config.http_method or "GET"
    headers = config.headers_json or {}
    params = config.query_params_json or {}

    try:
        # --- STEP 2: Fetch raw data ---
        response = requests.request(method, url, headers=headers, params=params, timeout=30)
        response.raise_for_status()

        records_created = 0

        if config.ingestion_type == "SCRAPE":
            records_created, job_logs = _run_scrape_pipeline(db, job, config, url, meta)
            job.job_logs_json = job_logs

        elif config.ingestion_type == "API":
            records_created, job_logs = _run_api_pipeline(db, job, url, response)
            job.job_logs_json = job_logs

        # --- STEP 6: Complete job ---
        job.job_status = "SUCCESS"
        job.completed_at = datetime.utcnow()
        job.records_processed = records_created
        job.records_created = records_created
        db.commit()

    except Exception as e:
        logger.error(f"Ingestion job failed: {e}")
        db.rollback()

        failed_job = db.query(IngestionJob).filter(IngestionJob.id == job.id).first()
        if failed_job:
            failed_job.job_status = "FAILED"
            failed_job.error_message = str(e)
            failed_job.completed_at = datetime.utcnow()
            db.commit()

    return job


# --- Orchestrator sub-pipelines ------------------------------------------------

def _run_scrape_pipeline(
    db: Session,
    job: IngestionJob,
    config: IngestionConfig,
    url: str,
    meta: dict
) -> tuple[int, dict]:
    """SCRAPE pipeline: fetch page, extract with LLM, reconcile.

    Supports two modes:
        - Standard (single-pass): extract all company data from the directory page in one LLM call.
        - Deep scrape (two-pass): extract profile links first, then visit each for rich firmographics.
    """
    ds = config.data_source
    custom_instruction = meta.get("llm_instruction", "")
    is_deep_scrape = meta.get("is_deep_scrape", False)

    # --- STEP 2: Fetch and save source ---
    logger.info(f"[Scrape] Fetching page: {url}")
    text_content = fetch_page_text(url)

    job.source_content = text_content
    db.commit()

    all_usage = []

    if is_deep_scrape:
        # --- DEEP SCRAPE (TWO-PASS) ---
        logger.info("[Scrape] Deep scrape mode enabled — Pass 1: extracting directory links…")

        # Pass 1: Extract company stubs + profile URLs from directory
        stubs, usage_p1 = llm_extract_directory(text_content, custom_instruction)
        all_usage.append({"pass": "directory", **usage_p1})
        logger.info(f"[Scrape] Pass 1 found {len(stubs)} companies with profile links.")

        # Pass 2: Visit each profile page for rich firmographics
        companies = []
        for i, stub in enumerate(stubs):
            if not isinstance(stub, dict):
                continue

            profile_path = stub.get("profile_path", "")
            company_name = stub.get("name", "Unknown")

            # Start with the directory-level data
            company = {
                "name": company_name,
                "description": stub.get("description", ""),
                "program_tags": stub.get("program_tags", []),
            }

            if profile_path:
                profile_url = urljoin(url, profile_path)
                logger.info(f"[Scrape] Pass 2 ({i+1}/{len(stubs)}): visiting {profile_url}")
                try:
                    profile_text = fetch_page_text(profile_url)
                    detail, usage_p2 = llm_extract_profile(profile_text, company_name)
                    all_usage.append({"pass": f"profile_{company_name}", **usage_p2})

                    # Merge detail into stub (detail wins for non-empty values)
                    for key in ["url", "description", "industry", "founded_year", "linkedin_url", "founders"]:
                        if detail.get(key):
                            company[key] = detail[key]
                except Exception as e:
                    logger.warning(f"[Scrape] Failed to deep-scrape {profile_url}: {e}")

            companies.append(company)

    else:
        # --- STANDARD SINGLE-PASS ---
        logger.info("[Scrape] Standard single-pass extraction…")
        companies, usage = llm_extract_companies(text_content, custom_instruction)
        all_usage.append(usage)

    # --- STEP 5: Reconcile all extracted companies ---
    records_created = 0
    for c in companies:
        if not isinstance(c, dict):
            continue
        org, _ = reconcile_company(db, c, job, ds.source_name, url)
        records_created += 1
        
        # --- STEP 5b: Auto-link to a configured Program ---
        link_program_id = meta.get("link_program_id")
        if link_program_id:
            existing = db.query(ProgramMembership).filter_by(
                company_organization_id=org.id, program_id=link_program_id
            ).first()
            if not existing:
                db.add(ProgramMembership(
                    company_organization_id=org.id,
                    program_id=link_program_id,
                    is_active=True
                ))

    job_logs = {"llm_usage": all_usage, "extracted_companies": companies}
    return records_created, job_logs


def _run_api_pipeline(
    db: Session,
    job: IngestionJob,
    url: str,
    response: requests.Response
) -> tuple[int, dict]:
    """API pipeline: parse JSON response, save each item as a RawEntity."""
    data = response.json()

    job.source_content = json.dumps(data)
    db.commit()

    records_created = 0
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
        job_logs = {"status": "success", "data": data[:10]}
    else:
        job_logs = {"status": "success", "data": data}

    return records_created, job_logs
