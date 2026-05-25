"""
Reconciler — Business Logic / CRM Reconciliation
==================================================
Maps extracted entities to CRM records (Organization, Person, etc.).
All CRM changes are logged to CanonicalMutation.
No LLM calls — that's the extractor's job.
"""

import logging
from datetime import datetime
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from market_comps.db.models import (
    Organization, CompanyProfile, InvestorProfile, Person, PersonOrganizationRole,
    PersonEmail, ProgramMembership, ProgramCohort,
    ExtractedEntity, ExtractedRelationship, ExtractionJob,
    IngestionRun, EntityMatch, EntityAttributeEvidence, CanonicalMutation
)

logger = logging.getLogger(__name__)


# ==============================================================================
# MUTATION HELPER
# ==============================================================================

def log_mutation(
    db: Session,
    entity_type: str,
    entity_id,
    action: str,
    field_name: str = None,
    old_value: str = None,
    new_value: str = None,
    source: str = None,
    extraction_job_id: int = None,
    created_by: str = None,
):
    """Write a single mutation entry."""
    db.add(CanonicalMutation(
        canonical_entity_type=entity_type,
        canonical_entity_id=str(entity_id),
        mutation_type=action,
        field_name=field_name,
        old_value=old_value,
        new_value=new_value,
        source=source,
        extraction_job_id=extraction_job_id,
        created_by=created_by or "SYSTEM",
    ))


# ==============================================================================
# ENTITY RECONCILIATION
# ==============================================================================

def reconcile_organization(
    db: Session,
    extracted_entity: ExtractedEntity,
    run: IngestionRun,
    org_type: str = "COMPANY",
) -> Organization:
    """Find or create an Organization + profile from an extracted entity.

    org_type determines the organization_type and which profile to create:
        COMPANY → CompanyProfile
        INVESTOR → InvestorProfile

    Returns the Organization.
    """
    payload = extracted_entity.extracted_payload_json or {}
    name = payload.get("name") or extracted_entity.raw_name or "Unknown"
    job_id = extracted_entity.extraction_job_id
    
    # Resolve company URL (support multiple common keys)
    company_url = payload.get("url") or payload.get("website") or payload.get("company_website") or ""
    linkedin_url = payload.get("linkedin_url") or payload.get("linkedin") or ""

    # Resolve domain for matching
    domain = ""
    if company_url:
        parsed_url = urlparse(company_url)
        domain = parsed_url.netloc.replace("www.", "")

    # Find existing org
    org = None
    if domain:
        org = db.query(Organization).filter_by(primary_domain=domain).first()
    if not org:
        org = db.query(Organization).filter_by(normalized_name=name.lower()).first()

    # Upsert
    is_new = False
    if org:
        # Update fields if we have better data
        if payload.get("description") and org.description != payload["description"]:
            log_mutation(db, "ORGANIZATION", org.id, "UPDATE",
                      field_name="description", old_value=org.description, new_value=payload["description"],
                      source="PIPELINE_FILL", extraction_job_id=job_id)
            org.description = payload["description"]
        if linkedin_url and not org.linkedin_url:
            log_mutation(db, "ORGANIZATION", org.id, "UPDATE",
                      field_name="linkedin_url", new_value=linkedin_url,
                      source="PIPELINE_FILL", extraction_job_id=job_id)
            org.linkedin_url = linkedin_url
        if company_url and not org.website_url:
            log_mutation(db, "ORGANIZATION", org.id, "UPDATE",
                      field_name="website_url", new_value=company_url,
                      source="PIPELINE_FILL", extraction_job_id=job_id)
            org.website_url = company_url
    else:
        is_new = True
        org = Organization(
            name=name,
            normalized_name=name.lower(),
            primary_domain=domain or None,
            website_url=company_url,
            description=payload.get("description"),
            linkedin_url=linkedin_url,
            organization_type=org_type
        )
        db.add(org)
        db.flush()
        log_mutation(db, "ORGANIZATION", org.id, "CREATE",
                  source="PIPELINE_AUTO_CREATE", extraction_job_id=job_id)

    db.flush()

    # Upsert profile based on org_type
    if org_type == "INVESTOR":
        inv_profile = db.query(InvestorProfile).filter_by(organization_id=org.id).first()
        if not inv_profile:
            inv_profile = InvestorProfile(
                organization_id=org.id,
                investor_type=payload.get("investor_type"),
                preferred_stage=payload.get("preferred_stage")
            )
            db.add(inv_profile)
            db.flush()
            log_mutation(db, "INVESTOR_PROFILE", str(inv_profile.id), "CREATE", source="PIPELINE_AUTO_CREATE", extraction_job_id=job_id)
    else:
        profile = db.query(CompanyProfile).filter_by(organization_id=org.id).first()
        if not profile:
            profile = CompanyProfile(organization_id=org.id)
            db.add(profile)
            db.flush()
            log_mutation(db, "COMPANY_PROFILE", str(profile.id), "CREATE", source="PIPELINE_AUTO_CREATE", extraction_job_id=job_id)
        if payload.get("industry") and not profile.industry:
            profile.industry = payload["industry"]
            log_mutation(db, "COMPANY_PROFILE", str(profile.id), "UPDATE", field_name="industry", new_value=payload["industry"], source="PIPELINE_FILL", extraction_job_id=job_id)
        if payload.get("founded_year") and not profile.founded_year:
            profile.founded_year = payload["founded_year"]
            log_mutation(db, "COMPANY_PROFILE", str(profile.id), "UPDATE", field_name="founded_year", new_value=str(payload["founded_year"]), source="PIPELINE_FILL", extraction_job_id=job_id)

    # Auto-parse founders if present in the payload
    founders_data = payload.get("founders")
    if founders_data:
        # Fallback for old comma-separated string format
        if isinstance(founders_data, str):
            founder_names = [f.strip() for f in founders_data.split(",") if f.strip()]
            founders_list = [{"first_name": fn.split(" ", 1)[0], "last_name": fn.split(" ", 1)[1] if " " in fn else "", "title": "Founder"} for fn in founder_names]
        elif isinstance(founders_data, list):
            founders_list = founders_data
        else:
            founders_list = []
            
        for f_dict in founders_list:
            if not isinstance(f_dict, dict):
                continue
                
            fname = f_dict.get("first_name", "")
            lname = f_dict.get("last_name", "")
            if not fname and not lname:
                continue
                
            full_name = f"{fname} {lname}".strip()
            
            person = db.query(Person).filter_by(first_name=fname, last_name=lname).first()
            if not person:
                person = Person(
                    first_name=fname,
                    last_name=lname,
                    full_name=full_name
                )
                db.add(person)
                db.flush()
                log_mutation(db, "PERSON", str(person.id), "CREATE", source="PIPELINE_AUTO_FOUNDER", extraction_job_id=job_id)
            
            # Save Email if present
            email_val = f_dict.get("email")
            if email_val:
                existing_email = db.query(PersonEmail).filter_by(email=email_val).first()
                if not existing_email:
                    db.add(PersonEmail(
                        person_id=person.id,
                        email=email_val,
                        organization_id=org.id,
                        is_primary=True
                    ))
                    log_mutation(db, "PERSON_EMAIL", str(person.id), "CREATE", field_name="email", new_value=email_val, source="PIPELINE_AUTO_FOUNDER", extraction_job_id=job_id)
            
            # Link to org
            title = f_dict.get("title") or "Founder"
            existing_role = db.query(PersonOrganizationRole).filter_by(
                person_id=person.id, organization_id=org.id, title=title
            ).first()
            
            if not existing_role:
                # Handle start/end dates
                start_year = f_dict.get("start_year")
                end_year = f_dict.get("end_year")
                
                from datetime import datetime
                s_date = datetime(start_year, 1, 1) if isinstance(start_year, int) else None
                e_date = datetime(end_year, 12, 31) if isinstance(end_year, int) else None
                is_current = True if not e_date else False
                
                role = PersonOrganizationRole(
                    person_id=person.id,
                    organization_id=org.id,
                    title=title,
                    start_date=s_date,
                    end_date=e_date,
                    is_current=is_current
                )
                db.add(role)
                db.flush()
                log_mutation(db, "PERSON_ROLE", str(role.id), "CREATE", source="PIPELINE_AUTO_FOUNDER", extraction_job_id=job_id)

    # Link extracted entity back using EntityMatch
    db.add(EntityMatch(
        extracted_entity_id=extracted_entity.id,
        canonical_entity_type="Organization",
        canonical_entity_id=str(org.id),
        match_confidence=1.0,
        match_method="PIPELINE_AUTO",
        created_by="SYSTEM"
    ))

    return org


def reconcile_person(
    db: Session,
    extracted_entity: ExtractedEntity,
    run: IngestionRun,
) -> Person:
    """Find or create a Person from an extracted entity.

    Returns the Person.
    """
    payload = extracted_entity.extracted_payload_json or {}
    fname = payload.get("first_name", "")
    lname = payload.get("last_name", "")
    full_name = f"{fname} {lname}".strip()
    job_id = extracted_entity.extraction_job_id

    person = db.query(Person).filter_by(first_name=fname, last_name=lname).first()

    if not person:
        person = Person(
            first_name=fname,
            last_name=lname,
            full_name=full_name,
            linkedin_url=payload.get("linkedin_url")
        )
        db.add(person)
        db.flush()
        log_mutation(db, "PERSON", str(person.id), "CREATE",
                  source="PIPELINE_AUTO_CREATE", extraction_job_id=job_id)
    else:
        if payload.get("linkedin_url") and not person.linkedin_url:
            log_mutation(db, "PERSON", str(person.id), "UPDATE",
                      field_name="linkedin_url", new_value=payload["linkedin_url"],
                      source="PIPELINE_FILL", extraction_job_id=job_id)
            person.linkedin_url = payload["linkedin_url"]

    # Save email
    email_val = payload.get("email")
    if email_val:
        existing = db.query(PersonEmail).filter_by(email=email_val).first()
        if not existing:
            db.add(PersonEmail(
                person_id=person.id,
                email=email_val,
                organization_id=None,
                is_primary=True
            ))

    # Link extracted entity back using EntityMatch
    db.add(EntityMatch(
        extracted_entity_id=extracted_entity.id,
        canonical_entity_type="Person",
        canonical_entity_id=str(person.id),
        match_confidence=1.0,
        match_method="PIPELINE_AUTO",
        created_by="SYSTEM"
    ))

    return person


# ==============================================================================
# RELATIONSHIP RECONCILIATION
# ==============================================================================

def reconcile_relationship(
    db: Session,
    extracted_rel: ExtractedRelationship,
    run: IngestionRun,
):
    """Reconcile an extracted relationship into the CRM.

    Handles:
        FOUNDER_OF → PersonOrganizationRole (person → org)
        MEMBER_OF_COHORT → ProgramMembership (org → cohort)
    """
    rel_type = extracted_rel.relationship_type
    payload = extracted_rel.relationship_payload_json or {}

    if rel_type == "FOUNDER_OF":
        _reconcile_founder_of(db, extracted_rel, run, payload)
    elif rel_type == "MEMBER_OF_COHORT":
        _reconcile_member_of_cohort(db, extracted_rel, run, payload)
    else:
        logger.warning(f"[Reconciler] Unhandled relationship type: {rel_type}")


def _reconcile_founder_of(db, extracted_rel, run, payload):
    """Create a PersonOrganizationRole linking person to org."""
    source = extracted_rel.source_extracted_entity
    target = extracted_rel.target_extracted_entity

    if not source or not target:
        return
        
    source_match = db.query(EntityMatch).filter_by(extracted_entity_id=source.id, canonical_entity_type="Person").first()
    target_match = db.query(EntityMatch).filter_by(extracted_entity_id=target.id, canonical_entity_type="Organization").first()

    if not source_match or not target_match:
        return

    person_id = source_match.canonical_entity_id
    org_id = target_match.canonical_entity_id

    existing = db.query(PersonOrganizationRole).filter_by(
        person_id=person_id, organization_id=org_id
    ).first()

    title = payload.get("title", "Founder")
    if not existing:
        role = PersonOrganizationRole(
            person_id=person_id,
            organization_id=org_id,
            title=title,
            is_current=True
        )
        db.add(role)
        log_mutation(db, "PERSON_ORGANIZATION_ROLE", f"{person_id}_{org_id}", "CREATE",
                  source="PIPELINE_AUTO_CREATE", extraction_job_id=extracted_rel.extraction_job_id)
    else:
        existing.title = title


def _reconcile_member_of_cohort(db, extracted_rel, run, payload):
    """Create a ProgramMembership linking org to cohort."""
    org_id = extracted_rel.source_entity_id
    cohort_id = extracted_rel.target_entity_id

    if not org_id or not cohort_id:
        return

    org_id = int(org_id)
    cohort_id = int(cohort_id)

    existing = db.query(ProgramMembership).filter_by(
        company_organization_id=org_id, program_cohort_id=cohort_id
    ).first()

    if not existing:
        db.add(ProgramMembership(
            company_organization_id=org_id,
            program_cohort_id=cohort_id,
            is_active=True
        ))
        log_mutation(db, "PROGRAM_MEMBERSHIP", f"{org_id}_{cohort_id}", "CREATE",
                  source="PIPELINE_AUTO_CREATE", extraction_job_id=extracted_rel.extraction_job_id)


# ==============================================================================
# BULK RECONCILIATION
# ==============================================================================

def reconcile_all(db: Session, run: IngestionRun, pipeline) -> dict:
    """Reconcile all extracted entities and relationships for a pipeline run.

    Also auto-links organizations to the pipeline's cohort if configured.

    Returns stats dict.
    """
    orgs_created = 0
    orgs_updated = 0
    people_created = 0

    # Determine org_type from pipeline type
    pipeline_type = pipeline.pipeline_type if pipeline else ""
    org_type = "INVESTOR" if pipeline_type == "INVESTOR_PORTFOLIO_PAGE" else "COMPANY"

    # Step 1: Reconcile entities
    entities = db.query(ExtractedEntity).join(ExtractionJob).filter(ExtractionJob.ingestion_run_id == run.id).all()
    logger.info(f"[Reconciler] Found {len(entities)} entities to reconcile for run {run.id}")
    
    for entity in entities:
        if entity.entity_type == "ORGANIZATION":
            org = reconcile_organization(db, entity, run, org_type=org_type)
            if org:
                orgs_created += 1
        elif entity.entity_type == "PERSON":
            logger.info(f"[Reconciler] Reconciling person: {entity.raw_name}")
            person = reconcile_person(db, entity, run)
            if person:
                people_created += 1

    db.flush()

    # Step 2: Reconcile relationships
    relationships = db.query(ExtractedRelationship).join(ExtractionJob).filter(ExtractionJob.ingestion_run_id == run.id).all()
    logger.info(f"[Reconciler] Found {len(relationships)} relationships to reconcile for run {run.id}")
    for rel in relationships:
        reconcile_relationship(db, rel, run)

    db.flush()

    # Step 3: Auto-link to pipeline's cohort
    cohort_id = pipeline.program_cohort_id if pipeline else None
    if cohort_id:
        org_entities = [e for e in entities if e.entity_type == "ORGANIZATION"]
        for entity in org_entities:
            match = db.query(EntityMatch).filter_by(extracted_entity_id=entity.id, canonical_entity_type="Organization").first()
            if not match:
                continue
            org_id = match.canonical_entity_id
            existing = db.query(ProgramMembership).filter_by(
                company_organization_id=org_id, program_cohort_id=cohort_id
            ).first()
            if not existing:
                db.add(ProgramMembership(
                    company_organization_id=org_id,
                    program_cohort_id=cohort_id,
                    is_active=True
                ))
                log_mutation(db, "PROGRAM_MEMBERSHIP", f"{org_id}_{cohort_id}", "CREATE",
                          source="PIPELINE_COHORT_LINK", extraction_job_id=entity.extraction_job_id)

    return {
        "orgs_reconciled": orgs_created,
        "people_reconciled": people_created,
        "relationships_reconciled": len(relationships),
    }
