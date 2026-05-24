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
    else:
        profile = db.query(CompanyProfile).filter_by(organization_id=org.id).first()
        if not profile:
            profile = CompanyProfile(organization_id=org.id)
            db.add(profile)
        if payload.get("industry") and not profile.industry:
            profile.industry = payload["industry"]
        if payload.get("founded_year") and not profile.founded_year:
            profile.founded_year = payload["founded_year"]

    # Auto-parse founders if present in the payload (comma-separated string)
    founders_str = payload.get("founders")
    if founders_str and isinstance(founders_str, str):
        founder_names = [f.strip() for f in founders_str.split(",") if f.strip()]
        for fn in founder_names:
            parts = fn.split(" ", 1)
            fname = parts[0]
            lname = parts[1] if len(parts) > 1 else ""
            
            person = db.query(Person).filter_by(first_name=fname, last_name=lname).first()
            if not person:
                person = Person(
                    first_name=fname,
                    last_name=lname,
                    full_name=fn
                )
                db.add(person)
                db.flush()
                log_mutation(db, "PERSON", str(person.id), "CREATE", source="PIPELINE_AUTO_FOUNDER", extraction_job_id=job_id)
            
            # Link to org
            existing_role = db.query(PersonOrganizationRole).filter_by(
                person_id=person.id, organization_id=org.id, title="Founder"
            ).first()
            
            if not existing_role:
                role = PersonOrganizationRole(
                    person_id=person.id,
                    organization_id=org.id,
                    title="Founder",
                    start_date=None,
                    is_current=True
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
