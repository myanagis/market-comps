"""
Reconciler — Business Logic / CRM Reconciliation
==================================================
Maps extracted entities to CRM records (Organization, Person, etc.).
All CRM changes are logged to entity_audit_trail.
No LLM calls — that's the extractor's job.
"""

import logging
from datetime import datetime
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from market_comps.db.models import (
    Organization, CompanyProfile, InvestorProfile, Person, PersonOrganizationRole,
    PersonEmail, ProgramMembership, ProgramCohort,
    ExtractedEntity, ExtractedRelationship,
    PipelineRun, EntityAuditTrail
)

logger = logging.getLogger(__name__)


# ==============================================================================
# AUDIT TRAIL HELPER
# ==============================================================================

def log_audit(
    db: Session,
    entity_type: str,
    entity_id,
    action: str,
    field_name: str = None,
    old_value: str = None,
    new_value: str = None,
    pipeline_run_id: int = None,
    extracted_entity_id: int = None,
    extracted_relationship_id: int = None,
    changed_by_user_id: str = None,
    reason: str = None,
    metadata_json: dict = None,
):
    """Write a single audit trail entry."""
    db.add(EntityAuditTrail(
        entity_type=entity_type,
        entity_id=str(entity_id),
        audit_action=action,
        field_name=field_name,
        old_value=old_value,
        new_value=new_value,
        pipeline_run_id=pipeline_run_id,
        extracted_entity_id=extracted_entity_id,
        extracted_relationship_id=extracted_relationship_id,
        changed_by_user_id=changed_by_user_id,
        reason=reason,
        metadata_json=metadata_json,
        created_at=datetime.utcnow()
    ))


# ==============================================================================
# ENTITY RECONCILIATION
# ==============================================================================

def reconcile_organization(
    db: Session,
    extracted_entity: ExtractedEntity,
    run: PipelineRun,
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
    company_url = payload.get("url", "")

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
            log_audit(db, "ORGANIZATION", org.id, "UPDATE",
                      field_name="description", old_value=org.description, new_value=payload["description"],
                      pipeline_run_id=run.id, extracted_entity_id=extracted_entity.id, reason="PIPELINE_FILL")
            org.description = payload["description"]
        if payload.get("linkedin_url") and not org.linkedin_url:
            log_audit(db, "ORGANIZATION", org.id, "UPDATE",
                      field_name="linkedin_url", new_value=payload["linkedin_url"],
                      pipeline_run_id=run.id, extracted_entity_id=extracted_entity.id, reason="PIPELINE_FILL")
            org.linkedin_url = payload["linkedin_url"]
        if payload.get("url") and not org.website_url:
            log_audit(db, "ORGANIZATION", org.id, "UPDATE",
                      field_name="website_url", new_value=payload["url"],
                      pipeline_run_id=run.id, extracted_entity_id=extracted_entity.id, reason="PIPELINE_FILL")
            org.website_url = payload["url"]
    else:
        is_new = True
        org = Organization(
            name=name,
            normalized_name=name.lower(),
            primary_domain=domain or None,
            website_url=company_url,
            description=payload.get("description"),
            linkedin_url=payload.get("linkedin_url"),
            organization_type=org_type
        )
        db.add(org)
        db.flush()
        log_audit(db, "ORGANIZATION", org.id, "CREATE",
                  pipeline_run_id=run.id, extracted_entity_id=extracted_entity.id, reason="PIPELINE_AUTO_CREATE")

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

    # Link extracted entity back to the matched org
    extracted_entity.matched_organization_id = org.id

    return org


def reconcile_person(
    db: Session,
    extracted_entity: ExtractedEntity,
    run: PipelineRun,
) -> Person:
    """Find or create a Person from an extracted entity.

    Returns the Person.
    """
    payload = extracted_entity.extracted_payload_json or {}
    fname = payload.get("first_name", "")
    lname = payload.get("last_name", "")
    full_name = f"{fname} {lname}".strip()

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
        log_audit(db, "PERSON", str(person.id), "CREATE",
                  pipeline_run_id=run.id, extracted_entity_id=extracted_entity.id, reason="PIPELINE_AUTO_CREATE")
    else:
        if payload.get("linkedin_url") and not person.linkedin_url:
            log_audit(db, "PERSON", str(person.id), "UPDATE",
                      field_name="linkedin_url", new_value=payload["linkedin_url"],
                      pipeline_run_id=run.id, extracted_entity_id=extracted_entity.id, reason="PIPELINE_FILL")
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

    # Link extracted entity back
    extracted_entity.matched_person_id = person.id

    return person


# ==============================================================================
# RELATIONSHIP RECONCILIATION
# ==============================================================================

def reconcile_relationship(
    db: Session,
    extracted_rel: ExtractedRelationship,
    run: PipelineRun,
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
    if not source.matched_person_id or not target.matched_organization_id:
        return

    person_id = source.matched_person_id
    org_id = target.matched_organization_id

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
        log_audit(db, "PERSON_ORGANIZATION_ROLE", f"{person_id}_{org_id}", "CREATE",
                  pipeline_run_id=run.id, extracted_relationship_id=extracted_rel.id,
                  reason="PIPELINE_AUTO_CREATE")
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
        log_audit(db, "PROGRAM_MEMBERSHIP", f"{org_id}_{cohort_id}", "CREATE",
                  pipeline_run_id=run.id, extracted_relationship_id=extracted_rel.id,
                  reason="PIPELINE_AUTO_CREATE")


# ==============================================================================
# BULK RECONCILIATION
# ==============================================================================

def reconcile_all(db: Session, run: PipelineRun, pipeline) -> dict:
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
    entities = db.query(ExtractedEntity).filter_by(pipeline_run_id=run.id).all()
    for entity in entities:
        if entity.entity_type == "ORGANIZATION":
            org = reconcile_organization(db, entity, run, org_type=org_type)
            if org:
                orgs_created += 1
        elif entity.entity_type == "PERSON":
            person = reconcile_person(db, entity, run)
            if person:
                people_created += 1

    db.flush()

    # Step 2: Reconcile relationships
    relationships = db.query(ExtractedRelationship).filter_by(pipeline_run_id=run.id).all()
    for rel in relationships:
        reconcile_relationship(db, rel, run)

    db.flush()

    # Step 3: Auto-link to pipeline's cohort
    cohort_id = pipeline.program_cohort_id
    if cohort_id:
        org_entities = [e for e in entities if e.entity_type == "ORGANIZATION" and e.matched_organization_id]
        for entity in org_entities:
            org_id = entity.matched_organization_id
            existing = db.query(ProgramMembership).filter_by(
                company_organization_id=org_id, program_cohort_id=cohort_id
            ).first()
            if not existing:
                db.add(ProgramMembership(
                    company_organization_id=org_id,
                    program_cohort_id=cohort_id,
                    is_active=True
                ))
                log_audit(db, "PROGRAM_MEMBERSHIP", f"{org_id}_{cohort_id}", "CREATE",
                          pipeline_run_id=run.id, extracted_entity_id=entity.id,
                          reason="PIPELINE_COHORT_LINK")

    return {
        "orgs_reconciled": orgs_created,
        "people_reconciled": people_created,
        "relationships_reconciled": len(relationships),
    }
