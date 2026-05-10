import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey, JSON, Float, Text
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()

class TimestampMixin:
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


# ==============================================================================
# CORE CRM MODELS
# ==============================================================================

class Organization(Base, TimestampMixin):
    __tablename__ = 'organizations'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    normalized_name = Column(String, index=True)
    website_url = Column(String)
    primary_domain = Column(String, unique=True, index=True)
    linkedin_url = Column(String)
    city = Column(String)
    state = Column(String)
    country = Column(String)
    organization_type = Column(String)
    description = Column(String)
    status = Column(String)
    is_active = Column(Boolean, default=True)

    company_profile = relationship("CompanyProfile", back_populates="organization", uselist=False)
    investor_profile = relationship("InvestorProfile", back_populates="organization", uselist=False)
    fund_profiles = relationship("FundProfile", back_populates="parent_organization")
    program_profiles = relationship("ProgramProfile", back_populates="parent_organization")
    roles = relationship("PersonOrganizationRole", back_populates="organization")
    program_memberships = relationship("ProgramMembership", back_populates="company")
    pipelines = relationship("Pipeline", back_populates="organization")


class CompanyProfile(Base, TimestampMixin):
    __tablename__ = 'company_profiles'

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False, unique=True)
    founded_year = Column(Integer)
    industry = Column(String)
    subindustry = Column(String)
    company_stage = Column(String)

    organization = relationship("Organization", back_populates="company_profile")


class InvestorProfile(Base, TimestampMixin):
    __tablename__ = 'investor_profiles'

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False, unique=True)
    investor_type = Column(String)
    preferred_stage = Column(String)

    organization = relationship("Organization", back_populates="investor_profile")


class FundProfile(Base, TimestampMixin):
    __tablename__ = 'fund_profiles'

    id = Column(Integer, primary_key=True, index=True)
    parent_organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)
    fund_name = Column(String, nullable=False)
    fund_type = Column(String)
    vintage_year = Column(Integer)
    fund_size = Column(String)
    status = Column(String)
    description = Column(String)

    parent_organization = relationship("Organization", back_populates="fund_profiles")


class ProgramProfile(Base, TimestampMixin):
    __tablename__ = 'program_profiles'

    id = Column(Integer, primary_key=True, index=True)
    parent_organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)
    program_name = Column(String, nullable=False)
    program_type = Column(String) # ACCELERATOR, GRANT, INCUBATOR, ...
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    status = Column(String)
    description = Column(String)

    parent_organization = relationship("Organization", back_populates="program_profiles")
    cohorts = relationship("ProgramCohort", back_populates="program")


class ProgramCohort(Base, TimestampMixin):
    __tablename__ = 'program_cohorts'

    id = Column(Integer, primary_key=True, index=True)
    program_id = Column(Integer, ForeignKey('program_profiles.id'), nullable=False)
    cohort_name = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    description = Column(String, nullable=True)

    program = relationship("ProgramProfile", back_populates="cohorts")
    memberships = relationship("ProgramMembership", back_populates="cohort")


class ProgramMembership(Base, TimestampMixin):
    __tablename__ = 'program_memberships'

    id = Column(Integer, primary_key=True, index=True)
    company_organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)
    program_cohort_id = Column(Integer, ForeignKey('program_cohorts.id'), nullable=True)
    
    is_active = Column(Boolean, default=True)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    notes = Column(String)
    metadata_json = Column(JSON)

    company = relationship("Organization", back_populates="program_memberships")
    cohort = relationship("ProgramCohort", back_populates="memberships")


class Person(Base, TimestampMixin):
    __tablename__ = 'people'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    first_name = Column(String)
    last_name = Column(String)
    full_name = Column(String)
    linkedin_url = Column(String)
    twitter_url = Column(String)
    city = Column(String)
    state = Column(String)
    country = Column(String)
    bio = Column(String)

    emails = relationship("PersonEmail", back_populates="person")
    roles = relationship("PersonOrganizationRole", back_populates="person")


class PersonEmail(Base, TimestampMixin):
    __tablename__ = 'person_emails'

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(UUID(as_uuid=True), ForeignKey('people.id'), nullable=False)
    email = Column(String, unique=True, nullable=False)
    email_type = Column(String)
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=True)
    is_primary = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    email_status = Column(String)

    person = relationship("Person", back_populates="emails")
    organization = relationship("Organization")


class PersonOrganizationRole(Base, TimestampMixin):
    __tablename__ = 'person_organization_roles'

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(UUID(as_uuid=True), ForeignKey('people.id'), nullable=False)
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)
    title = Column(String)
    seniority_level = Column(String)
    role_type = Column(String)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    is_current = Column(Boolean, default=True)
    is_primary_role = Column(Boolean, default=False)
    source = Column(String)
    metadata_json = Column(JSON)

    person = relationship("Person", back_populates="roles")
    organization = relationship("Organization", back_populates="roles")


# ==============================================================================
# PIPELINE FRAMEWORK
# ==============================================================================

class Pipeline(Base, TimestampMixin):
    __tablename__ = 'pipelines'

    id = Column(Integer, primary_key=True, index=True)
    pipeline_name = Column(String, nullable=False)
    pipeline_type = Column(String, nullable=False)
    # Pipeline types:
    #   PROGRAM_COMPANY_PAGE — extracts companies, people/emails, creates program memberships + people-company links
    #   INVESTOR_PORTFOLIO_PAGE — extracts companies, creates investment relationships + people-company links
    #   API_COMPANY_SEARCH — extracts companies from an API response
    #   CSV_IMPORT — imports entities from a CSV file
    #   INVESTOR_PEOPLE_PAGE — extracts people/emails from an investor's team page

    source_url = Column(String)

    # Context links — what org/program/cohort/fund does this pipeline relate to?
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=True)
    program_id = Column(Integer, ForeignKey('program_profiles.id'), nullable=True)
    program_cohort_id = Column(Integer, ForeignKey('program_cohorts.id'), nullable=True)
    fund_id = Column(Integer, ForeignKey('fund_profiles.id'), nullable=True)

    schedule_type = Column(String) # MANUAL, DAILY, WEEKLY, MONTHLY
    next_run_at = Column(DateTime, nullable=True)
    last_run_at = Column(DateTime, nullable=True)
    last_success_at = Column(DateTime, nullable=True)

    is_active = Column(Boolean, default=True)
    owner_user_id = Column(String, nullable=True)

    config_json = Column(JSON) # deep_scrape, llm_instruction, headers, etc.

    organization = relationship("Organization", back_populates="pipelines")
    program = relationship("ProgramProfile")
    program_cohort = relationship("ProgramCohort")
    fund = relationship("FundProfile")
    runs = relationship("PipelineRun", back_populates="pipeline", order_by="PipelineRun.started_at.desc()")


class PipelineRun(Base):
    __tablename__ = 'pipeline_runs'

    id = Column(Integer, primary_key=True, index=True)
    pipeline_id = Column(Integer, ForeignKey('pipelines.id'), nullable=False)

    run_status = Column(String) # RUNNING, SUCCESS, FAILED

    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    records_processed = Column(Integer, default=0)
    records_created = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)

    error_message = Column(String)
    logs_json = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    pipeline = relationship("Pipeline", back_populates="runs")
    extracted_data = relationship("ExtractedDataRaw", back_populates="pipeline_run")
    extracted_entities = relationship("ExtractedEntity", back_populates="pipeline_run")
    extracted_relationships = relationship("ExtractedRelationship", back_populates="pipeline_run")


class ExtractedDataRaw(Base):
    __tablename__ = 'extracted_data_raw'

    id = Column(Integer, primary_key=True, index=True)
    pipeline_run_id = Column(Integer, ForeignKey('pipeline_runs.id'), nullable=False)

    data_type = Column(String) # PAGE_TEXT, API_JSON, CSV_ROW, PROFILE_TEXT
    source_url = Column(String)
    raw_content = Column(Text)
    content_hash = Column(String, index=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    pipeline_run = relationship("PipelineRun", back_populates="extracted_data")
    extracted_entities = relationship("ExtractedEntity", back_populates="extracted_data_raw")
    extracted_relationships = relationship("ExtractedRelationship", back_populates="extracted_data_raw")


class ExtractedEntity(Base):
    __tablename__ = 'extracted_entities'

    id = Column(Integer, primary_key=True, index=True)
    pipeline_run_id = Column(Integer, ForeignKey('pipeline_runs.id'), nullable=False)
    extracted_data_raw_id = Column(Integer, ForeignKey('extracted_data_raw.id'), nullable=True)

    entity_type = Column(String) # ORGANIZATION, PERSON, PROGRAM_COHORT
    raw_name = Column(String)
    normalized_name = Column(String, index=True)

    extracted_payload_json = Column(JSON)

    matched_organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=True, index=True)
    matched_person_id = Column(UUID(as_uuid=True), ForeignKey('people.id'), nullable=True, index=True)

    extraction_confidence = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    pipeline_run = relationship("PipelineRun", back_populates="extracted_entities")
    extracted_data_raw = relationship("ExtractedDataRaw", back_populates="extracted_entities")
    matched_organization = relationship("Organization")
    matched_person = relationship("Person")


class ExtractedRelationship(Base):
    __tablename__ = 'extracted_relationships'

    id = Column(Integer, primary_key=True, index=True)
    pipeline_run_id = Column(Integer, ForeignKey('pipeline_runs.id'), nullable=False)
    extracted_data_raw_id = Column(Integer, ForeignKey('extracted_data_raw.id'), nullable=True)

    relationship_type = Column(String) # FOUNDER_OF, EMPLOYEE_OF, MEMBER_OF_COHORT, INVESTED_IN

    source_extracted_entity_id = Column(Integer, ForeignKey('extracted_entities.id'), nullable=True)
    source_entity_type = Column(String, nullable=True) # ORGANIZATION, PERSON
    source_entity_id = Column(String, nullable=True) # Polymorphic — int or UUID as string

    target_extracted_entity_id = Column(Integer, ForeignKey('extracted_entities.id'), nullable=True)
    target_entity_type = Column(String, nullable=True)
    target_entity_id = Column(String, nullable=True)

    relationship_payload_json = Column(JSON)
    extraction_confidence = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    pipeline_run = relationship("PipelineRun", back_populates="extracted_relationships")
    extracted_data_raw = relationship("ExtractedDataRaw", back_populates="extracted_relationships")
    source_extracted_entity = relationship("ExtractedEntity", foreign_keys=[source_extracted_entity_id])
    target_extracted_entity = relationship("ExtractedEntity", foreign_keys=[target_extracted_entity_id])


# ==============================================================================
# AUDIT TRAIL
# ==============================================================================

class EntityAuditTrail(Base):
    __tablename__ = 'entity_audit_trail'

    id = Column(Integer, primary_key=True, index=True)

    entity_type = Column(String, nullable=False) # ORGANIZATION, PERSON, COMPANY_PROFILE, etc.
    entity_id = Column(String, nullable=False, index=True) # Polymorphic — int or UUID as string

    audit_action = Column(String, nullable=False) # CREATE, UPDATE, DELETE

    field_name = Column(String, nullable=True)
    old_value = Column(String, nullable=True)
    new_value = Column(String, nullable=True)

    pipeline_run_id = Column(Integer, ForeignKey('pipeline_runs.id'), nullable=True)
    extracted_entity_id = Column(Integer, ForeignKey('extracted_entities.id'), nullable=True)
    extracted_relationship_id = Column(Integer, ForeignKey('extracted_relationships.id'), nullable=True)

    changed_by_user_id = Column(String, nullable=True)
    reason = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    pipeline_run = relationship("PipelineRun")
    extracted_entity = relationship("ExtractedEntity")
    extracted_relationship = relationship("ExtractedRelationship")
