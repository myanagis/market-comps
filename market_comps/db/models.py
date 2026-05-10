import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()

class TimestampMixin:
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

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
    raw_entities = relationship("RawEntity", back_populates="organization")
    updates = relationship("EntityUpdate", back_populates="organization")
    program_memberships = relationship("ProgramMembership", back_populates="company")


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
    fund_size = Column(String) # E.g., "100M" or maybe store as numeric? The prompt didn't specify numeric, String is safe.
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
    raw_entities = relationship("RawEntity", back_populates="person")
    updates = relationship("EntityUpdate", back_populates="person")


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


class DataSource(Base, TimestampMixin):
    __tablename__ = 'data_sources'

    id = Column(Integer, primary_key=True, index=True)
    source_name = Column(String, nullable=False)
    source_type = Column(String)
    base_url = Column(String)
    description = Column(String)
    auth_type = Column(String)
    is_active = Column(Boolean, default=True)

    configs = relationship("IngestionConfig", back_populates="data_source")

class IngestionConfig(Base, TimestampMixin):
    __tablename__ = 'ingestion_configs'

    id = Column(Integer, primary_key=True, index=True)
    data_source_id = Column(Integer, ForeignKey('data_sources.id'), nullable=False)
    config_name = Column(String, nullable=False)
    ingestion_type = Column(String) # API, SCRAPE, IMPORT
    endpoint_url = Column(String)
    http_method = Column(String)
    query_params_json = Column(JSON)
    headers_json = Column(JSON)
    schedule_type = Column(String) # MANUAL, DAILY, WEEKLY, EVERY_OTHER_WEEK, MONTHLY
    next_run_at = Column(DateTime)
    last_run_at = Column(DateTime)
    last_success_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    owner_user_id = Column(String, nullable=True)
    metadata_json = Column(JSON)

    data_source = relationship("DataSource", back_populates="configs")
    jobs = relationship("IngestionJob", back_populates="config")

class IngestionJob(Base, TimestampMixin):
    __tablename__ = 'ingestion_jobs'

    id = Column(Integer, primary_key=True, index=True)
    ingestion_config_id = Column(Integer, ForeignKey('ingestion_configs.id'), nullable=False)
    job_status = Column(String)
    triggered_by = Column(String) # SCEDHULER, MANUAL, ETC.
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    records_processed = Column(Integer, default=0)
    records_created = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    error_message = Column(String)
    source_content = Column(String) # Raw scraped text/HTML content
    job_logs_json = Column(JSON)
    metadata_json = Column(JSON)

    config = relationship("IngestionConfig", back_populates="jobs")
    raw_entities = relationship("RawEntity", back_populates="job")
    entity_updates = relationship("EntityUpdate", back_populates="job")

class RawEntity(Base, TimestampMixin):
    __tablename__ = 'raw_entities'

    id = Column(Integer, primary_key=True, index=True)
    ingestion_job_id = Column(Integer, ForeignKey('ingestion_jobs.id'), nullable=False)
    entity_type = Column(String) # ORGANIZATION, PERSON
    
    matched_organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=True, index=True)
    matched_person_id = Column(UUID(as_uuid=True), ForeignKey('people.id'), nullable=True, index=True)
    
    raw_name = Column(String)
    normalized_name = Column(String, index=True)
    source_url = Column(String)
    source_identifier = Column(String)
    raw_payload_json = Column(JSON)
    detected_at = Column(DateTime)

    job = relationship("IngestionJob", back_populates="raw_entities")
    organization = relationship("Organization", back_populates="raw_entities")
    person = relationship("Person", back_populates="raw_entities")
    updates = relationship("EntityUpdate", back_populates="raw_entity")

class EntityUpdate(Base, TimestampMixin):
    __tablename__ = 'entity_updates'

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=True, index=True)
    person_id = Column(UUID(as_uuid=True), ForeignKey('people.id'), nullable=True, index=True)
    raw_entity_id = Column(Integer, ForeignKey('raw_entities.id'), nullable=False, index=True)
    ingestion_job_id = Column(Integer, ForeignKey('ingestion_jobs.id'), nullable=True)
    
    update_reason = Column(String) # FILL_EMPTY, SOURCE_PRIORITY, MANUAL_OVERRIDE, AUTO_CREATE
    field_name = Column(String)
    old_value = Column(String)
    new_value = Column(String)
    update_action = Column(String)
    update_status = Column(String)
    source = Column(String)

    organization = relationship("Organization", back_populates="updates")
    person = relationship("Person", back_populates="updates")
    raw_entity = relationship("RawEntity", back_populates="updates")
    job = relationship("IngestionJob", back_populates="entity_updates")
