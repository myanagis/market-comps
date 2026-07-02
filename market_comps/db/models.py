import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey, JSON, Float, Text
from sqlalchemy.orm import declarative_base, relationship, validates
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
    street1 = Column(String)
    street2 = Column(String)
    zip_code = Column(String)
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
    investments_made = relationship("Investment", foreign_keys="[Investment.investor_organization_id]", back_populates="investor")
    investments_received = relationship("Investment", foreign_keys="[Investment.company_organization_id]", back_populates="company")

    @validates('primary_domain')
    def validate_primary_domain(self, key, value):
        if not value:
            return value
            
        value = value.lower().strip()
        if "://" in value:
            value = value.split("://")[-1]
        if "/" in value:
            value = value.split("/")[0]
        if value.startswith("www."):
            value = value[4:]
            
        return value if value else None


class CompanyProfile(Base, TimestampMixin):
    __tablename__ = 'company_profiles'

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False, unique=True)
    founded_year = Column(Integer)
    industry = Column(String)
    subindustry = Column(String)
    company_stage = Column(String)
    themes = Column(JSON)

    organization = relationship("Organization", back_populates="company_profile")


class InvestorProfile(Base, TimestampMixin):
    __tablename__ = 'investor_profiles'

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False, unique=True)
    investor_type = Column(String)
    preferred_stage = Column(String)
    founded_year = Column(Integer)
    themes = Column(JSON)
    user_notes = Column(String)

    organization = relationship("Organization", back_populates="investor_profile")


class FundProfile(Base, TimestampMixin):
    __tablename__ = 'fund_profiles'

    id = Column(Integer, primary_key=True, index=True)
    parent_organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)
    fund_name = Column(String, nullable=False)
    fund_type = Column(String)
    investment_fund_type = Column(String) # e.g. Venture Capital, Hedge Fund (from SEC Form D)
    vintage_year = Column(Integer)
    fund_size_raised = Column(String)
    fund_size_target = Column(String)
    status = Column(String)
    description = Column(String)
    accession_number = Column(String)

    # Address fields
    street1 = Column(String)
    street2 = Column(String)
    city = Column(String)
    state = Column(String)
    country = Column(String)
    zip_code = Column(String)

    themes = Column(JSON)
    market_reputation = Column(String)
    user_notes = Column(String)

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


class Investment(Base, TimestampMixin):
    __tablename__ = 'investments'

    id = Column(Integer, primary_key=True, index=True)
    investor_organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)
    company_organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)
    
    investment_date = Column(DateTime)
    round_type = Column(String) # e.g., Seed, Series A
    amount = Column(String) # legacy / general amount string
    total_round_amount = Column(String)
    firm_investment_amount = Column(String)
    is_lead = Column(Boolean, default=False)
    fund_id = Column(Integer, ForeignKey('fund_profiles.id'), nullable=True)
    source_document_id = Column(Integer, ForeignKey('source_documents.id'), nullable=True)
    metadata_json = Column(JSON)

    investor = relationship("Organization", foreign_keys=[investor_organization_id], back_populates="investments_made")
    company = relationship("Organization", foreign_keys=[company_organization_id], back_populates="investments_received")
    fund = relationship("FundProfile")
    source_document = relationship("SourceDocument")


# ==============================================================================
# PIPELINE FRAMEWORK
# ==============================================================================

class Pipeline(Base, TimestampMixin):
    __tablename__ = 'pipelines'

    id = Column(Integer, primary_key=True, index=True)
    pipeline_name = Column(String, nullable=False)
    
    # Categories for pipeline execution strategy
    connector_type = Column(String)     # HOW TO FETCH (e.g. SEC_DAILY_INDEX, WEB_PAGE)
    parser_type = Column(String)        # HOW TO PARSE (e.g. SEC_FORM_D_XML, HTML_TO_MARKDOWN)
    normalizer_type = Column(String)    # HOW TO MAP TO CANONICAL (e.g. SEC_FORM_D_TO_ORGS, PORTFOLIO_TO_INVESTMENTS)

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
    pipeline_id = Column(Integer, ForeignKey('pipelines.id'), nullable=True)

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
    source_documents = relationship("SourceDocument", back_populates="pipeline_run")
    extraction_jobs = relationship("ExtractionJob", back_populates="pipeline_run")
    steps = relationship("PipelineRunStep", back_populates="pipeline_run", order_by="PipelineRunStep.step_order.asc()")


class PipelineRunStep(Base, TimestampMixin):
    __tablename__ = 'pipeline_run_steps'

    id = Column(Integer, primary_key=True, index=True)
    pipeline_run_id = Column(Integer, ForeignKey('pipeline_runs.id'), nullable=False)
    
    step_order = Column(Integer, nullable=False)
    step_name = Column(String, nullable=False)
    step_type = Column(String) # FETCH, FILTER, PARSE, EXTRACT, NORMALIZE, CANONICAL_WRITE
    method = Column(String)
    
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    output_count = Column(Integer, default=0)
    records_created = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    
    status = Column(String) # SUCCESS, FAILED, RUNNING
    error_message = Column(String)

    pipeline_run = relationship("PipelineRun", back_populates="steps")


class SourceDocument(Base, TimestampMixin):
    __tablename__ = 'source_documents'

    id = Column(Integer, primary_key=True, index=True)
    pipeline_run_id = Column(Integer, ForeignKey('pipeline_runs.id'), nullable=False)
    
    source_type = Column(String, default="EXA_SEARCH") # EXA_SEARCH, MANUAL_UPLOAD

    document_type = Column(String) # PDF, WEB_PAGE, IMAGE, API_RESPONSE, CSV_FILE, DOCSEND, etc.
    document_class = Column(String) # startup_pitch_deck, legal_contract, etc.
    classification_result_json = Column(JSON)
    document_date = Column(String, nullable=True)
    
    title = Column(String, nullable=True)
    source_name = Column(String, nullable=True)
    source_url = Column(String)
    file_path = Column(String)
    source_identifier = Column(String) # external ID / checksum / docsend ID / etc.
    content_hash = Column(String, index=True)

    pipeline_run = relationship("PipelineRun", back_populates="source_documents")
    document_texts = relationship("DocumentText", back_populates="source_document")


class DocumentText(Base, TimestampMixin):
    __tablename__ = 'document_texts'

    id = Column(Integer, primary_key=True, index=True)
    source_document_id = Column(Integer, ForeignKey('source_documents.id'), nullable=False)

    data_type = Column(String) # PAGE_TEXT, API_JSON, CSV_ROW, PROFILE_TEXT
    raw_content = Column(Text)
    content_hash = Column(String, index=True)

    source_document = relationship("SourceDocument", back_populates="document_texts")
    extraction_jobs = relationship("ExtractionJob", back_populates="document_text")


class ExtractionJob(Base, TimestampMixin):
    __tablename__ = 'extraction_jobs'

    id = Column(Integer, primary_key=True, index=True)
    pipeline_run_id = Column(Integer, ForeignKey('pipeline_runs.id'), nullable=False)
    document_text_id = Column(Integer, ForeignKey('document_texts.id'), nullable=False)

    schema_name = Column(String)
    schema_version = Column(String)
    prompt_name = Column(String)
    prompt_version = Column(String)

    status = Column(String) # PENDING, IN_PROGRESS, SUCCESS, FAILED
    llm_usage_json = Column(JSON)
    error_message = Column(String)

    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    pipeline_run = relationship("PipelineRun", back_populates="extraction_jobs")
    document_text = relationship("DocumentText", back_populates="extraction_jobs")
    extracted_entities = relationship("ExtractedEntity", back_populates="extraction_job")
    extracted_relationships = relationship("ExtractedRelationship", back_populates="extraction_job")


class ExtractedEntity(Base, TimestampMixin):
    __tablename__ = 'extracted_entities'

    id = Column(Integer, primary_key=True, index=True)
    extraction_job_id = Column(Integer, ForeignKey('extraction_jobs.id'), nullable=False)

    entity_type = Column(String) # ORGANIZATION, PERSON, PROGRAM_COHORT
    raw_name = Column(String)
    normalized_name = Column(String, index=True)

    extracted_payload_json = Column(JSON)
    extraction_confidence = Column(Float, nullable=True)

    extraction_job = relationship("ExtractionJob", back_populates="extracted_entities")
    matches = relationship("EntityMatch", back_populates="extracted_entity")
    attribute_evidences = relationship("EntityAttributeEvidence", back_populates="extracted_entity")


class ExtractedRelationship(Base, TimestampMixin):
    __tablename__ = 'extracted_relationships'

    id = Column(Integer, primary_key=True, index=True)
    extraction_job_id = Column(Integer, ForeignKey('extraction_jobs.id'), nullable=False)

    relationship_type = Column(String) # FOUNDER_OF, EMPLOYEE_OF, MEMBER_OF_COHORT, INVESTED_IN

    source_extracted_entity_id = Column(Integer, ForeignKey('extracted_entities.id'), nullable=True)
    source_entity_type = Column(String, nullable=True) # ORGANIZATION, PERSON
    source_entity_id = Column(String, nullable=True) # Polymorphic — int or UUID as string

    target_extracted_entity_id = Column(Integer, ForeignKey('extracted_entities.id'), nullable=True)
    target_entity_type = Column(String, nullable=True)
    target_entity_id = Column(String, nullable=True)

    relationship_payload_json = Column(JSON)
    extraction_confidence = Column(Float, nullable=True)

    extraction_job = relationship("ExtractionJob", back_populates="extracted_relationships")
    source_extracted_entity = relationship("ExtractedEntity", foreign_keys=[source_extracted_entity_id])
    target_extracted_entity = relationship("ExtractedEntity", foreign_keys=[target_extracted_entity_id])


# ==============================================================================
# AUDIT TRAIL
# ==============================================================================

class EntityMatch(Base, TimestampMixin):
    __tablename__ = 'entity_matches'

    id = Column(Integer, primary_key=True, index=True)
    extracted_entity_id = Column(Integer, ForeignKey('extracted_entities.id'), nullable=False)
    
    canonical_entity_type = Column(String, nullable=False) # Organization, Person, FundProfile, ProgramProfile, etc.
    canonical_entity_id = Column(String, nullable=False, index=True) # Int or UUID as string
    
    match_confidence = Column(Float, nullable=True)
    match_method = Column(String) # EXACT_NAME_MATCH, LLM_SIMILARITY, MANUAL
    created_by = Column(String) # SYSTEM, USER_ID
    
    extracted_entity = relationship("ExtractedEntity", back_populates="matches")


class EntityAttributeEvidence(Base, TimestampMixin):
    __tablename__ = 'entity_attribute_evidences'

    id = Column(Integer, primary_key=True, index=True)
    
    canonical_entity_type = Column(String, nullable=False)
    canonical_entity_id = Column(String, nullable=False, index=True)
    
    attribute_name = Column(String, nullable=False)
    attribute_value_json = Column(JSON)
    
    source_document_id = Column(Integer, ForeignKey('source_documents.id'), nullable=True)
    extracted_entity_id = Column(Integer, ForeignKey('extracted_entities.id'), nullable=True)

    source_document = relationship("SourceDocument")
    extracted_entity = relationship("ExtractedEntity", back_populates="attribute_evidences")


class AuditTrail(Base, TimestampMixin):
    __tablename__ = 'audit_trails'

    id = Column(Integer, primary_key=True, index=True)
    
    canonical_entity_type = Column(String, nullable=False)
    canonical_entity_id = Column(String, nullable=False, index=True)
    
    mutation_type = Column(String, nullable=False) # CREATE, UPDATE, DELETE
    
    field_name = Column(String, nullable=True)
    old_value = Column(String, nullable=True)
    new_value = Column(String, nullable=True)
    
    source = Column(String) # PIPELINE, USER_EDIT
    created_by = Column(String, nullable=True) # User email or SYSTEM
    extraction_job_id = Column(Integer, ForeignKey('extraction_jobs.id'), nullable=True)
    
    extraction_job = relationship("ExtractionJob")

class CompanyAugmentationReport(Base, TimestampMixin):
    __tablename__ = 'company_augmentation_reports'

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)
    
    pipeline_run_id = Column(Integer, ForeignKey('pipeline_runs.id'), nullable=True)
    
    schema_version = Column(String)
    extracted_data_json = Column(JSON)
    scoring_json = Column(JSON)
    
    status = Column(String) # RUNNING, SUCCESS, FAILED
    error_message = Column(String)

    organization = relationship("Organization")
    pipeline_run = relationship("PipelineRun")
