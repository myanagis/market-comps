import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey, JSON, Float, Text, UniqueConstraint
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
    organization_type = Column(String, nullable=False, default="COMPANY")
    ownership_type = Column(String, default="PRIVATE") # public, private
    ticker = Column(String)
    exchange = Column(String)
    
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

    # New relationships for the CRM overhaul
    metric_observations = relationship("MetricObservation", back_populates="company", cascade="all, delete-orphan")
    financing_rounds = relationship("FinancingRound", back_populates="company", cascade="all, delete-orphan")
    investments_made = relationship("RoundInvestor", back_populates="investor", cascade="all, delete-orphan")

    transactions_as_target = relationship("Transaction", foreign_keys="[Transaction.target_company_id]", back_populates="target_company", cascade="all, delete-orphan")
    transactions_as_acquirer = relationship("Transaction", foreign_keys="[Transaction.acquirer_company_id]", back_populates="acquirer_company", cascade="all, delete-orphan")

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


class Sector(Base, TimestampMixin):
    __tablename__ = 'sectors'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(String)


class CompanyProfile(Base, TimestampMixin):
    __tablename__ = 'company_profiles'

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False, unique=True)
    founded_year = Column(Integer)
    industry = Column(String)
    subindustry = Column(String)
    company_stage = Column(String)
    themes = Column(JSON)
    sectors = Column(JSON)

    organization = relationship("Organization", back_populates="company_profile")


class InvestorProfile(Base, TimestampMixin):
    __tablename__ = 'investor_profiles'

    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False, unique=True)
    investor_type = Column(String)
    preferred_stage = Column(String)
    founded_year = Column(Integer)
    themes = Column(JSON)
    sectors = Column(JSON)
    stages = Column(JSON)
    specialties = Column(JSON)
    check_size_min = Column(Float)
    check_size_max = Column(Float)
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


# ==============================================================================
# CRM METRICS & FINANCING
# ==============================================================================

class MetricType(Base, TimestampMixin):
    __tablename__ = 'metric_types'

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, nullable=False, unique=True, index=True)
    display_name = Column(String, nullable=False)
    value_type = Column(String, nullable=False) # currency, integer, decimal, percentage, multiple, text, boolean
    default_unit = Column(String)
    default_currency = Column(String)
    description = Column(String)
    is_point_in_time = Column(Boolean, nullable=False, default=False)

    observations = relationship("MetricObservation", back_populates="metric_type", cascade="all, delete-orphan")


class MetricObservation(Base):
    __tablename__ = 'metric_observations'

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)
    metric_type_id = Column(Integer, ForeignKey('metric_types.id'), nullable=False)

    value_numeric = Column(Float)
    value_text = Column(String)
    currency_code = Column(String)
    unit = Column(String)

    period_start = Column(DateTime)
    period_end = Column(DateTime)
    as_of_date = Column(DateTime)

    observation_status = Column(String, nullable=False) # actual, company_estimate, company_guidance, external_estimate, internal_estimate, derived, unverified
    reporting_basis = Column(String) # fiscal_year, calendar_year, quarter, month, trailing_twelve_months, run_rate, point_in_time, transaction

    reported_at = Column(DateTime)
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    confidence_score = Column(Float)
    supersedes_observation_id = Column(Integer, ForeignKey('metric_observations.id'))

    notes = Column(String)
    metadata_json = Column(JSON, nullable=False, default={})
    created_by = Column(String) # UUID or email string

    company = relationship("Organization", back_populates="metric_observations")
    metric_type = relationship("MetricType", back_populates="observations")
    observation_sources = relationship("ObservationSource", back_populates="observation", cascade="all, delete-orphan")
    supersedes = relationship("MetricObservation", remote_side=[id])


class ObservationSource(Base):
    __tablename__ = 'observation_sources'

    observation_id = Column(Integer, ForeignKey('metric_observations.id', ondelete='CASCADE'), primary_key=True)
    source_id = Column(Integer, ForeignKey('source_documents.id', ondelete='CASCADE'), primary_key=True)

    relationship_type = Column(String, nullable=False) # primary, supporting, corroborating, contradicting, derived_from
    page_number = Column(Integer)
    section_name = Column(String)
    source_excerpt = Column(String)

    observation = relationship("MetricObservation", back_populates="observation_sources")
    source_document = relationship("SourceDocument")


class FinancingRound(Base, TimestampMixin):
    __tablename__ = 'financing_rounds'

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)
    round_name = Column(String)
    status = Column(String, nullable=False, default='rumored') # rumored, raising, announced, closed, cancelled

    company = relationship("Organization", back_populates="financing_rounds")
    facts = relationship("FinancingRoundFact", back_populates="round", cascade="all, delete-orphan")
    investors = relationship("RoundInvestor", back_populates="round", cascade="all, delete-orphan")


class FinancingRoundFact(Base):
    __tablename__ = 'financing_round_facts'

    id = Column(Integer, primary_key=True, index=True)
    financing_round_id = Column(Integer, ForeignKey('financing_rounds.id', ondelete='CASCADE'), nullable=False)
    
    fact_type = Column(String, nullable=False) # target_raise, amount_raised, pre_money_valuation, post_money_valuation, etc
    value_numeric = Column(Float)
    value_text = Column(String)
    value_date = Column(DateTime)
    currency_code = Column(String)
    
    certainty = Column(String, nullable=False, default='unknown') # rumored, estimated, company_stated, announced, confirmed, disputed
    
    reported_at = Column(DateTime)
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    source_id = Column(Integer, ForeignKey('source_documents.id'))
    notes = Column(String)

    round = relationship("FinancingRound", back_populates="facts")
    source_document = relationship("SourceDocument")


class RoundInvestor(Base):
    __tablename__ = 'round_investors'

    id = Column(Integer, primary_key=True, index=True)
    financing_round_id = Column(Integer, ForeignKey('financing_rounds.id', ondelete='CASCADE'), nullable=False)
    investor_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)

    role = Column(String) # lead, co-lead, participant, strategic, unknown
    status = Column(String, nullable=False, default='rumored') # rumored, considering, committed, invested, withdrew, denied
    
    amount_numeric = Column(Float)
    currency_code = Column(String)
    amount_certainty = Column(String)

    reported_at = Column(DateTime)
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    source_id = Column(Integer, ForeignKey('source_documents.id'))
    notes = Column(String)

    round = relationship("FinancingRound", back_populates="investors")
    investor = relationship("Organization", back_populates="investments_made")
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
    
    llm_total_tokens = Column(Integer, default=0)
    llm_estimated_cost_usd = Column(Float, default=0.0)
    exa_calls = Column(Integer, default=0)
    exa_estimated_cost_usd = Column(Float, default=0.0)

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
    
    # Newly added fields for CRM sources & pipeline status
    source_tier = Column(Integer) # 1-5
    publisher = Column(String)
    published_at = Column(DateTime)
    llm_model_used = Column(String)
    extraction_status = Column(String, default="SUCCESS") # SUCCESS, FAILED_JUNK, FAILED_RELEVANCE, FAILED_FETCH
    extraction_error = Column(String, nullable=True)
    
    deleted_at = Column(DateTime, nullable=True)
    deleted_by = Column(String, nullable=True)

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

# ==============================================================================
# MARKETS AND COMPETITIVE LANDSCAPE
# ==============================================================================

class Market(Base, TimestampMixin):
    __tablename__ = 'markets'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(String)
    sectors = Column(JSON)
    parent_market_id = Column(Integer, ForeignKey('markets.id'), nullable=True)

    parent_market = relationship("Market", remote_side=[id])
    segments = relationship("MarketSegment", back_populates="market", cascade="all, delete-orphan")
    competitive_analyses = relationship("CompetitiveAnalysis", back_populates="market")


class MarketSegment(Base, TimestampMixin):
    __tablename__ = 'market_segments'

    id = Column(Integer, primary_key=True, index=True)
    market_id = Column(Integer, ForeignKey('markets.id'), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    segment_type = Column(String, nullable=True)
    parent_segment_id = Column(Integer, ForeignKey('market_segments.id'), nullable=True)
    sort_order = Column(Integer, nullable=True)

    market = relationship("Market", back_populates="segments")
    parent_segment = relationship("MarketSegment", remote_side=[id])
    company_segments = relationship("MarketSegmentCompanyLink", back_populates="market_segment", cascade="all, delete-orphan")


class MarketSegmentCompanyLink(Base, TimestampMixin):
    __tablename__ = 'market_segment_company_links'

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)
    market_segment_id = Column(Integer, ForeignKey('market_segments.id'), nullable=False)
    
    differentiation = Column(String)
    notes = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    source_id = Column(Integer, ForeignKey('source_documents.id'), nullable=True)
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    company = relationship("Organization")
    market_segment = relationship("MarketSegment", back_populates="company_segments")
    source_document = relationship("SourceDocument")


class CompetitiveAnalysis(Base, TimestampMixin):
    __tablename__ = 'competitive_analyses'

    id = Column(Integer, primary_key=True, index=True)
    subject_company_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)
    market_id = Column(Integer, ForeignKey('markets.id'), nullable=False)
    title = Column(String, nullable=False, default="Competitive Landscape")
    summary = Column(String)
    status = Column(String, default="draft") # draft, reviewed, published, archived

    subject_company = relationship("Organization")
    market = relationship("Market", back_populates="competitive_analyses")
    analysis_segments = relationship("CompetitiveAnalysisSegment", back_populates="competitive_analysis", cascade="all, delete-orphan")
    analysis_companies = relationship("CompetitiveAnalysisCompany", back_populates="competitive_analysis", cascade="all, delete-orphan")


class CompetitiveAnalysisSegment(Base, TimestampMixin):
    __tablename__ = 'competitive_analysis_segments'

    id = Column(Integer, primary_key=True, index=True)
    competitive_analysis_id = Column(Integer, ForeignKey('competitive_analyses.id'), nullable=False)
    market_segment_id = Column(Integer, ForeignKey('market_segments.id'), nullable=False)
    
    threat_level = Column(String, nullable=True) # enum-like: High, Medium/High, Medium, Medium/Low, Low, N/A
    analysis_notes = Column(String, nullable=True)
    sort_order = Column(Integer, nullable=True)

    competitive_analysis = relationship("CompetitiveAnalysis", back_populates="analysis_segments")
    market_segment = relationship("MarketSegment")


class CompetitiveAnalysisCompany(Base, TimestampMixin):
    __tablename__ = 'competitive_analysis_companies'

    id = Column(Integer, primary_key=True, index=True)
    competitive_analysis_id = Column(Integer, ForeignKey('competitive_analyses.id'), nullable=False)
    competitor_company_id = Column(Integer, ForeignKey('organizations.id'), nullable=False)
    market_segment_id = Column(Integer, ForeignKey('market_segments.id'), nullable=True)
    
    relationship_type = Column(String, nullable=False, default="direct_competitor") # direct_competitor, indirect_competitor, substitute, incumbent, adjacent, potential_entrant, partner_competitor
    competitive_notes = Column(String)
    included = Column(Boolean, default=True)
    sort_order = Column(Integer, nullable=True)

    competitive_analysis = relationship("CompetitiveAnalysis", back_populates="analysis_companies")
    competitor_company = relationship("Organization")
    market_segment = relationship("MarketSegment")

class ComparisonSet(Base, TimestampMixin):
    __tablename__ = 'comparison_sets'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    set_type = Column(String, nullable=False) # e.g. "Public Comps", "M&A Precedents", "Financing Comps" - Should be controlled via application logic
    
    market_links = relationship("MarketComparisonSetLink", back_populates="comparison_set", cascade="all, delete-orphan")
    organization_links = relationship("ComparisonSetOrganizationLink", back_populates="comparison_set", cascade="all, delete-orphan")

class MarketComparisonSetLink(Base, TimestampMixin):
    __tablename__ = 'market_comparison_set_links'
    
    id = Column(Integer, primary_key=True, index=True)
    market_id = Column(Integer, ForeignKey('markets.id', ondelete="CASCADE"), nullable=False)
    comparison_set_id = Column(Integer, ForeignKey('comparison_sets.id', ondelete="CASCADE"), nullable=False)
    
    notes = Column(String)
    
    market = relationship("Market")
    comparison_set = relationship("ComparisonSet", back_populates="market_links")
    
    __table_args__ = (
        UniqueConstraint('market_id', 'comparison_set_id', name='uq_market_comparison_set'),
    )

class ComparisonSetOrganizationLink(Base, TimestampMixin):
    __tablename__ = 'comparison_set_organization_links'
    
    id = Column(Integer, primary_key=True, index=True)
    comparison_set_id = Column(Integer, ForeignKey('comparison_sets.id', ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey('organizations.id', ondelete="CASCADE"), nullable=False)
    
    notes = Column(String)
    included = Column(Boolean, default=True)
    
    comparison_set = relationship("ComparisonSet", back_populates="organization_links")
    organization = relationship("Organization")
    
    __table_args__ = (
        UniqueConstraint('comparison_set_id', 'organization_id', name='uq_comparison_set_organization'),
    )

class Transaction(Base, TimestampMixin):
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_name = Column(String, nullable=False)
    transaction_type = Column(String, nullable=False) # ACQUISITION, MERGE, IPO, SPAC, BUYOUT, SPINOFF
    status = Column(String) # RUMORED, ANNOUNCED, CLOSED, CANCELLED
    
    announced_date = Column(DateTime)
    closed_date = Column(DateTime)
    
    target_company_id = Column(Integer, ForeignKey('organizations.id', ondelete="CASCADE"))
    acquirer_company_id = Column(Integer, ForeignKey('organizations.id', ondelete="CASCADE"))
    
    transaction_value_numeric = Column(Float)
    transaction_value_text = Column(String)
    currency_code = Column(String)
    
    description = Column(String)
    notes = Column(String)
    
    target_company = relationship("Organization", foreign_keys=[target_company_id], back_populates="transactions_as_target")
    acquirer_company = relationship("Organization", foreign_keys=[acquirer_company_id], back_populates="transactions_as_acquirer")
