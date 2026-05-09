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
