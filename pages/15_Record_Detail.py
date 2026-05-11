import streamlit as st
from market_comps.db.session import get_db
from market_comps.db.models import (
    Organization, Person, CompanyProfile, InvestorProfile, 
    FundProfile, ProgramProfile, ProgramMembership, PersonOrganizationRole,
    PersonEmail, EntityAuditTrail
)
from sqlalchemy.orm import joinedload
import pandas as pd

st.set_page_config(page_title="Record Detail", page_icon="📄", layout="wide")

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# Get record type and ID from query params or session state
params = st.query_params
record_type = params.get("type", "ORGANIZATION")
record_id = params.get("id")

if not record_id:
    st.warning("No record ID provided.")
    if st.button("Go to Directory"):
        st.switch_page("pages/12_CRM_Directory.py")
    st.stop()

if record_type == "ORGANIZATION":
    org = db.query(Organization).options(
        joinedload(Organization.company_profile),
        joinedload(Organization.investor_profile),
        joinedload(Organization.fund_profiles),
        joinedload(Organization.program_profiles),
        joinedload(Organization.program_memberships),
        joinedload(Organization.roles).joinedload(PersonOrganizationRole.person)
    ).filter(Organization.id == int(record_id)).first()

    if not org:
        st.error(f"Organization with ID {record_id} not found.")
        st.stop()

    st.title(f"🏢 {org.name}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Basic Information")
        st.write(f"**Name:** {org.name}")
        st.write(f"**Domain:** {org.primary_domain}")
        st.write(f"**Website:** {org.website_url}")
        st.write(f"**LinkedIn:** {org.linkedin_url}")
        st.write(f"**City:** {org.city} | **State:** {org.state} | **Country:** {org.country}")
        st.write(f"**Type:** {org.organization_type}")
        st.write(f"**Status:** {org.status}")
        
        if org.description:
            st.info(org.description)

        if org.company_profile:
            st.divider()
            st.subheader("Company Profile")
            st.write(f"**Industry:** {org.company_profile.industry}")
            st.write(f"**Sub-Industry:** {org.company_profile.subindustry}")
            st.write(f"**Stage:** {org.company_profile.company_stage}")
            st.write(f"**Founded:** {org.company_profile.founded_year}")

        if org.investor_profile:
            st.divider()
            st.subheader("Investor Profile")
            st.write(f"**Investor Type:** {org.investor_profile.investor_type}")
            st.write(f"**Preferred Stage:** {org.investor_profile.preferred_stage}")

    with col2:
        if org.program_memberships:
            st.subheader("Program Memberships")
            for m in org.program_memberships:
                if m.cohort:
                    st.write(f"- 🎯 **{m.cohort.program.program_name}** — {m.cohort.cohort_name}")
                else:
                    st.write(f"- 🎯 (unlinked membership)")

        if org.fund_profiles:
            st.subheader("Funds")
            for fund in org.fund_profiles:
                with st.expander(f"💰 {fund.fund_name}"):
                    st.write(f"**Type:** {fund.fund_type}")
                    st.write(f"**Vintage:** {fund.vintage_year}")
                    st.write(f"**Size:** {fund.fund_size}")
                    st.write(f"**Status:** {fund.status}")
                    st.write(fund.description)

    st.divider()
    st.subheader("People & Roles")
    if org.roles:
        for role in org.roles:
            p = role.person
            if p:
                name = p.full_name or f"{p.first_name} {p.last_name}"
                with st.expander(f"👤 {name} — {role.title or 'No Title'}"):
                    st.write(f"**Title:** {role.title}")
                    st.write(f"**LinkedIn:** {p.linkedin_url}")
                    if p.emails:
                        st.write("**Emails:**")
                        for e in p.emails:
                            st.write(f"- {e.email} ({e.email_type})")
                    if st.button("View Person Detail", key=f"btn_p_{p.id}"):
                        st.query_params["type"] = "PERSON"
                        st.query_params["id"] = str(p.id)
                        st.rerun()
    else:
        st.info("No people linked to this organization.")

    st.divider()
    st.subheader("Audit Trail")
    audit = db.query(EntityAuditTrail).filter_by(
        entity_type="ORGANIZATION", entity_id=str(org.id)
    ).order_by(EntityAuditTrail.created_at.desc()).all()
    
    if audit:
        audit_data = []
        for a in audit:
            audit_data.append({
                "Date": a.created_at.strftime("%Y-%m-%d %H:%M"),
                "Action": a.audit_action,
                "Field": a.field_name or "",
                "Old Value": a.old_value or "",
                "New Value": a.new_value or "",
                "Reason": a.reason or ""
            })
        st.table(pd.DataFrame(audit_data))
    else:
        st.info("No audit trail entries found.")

elif record_type == "PERSON":
    import uuid
    try:
        u_id = uuid.UUID(record_id)
    except:
        st.error(f"Invalid Person ID: {record_id}")
        st.stop()
        
    person = db.query(Person).options(
        joinedload(Person.emails),
        joinedload(Person.roles).joinedload(PersonOrganizationRole.organization)
    ).filter(Person.id == u_id).first()

    if not person:
        st.error(f"Person with ID {record_id} not found.")
        st.stop()

    st.title(f"👤 {person.full_name or person.first_name + ' ' + person.last_name}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Basic Information")
        st.write(f"**Name:** {person.full_name}")
        st.write(f"**LinkedIn:** {person.linkedin_url}")
        st.write(f"**Twitter:** {person.twitter_url}")
        st.write(f"**Location:** {person.city}, {person.state}, {person.country}")
        if person.bio:
            st.info(person.bio)

        if person.emails:
            st.subheader("Emails")
            for e in person.emails:
                st.write(f"- {e.email} ({e.email_type}) {'(Primary)' if e.is_primary else ''}")

    with col2:
        st.subheader("Roles & Organizations")
        if person.roles:
            for role in person.roles:
                org = role.organization
                if org:
                    with st.expander(f"🏢 {org.name} — {role.title or 'No Title'}"):
                        st.write(f"**Title:** {role.title}")
                        st.write(f"**Type:** {role.role_type}")
                        st.write(f"**Current:** {role.is_current}")
                        if st.button("View Organization Detail", key=f"btn_org_{org.id}"):
                            st.query_params["type"] = "ORGANIZATION"
                            st.query_params["id"] = str(org.id)
                            st.rerun()
        else:
            st.info("No organizations linked to this person.")

    st.divider()
    st.subheader("Audit Trail")
    audit = db.query(EntityAuditTrail).filter_by(
        entity_type="PERSON", entity_id=str(person.id)
    ).order_by(EntityAuditTrail.created_at.desc()).all()
    
    if audit:
        audit_data = []
        for a in audit:
            audit_data.append({
                "Date": a.created_at.strftime("%Y-%m-%d %H:%M"),
                "Action": a.audit_action,
                "Field": a.field_name or "",
                "Old Value": a.old_value or "",
                "New Value": a.new_value or "",
                "Reason": a.reason or ""
            })
        st.table(pd.DataFrame(audit_data))
    else:
        st.info("No audit trail entries found.")

if st.button("Back to Directory"):
    st.switch_page("pages/12_CRM_Directory.py")
