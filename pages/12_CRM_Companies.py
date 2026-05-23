import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from sqlalchemy import or_
from market_comps.db.session import get_db
from market_comps.db.models import (
    Organization, Person, CompanyProfile, FundProfile,
    ProgramMembership, PersonOrganizationRole, CanonicalMutation,
    ProgramProfile, ProgramCohort, EntityMatch, ExtractedEntity, ExtractionJob, DocumentText, SourceDocument
)

st.set_page_config(page_title="Companies Directory", page_icon="🏢", layout="wide")
st.title("🏢 Companies Directory")

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# Helper to render company details below the table
def display_company_details(company_id):
    org = db.query(Organization).options(
        joinedload(Organization.company_profile),
        joinedload(Organization.fund_profiles),
        joinedload(Organization.program_profiles).joinedload(ProgramProfile.cohorts),
        joinedload(Organization.program_memberships).joinedload(ProgramMembership.cohort).joinedload(ProgramCohort.program),
        joinedload(Organization.roles).joinedload(PersonOrganizationRole.person)
    ).filter(Organization.id == int(company_id)).first()

    if not org:
        st.error(f"Company with ID {company_id} not found.")
        return

    with st.container(border=True):
        st.subheader(f"🏢 {org.name}")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Basic Information")
            st.write(f"**Domain:** {org.primary_domain} | **Website:** {org.website_url}")
            st.write(f"**LinkedIn:** {org.linkedin_url}")
            st.write(f"**Location:** {org.city}, {org.state}, {org.country}")
            st.write(f"**Status:** {org.status}")
            if org.description:
                st.info(org.description)
                
            if org.company_profile:
                st.divider()
                st.markdown("#### Company Profile")
                st.write(f"**Industry:** {org.company_profile.industry} | **Stage:** {org.company_profile.company_stage}")
                st.write(f"**Sub-Industry:** {org.company_profile.subindustry} | **Founded:** {org.company_profile.founded_year}")
                
        with col2:
            if org.program_memberships:
                st.markdown("#### Program Memberships")
                for m in org.program_memberships:
                    if m.cohort and m.cohort.program:
                        st.write(f"- 🎯 **{m.cohort.program.program_name}** — {m.cohort.cohort_name}")
                    elif m.cohort:
                        st.write(f"- 🎯 {m.cohort.cohort_name}")
                    else:
                        st.write("- 🎯 (Unlinked Membership)")
                        
            if org.fund_profiles:
                st.markdown("#### Fund Profiles")
                for fund in org.fund_profiles:
                    with st.expander(f"💰 {fund.fund_name}"):
                        st.write(f"**Vintage:** {fund.vintage_year} | **Size:** {fund.fund_size}")
                        st.write(f"**Type:** {fund.fund_type} | **Status:** {fund.status}")
                        if fund.description:
                            st.caption(fund.description)

            if org.program_profiles:
                st.markdown("#### Programs & Cohorts")
                for prog in org.program_profiles:
                    with st.expander(f"🎯 {prog.program_name} ({prog.program_type or 'Program'})"):
                        st.write(f"**Status:** {prog.status or 'N/A'}")
                        if prog.description:
                            st.write(prog.description)
                        
                        if prog.cohorts:
                            st.write("**Cohorts:**")
                            for cohort in prog.cohorts:
                                date_str = ""
                                if cohort.start_date:
                                    date_str = f" ({cohort.start_date.strftime('%b %Y')} - {cohort.end_date.strftime('%b %Y') if cohort.end_date else 'Present'})"
                                st.write(f"- 📦 **{cohort.cohort_name}**{date_str}")
                                if cohort.description:
                                    st.caption(f"  {cohort.description}")
                        else:
                            st.caption("No cohorts defined.")

        # People & Roles
        st.divider()
        st.subheader("👥 People & Roles")
        if org.roles:
            for role in org.roles:
                p = role.person
                if p:
                    name = p.full_name or f"{p.first_name} {p.last_name}"
                    with st.expander(f"👤 {name} — {role.title or 'No Title'}"):
                        st.write(f"**Seniority:** {role.seniority_level or 'N/A'} | **Type:** {role.role_type or 'N/A'}")
                        st.write(f"**LinkedIn:** {p.linkedin_url or 'N/A'}")
                        st.write(f"**Location:** {p.city or 'N/A'}, {p.state or 'N/A'}")
                        if p.emails:
                            st.write("**Emails:**")
                            for e in p.emails:
                                st.write(f"- {e.email} ({e.email_type})")
        else:
            st.info("No people linked to this company.")

        # Linked Source Documents
        st.divider()
        st.subheader("📄 Linked Source Documents")
        
        docs = db.query(SourceDocument).join(DocumentText).join(ExtractionJob).join(ExtractedEntity).join(EntityMatch).filter(
            EntityMatch.canonical_entity_type == "Organization",
            EntityMatch.canonical_entity_id == str(org.id)
        ).distinct().all()
        
        if docs:
            for doc in docs:
                st.write(f"- **{doc.document_type}**: {doc.source_url} (Processed: {doc.created_at.strftime('%Y-%m-%d')})")
        else:
            st.info("No documents linked to this company.")

        # Audit Trail
        st.divider()
        st.subheader("📜 Mutation History")
        audit = db.query(CanonicalMutation).filter_by(
            canonical_entity_type="ORGANIZATION", canonical_entity_id=str(org.id)
        ).order_by(CanonicalMutation.created_at.desc()).limit(10).all()
        
        if audit:
            audit_data = []
            for a in audit:
                audit_data.append({
                    "Date": a.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Action": a.mutation_type,
                    "Field": a.field_name or "",
                    "Old Value": a.old_value or "",
                    "New Value": a.new_value or "",
                    "Source": a.source or ""
                })
            st.dataframe(pd.DataFrame(audit_data), use_container_width=True, hide_index=True)
        else:
            st.info("No mutation entries found.")

# Fetch and query data
search_query = st.text_input("Search Companies...", placeholder="Search by name, domain, or website...")

q = db.query(Organization).options(
    joinedload(Organization.company_profile)
).filter(Organization.organization_type == "COMPANY")

if search_query:
    search_filter = f"%{search_query}%"
    q = q.filter(
        or_(
            Organization.name.ilike(search_filter),
            Organization.normalized_name.ilike(search_filter),
            Organization.primary_domain.ilike(search_filter)
        )
    )

orgs = q.order_by(Organization.created_at.desc()).limit(200).all()

# Prepare Dataframe
data = []
for o in orgs:
    row = {
        "ID": o.id,
        "Name": o.name,
        "Domain": o.primary_domain,
        "Website": o.website_url,
        "City": o.city,
        "Status": o.status,
        "Created": o.created_at.strftime("%Y-%m-%d") if o.created_at else ""
    }
    if o.company_profile:
        row["Industry"] = o.company_profile.industry
        row["Stage"] = o.company_profile.company_stage
    data.append(row)

df = pd.DataFrame(data)

if df.empty:
    st.info("No company records found.")
else:
    st.write("👆 *Select a company row below to inspect full details.*")
    
    event = st.dataframe(
        df,
        key="grid_companies",
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        use_container_width=True,
        column_config={
            "ID": None, # Hide ID
            "Website": st.column_config.LinkColumn("Website"),
            "Created": st.column_config.DateColumn("Created")
        }
    )
    
    selection = event.get("selection", {})
    rows = selection.get("rows", [])
    
    if rows:
        st.divider()
        selected_row_idx = rows[0]
        selected_company_id = df.iloc[selected_row_idx]["ID"]
        display_company_details(selected_company_id)
