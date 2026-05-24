import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from sqlalchemy import or_
from market_comps.db.session import get_db
from market_comps.db.models import (
    Organization, Person, InvestorProfile, FundProfile,
    ProgramMembership, PersonOrganizationRole, CanonicalMutation,
    ProgramProfile, ProgramCohort, EntityMatch, ExtractedEntity, ExtractionJob, DocumentText, SourceDocument
)

st.set_page_config(page_title="Investment Firms Directory", page_icon="🏦", layout="wide")

st.title("🏦 Investment Firms Directory")
st.markdown("Browse and search investment firms in the CRM.")

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# Helper to render investor details below the table
def display_investor_details(investor_id):
    org = db.query(Organization).options(
        joinedload(Organization.investor_profile),
        joinedload(Organization.fund_profiles),
        joinedload(Organization.program_profiles).joinedload(ProgramProfile.cohorts),
        joinedload(Organization.program_memberships).joinedload(ProgramMembership.cohort).joinedload(ProgramCohort.program),
        joinedload(Organization.roles).joinedload(PersonOrganizationRole.person)
    ).filter(Organization.id == int(investor_id)).first()

    if not org:
        st.error(f"Investor with ID {investor_id} not found.")
        return

    with st.container(border=True):
        st.subheader(f"🏦 {org.name}")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Basic Information")
            st.write(f"**Domain:** {org.primary_domain} | **Website:** {org.website_url}")
            st.write(f"**LinkedIn:** {org.linkedin_url}")
            st.write(f"**Location:** {org.city}, {org.state}, {org.country}")
            st.write(f"**Status:** {org.status}")
            if org.description:
                st.info(org.description)
                
            if org.investor_profile:
                st.divider()
                st.markdown("#### Investor Profile")
                st.write(f"**Investor Type:** {org.investor_profile.investor_type}")
                st.write(f"**Preferred Stage:** {org.investor_profile.preferred_stage}")
                
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
            role_data = []
            for role in org.roles:
                p = role.person
                if p:
                    name = p.full_name or f"{p.first_name} {p.last_name}"
                    emails = ", ".join([e.email for e in p.emails]) if p.emails else ""
                    location = f"{p.city or ''}, {p.state or ''}".strip(", ")
                    if location == ",": location = ""
                    
                    years = ""
                    if role.start_date:
                        start_str = role.start_date.strftime("%Y")
                        end_str = role.end_date.strftime("%Y") if role.end_date else "Present"
                        years = f"{start_str} - {end_str}"
                    elif role.end_date:
                        years = f"Until {role.end_date.strftime('%Y')}"
                        
                    role_data.append({
                        "Name": name,
                        "Title": role.title or "",
                        "Years": years,
                        "Seniority": role.seniority_level or "",
                        "Location": location,
                        "Email": emails,
                        "LinkedIn": p.linkedin_url or ""
                    })
            if role_data:
                st.dataframe(role_data, use_container_width=True, hide_index=True)
            else:
                st.info("No people linked to this investor.")
        else:
            st.info("No people linked to this investor.")

        # Linked Source Documents
        st.divider()
        st.subheader("📄 Linked Source Documents")
        
        doc_ids_subquery = db.query(SourceDocument.id).join(DocumentText).join(ExtractionJob).join(ExtractedEntity).join(EntityMatch).filter(
            EntityMatch.canonical_entity_type == "Organization",
            EntityMatch.canonical_entity_id == str(org.id)
        ).subquery()
        
        docs = db.query(SourceDocument).filter(SourceDocument.id.in_(doc_ids_subquery)).all()
        
        if docs:
            from market_comps.config import get_supabase_url
            import zoneinfo
            eastern = zoneinfo.ZoneInfo("America/New_York")
            
            for doc in docs:
                signed_url = get_supabase_url(doc.file_path) if doc.file_path else ""
                if signed_url:
                    url_display = f"{doc.source_url} [(View)]({signed_url})"
                else:
                    url_display = f"[{doc.source_url}]({doc.source_url})" if str(doc.source_url).startswith("http") else doc.source_url
                
                tz_time = doc.created_at.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(eastern).strftime('%Y-%m-%d %I:%M %p ET') if doc.created_at else "Unknown Time"
                st.markdown(f"- **{doc.document_type}**: {url_display} (Processed: {tz_time})")
        else:
            st.info("No documents linked to this investor.")

        # Raw Extracted Data
        st.divider()
        st.subheader("🧠 Raw Extracted Data")
        
        matches = db.query(EntityMatch).filter_by(
            canonical_entity_type="Organization",
            canonical_entity_id=str(org.id)
        ).all()
        
        extracted_entities = [m.extracted_entity for m in matches if m.extracted_entity]
        if extracted_entities:
            for ent in extracted_entities:
                job = ent.extraction_job
                schema_label = job.schema_name if job else "Unknown Schema"
                with st.expander(f"{schema_label} Data (Entity ID: {ent.id})"):
                    st.json(ent.extracted_payload_json)
        else:
            st.info("No raw extracted data found.")

        # Audit Trail
        st.divider()
        st.subheader("📜 Audit Trail")
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
search_query = st.text_input("Search Investors...", placeholder="Search by name, domain, or website...")

q = db.query(Organization).options(
    joinedload(Organization.investor_profile)
).filter(Organization.organization_type == "INVESTOR")

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
    if o.investor_profile:
        row["Inv Type"] = o.investor_profile.investor_type
        row["Pref Stage"] = o.investor_profile.preferred_stage
    data.append(row)

df = pd.DataFrame(data)

if df.empty:
    st.info("No investor records found.")
else:
    st.write("👆 *Select an investor row below to inspect full details.*")
    
    event = st.dataframe(
        df,
        key="grid_investors",
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
        selected_investor_id = df.iloc[selected_row_idx]["ID"]
        display_investor_details(selected_investor_id)
