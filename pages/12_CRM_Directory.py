import streamlit as st
from market_comps.db.session import get_db
from market_comps.db.models import (
    Organization, Person, CompanyProfile, InvestorProfile, 
    FundProfile, ProgramProfile, ProgramMembership, PersonOrganizationRole,
    PersonEmail, EntityAuditTrail
)
from sqlalchemy.orm import joinedload
from sqlalchemy import or_
import pandas as pd
import uuid

st.set_page_config(page_title="CRM Directory", page_icon="🗂️", layout="wide")
st.title("🗂️ CRM Directory")

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed. Did you configure secrets.toml? Error: {e}")
    st.stop()

def display_full_record_details(record_type, record_id):
    """Helper to display full record details inline."""
    if record_type == "ORGANIZATION":
        org = db.query(Organization).options(
            joinedload(Organization.company_profile),
            joinedload(Organization.investor_profile),
            joinedload(Organization.fund_profiles),
            joinedload(Organization.program_profiles),
            joinedload(Organization.program_memberships).joinedload(ProgramMembership.cohort),
            joinedload(Organization.roles).joinedload(PersonOrganizationRole.person)
        ).filter(Organization.id == int(record_id)).first()

        if not org:
            st.error(f"Organization with ID {record_id} not found.")
            return

        with st.container(border=True):
            st.subheader(f"🏢 {org.name}")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Domain:** {org.primary_domain} | **Website:** {org.website_url}")
                st.write(f"**LinkedIn:** {org.linkedin_url}")
                st.write(f"**Location:** {org.city}, {org.state}, {org.country}")
                if org.description:
                    st.info(org.description)

                if org.company_profile:
                    st.write(f"**Industry:** {org.company_profile.industry} | **Stage:** {org.company_profile.company_stage}")

            with col2:
                if org.program_memberships:
                    st.write("**Programs:**")
                    for m in org.program_memberships:
                        if m.cohort:
                            st.write(f"- 🎯 {m.cohort.program.program_name} ({m.cohort.cohort_name})")

            # People
            if org.roles:
                st.write("**People & Roles:**")
                for role in org.roles:
                    p = role.person
                    if p:
                        name = p.full_name or f"{p.first_name} {p.last_name}"
                        st.write(f"- 👤 **{name}** — {role.title or 'No Title'} {'([LinkedIn](' + p.linkedin_url + '))' if p.linkedin_url else ''}")
            
            # Audit Trail (Expandable)
            with st.expander("Audit Trail"):
                audit = db.query(EntityAuditTrail).filter_by(
                    entity_type="ORGANIZATION", entity_id=str(org.id)
                ).order_by(EntityAuditTrail.created_at.desc()).limit(10).all()
                if audit:
                    audit_data = []
                    for a in audit:
                        audit_data.append({
                            "Date": a.created_at.strftime("%Y-%m-%d %H:%M"),
                            "Action": a.audit_action,
                            "Field": a.field_name or "",
                            "Old": a.old_value or "",
                            "New": a.new_value or ""
                        })
                    st.dataframe(pd.DataFrame(audit_data), use_container_width=True, hide_index=True)

    elif record_type == "PERSON":
        person = db.query(Person).options(
            joinedload(Person.emails),
            joinedload(Person.roles).joinedload(PersonOrganizationRole.organization)
        ).filter(Person.id == (uuid.UUID(record_id) if isinstance(record_id, str) else record_id)).first()

        if not person:
            st.error(f"Person with ID {record_id} not found.")
            return

        with st.container(border=True):
            st.subheader(f"👤 {person.full_name or person.first_name + ' ' + person.last_name}")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**LinkedIn:** {person.linkedin_url}")
                st.write(f"**Location:** {person.city}, {person.state}")
                if person.bio:
                    st.info(person.bio)
            with col2:
                if person.roles:
                    st.write("**Roles:**")
                    for role in person.roles:
                        if role.organization:
                            st.write(f"- 🏢 **{role.organization.name}** — {role.title or 'No Title'}")

def get_org_df(org_type, search_text):
    q = db.query(Organization).options(
        joinedload(Organization.company_profile),
        joinedload(Organization.investor_profile),
    ).filter(Organization.organization_type == org_type)
    
    if search_text:
        search_filter = f"%{search_text}%"
        q = q.filter(
            or_(
                Organization.name.ilike(search_filter),
                Organization.normalized_name.ilike(search_filter),
                Organization.primary_domain.ilike(search_filter)
            )
        )
    orgs = q.order_by(Organization.created_at.desc()).limit(200).all()
    
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
        if o.investor_profile:
            row["Inv Type"] = o.investor_profile.investor_type
        data.append(row)
    
    return pd.DataFrame(data)

def render_directory_table(df, key, record_type="ORGANIZATION"):
    if df.empty:
        st.info("No records found.")
        return
    
    st.write("👆 *Select one or more rows to view details below.*")
    
    # Use native Streamlit selection with multi-row
    event = st.dataframe(
        df,
        key=key,
        on_select="rerun",
        selection_mode="multi-row",
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
        st.header(f"🔍 Details for {len(rows)} selected record(s)")
        for row_idx in rows:
            selected_row = df.iloc[row_idx]
            display_full_record_details(record_type, selected_row["ID"])

tab_companies, tab_investors, tab_people = st.tabs(["🏢 Companies", "🏦 Investors", "👤 People"])

# --- COMPANIES TAB ---
with tab_companies:
    c_search = st.text_input("Search Companies...", placeholder="e.g. Acme Corp", key="search_companies")
    df_companies = get_org_df("COMPANY", c_search)
    render_directory_table(df_companies, "grid_companies", "ORGANIZATION")

# --- INVESTORS TAB ---
with tab_investors:
    i_search = st.text_input("Search Investors...", placeholder="e.g. Sequoia", key="search_investors")
    df_investors = get_org_df("INVESTOR", i_search)
    render_directory_table(df_investors, "grid_investors", "ORGANIZATION")

# --- PEOPLE TAB ---
with tab_people:
    p_search = st.text_input("Search People...", placeholder="e.g. Jane Doe", key="search_people")
    
    q = db.query(Person)
    if p_search:
        search_filter = f"%{p_search}%"
        q = q.filter(
            or_(
                Person.full_name.ilike(search_filter),
                Person.first_name.ilike(search_filter),
                Person.last_name.ilike(search_filter)
            )
        )
    people = q.order_by(Person.created_at.desc()).limit(200).all()
    
    data = []
    for p in people:
        data.append({
            "ID": p.id,
            "Name": p.full_name or f"{p.first_name} {p.last_name}",
            "LinkedIn": p.linkedin_url,
            "City": p.city,
            "State": p.state,
            "Created": p.created_at.strftime("%Y-%m-%d") if p.created_at else ""
        })
    df_people = pd.DataFrame(data)
    render_directory_table(df_people, "grid_people", "PERSON")

