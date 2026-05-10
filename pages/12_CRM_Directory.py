import streamlit as st
from market_comps.db.session import get_db
from market_comps.db.models import Organization, Person, EntityAuditTrail
from sqlalchemy.orm import joinedload
from sqlalchemy import or_

st.set_page_config(page_title="CRM Directory", page_icon="🗂️", layout="wide")
st.title("🗂️ CRM Directory")

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed. Did you configure secrets.toml? Error: {e}")
    st.stop()

tab_companies, tab_investors, tab_people = st.tabs(["🏢 Companies", "🏦 Investors", "👤 People"])

def display_orgs(orgs):
    if not orgs:
        st.info("No matching organizations found.")
        return
        
    st.caption(f"Showing {len(orgs)} organizations")
    for org in orgs:
        with st.expander(f"**{org.name}**", expanded=False):
            st.write(f"**Domain:** {org.primary_domain} | **City:** {org.city}")
            
            if org.company_profile:
                st.caption("COMPANY PROFILE")
                st.write(f"- Industry: {org.company_profile.industry} | Stage: {org.company_profile.company_stage}")
                
            if org.investor_profile:
                st.caption("INVESTOR PROFILE")
                st.write(f"- Type: {org.investor_profile.investor_type} | Preferred Stage: {org.investor_profile.preferred_stage}")
            
            if org.fund_profiles:
                st.caption("FUNDS")
                for fund in org.fund_profiles:
                    st.write(f"- 💰 **{fund.fund_name}** ({fund.vintage_year}) — {fund.fund_size}")
            
            if org.program_profiles:
                st.caption("PROGRAMS")
                for prog in org.program_profiles:
                    st.write(f"- 🚀 **{prog.program_name}** ({prog.program_type})")

            # Show audit trail
            audit = db.query(EntityAuditTrail).filter_by(
                entity_type="ORGANIZATION", entity_id=str(org.id)
            ).order_by(EntityAuditTrail.created_at.desc()).limit(5).all()
            if audit:
                st.caption("RECENT CHANGES (Audit Trail)")
                for a in audit:
                    field_str = f" ({a.field_name})" if a.field_name else ""
                    value_str = f": {a.old_value} → {a.new_value}" if a.field_name else ""
                    st.write(f"- [{a.created_at.strftime('%Y-%m-%d %H:%M')}] **{a.audit_action}**{field_str}{value_str} — {a.reason or ''}")

def get_org_query(org_type, search_text):
    q = db.query(Organization).options(
        joinedload(Organization.company_profile),
        joinedload(Organization.investor_profile),
        joinedload(Organization.fund_profiles),
        joinedload(Organization.program_profiles),
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
    return q.order_by(Organization.created_at.desc()).limit(50).all()

# --- COMPANIES TAB ---
with tab_companies:
    c_search = st.text_input("Search Companies...", placeholder="e.g. Acme Corp", key="search_companies")
    orgs = get_org_query("COMPANY", c_search)
    display_orgs(orgs)

# --- INVESTORS TAB ---
with tab_investors:
    i_search = st.text_input("Search Investors...", placeholder="e.g. Sequoia", key="search_investors")
    orgs = get_org_query("INVESTOR", i_search)
    display_orgs(orgs)

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
    people = q.order_by(Person.created_at.desc()).limit(50).all()
    
    if not people:
        st.info("No matching people found.")
    else:
        st.caption(f"Showing {len(people)} people")
        for p in people:
            with st.expander(f"**{p.full_name or p.first_name + ' ' + p.last_name}**", expanded=False):
                st.write(f"**LinkedIn:** {p.linkedin_url}")
                st.write(f"**Location:** {p.city}, {p.state}")
                if p.bio:
                    st.write(f"**Bio:** {p.bio}")
