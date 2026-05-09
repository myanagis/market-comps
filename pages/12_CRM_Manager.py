import streamlit as st
from market_comps.db.session import get_db
from market_comps.db.models import Organization, CompanyProfile, InvestorProfile, Person, FundProfile, ProgramProfile
from sqlalchemy.orm import joinedload
from sqlalchemy import or_

st.set_page_config(page_title="CRM Manager", page_icon="🗄️", layout="wide")
st.title("🗄️ CRM Manager")

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed. Did you configure secrets.toml? Error: {e}")
    st.stop()

# --- SECTION 1: GLOBAL SEARCH ---
st.header("🔍 Global Search")
search_query = st.text_input("Search for an Organization or Person...", placeholder="e.g. Acme Corp, Jane Doe")

if search_query:
    query = f"%{search_query}%"
    
    # Search Organizations
    orgs = db.query(Organization).options(
        joinedload(Organization.company_profile),
        joinedload(Organization.investor_profile),
        joinedload(Organization.fund_profiles),
        joinedload(Organization.program_profiles)
    ).filter(
        or_(
            Organization.name.ilike(query),
            Organization.normalized_name.ilike(query),
            Organization.primary_domain.ilike(query)
        )
    ).all()
    
    # Search People
    people = db.query(Person).filter(
        or_(
            Person.full_name.ilike(query),
            Person.first_name.ilike(query),
            Person.last_name.ilike(query)
        )
    ).all()
    
    if not orgs and not people:
        st.info("No matching records found.")
    
    if orgs:
        st.subheader(f"🏢 Organizations ({len(orgs)})")
        for org in orgs:
            with st.expander(f"**{org.name}** ({org.organization_type or 'Unknown Type'})", expanded=False):
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

    if people:
        st.subheader(f"👤 People ({len(people)})")
        for p in people:
            with st.expander(f"**{p.full_name or p.first_name + ' ' + p.last_name}**", expanded=False):
                st.write(f"**LinkedIn:** {p.linkedin_url}")
                st.write(f"**Location:** {p.city}, {p.state}")
                if p.bio:
                    st.write(f"**Bio:** {p.bio}")

st.divider()

# --- SECTION 2: DATA ENTRY ---
st.header("➕ Add New Record")
entity_type = st.selectbox("Record Type", ["Company", "Investor", "Person", "Fund", "Program"])

if entity_type == "Company":
    with st.form("company_form", clear_on_submit=True):
        st.subheader("Organization Details")
        col1, col2 = st.columns(2)
        name = col1.text_input("Company Name *")
        domain = col2.text_input("Primary Domain (Unique) *")
        city = col1.text_input("City")
        desc = st.text_area("Description")
        
        st.subheader("Company Profile")
        col3, col4 = st.columns(2)
        founded = col3.number_input("Founded Year", min_value=1800, max_value=2100, value=2024)
        industry = col4.text_input("Industry")
        stage = col3.text_input("Company Stage")
        
        submitted = st.form_submit_button("Create Company")
        if submitted:
            if not name or not domain:
                st.error("Name and Domain are required.")
            else:
                try:
                    org = Organization(name=name, normalized_name=name.lower(), primary_domain=domain, city=city, description=desc, organization_type="COMPANY")
                    db.add(org)
                    db.flush() # get ID
                    
                    profile = CompanyProfile(organization_id=org.id, founded_year=founded, industry=industry, company_stage=stage)
                    db.add(profile)
                    db.commit()
                    st.success(f"Successfully created company: {name}!")
                except Exception as e:
                    db.rollback()
                    st.error(f"Error saving to DB: {str(e)}")

elif entity_type == "Investor":
    with st.form("investor_form", clear_on_submit=True):
        st.subheader("Organization Details")
        col1, col2 = st.columns(2)
        name = col1.text_input("Firm Name *")
        domain = col2.text_input("Primary Domain (Unique) *")
        city = col1.text_input("City")
        desc = st.text_area("Description")
        
        st.subheader("Investor Profile")
        col3, col4 = st.columns(2)
        inv_type = col3.selectbox("Investor Type", ["VC", "PE", "Angel", "CVC", "Family Office"])
        pref_stage = col4.text_input("Preferred Stage (e.g. Seed, Series A)")
        
        submitted = st.form_submit_button("Create Investor")
        if submitted:
            if not name or not domain:
                st.error("Name and Domain are required.")
            else:
                try:
                    org = Organization(name=name, normalized_name=name.lower(), primary_domain=domain, city=city, description=desc, organization_type="INVESTOR")
                    db.add(org)
                    db.flush()
                    
                    profile = InvestorProfile(organization_id=org.id, investor_type=inv_type, preferred_stage=pref_stage)
                    db.add(profile)
                    db.commit()
                    st.success(f"Successfully created investor: {name}!")
                except Exception as e:
                    db.rollback()
                    st.error(f"Error saving to DB: {str(e)}")

elif entity_type == "Person":
    with st.form("person_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        first_name = col1.text_input("First Name *")
        last_name = col2.text_input("Last Name *")
        linkedin = st.text_input("LinkedIn URL")
        city = col1.text_input("City")
        bio = st.text_area("Bio")
        
        submitted = st.form_submit_button("Create Person")
        if submitted:
            if not first_name or not last_name:
                st.error("First and Last Name are required.")
            else:
                try:
                    full_name = f"{first_name} {last_name}"
                    p = Person(first_name=first_name, last_name=last_name, full_name=full_name, linkedin_url=linkedin, city=city, bio=bio)
                    db.add(p)
                    db.commit()
                    st.success(f"Successfully created person: {full_name}!")
                except Exception as e:
                    db.rollback()
                    st.error(f"Error saving to DB: {str(e)}")

elif entity_type in ["Fund", "Program"]:
    # Fetch organizations for dropdown
    orgs = db.query(Organization).order_by(Organization.name).all()
    org_options = {org.id: f"{org.name} ({org.primary_domain})" for org in orgs}
    
    if not org_options:
        st.warning("You must create an Organization first before creating a Fund or Program.")
    else:
        with st.form(f"{entity_type.lower()}_form", clear_on_submit=True):
            parent_id = st.selectbox("Parent Organization *", options=list(org_options.keys()), format_func=lambda x: org_options[x])
            
            col1, col2 = st.columns(2)
            name = col1.text_input(f"{entity_type} Name *")
            
            if entity_type == "Fund":
                f_type = col2.text_input("Fund Type (e.g. Flagship, Opportunity)")
                vintage = col1.number_input("Vintage Year", min_value=1980, max_value=2100, value=2024)
                size = col2.text_input("Fund Size (e.g. 500M)")
            else:
                p_type = col2.text_input("Program Type (e.g. ACCELERATOR, GRANT)")
                
            desc = st.text_area("Description")
            
            submitted = st.form_submit_button(f"Create {entity_type}")
            if submitted:
                if not name:
                    st.error(f"{entity_type} Name is required.")
                else:
                    try:
                        if entity_type == "Fund":
                            record = FundProfile(parent_organization_id=parent_id, fund_name=name, fund_type=f_type, vintage_year=vintage, fund_size=size, description=desc)
                        else:
                            record = ProgramProfile(parent_organization_id=parent_id, program_name=name, program_type=p_type, description=desc)
                        
                        db.add(record)
                        db.commit()
                        st.success(f"Successfully created {entity_type}: {name}!")
                    except Exception as e:
                        db.rollback()
                        st.error(f"Error saving to DB: {str(e)}")
