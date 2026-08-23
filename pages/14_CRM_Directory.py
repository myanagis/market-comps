import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from sqlalchemy import or_, func
from market_comps.db.session import get_db
from market_comps.db.models import (
    Organization, Person, ProgramProfile, ProgramCohort
)

st.set_page_config(page_title="Unified CRM Directory", page_icon="📁", layout="wide")
st.title("📁 Unified CRM Directory")

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

tab_search, tab_add_co, tab_add_firm, tab_add_person, tab_add_prog = st.tabs([
    "🔍 Directory Search", "Add Company", "Add Investment Firm", "Add Person", "Add Program & Cohort"
])

with tab_search:
    search_query = st.text_input("Search CRM...", placeholder="Search across companies, firms, people, and programs...", label_visibility="collapsed")
    st.divider()
    
    if not search_query:
        st.info("Start typing in the search bar above to find records.")
    else:
        col_co, col_firm, col_pe, col_pr = st.columns(4)
    
        # --- Companies ---
        with col_co:
            st.subheader("🏢 Companies")
            q_co = db.query(Organization).options(joinedload(Organization.company_profile)).filter(Organization.organization_type == "COMPANY")
            sf = f"%{search_query}%"
            q_co = q_co.filter(or_(Organization.name.ilike(sf), Organization.primary_domain.ilike(sf)))
            
            orgs = q_co.order_by(Organization.name).limit(100).all()
            st.caption(f"{len(orgs)} found")
            for o in orgs:
                with st.container(border=True):
                    st.markdown(f"<h5 style='margin-bottom:0; margin-top:0;'>{o.name}</h5>", unsafe_allow_html=True)
                    
                    meta_tags = []
                    if o.company_profile and o.company_profile.industry: meta_tags.append(o.company_profile.industry)
                    if o.city: meta_tags.append(o.city)
                    if o.company_profile and o.company_profile.company_stage: meta_tags.append(o.company_profile.company_stage)
                    
                    if o.ownership_type and o.ownership_type.upper() == "PUBLIC":
                        meta_tags.append("Public")
                    else:
                        meta_tags.append("Private")
                        
                    st.markdown(f"<div style='margin-bottom:8px; color:gray; font-size:0.8em;'>{' &middot; '.join(meta_tags)}</div>", unsafe_allow_html=True)
                    
                    domain_md = f"<a href='https://{o.primary_domain}' target='_blank'>{o.primary_domain}</a>" if o.primary_domain else ""
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.markdown(f"<div style='font-size:0.9em;'>{domain_md}</div>", unsafe_allow_html=True)
                    with c2:
                        st.page_link("pages/13_CRM_Company_Details.py", label="View profile →")
                        
        # --- Investment Firms ---
        with col_firm:
            st.subheader("🏦 Inv Firms")
            q_firm = db.query(Organization).options(joinedload(Organization.investor_profile)).filter(Organization.organization_type == "INVESTOR")
            sf = f"%{search_query}%"
            q_firm = q_firm.filter(or_(Organization.name.ilike(sf), Organization.primary_domain.ilike(sf)))
                
            firms = q_firm.order_by(Organization.name).limit(100).all()
            st.caption(f"{len(firms)} found")
            for f in firms:
                with st.container(border=True):
                    st.markdown(f"<h5 style='margin-bottom:0; margin-top:0;'>{f.name}</h5>", unsafe_allow_html=True)
                    
                    meta_tags = []
                    if f.investor_profile and f.investor_profile.investor_type: meta_tags.append(f.investor_profile.investor_type)
                    if f.city: meta_tags.append(f.city)
                    
                    st.markdown(f"<div style='margin-bottom:8px; color:gray; font-size:0.8em;'>{' &middot; '.join(meta_tags)}</div>", unsafe_allow_html=True)
                    
                    domain_md = f"<a href='https://{f.primary_domain}' target='_blank'>{f.primary_domain}</a>" if f.primary_domain else ""
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.markdown(f"<div style='font-size:0.9em;'>{domain_md}</div>", unsafe_allow_html=True)
                    with c2:
                        st.page_link("pages/15_CRM_Investment_Firm_Details.py", label="View profile →")
                        
        # --- People ---
        with col_pe:
            st.subheader("👤 People")
            q_pe = db.query(Person)
            sf = f"%{search_query}%"
            q_pe = q_pe.filter(or_(Person.full_name.ilike(sf), Person.first_name.ilike(sf), Person.last_name.ilike(sf)))
                
            people = q_pe.order_by(Person.first_name).limit(100).all()
            st.caption(f"{len(people)} found")
            for p in people:
                with st.container(border=True):
                    name = p.full_name or f"{p.first_name} {p.last_name}"
                    st.markdown(f"<h5 style='margin-bottom:0; margin-top:0;'>{name}</h5>", unsafe_allow_html=True)
                    
                    meta_tags = []
                    if p.city: meta_tags.append(p.city)
                    
                    st.markdown(f"<div style='margin-bottom:8px; color:gray; font-size:0.8em;'>{' &middot; '.join(meta_tags)}</div>", unsafe_allow_html=True)
                    
                    domain_md = f"<a href='{p.linkedin_url}' target='_blank'>LinkedIn</a>" if p.linkedin_url else ""
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.markdown(f"<div style='font-size:0.9em;'>{domain_md}</div>", unsafe_allow_html=True)
                    with c2:
                        st.page_link("pages/16_CRM_Person_Details.py", label="View profile →")
                        
        # --- Programs ---
        with col_pr:
            st.subheader("🚀 Programs")
            q_pr = db.query(ProgramProfile).options(joinedload(ProgramProfile.parent_organization))
            sf = f"%{search_query}%"
            q_pr = q_pr.filter(ProgramProfile.program_name.ilike(sf))
                
            programs = q_pr.order_by(ProgramProfile.program_name).limit(100).all()
            st.caption(f"{len(programs)} found")
            for p in programs:
                with st.container(border=True):
                    st.markdown(f"<h5 style='margin-bottom:0; margin-top:0;'>{p.program_name}</h5>", unsafe_allow_html=True)
                    
                    meta_tags = []
                    if p.program_type: meta_tags.append(p.program_type)
                    if p.parent_organization: meta_tags.append(p.parent_organization.name)
                    
                    st.markdown(f"<div style='margin-bottom:8px; color:gray; font-size:0.8em;'>{' &middot; '.join(meta_tags)}</div>", unsafe_allow_html=True)
                    
                    c1, c2 = st.columns([1, 1])
                    with c2:
                        st.page_link("pages/17_CRM_Program_Details.py", label="View profile →")


# --- ADD COMPANY ---
with tab_add_co:
    with st.form("company_form", clear_on_submit=True):
        st.subheader("Organization Details")
        c1, c2 = st.columns(2)
        name = c1.text_input("Company Name *")
        domain = c2.text_input("Primary Domain (Unique) *")
        city = c1.text_input("City")
        desc = st.text_area("Description")
        
        submitted = st.form_submit_button("Create Company")
        if submitted:
            if not name or not domain:
                st.error("Name and Domain are required.")
            else:
                from market_comps.db.models import Organization
                try:
                    org = Organization(name=name, normalized_name=name.lower(), primary_domain=domain, city=city, description=desc, organization_type="COMPANY")
                    db.add(org)
                    db.commit()
                    st.success(f"Successfully created company: {name}!")
                except Exception as e:
                    db.rollback()
                    st.error(f"Error saving to DB: {str(e)}")

# --- ADD FIRM ---
with tab_add_firm:
    with st.form("firm_form", clear_on_submit=True):
        st.subheader("Firm Details")
        c1, c2 = st.columns(2)
        name = c1.text_input("Firm Name *")
        domain = c2.text_input("Primary Domain (Unique) *")
        city = c1.text_input("City")
        desc = st.text_area("Description")
        
        submitted = st.form_submit_button("Create Firm")
        if submitted:
            if not name or not domain:
                st.error("Name and Domain are required.")
            else:
                from market_comps.db.models import Organization, InvestorProfile
                try:
                    org = Organization(name=name, normalized_name=name.lower(), primary_domain=domain, city=city, description=desc, organization_type="INVESTOR")
                    db.add(org)
                    db.flush()
                    prof = InvestorProfile(organization_id=org.id)
                    db.add(prof)
                    db.commit()
                    st.success(f"Successfully created firm: {name}!")
                except Exception as e:
                    db.rollback()
                    st.error(f"Error saving to DB: {str(e)}")

# --- ADD PERSON ---
with tab_add_person:
    with st.form("person_form", clear_on_submit=True):
        st.subheader("Person Details")
        c1, c2 = st.columns(2)
        first_name = c1.text_input("First Name *")
        last_name = c2.text_input("Last Name *")
        linkedin = st.text_input("LinkedIn URL")
        city = c1.text_input("City")
        bio = st.text_area("Bio")
        
        submitted = st.form_submit_button("Create Person")
        if submitted:
            if not first_name or not last_name:
                st.error("First and Last Name are required.")
            else:
                from market_comps.db.models import Person
                try:
                    full_name = f"{first_name} {last_name}"
                    p = Person(first_name=first_name, last_name=last_name, full_name=full_name, linkedin_url=linkedin, city=city, bio=bio)
                    db.add(p)
                    db.commit()
                    st.success(f"Successfully created person: {full_name}!")
                except Exception as e:
                    db.rollback()
                    st.error(f"Error saving to DB: {str(e)}")

# --- ADD PROGRAM ---
with tab_add_prog:
    orgs = db.query(Organization).order_by(Organization.name).all()
    org_options = {org.id: f"{org.name} ({org.primary_domain})" for org in orgs}
    
    with st.form("program_form", clear_on_submit=True):
        st.subheader("Program Details")
        parent_id = st.selectbox("Parent Organization", options=[None] + list(org_options.keys()), format_func=lambda x: org_options[x] if x else "-- None --")
        
        c1, c2 = st.columns(2)
        name = c1.text_input("Program Name *")
        p_type = c2.text_input("Program Type (e.g. ACCELERATOR, GRANT)")
        desc = st.text_area("Description")
        
        submitted = st.form_submit_button("Create Program")
        if submitted:
            if not name:
                st.error("Program Name is required.")
            else:
                from market_comps.db.models import ProgramProfile
                try:
                    record = ProgramProfile(parent_organization_id=parent_id, program_name=name, program_type=p_type, description=desc)
                    db.add(record)
                    db.commit()
                    st.success(f"Successfully created Program: {name}!")
                except Exception as e:
                    db.rollback()
                    st.error(f"Error saving to DB: {str(e)}")
