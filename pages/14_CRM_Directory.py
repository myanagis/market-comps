import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from sqlalchemy import or_, func
from market_comps.db.session import get_db
from market_comps.db.models import (
    Organization, Person, ProgramProfile, ProgramCohort,
    CompanyProfile, InvestorProfile, FinancingRound, RoundInvestor, PersonOrganizationRole
)

st.set_page_config(page_title="Unified CRM Directory", page_icon="📁", layout="wide")
st.title("📁 Unified CRM Directory")

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

col_search1, col_search2 = st.columns([3, 1])
with col_search1:
    search_query = st.text_input("Search CRM...", placeholder="Search across companies, firms, people, and programs...", label_visibility="collapsed")
with col_search2:
    date_added = st.date_input("Date Added (Created On)", value=None, help="Filter by when the record was created in the system")
    
st.divider()

tab_all, tab_co, tab_firm, tab_pe, tab_pr, tab_add_co, tab_add_firm, tab_add_person, tab_add_prog = st.tabs([
    "🔍 All", "🏢 Companies", "🏦 Investors", "👤 People", "🚀 Programs", "Add Company", "Add Firm", "Add Person", "Add Program"
])

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

with tab_all:
    if not search_query and not date_added:
        st.info("Start typing in the search bar above to find detailed records.")
    else:
        sf = f"%{search_query}%" if search_query else "%"
        
        # --- SEARCH ORGANIZATIONS (Record Lookup style) ---
        q_orgs = db.query(Organization).options(
            joinedload(Organization.company_profile),
            joinedload(Organization.investor_profile),
            joinedload(Organization.investments_made).joinedload(RoundInvestor.round).joinedload(FinancingRound.company),
            joinedload(Organization.financing_rounds).joinedload(FinancingRound.investors).joinedload(RoundInvestor.investor)
        ).filter(
            or_(
                Organization.name.ilike(sf),
                Organization.normalized_name.ilike(sf),
                Organization.primary_domain.ilike(sf)
            )
        )
        if date_added: q_orgs = q_orgs.filter(func.date(Organization.created_at) == date_added)
        orgs = q_orgs.order_by(Organization.created_at.desc()).limit(50).all()
        
        # --- SEARCH PEOPLE (Record Lookup style) ---
        q_peop = db.query(Person).options(
            joinedload(Person.roles).joinedload(PersonOrganizationRole.organization)
        ).filter(
            or_(
                Person.full_name.ilike(sf),
                Person.first_name.ilike(sf),
                Person.last_name.ilike(sf)
            )
        )
        if date_added: q_peop = q_peop.filter(func.date(Person.created_at) == date_added)
        people = q_peop.order_by(Person.created_at.desc()).limit(50).all()
        
        total_results = len(orgs) + len(people)
        
        if total_results == 0:
            st.info("No records found.")
        else:
            st.success(f"Found {total_results} matching records.")
            
            companies = [o for o in orgs if o.organization_type == "COMPANY"]
            if companies:
                st.subheader("🏢 Companies")
                for c in companies:
                    with st.container(border=True):
                        st.markdown(f"**[{c.name}](/company?id={c.id})**")
                        stage = c.company_profile.company_stage if c.company_profile else "N/A"
                        ind = c.company_profile.industry if c.company_profile else "N/A"
                        st.caption(f"Domain: {c.primary_domain or 'N/A'} | Stage: {stage} | Industry: {ind}")
                        
                        if c.financing_rounds:
                            st.markdown("**Financing Rounds:**")
                            for rnd in c.financing_rounds:
                                investors = ", ".join([inv.investor.name for inv in rnd.investors if inv.investor])
                                st.write(f"- **{rnd.round_name or 'Unknown Round'}**: {investors or 'No investors recorded'}")
                        else:
                            st.write("**Investors:** None recorded")

            investors = [o for o in orgs if o.organization_type == "INVESTOR"]
            if investors:
                st.subheader("🏦 Investment Firms")
                for i in investors:
                    with st.container(border=True):
                        st.markdown(f"**[{i.name}](/investment_firm?id={i.id})**")
                        itype = i.investor_profile.investor_type if i.investor_profile else "N/A"
                        pstage = i.investor_profile.preferred_stage if i.investor_profile else "N/A"
                        st.caption(f"Domain: {i.primary_domain or 'N/A'} | Type: {itype} | Pref Stage: {pstage}")
                        
                        if i.investments_made:
                            st.markdown("**Sample Investments Made:**")
                            for inv in i.investments_made[:5]:
                                comp_name = inv.round.company.name if inv.round and inv.round.company else "Unknown"
                                rnd_name = inv.round.round_name if inv.round else "Unknown Round"
                                inv_str = f"- **{comp_name}** ({rnd_name})"
                                if inv.amount_numeric: inv_str += f" ({inv.currency_code or ''} {inv.amount_numeric})"
                                st.write(inv_str)
                        else:
                            st.write("**Investments:** None recorded")

            if people:
                st.subheader("👤 People")
                for p in people:
                    with st.container(border=True):
                        st.markdown(f"**[{p.full_name or p.first_name + ' ' + p.last_name}](/person?id={p.id})**")
                        st.caption(f"Location: {p.city or 'N/A'}, {p.state or 'N/A'}")
                        
                        if p.roles:
                            st.write("**Roles:**")
                            for r in p.roles:
                                curr_badge = "🟢 (Current)" if r.is_current else "⚪ (Past)"
                                org_name = r.organization.name if r.organization else "Unknown Org"
                                st.write(f"- {r.title or 'Unknown Title'} at **{org_name}** {curr_badge}")
                        else:
                            st.write("**Roles:** None recorded")

with tab_co:
    st.subheader("🏢 Companies")
    q_co = db.query(Organization).options(joinedload(Organization.company_profile)).filter(Organization.organization_type == "COMPANY")
    if search_query:
        sf = f"%{search_query}%"
        q_co = q_co.filter(or_(Organization.name.ilike(sf), Organization.primary_domain.ilike(sf)))
    if date_added:
        q_co = q_co.filter(func.date(Organization.created_at) == date_added)
        
    orgs = q_co.order_by(Organization.name).limit(100).all()
    st.caption(f"{len(orgs)} found")
    
    for row in chunks(orgs, 3):
        cols = st.columns(3)
        for i, o in enumerate(row):
            with cols[i]:
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
                    with c1: st.markdown(f"<div style='font-size:0.9em;'>{domain_md}</div>", unsafe_allow_html=True)
                    with c2: st.markdown(f"[**View profile →**](/company?id={o.id})")

with tab_firm:
    st.subheader("🏦 Investment Firms")
    q_firm = db.query(Organization).options(joinedload(Organization.investor_profile)).filter(Organization.organization_type == "INVESTOR")
    if search_query:
        sf = f"%{search_query}%"
        q_firm = q_firm.filter(or_(Organization.name.ilike(sf), Organization.primary_domain.ilike(sf)))
    if date_added:
        q_firm = q_firm.filter(func.date(Organization.created_at) == date_added)
        
    firms = q_firm.order_by(Organization.name).limit(100).all()
    st.caption(f"{len(firms)} found")
    
    for row in chunks(firms, 3):
        cols = st.columns(3)
        for i, f in enumerate(row):
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"<h5 style='margin-bottom:0; margin-top:0;'>{f.name}</h5>", unsafe_allow_html=True)
                    meta_tags = []
                    if f.investor_profile and f.investor_profile.investor_type: meta_tags.append(f.investor_profile.investor_type)
                    if f.city: meta_tags.append(f.city)
                    st.markdown(f"<div style='margin-bottom:8px; color:gray; font-size:0.8em;'>{' &middot; '.join(meta_tags)}</div>", unsafe_allow_html=True)
                    domain_md = f"<a href='https://{f.primary_domain}' target='_blank'>{f.primary_domain}</a>" if f.primary_domain else ""
                    c1, c2 = st.columns([1, 1])
                    with c1: st.markdown(f"<div style='font-size:0.9em;'>{domain_md}</div>", unsafe_allow_html=True)
                    with c2: st.markdown(f"[**View profile →**](/investment_firm?id={f.id})")

with tab_pe:
    st.subheader("👤 People")
    q_pe = db.query(Person)
    if search_query:
        sf = f"%{search_query}%"
        q_pe = q_pe.filter(or_(Person.full_name.ilike(sf), Person.first_name.ilike(sf), Person.last_name.ilike(sf)))
    if date_added:
        q_pe = q_pe.filter(func.date(Person.created_at) == date_added)
        
    people = q_pe.order_by(Person.first_name).limit(100).all()
    st.caption(f"{len(people)} found")
    
    for row in chunks(people, 3):
        cols = st.columns(3)
        for i, p in enumerate(row):
            with cols[i]:
                with st.container(border=True):
                    name = p.full_name or f"{p.first_name} {p.last_name}"
                    st.markdown(f"<h5 style='margin-bottom:0; margin-top:0;'>{name}</h5>", unsafe_allow_html=True)
                    meta_tags = []
                    if p.city: meta_tags.append(p.city)
                    st.markdown(f"<div style='margin-bottom:8px; color:gray; font-size:0.8em;'>{' &middot; '.join(meta_tags)}</div>", unsafe_allow_html=True)
                    domain_md = f"<a href='{p.linkedin_url}' target='_blank'>LinkedIn</a>" if p.linkedin_url else ""
                    c1, c2 = st.columns([1, 1])
                    with c1: st.markdown(f"<div style='font-size:0.9em;'>{domain_md}</div>", unsafe_allow_html=True)
                    with c2: st.markdown(f"[**View profile →**](/person?id={p.id})")

with tab_pr:
    st.subheader("🚀 Programs")
    q_pr = db.query(ProgramProfile).options(joinedload(ProgramProfile.parent_organization))
    if search_query:
        sf = f"%{search_query}%"
        q_pr = q_pr.filter(ProgramProfile.program_name.ilike(sf))
        
    programs = q_pr.order_by(ProgramProfile.program_name).limit(100).all()
    st.caption(f"{len(programs)} found")
    
    for row in chunks(programs, 3):
        cols = st.columns(3)
        for i, p in enumerate(row):
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"<h5 style='margin-bottom:0; margin-top:0;'>{p.program_name}</h5>", unsafe_allow_html=True)
                    meta_tags = []
                    if p.program_type: meta_tags.append(p.program_type)
                    if p.parent_organization: meta_tags.append(p.parent_organization.name)
                    st.markdown(f"<div style='margin-bottom:8px; color:gray; font-size:0.8em;'>{' &middot; '.join(meta_tags)}</div>", unsafe_allow_html=True)
                    c1, c2 = st.columns([1, 1])
                    with c2: st.markdown(f"[**View profile →**](/program?id={p.id})")

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
