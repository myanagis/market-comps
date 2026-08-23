import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from sqlalchemy import or_, func
from market_comps.db.session import get_db, get_db_context
from market_comps.db.models import (
    Organization, Person, CompanyProfile, FundProfile,
    ProgramMembership, PersonOrganizationRole, AuditTrail,
    ProgramProfile, ProgramCohort, EntityMatch, ExtractedEntity, ExtractionJob, PipelineRun, DocumentText, SourceDocument, FinancingRound, FinancingRoundFact, RoundInvestor, MetricType, MetricObservation,
    Market, MarketSegment, MarketSegmentCompanyLink, CompetitiveAnalysis, CompetitiveAnalysisSegment, CompetitiveAnalysisCompany
)
from market_comps.ingestion.reconciler import log_mutation
from market_comps.crm.competitor_manager import (
    get_company_segments, get_all_markets, get_market_segments, add_company_to_segment,
    get_or_create_competitive_analysis, add_competitive_analysis_company,
    THREAT_LEVELS, RELATIONSHIP_TYPES
)

st.set_page_config(page_title="Companies Directory", page_icon="🏢", layout="wide")
st.title("🏢 Companies Directory")

tab_dir, tab_add = st.tabs(["Companies Directory", "Add Company"])

# Get database session
try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

def get_all_company_themes(db):
    profiles = db.query(CompanyProfile.themes).filter(CompanyProfile.themes.is_not(None)).all()
    themes = set()
    for (t,) in profiles:
        if t and isinstance(t, list):
            themes.update(t)
    return sorted(list(themes))

# Fetch and query data
with tab_dir:
    st.markdown("### Search")
    search_query = st.text_input("Search Companies...", placeholder="Search by name, domain, or website...", label_visibility="collapsed")
    st.divider()
    
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

    # Sort alphabetically
    orgs = q.order_by(Organization.name).limit(100).all()

    if not orgs:
        st.info("No company records found.")
    else:
        st.markdown(f"**Showing {len(orgs)} companies**")
        
        # Build Custom Narrow List View
        col_list, col_pad = st.columns([2, 1])
        with col_list:
            for o in orgs:
                with st.container(border=True):
                    st.markdown(f"<h4 style='margin-bottom:0; margin-top:0;'>{o.name}</h4>", unsafe_allow_html=True)
                    
                    # Construct meta tags
                    meta_tags = []
                    if o.company_profile and o.company_profile.industry: 
                        meta_tags.append(o.company_profile.industry)
                        
                    locs = [l for l in [o.city, o.state, o.country] if l]
                    if locs:
                        meta_tags.append(", ".join(locs))
                        
                    if o.company_profile and o.company_profile.company_stage: 
                        meta_tags.append(o.company_profile.company_stage)
                        
                    if o.ownership_type and o.ownership_type.upper() == "PUBLIC":
                        ticker_str = f" ({o.exchange}: {o.ticker})" if o.ticker and o.exchange else (f" ({o.ticker})" if o.ticker else "")
                        meta_tags.append(f"Public{ticker_str}")
                    else:
                        meta_tags.append("Private")
                        
                    st.markdown(f"<div style='margin-bottom:8px; color:gray; font-size:0.9em;'>{' &middot; '.join(meta_tags)}</div>", unsafe_allow_html=True)
                    
                    domain_md = f"[{o.primary_domain}](https://{o.primary_domain})" if o.primary_domain else ""
                    st.markdown(f"<div style='font-size:0.95em;'>{domain_md} &nbsp;&nbsp;&nbsp; <a href='/company?id={o.id}' target='_self'>View profile →</a></div>", unsafe_allow_html=True)

with tab_add:
    with st.form("company_form", clear_on_submit=True):
        st.subheader("Organization Details")
        col1, col2 = st.columns(2)
        name = col1.text_input("Company Name *")
        domain = col2.text_input("Primary Domain (Unique) *")
        city = col1.text_input("City")
        desc = st.text_area("Description")
        
        st.subheader("Company Profile")
        col3, col4 = st.columns(2)
        founded = col3.number_input("Founded Year", min_value=1800, max_value=2100, value=None, placeholder="YYYY")
        industry = col4.text_input("Industry")
        stage = col3.text_input("Company Stage")
        
        all_themes = get_all_company_themes(db)
        selected_themes = st.multiselect("Themes", options=all_themes)
        new_themes = st.text_input("Add New Themes (comma separated)")
        
        from market_comps.db.models import FinancingRound, FinancingRoundFact, RoundInvestor
        st.subheader("Add Investment Firm (Optional)")
        investors = db.query(Organization).filter_by(organization_type="INVESTOR").order_by(Organization.name).all()
        investor_opts = {0: "-- None --"}
        investor_opts.update({i.id: i.name for i in investors})
        
        col5, col6 = st.columns(2)
        linked_investor_id = col5.selectbox("Select Investment Firm", options=list(investor_opts.keys()), format_func=lambda x: investor_opts[x])
        inv_round = col6.text_input("Round (e.g. Seed, Series A)")
        inv_amount = col5.text_input("Amount (e.g. $2M)")
        inv_date = col6.date_input("Investment Date", value=None)
        
        submitted = st.form_submit_button("Create Company")
        if submitted:
            if not name or not domain:
                st.error("Name and Domain are required.")
            else:
                try:
                    org = db.query(Organization).filter_by(primary_domain=domain).first()
                    action_str = "updated" if org else "created"
                    
                    if org:
                        org.name = name
                        org.normalized_name = name.lower()
                        org.city = city
                        if desc: org.description = desc
                        org.organization_type = "COMPANY"
                    else:
                        org = Organization(name=name, normalized_name=name.lower(), primary_domain=domain, city=city, description=desc, organization_type="COMPANY")
                        db.add(org)
                        db.flush()
                        log_mutation(db, "ORGANIZATION", str(org.id), "CREATE", source="USER_EDIT", created_by=st.session_state.get("user_email", "SYSTEM"))
                        
                    db.flush() # get ID
                    
                    profile = db.query(CompanyProfile).filter_by(organization_id=org.id).first()
                    
                    final_themes = list(selected_themes)
                    if new_themes:
                        final_themes.extend([t.strip() for t in new_themes.split(",") if t.strip()])
                    final_themes = list(set(final_themes))
                    
                    if profile:
                        if founded: profile.founded_year = founded
                        if industry: profile.industry = industry
                        if stage: profile.company_stage = stage
                        if final_themes: profile.themes = final_themes
                    else:
                        profile = CompanyProfile(organization_id=org.id, founded_year=founded, industry=industry, company_stage=stage, themes=final_themes if final_themes else None)
                        db.add(profile)
                        db.flush()
                        log_mutation(db, "COMPANY_PROFILE", str(profile.id), "CREATE", source="USER_EDIT", created_by=st.session_state.get("user_email", "SYSTEM"))
                        
                    db.commit()
                    
                    if linked_investor_id != 0:
                        rnd = FinancingRound(
                            company_id=org.id,
                            round_name=inv_round,
                            status='closed'
                        )
                        db.add(rnd)
                        db.flush()
                        
                        if inv_amount:
                            fact = FinancingRoundFact(
                                financing_round_id=rnd.id,
                                fact_type='amount_raised',
                                value_text=inv_amount,
                                certainty='company_stated',
                                value_date=inv_date
                            )
                            db.add(fact)
                            
                        rinv = RoundInvestor(
                            financing_round_id=rnd.id,
                            investor_id=linked_investor_id,
                            role='participant',
                            status='invested',
                            notes=f"Amount: {inv_amount}" if inv_amount else None
                        )
                        db.add(rinv)
                        db.commit()
                        st.success(f"Successfully added investment from {investor_opts[linked_investor_id]}!")
                        
                    st.success(f"Successfully {action_str} company: {name}!")
                except Exception as e:
                    db.rollback()
                    st.error(f"Error saving to DB: {str(e)}")
