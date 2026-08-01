import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from sqlalchemy import or_
from market_comps.db.session import get_db
from market_comps.db.models import (
    Organization, Person, CompanyProfile, InvestorProfile,
    FinancingRound, RoundInvestor, PersonOrganizationRole
)

st.set_page_config(page_title="CRM Record Lookup", page_icon="🔍", layout="wide")
st.title("🔍 CRM Record Lookup")
st.markdown("Search across all Companies, Investors, and People in the CRM.")

try:
    db = next(get_db())
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

search_query = st.text_input("Enter search term (name, domain, etc.)", placeholder="e.g. Acme Corp, Jane Doe, a16z")

if search_query:
    st.divider()
    search_filter = f"%{search_query}%"
    
    # --- SEARCH ORGANIZATIONS ---
    orgs = db.query(Organization).options(
        joinedload(Organization.company_profile),
        joinedload(Organization.investor_profile),
        joinedload(Organization.investments_made).joinedload(RoundInvestor.round).joinedload(FinancingRound.company),
        joinedload(Organization.financing_rounds).joinedload(FinancingRound.investors).joinedload(RoundInvestor.investor)
    ).filter(
        or_(
            Organization.name.ilike(search_filter),
            Organization.normalized_name.ilike(search_filter),
            Organization.primary_domain.ilike(search_filter)
        )
    ).order_by(Organization.created_at.desc()).limit(50).all()
    
    # --- SEARCH PEOPLE ---
    people = db.query(Person).options(
        joinedload(Person.roles).joinedload(PersonOrganizationRole.organization)
    ).filter(
        or_(
            Person.full_name.ilike(search_filter),
            Person.first_name.ilike(search_filter),
            Person.last_name.ilike(search_filter)
        )
    ).order_by(Person.created_at.desc()).limit(50).all()
    
    total_results = len(orgs) + len(people)
    
    if total_results == 0:
        st.info(f"No records found matching '{search_query}'.")
    else:
        st.success(f"Found {total_results} matching records.")
        
        # Display Companies
        companies = [o for o in orgs if o.organization_type == "COMPANY"]
        if companies:
            st.subheader("🏢 Companies")
            for c in companies:
                with st.container(border=True):
                    st.markdown(f"**{c.name}**")
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

        # Display Investors
        investors = [o for o in orgs if o.organization_type == "INVESTOR"]
        if investors:
            st.subheader("🏦 Investment Firms")
            for i in investors:
                with st.container(border=True):
                    st.markdown(f"**{i.name}**")
                    itype = i.investor_profile.investor_type if i.investor_profile else "N/A"
                    pstage = i.investor_profile.preferred_stage if i.investor_profile else "N/A"
                    st.caption(f"Domain: {i.primary_domain or 'N/A'} | Type: {itype} | Pref Stage: {pstage}")
                    
                    if i.investments_made:
                        st.markdown("**Sample Investments Made:**")
                        for inv in i.investments_made[:5]: # show up to 5
                            comp_name = inv.round.company.name if inv.round and inv.round.company else "Unknown"
                            rnd_name = inv.round.round_name if inv.round else "Unknown Round"
                            inv_str = f"- **{comp_name}** ({rnd_name})"
                            if inv.amount_numeric: inv_str += f" ({inv.currency_code or ''} {inv.amount_numeric})"
                            st.write(inv_str)
                    else:
                        st.write("**Investments:** None recorded")

        # Display People
        if people:
            st.subheader("👤 People")
            for p in people:
                with st.container(border=True):
                    st.markdown(f"**{p.full_name or p.first_name + ' ' + p.last_name}**")
                    st.caption(f"Location: {p.city or 'N/A'}, {p.state or 'N/A'}")
                    
                    if p.roles:
                        st.write("**Roles:**")
                        for r in p.roles:
                            curr_badge = "🟢 (Current)" if r.is_current else "⚪ (Past)"
                            org_name = r.organization.name if r.organization else "Unknown Org"
                            st.write(f"- {r.title or 'Unknown Title'} at **{org_name}** {curr_badge}")
                    else:
                        st.write("**Roles:** None recorded")
