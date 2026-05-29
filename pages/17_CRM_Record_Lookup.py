import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from sqlalchemy import or_
from market_comps.db.session import get_db
from market_comps.db.models import (
    Organization, Person, CompanyProfile, InvestorProfile,
    Investment, PersonOrganizationRole
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
        joinedload(Organization.investments_made).joinedload(Investment.company),
        joinedload(Organization.investments_made).joinedload(Investment.fund),
        joinedload(Organization.investments_received).joinedload(Investment.investor),
        joinedload(Organization.investments_received).joinedload(Investment.fund)
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
                    
                    if c.investments_received:
                        st.markdown("**Investments Received:**")
                        for inv in c.investments_received:
                            fund_str = f" via {inv.fund.fund_name}" if inv.fund else ""
                            st.write(f"- **{inv.investor.name if inv.investor else 'Unknown'}**{fund_str} ({inv.round_type or 'Unknown Round'})")
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
                            fund_str = f" (via {inv.fund.fund_name})" if inv.fund else ""
                            comp_name = inv.company.name if inv.company else "Unknown"
                            inv_str = f"- **{comp_name}**{fund_str}"
                            if inv.investment_date: inv_str += f" on {inv.investment_date.strftime('%Y-%m-%d')}"
                            if inv.amount: inv_str += f" ({inv.amount})"
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
