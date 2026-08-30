import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from sqlalchemy import or_
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

st.set_page_config(page_title="Company Details", page_icon="🏢", layout="wide")
st.title("🏢 Company Details")

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

def get_all_sectors(db):
    from market_comps.db.models import Sector
    sectors = db.query(Sector).order_by(Sector.name).all()
    return [s.name for s in sectors]

@st.dialog("Edit Company")
def edit_company_dialog(org):
    with st.form("edit_company"):
        st.write(f"Edit details for **{org.name}**")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", value=org.name or "")
            domain = st.text_input("Domain", value=org.primary_domain or "")
            website = st.text_input("Website URL", value=org.website_url or "")
            linkedin = st.text_input("LinkedIn URL", value=org.linkedin_url or "")
        with col2:
            city = st.text_input("City", value=org.city or "")
            state = st.text_input("State", value=org.state or "")
            country = st.text_input("Country", value=org.country or "")
            status_opts = ["ACTIVE", "INACTIVE", "ACQUIRED", "DEFUNCT"]
            status = st.selectbox("Status", status_opts, index=status_opts.index(org.status) if org.status in status_opts else 0)
            ownership_opts = ["PRIVATE", "PUBLIC"]
            ownership_type = st.selectbox("Ownership", ownership_opts, index=ownership_opts.index((org.ownership_type or "PRIVATE").upper()) if (org.ownership_type or "PRIVATE").upper() in ownership_opts else 0)
            ticker = st.text_input("Ticker", value=org.ticker or "")
            exchange = st.text_input("Exchange", value=org.exchange or "")
            
        desc = st.text_area("Description", value=org.description or "")
        
        st.divider()
        st.write("Company Profile")
        col3, col4 = st.columns(2)
        prof = org.company_profile
        with col3:
            industry = st.text_input("Industry", value=prof.industry if prof else "")
            subind = st.text_input("Sub-Industry", value=prof.subindustry if prof else "")
        with col4:
            stage = st.text_input("Company Stage", value=prof.company_stage if prof else "")
            founded = st.number_input("Founded Year", value=prof.founded_year if prof and prof.founded_year else None, step=1, placeholder="YYYY")
            
        all_themes = get_all_company_themes(db)
        selected_themes = st.multiselect("Themes", options=all_themes, default=prof.themes if prof and prof.themes else [])
        new_themes = st.text_input("Add New Themes (comma separated)")
        
        all_sectors = get_all_sectors(db)
        selected_sectors = st.multiselect("Sectors", options=all_sectors, default=prof.sectors if prof and prof.sectors else [])
        new_sectors = st.text_input("Add New Sectors (comma separated)")
            
        if st.form_submit_button("Save Changes"):
            user = st.session_state.get("user_email", "SYSTEM")
            
            def check_and_update(entity_type, entity_id, field_name, old_val, new_val, obj):
                if str(old_val) != str(new_val) and (old_val or new_val):
                    log_mutation(
                        db, entity_type, entity_id, "UPDATE",
                        field_name=field_name,
                        old_value=str(old_val),
                        new_value=str(new_val),
                        source="USER_EDIT",
                        created_by=user
                    )
                    setattr(obj, field_name, new_val)
                    
            check_and_update("ORGANIZATION", org.id, "name", org.name, name, org)
            check_and_update("ORGANIZATION", org.id, "primary_domain", org.primary_domain, domain, org)
            check_and_update("ORGANIZATION", org.id, "website_url", org.website_url, website, org)
            check_and_update("ORGANIZATION", org.id, "linkedin_url", org.linkedin_url, linkedin, org)
            check_and_update("ORGANIZATION", org.id, "city", org.city, city, org)
            check_and_update("ORGANIZATION", org.id, "state", org.state, state, org)
            check_and_update("ORGANIZATION", org.id, "country", org.country, country, org)
            check_and_update("ORGANIZATION", org.id, "status", org.status, status, org)
            check_and_update("ORGANIZATION", org.id, "ownership_type", org.ownership_type, ownership_type, org)
            check_and_update("ORGANIZATION", org.id, "ticker", org.ticker, ticker, org)
            check_and_update("ORGANIZATION", org.id, "exchange", org.exchange, exchange, org)
            check_and_update("ORGANIZATION", org.id, "description", org.description, desc, org)
            
            if not prof:
                prof = CompanyProfile(organization_id=org.id)
                db.add(prof)
                db.flush()
                log_mutation(
                    db, "COMPANY_PROFILE", prof.id, "CREATE",
                    source="USER_EDIT", created_by=user
                )
                org.company_profile = prof
                
            check_and_update("COMPANY_PROFILE", prof.id, "industry", prof.industry, industry, prof)
            check_and_update("COMPANY_PROFILE", prof.id, "subindustry", prof.subindustry, subind, prof)
            check_and_update("COMPANY_PROFILE", prof.id, "company_stage", prof.company_stage, stage, prof)
            check_and_update("COMPANY_PROFILE", prof.id, "founded_year", prof.founded_year, founded if founded else None, prof)
            
            final_themes = list(selected_themes)
            if new_themes:
                final_themes.extend([t.strip() for t in new_themes.split(",") if t.strip()])
            final_themes = list(set(final_themes))
            
            final_sectors = list(selected_sectors)
            if new_sectors:
                new_sec_list = [s.strip() for s in new_sectors.split(",") if s.strip()]
                final_sectors.extend(new_sec_list)
                from market_comps.db.models import Sector
                for ns in new_sec_list:
                    if not db.query(Sector).filter_by(name=ns).first():
                        db.add(Sector(name=ns))
            final_sectors = list(set(final_sectors))
            
            check_and_update("COMPANY_PROFILE", prof.id, "themes", prof.themes, final_themes, prof)
            check_and_update("COMPANY_PROFILE", prof.id, "sectors", prof.sectors, final_sectors, prof)
            
            db.commit()
            st.rerun()

@st.dialog("Metric History")
def view_metric_history_dialog(company_id, metric_type_code):
    metric = db.query(MetricType).filter_by(code=metric_type_code).first()
    if not metric:
        st.error("Metric type not found.")
        return
        
    st.write(f"### {metric.display_name} History")
    observations = db.query(MetricObservation).filter_by(company_id=company_id, metric_type_id=metric.id).order_by(MetricObservation.recorded_at.desc()).all()
    
    if not observations:
        st.info("No historical data found.")
        return
        
    data = []
    for obs in observations:
        val_str = f"{obs.currency_code or ''} {obs.value_numeric}" if obs.value_numeric else obs.value_text
        date_str = obs.as_of_date.strftime("%Y-%m-%d") if obs.as_of_date else (obs.period_end.strftime("%Y-%m-%d") if obs.period_end else "")
        data.append({
            "Date/Period": date_str,
            "Value": val_str,
            "Status": obs.observation_status,
            "Certainty": f"{obs.confidence_score*100}%" if obs.confidence_score else "N/A",
            "Recorded": obs.recorded_at.strftime("%Y-%m-%d") if obs.recorded_at else ""
        })
    st.dataframe(data, hide_index=True, use_container_width=True)

@st.dialog("Add Transaction")
def add_transaction_dialog(company_id):
    with st.form("add_tx_form"):
        tx_name = st.text_input("Transaction Name", placeholder="e.g. Acme Corp Acquisition")
        tx_type = st.selectbox("Type", ["ACQUISITION", "MERGER", "IPO", "SPAC", "BUYOUT", "SPINOFF"])
        tx_status = st.selectbox("Status", ["ANNOUNCED", "CLOSED", "RUMORED", "CANCELLED"])
        
        all_orgs = db.query(Organization).order_by(Organization.name).all()
        org_opts = {0: "-- None / Unknown --"}
        org_opts.update({o.id: o.name for o in all_orgs})
        
        col1, col2 = st.columns(2)
        acq_index = list(org_opts.keys()).index(company_id) if company_id in org_opts else 0
        tgt_index = list(org_opts.keys()).index(company_id) if company_id in org_opts else 0
        
        acquirer_id = col1.selectbox("Acquirer", options=list(org_opts.keys()), format_func=lambda x: org_opts[x], index=acq_index)
        target_id = col2.selectbox("Target", options=list(org_opts.keys()), format_func=lambda x: org_opts[x], index=tgt_index)
        
        col3, col4 = st.columns(2)
        currency = col3.text_input("Currency", value="USD")
        val_num = col4.number_input("Value (Numeric)", min_value=0.0, format="%f", value=None)
        val_txt = col4.text_input("Value Text", placeholder="e.g. $50M")
        
        col5, col6 = st.columns(2)
        announced = col5.date_input("Announced Date", value=None)
        closed = col6.date_input("Closed Date", value=None)
        
        desc = st.text_area("Description")
        
        if st.form_submit_button("Save Transaction"):
            if tx_name:
                from market_comps.db.models import Transaction
                new_tx = Transaction(
                    transaction_name=tx_name,
                    transaction_type=tx_type,
                    status=tx_status,
                    acquirer_company_id=acquirer_id if acquirer_id != 0 else None,
                    target_company_id=target_id if target_id != 0 else None,
                    currency_code=currency,
                    transaction_value_numeric=val_num if val_num else None,
                    transaction_value_text=val_txt,
                    announced_date=announced,
                    closed_date=closed,
                    description=desc
                )
                db.add(new_tx)
                db.commit()
                st.success("Transaction added!")
                st.rerun()
            else:
                st.error("Transaction Name is required.")

# Helper to render company details below the table
def display_company_details(company_id):
    org = db.query(Organization).options(
        joinedload(Organization.company_profile),
        joinedload(Organization.fund_profiles),
        joinedload(Organization.program_profiles).joinedload(ProgramProfile.cohorts),
        joinedload(Organization.program_memberships).joinedload(ProgramMembership.cohort).joinedload(ProgramCohort.program),
        joinedload(Organization.roles).joinedload(PersonOrganizationRole.person),
        joinedload(Organization.metric_observations).joinedload(MetricObservation.metric_type),
        joinedload(Organization.financing_rounds).joinedload(FinancingRound.facts),
        joinedload(Organization.financing_rounds).joinedload(FinancingRound.investors).joinedload(RoundInvestor.investor),
        joinedload(Organization.transactions_as_target),
        joinedload(Organization.transactions_as_acquirer),
        joinedload(Organization.source_links).joinedload("source")
    ).filter(Organization.id == int(company_id)).first()

    if not org:
        st.error(f"Company with ID {company_id} not found.")
        return

    with st.container(border=True):
        col_h1, col_h2 = st.columns([5, 1])
        with col_h1:
            st.subheader(f"🏢 {org.name}")
        with col_h2:
            if st.button("✏️ Edit", key=f"edit_org_{org.id}", use_container_width=True):
                edit_company_dialog(org)
        
        location_str = ", ".join(filter(None, [org.city, org.state, org.country]))
        
        st.write(f"**Domain:** {org.primary_domain} | **Website:** {org.website_url} | **LinkedIn:** {org.linkedin_url}")
        
        info_parts = [f"**Location:** {location_str}"]
        
        if org.ownership_type and org.ownership_type.upper() == "PUBLIC":
            ticker_str = f" ({org.exchange}: {org.ticker})" if org.ticker and org.exchange else (f" ({org.ticker})" if org.ticker else "")
            info_parts.append(f"**Ownership:** Public{ticker_str}")
        else:
            info_parts.append("**Ownership:** Private")
            
        if org.company_profile:
            if org.company_profile.industry:
                info_parts.append(f"**Industry:** {org.company_profile.industry}")
            if org.company_profile.subindustry:
                info_parts.append(f"**Sub-Industry:** {org.company_profile.subindustry}")
            if org.company_profile.company_stage:
                info_parts.append(f"**Stage:** {org.company_profile.company_stage}")
            if org.company_profile.founded_year:
                info_parts.append(f"**Founded:** {org.company_profile.founded_year}")
                
        st.write(" | ".join(info_parts))
        
        if org.company_profile:
            if org.company_profile.sectors:
                st.write(f"**Sectors:**")
                st.markdown(" ".join([f"`{s}`" for s in org.company_profile.sectors]))
            if org.company_profile.themes:
                st.write(f"**Themes:** {', '.join(org.company_profile.themes)}")

        if org.status and org.status.upper() != "ACTIVE":
            st.warning(f"**Status:** {org.status}")
            
        if org.description:
            st.info(org.description)
            
        if org.company_profile and org.company_profile.themes:
            st.write(f"**Themes:** {', '.join(org.company_profile.themes)}")
            
        if org.program_memberships:
            st.divider()
            st.markdown("#### Program Memberships")
            for m in org.program_memberships:
                if m.cohort and m.cohort.program:
                    st.write(f"- 🎯 **{m.cohort.program.program_name}** — {m.cohort.cohort_name}")
                elif m.cohort:
                    st.write(f"- 🎯 {m.cohort.cohort_name}")
                else:
                    st.write("- 🎯 (Unlinked Membership)")
                    
        if org.fund_profiles:
            st.divider()
            st.markdown("#### Fund Profiles")
            for fund in org.fund_profiles:
                with st.expander(f"💰 {fund.fund_name}"):
                    st.write(f"**Vintage:** {fund.vintage_year} | **Raised:** {fund.fund_size_raised} | **Target:** {fund.fund_size_target}")
                    st.write(f"**Type:** {fund.fund_type} | **Status:** {fund.status}")
                    if fund.description:
                        st.caption(fund.description)

        if org.program_profiles:
            st.divider()
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

        # Metrics & KPIs
        st.divider()
        st.subheader("📊 Metrics & KPIs")
        if org.metric_observations:
            # Group by metric_type to get the latest
            metrics_dict = {}
            for obs in org.metric_observations:
                code = obs.metric_type.code
                if code not in metrics_dict or (obs.recorded_at and metrics_dict[code].recorded_at and obs.recorded_at > metrics_dict[code].recorded_at):
                    metrics_dict[code] = obs
                    
            if metrics_dict:
                cols = st.columns(min(len(metrics_dict), 4))
                for i, (code, obs) in enumerate(metrics_dict.items()):
                    with cols[i % 4]:
                        val_str = f"{obs.currency_code or ''} {obs.value_numeric}" if obs.value_numeric else obs.value_text
                        st.metric(label=obs.metric_type.display_name, value=val_str, help=obs.observation_status)
                        if st.button("View History", key=f"hist_{obs.id}"):
                            view_metric_history_dialog(org.id, code)
            else:
                st.info("No metric data available.")
        else:
            st.info("No metric data available.")

        # Market & Competitors
        st.divider()
        col_mhead1, col_mhead2 = st.columns([3, 1])
        with col_mhead1:
            st.subheader("🗺️ Market & Competitors")
        with col_mhead2:
            st.page_link("pages/21_Market_Map.py", label="Open Market Map", icon="🗺️")
        
        # Determine all markets this company participates in or analyzes
        my_segs = get_company_segments(db, org.id)
        linked_market_ids = {link.market_segment.market_id for link in my_segs if link.market_segment and link.market_segment.market_id}
        ca_list = db.query(CompetitiveAnalysis).filter_by(subject_company_id=org.id).all()
        ca_market_ids = {ca.market_id for ca in ca_list}
        all_my_market_ids = sorted(list(linked_market_ids | ca_market_ids))

        all_markets = get_all_markets(db)

        if not all_my_market_ids:
            st.info("This company is not yet mapped to any markets or segments.")

        for m_id in all_my_market_ids:
            market = db.query(Market).get(m_id)
            if not market: continue
            
            st.markdown(f"#### Market: {market.name}")
            ca = db.query(CompetitiveAnalysis).filter_by(subject_company_id=org.id, market_id=m_id).first()
            if not ca:
                ca = get_or_create_competitive_analysis(db, org.id, m_id, f"{org.name} {market.name} Landscape")
                db.commit()
                
            with st.expander(f"📝 Market Dynamics & Competition Notes ({market.name})"):
                with st.form(f"market_notes_form_{ca.id if ca else m_id}"):
                    m_notes = st.text_area(
                        "Market Dynamics, What Market Values Most & Key Standouts",
                        value=ca.summary if (ca and ca.summary) else "",
                        placeholder="e.g. What market info stands out, key trends, what buyers value most, market opportunities...",
                        help="Free-text notes on overall market observations."
                    )
                    if st.form_submit_button("Save Market Notes"):
                        if ca:
                            ca.summary = m_notes
                            db.commit()
                            st.success("Market notes saved!")
                            st.rerun()

            # -------------------------------------------------------------
            # ##### Segments
            # -------------------------------------------------------------
            st.markdown("##### Segments")
            market_segments = get_market_segments(db, m_id)
            ca_segs_map = {cas.market_segment_id: cas for cas in ca.analysis_segments} if ca else {}
            my_seg_links_map = {link.market_segment_id: link for link in my_segs if link.market_segment_id}
            
            if market_segments:
                seg_df_data = []
                for seg_obj in market_segments:
                    link = my_seg_links_map.get(seg_obj.id)
                    ca_seg = ca_segs_map.get(seg_obj.id)
                    
                    threat_val = ca_seg.threat_level if (ca_seg and ca_seg.threat_level in THREAT_LEVELS) else "N/A"
                    notes_val = ca_seg.analysis_notes if (ca_seg and ca_seg.analysis_notes) else ""
                    
                    seg_df_data.append({
                        "_seg_id": seg_obj.id,
                        "Segment (Read Only)": seg_obj.name,
                        "Differentiation / Info (Read Only)": link.differentiation if link else "",
                        "Threat Level": threat_val,
                        "Threat Notes": notes_val
                    })
                
                df_seg = pd.DataFrame(seg_df_data)
                
                edited_seg_df = st.data_editor(
                    df_seg,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "_seg_id": None,
                        "Segment (Read Only)": st.column_config.TextColumn(disabled=True),
                        "Differentiation / Info (Read Only)": st.column_config.TextColumn(disabled=True),
                        "Threat Level": st.column_config.SelectboxColumn(
                            options=THREAT_LEVELS,
                            required=True
                        ),
                        "Threat Notes": st.column_config.TextColumn(disabled=False)
                    },
                    key=f"data_editor_seg_{ca.id if ca else m_id}"
                )
                
                col_sb1, col_sb2 = st.columns([1, 1])
                with col_sb1:
                    if st.button("💾 Save Segment Edits", key=f"save_seg_btn_{ca.id if ca else m_id}"):
                        from market_comps.crm.competitor_manager import add_competitive_analysis_segment
                        for _, row in edited_seg_df.iterrows():
                            s_id = int(row["_seg_id"])
                            t_val = row["Threat Level"]
                            n_val = row["Threat Notes"]
                            add_competitive_analysis_segment(db, ca.id, s_id, t_val, n_val)
                        db.commit()
                        st.success("Segment edits saved!")
                        st.rerun()
                with col_sb2:
                    with st.popover("➕ Add Segment to Market"):
                        with st.form(f"add_seg_market_{ca.id if ca else m_id}"):
                            s_name = st.text_input("Segment Name")
                            s_desc = st.text_area("Description")
                            if st.form_submit_button("Create Segment"):
                                if s_name:
                                    create_market_segment(db, m_id, s_name, s_desc)
                                    db.commit()
                                    st.success(f"Segment '{s_name}' created!")
                                    st.rerun()
            else:
                st.info("No segments defined in this market yet.")
                with st.popover("➕ Add Segment to Market"):
                    with st.form(f"add_seg_market_empty_{ca.id if ca else m_id}"):
                        s_name = st.text_input("Segment Name")
                        s_desc = st.text_area("Description")
                        if st.form_submit_button("Create Segment"):
                            if s_name:
                                create_market_segment(db, m_id, s_name, s_desc)
                                db.commit()
                                st.success(f"Segment '{s_name}' created!")
                                st.rerun()

            # -------------------------------------------------------------
            # ##### Companies
            # -------------------------------------------------------------
            st.markdown("##### Companies")
            
            all_m_seg_ids = [s.id for s in market_segments]
            m_company_links = db.query(MarketSegmentCompanyLink).filter(
                MarketSegmentCompanyLink.market_segment_id.in_(all_m_seg_ids)
            ).all() if all_m_seg_ids else []
            
            m_company_links = [l for l in m_company_links if l.company_id != org.id]
            ca_companies_map = {(c.market_segment_id, c.competitor_company_id): c for c in ca.analysis_companies} if ca else {}
            
            rel_display_options = [r.replace("_", " ").title() for r in RELATIONSHIP_TYPES]
            rel_map_reverse = {r.replace("_", " ").title(): r for r in RELATIONSHIP_TYPES}
            
            if m_company_links:
                company_summary = {}
                for link in m_company_links:
                    comp_id = link.company_id
                    comp_org = link.company
                    if not comp_org: continue
                    if comp_id not in company_summary:
                        company_summary[comp_id] = {
                            "name": comp_org.name,
                            "segments": [],
                            "differentiation": [],
                            "relationship": "Direct Competitor",
                            "notes": "",
                            "seg_id": link.market_segment_id
                        }
                    if link.market_segment and link.market_segment.name not in company_summary[comp_id]["segments"]:
                        company_summary[comp_id]["segments"].append(link.market_segment.name)
                    if link.differentiation and link.differentiation not in company_summary[comp_id]["differentiation"]:
                        company_summary[comp_id]["differentiation"].append(link.differentiation)
                    
                    ca_comp = ca_companies_map.get((link.market_segment_id, comp_id))
                    if ca_comp:
                        if ca_comp.relationship_type:
                            company_summary[comp_id]["relationship"] = ca_comp.relationship_type.replace("_", " ").title()
                        if ca_comp.competitive_notes:
                            company_summary[comp_id]["notes"] = ca_comp.competitive_notes

                comp_df_data = []
                for cid, cdata in company_summary.items():
                    comp_df_data.append({
                        "_comp_id": cid,
                        "_seg_id": cdata["seg_id"],
                        "Company (Read Only)": cdata["name"],
                        "Segments (Read Only)": ", ".join(cdata["segments"]),
                        "Differentiation (Read Only)": " | ".join(cdata["differentiation"]),
                        "Relationship": cdata["relationship"] if cdata["relationship"] in rel_display_options else rel_display_options[0],
                        "Notes": cdata["notes"]
                    })
                
                df_comp = pd.DataFrame(comp_df_data)
                
                edited_comp_df = st.data_editor(
                    df_comp,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "_comp_id": None,
                        "_seg_id": None,
                        "Company (Read Only)": st.column_config.TextColumn(disabled=True),
                        "Segments (Read Only)": st.column_config.TextColumn(disabled=True),
                        "Differentiation (Read Only)": st.column_config.TextColumn(disabled=True),
                        "Relationship": st.column_config.SelectboxColumn(
                            options=rel_display_options,
                            required=True
                        ),
                        "Notes": st.column_config.TextColumn(disabled=False)
                    },
                    key=f"data_editor_comp_{ca.id if ca else m_id}"
                )
                
                col_cb1, col_cb2 = st.columns([1, 1])
                with col_cb1:
                    if st.button("💾 Save Competitor Edits", key=f"save_comp_btn_{ca.id if ca else m_id}"):
                        for _, row in edited_comp_df.iterrows():
                            c_id = int(row["_comp_id"])
                            s_id = int(row["_seg_id"]) if pd.notnull(row["_seg_id"]) else None
                            rel_str = rel_map_reverse.get(row["Relationship"], "direct_competitor")
                            n_str = row["Notes"]
                            add_competitive_analysis_company(db, ca.id, c_id, s_id, rel_str, None, None, n_str)
                        db.commit()
                        st.success("Competitor edits saved!")
                        st.rerun()
                with col_cb2:
                    with st.popover("➕ Add Competitor"):
                        all_orgs = db.query(Organization).order_by(Organization.name).all()
                        org_opts = {o.name: o.id for o in all_orgs if o.id != org.id}
                        seg_opts = {s.name: s.id for s in market_segments}
                        if seg_opts and org_opts:
                            with st.form(f"add_comp_form_{ca.id if ca else m_id}"):
                                comp_sel = st.selectbox("Company", options=list(org_opts.keys()))
                                seg_sel = st.selectbox("Segment", options=list(seg_opts.keys()))
                                rel_sel = st.selectbox("Relationship Type", options=RELATIONSHIP_TYPES, format_func=lambda x: x.replace("_", " ").title())
                                notes_text = st.text_area("Competitive Notes")
                                if st.form_submit_button("Add Competitor"):
                                    if comp_sel and seg_sel:
                                        comp_id = org_opts[comp_sel]
                                        s_id = seg_opts[seg_sel]
                                        add_company_to_segment(db, comp_id, s_id, differentiation="Mapped via Market Analysis")
                                        add_competitive_analysis_company(db, ca.id, comp_id, s_id, rel_sel, None, None, notes_text)
                                        db.commit()
                                        st.success("Competitor added!")
                                        st.rerun()
                        else:
                            st.write("Ensure segments exist in this market.")
            else:
                st.info("No competitor companies mapped to this market yet.")
                with st.popover("➕ Add Competitor"):
                    all_orgs = db.query(Organization).order_by(Organization.name).all()
                    org_opts = {o.name: o.id for o in all_orgs if o.id != org.id}
                    seg_opts = {s.name: s.id for s in market_segments}
                    if seg_opts and org_opts:
                        with st.form(f"add_comp_form_empty_{ca.id if ca else m_id}"):
                            comp_sel = st.selectbox("Company", options=list(org_opts.keys()))
                            seg_sel = st.selectbox("Segment", options=list(seg_opts.keys()))
                            rel_sel = st.selectbox("Relationship Type", options=RELATIONSHIP_TYPES, format_func=lambda x: x.replace("_", " ").title())
                            notes_text = st.text_area("Competitive Notes")
                            if st.form_submit_button("Add Competitor"):
                                if comp_sel and seg_sel:
                                    comp_id = org_opts[comp_sel]
                                    s_id = seg_opts[seg_sel]
                                    add_company_to_segment(db, comp_id, s_id, differentiation="Mapped via Market Analysis")
                                    add_competitive_analysis_company(db, ca.id, comp_id, s_id, rel_sel, None, None, notes_text)
                                    db.commit()
                                    st.success("Competitor added!")
                                    st.rerun()

            st.divider()

        # Global Action to Add New Market Analysis at the bottom
        unlinked_markets = [m for m in all_markets if m.id not in ca_market_ids]
        if unlinked_markets:
            with st.popover("➕ Add New Market Analysis"):
                with st.form(f"start_ca_bottom_{org.id}"):
                    m_opts = {m.name: m for m in unlinked_markets}
                    m_sel = st.selectbox("Select Market to Analyze", options=list(m_opts.keys()))
                    if st.form_submit_button("Create Market Analysis"):
                        if m_sel:
                            get_or_create_competitive_analysis(db, org.id, m_opts[m_sel].id, f"{org.name} {m_sel} Landscape")
                            db.commit()
                            st.success("Market Analysis created!")
                            st.rerun()

        # Financing Rounds
        st.divider()
        st.subheader("💸 Financing Rounds")
        if org.financing_rounds:
            from datetime import datetime as dt_cls
            def get_rnd_sort_key(r):
                dates = [inv.reported_at for inv in r.investors if inv.reported_at]
                return max(dates) if dates else (r.created_at or dt_cls.min)
            rounds_sorted = sorted(org.financing_rounds, key=get_rnd_sort_key, reverse=True)
            for rnd in rounds_sorted:
                dates = [inv.reported_at for inv in rnd.investors if inv.reported_at]
                date_str = f" ({max(dates).strftime('%b %Y')})" if dates else ""
                with st.expander(f"💰 {rnd.round_name or 'Unknown Round'}{date_str} - {rnd.status.upper()}"):
                    fact_amt = next((f.value_text or str(f.value_numeric) for f in rnd.facts if f.fact_type == 'amount_raised'), None)
                    if fact_amt:
                        st.write(f"**Amount Raised:** {fact_amt}")
                    
                    if rnd.investors:
                        st.write("**Investors:**")
                        inv_data = []
                        for inv in rnd.investors:
                            name = inv.investor.name if inv.investor else "Unknown"
                            inv_date_lbl = inv.reported_at.strftime("%Y-%m-%d") if inv.reported_at else "Unknown Date"
                            inv_data.append({
                                "Investor": name,
                                "Role": inv.role,
                                "Status": inv.status,
                                "Date": inv_date_lbl,
                                "Notes": inv.notes
                            })
                        st.dataframe(inv_data, hide_index=True, use_container_width=True)
                    else:
                        st.caption("No investors recorded.")
        else:
            st.info("No financing rounds recorded for this company.")

        # Transactions
        st.divider()
        col_tx1, col_tx2 = st.columns([3, 1])
        with col_tx1:
            st.subheader("🤝 Transactions (M&A, IPOs)")
        with col_tx2:
            if st.button("➕ Add Transaction", key=f"add_tx_{org.id}", use_container_width=True):
                add_transaction_dialog(org.id)
                
        all_txs = list(org.transactions_as_target) + list(org.transactions_as_acquirer)
        if all_txs:
            all_txs_sorted = sorted(all_txs, key=lambda x: x.announced_date or x.created_at, reverse=True)
            for tx in all_txs_sorted:
                role_label = "Target" if tx.target_company_id == org.id else "Acquirer"
                with st.expander(f"🤝 {tx.transaction_name} ({tx.transaction_type}) - {role_label} - {tx.status}"):
                    t_val = f"{tx.currency_code or ''} {tx.transaction_value_numeric or ''} {tx.transaction_value_text or ''}".strip()
                    if t_val:
                        st.write(f"**Value:** {t_val}")
                    if tx.announced_date:
                        st.write(f"**Announced:** {tx.announced_date.strftime('%Y-%m-%d')}")
                    if tx.description:
                        st.write(tx.description)
        else:
            st.info("No transactions recorded for this company.")

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
                st.info("No people linked to this company.")
        else:
            st.info("No people linked to this company.")

        # AI Profile Augmentation
        st.divider()
        col_aug1, col_aug2, col_aug3, col_aug4 = st.columns([2, 1.5, 1.5, 1])
        with col_aug1:
            st.subheader("🌐 Web Profile Augmentation")
        with col_aug2:
            if st.button("Augment via Web Search", key=f"btn_aug_{org.id}", use_container_width=True):
                with st.spinner("Searching the web and extracting evidence..."):
                    from market_comps.ingestion.company_augmentation import run_augmentation_pipeline
                    try:
                        run_augmentation_pipeline(org.id)
                        st.success("Augmentation Complete! Reloading...")
                    except Exception as e:
                        st.error(f"Augmentation error: {e}")
                import time
                time.sleep(1)
                st.rerun()
        with col_aug3:
            if st.button("🧠 Re-synthesize Data", key=f"btn_resynth_aug_{org.id}", use_container_width=True):
                with st.spinner("Re-synthesizing from existing sources..."):
                    from market_comps.ingestion.company_augmentation import re_synthesize_company_data
                    try:
                        re_synthesize_company_data(org.id)
                        st.success("Re-synthesized! Reloading...")
                    except Exception as e:
                        st.error(f"Error: {e}")
                import time
                time.sleep(1)
                st.rerun()
        with col_aug4:
            if st.button("🗑️ Clear Data", key=f"btn_clear_aug_{org.id}", use_container_width=True):
                with st.spinner("Clearing augmentation data..."):
                    from market_comps.ingestion.company_augmentation import clear_augmentation_data
                    clear_augmentation_data(org.id)
                st.success("Cleared! Reloading...")
                import time
                time.sleep(1)
                st.rerun()

        with st.expander("ℹ️ How Search & Source Filtering Works"):
            st.markdown("""
            **Exa Search Protocol & Guardrails:**
            1. **Query Generation**: LLM constructs 4 queries targeting Overview, Team, Traction, and Funding for `{org.name}`.
            2. **Exa Retrieval**: Fetches top web page contents.
            3. **Quality & Relevance Checks**:
               - ⚠️ **Junk / Error Filter**: Drops pages with <150 chars, 404s, or Cloudflare/Captcha blocks from LLM prompt inputs.
               - 🛑 **Entity Relevance Filter**: Verifies that page text/URL matches target company name or domain.
            4. **Data Auditability**: Flagged documents are saved with error badges so you maintain 100% data provenance without polluting AI extractions.
            """)
                
        st.markdown("**Manual URL Ingestion**")
        col_man1, col_man2 = st.columns([3, 1])
        with col_man1:
            manual_url = st.text_input("Enter URL to extract data from", key=f"man_url_{org.id}", label_visibility="collapsed", placeholder="https://techcrunch.com/...")
        with col_man2:
            if st.button("Ingest URL", key=f"btn_man_aug_{org.id}", use_container_width=True) and manual_url:
                with st.spinner("Reading URL and extracting data..."):
                    from market_comps.ingestion.company_augmentation import run_manual_url_augmentation
                    try:
                        run_manual_url_augmentation(org.id, manual_url)
                        st.success("Successfully ingested data! Refreshing...")
                        import time
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to ingest URL: {e}")
                
        from market_comps.db.models import CompanyAugmentationReport
        latest_report = db.query(CompanyAugmentationReport).filter_by(organization_id=org.id).order_by(CompanyAugmentationReport.created_at.desc()).first()
        
        if latest_report and latest_report.status == "SUCCESS" and latest_report.extracted_data_json:
            import zoneinfo
            eastern = zoneinfo.ZoneInfo("America/New_York")
            tz_time = latest_report.created_at.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(eastern).strftime('%Y-%m-%d %I:%M %p ET') if latest_report.created_at else "Unknown"
            st.caption(f"*(Last augmented: {tz_time})*")
            
            exec_summary = latest_report.extracted_data_json.get("executive_summary")
            if exec_summary:
                st.markdown("##### Executive Summary")
                st.info(exec_summary)
            
            score_data = latest_report.scoring_json or {}
            if score_data:
                st.markdown("##### LLM Opinion Scores")
                score_cols = st.columns(len(score_data))
                for i, (section, s_data) in enumerate(score_data.items()):
                    with score_cols[i]:
                        score_val = s_data.get("score", 0)
                        conf = s_data.get("confidence", "Unknown")
                        color = "green" if score_val and score_val >= 8 else "orange" if score_val and score_val >= 5 else "red"
                        st.markdown(f"**{section.replace('_', ' ').title()}**")
                        st.markdown(f"### :{color}[{score_val}/10]")
                        st.caption(f"Conf: {conf}")
                        if s_data.get("reasoning"):
                            st.write(s_data.get("reasoning"))
                            
            st.markdown("##### Evidenced Data")
            for section, d in latest_report.extracted_data_json.items():
                if section == "executive_summary":
                    continue
                with st.expander(f"📦 {section.replace('_', ' ').title()}"):
                    summary_text = d.get("summary")
                    if summary_text:
                        st.info(summary_text)
                    
                    for quote in d.get("evidenced_data", []):
                        if isinstance(quote, dict):
                            st.markdown(f"- \"{quote.get('quote')}\" (Source: {quote.get('source_url')})")
                        else:
                            st.markdown(f"- \"{quote}\"")
        elif latest_report and latest_report.status == "FAILED":
            st.error(f"Last augmentation failed: {latest_report.error_message}")

        # Linked Source Documents
        st.divider()
        st.subheader("📄 Linked Source Documents")
        
        doc_ids_subquery_1 = db.query(SourceDocument.id).join(DocumentText).join(ExtractionJob).join(ExtractedEntity).join(EntityMatch).filter(
            EntityMatch.canonical_entity_type == "Organization",
            EntityMatch.canonical_entity_id == str(org.id)
        ).subquery()
        
        from market_comps.db.models import CompanyAugmentationReport, PipelineRun
        doc_ids_subquery_2 = db.query(SourceDocument.id).join(PipelineRun, SourceDocument.pipeline_run_id == PipelineRun.id).join(CompanyAugmentationReport, PipelineRun.id == CompanyAugmentationReport.pipeline_run_id).filter(
            CompanyAugmentationReport.organization_id == org.id
        ).subquery()
        
        docs = db.query(SourceDocument).filter(
            ((SourceDocument.id.in_(doc_ids_subquery_1)) | (SourceDocument.id.in_(doc_ids_subquery_2))),
            SourceDocument.deleted_at == None
        ).all()
        
        @st.dialog("Source Details", width="large")
        def view_source_details(doc_id: int):
            from market_comps.db.models import SourceDocument, DocumentText, ObservationSource, FinancingRoundFact, RoundInvestor
            with get_db_context() as tdb:
                doc = tdb.query(SourceDocument).filter_by(id=doc_id).first()
                if not doc:
                    st.error("Document not found")
                    return
                st.subheader(f"{doc.title or doc.source_url}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Trustworthiness", f"Tier {doc.source_tier}" if doc.source_tier else "Unknown")
                col2.metric("LLM Used", doc.llm_model_used or "N/A")
                
                import zoneinfo
                eastern = zoneinfo.ZoneInfo("America/New_York")
                tz_time = doc.created_at.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(eastern).strftime('%Y-%m-%d %H:%M') if doc.created_at else "Unknown"
                col3.metric("Extracted At", tz_time)
                
                st.divider()
                st.markdown("### Extracted Data")
                
                found_data = False
                osrcs = tdb.query(ObservationSource).filter_by(source_id=doc_id).all()
                if osrcs:
                    found_data = True
                    for osrc in osrcs:
                        obs = osrc.observation
                        if obs and obs.metric_type:
                            st.markdown(f"- **Metric - {obs.metric_type.display_name}**: {obs.value_text} ({obs.observation_status}, {obs.reporting_basis or 'unknown basis'})")
                        if osrc.source_excerpt:
                            st.caption(f"Excerpt: \"{osrc.source_excerpt}\"")
                            
                facts = tdb.query(FinancingRoundFact).filter_by(source_id=doc_id).all()
                if facts:
                    found_data = True
                    for f in facts:
                        st.markdown(f"- **Financing Fact**: {f.fact_type} = {f.value_text}")
                        
                rinvs = tdb.query(RoundInvestor).filter_by(source_id=doc_id).all()
                if rinvs:
                    found_data = True
                    for r in rinvs:
                        org_name = r.investor.name if r.investor else "Unknown Investor"
                        st.markdown(f"- **Round Investor**: {org_name} ({r.role})")
                        
                if not found_data:
                    st.write("No specific facts or metrics linked to this source.")
                    
                st.divider()
                st.markdown("### Raw Content")
                txt = tdb.query(DocumentText).filter_by(source_document_id=doc_id, data_type="PAGE_TEXT").first()
                if txt and txt.raw_content:
                    st.text_area("Content", txt.raw_content, height=300, disabled=True, label_visibility="collapsed")
                else:
                    st.info("Raw content not available.")

        
        if docs:
            # Deduplicate by URL
            seen_urls = set()
            deduped_docs = []
            for doc in docs:
                url = str(doc.source_url).strip().lower()
                if url.endswith('/'): url = url[:-1]
                if url not in seen_urls:
                    seen_urls.add(url)
                    deduped_docs.append(doc)
            
            from market_comps.config import get_supabase_url
            import zoneinfo
            eastern = zoneinfo.ZoneInfo("America/New_York")
            
            for doc in deduped_docs:
                signed_url = get_supabase_url(doc.file_path) if doc.file_path else ""
                
                doc_label = doc.title if doc.title else doc.source_url
                
                if signed_url:
                    url_display = f"{doc_label} [(View)]({signed_url})"
                else:
                    url_display = f"[{doc_label}]({doc.source_url})" if str(doc.source_url).startswith("http") else doc_label
                
                if doc.document_date:
                    from datetime import datetime
                    if isinstance(doc.document_date, datetime):
                        tz_time = doc.document_date.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(eastern).strftime('%Y-%m-%d')
                    else:
                        tz_time = str(doc.document_date).split("T")[0]
                    date_display = f"Written: {tz_time}"
                else:
                    tz_time = doc.created_at.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(eastern).strftime('%Y-%m-%d') if doc.created_at else "Unknown Time"
                    date_display = f"Processed: {tz_time}"
                
                source_suffix = f" (via {doc.source_name})" if doc.source_name else ""
                
                status_val = getattr(doc, "extraction_status", "SUCCESS") or "SUCCESS"
                if status_val == "FAILED_JUNK":
                    badge_str = " ⚠️ `[Errored/Blocked Page]`"
                elif status_val == "FAILED_RELEVANCE":
                    badge_str = " 🛑 `[Name Mismatch]`"
                else:
                    badge_str = " 🟢"
                    
                err_detail = f" — *{doc.extraction_error}*" if getattr(doc, "extraction_error", None) else ""
                
                col_info, col_act1, col_act2 = st.columns([5, 1, 1])
                with col_info:
                    st.markdown(f"- **{doc.document_type}**{badge_str}: {url_display}{source_suffix} ({date_display}){err_detail}")
                with col_act1:
                    if st.button("📄 Details", key=f"view_doc_{doc.id}"):
                        view_source_details(doc.id)
                with col_act2:
                    with st.popover("🗑️ Remove"):
                        st.caption("Remove this source document from the database.")
                        if st.button("Confirm Delete", key=f"del_doc_{doc.id}", type="primary"):
                            try:
                                # Soft delete document
                                db.query(SourceDocument).filter(SourceDocument.id == doc.id).update({SourceDocument.deleted_at: datetime.utcnow(), SourceDocument.deleted_by: 'USER'}, synchronize_session=False)
                                db.commit()
                                st.rerun()
                            except Exception as e:
                                db.rollback()
                                st.error(f"Failed to delete: {e}")
        else:
            st.info("No documents linked to this company.")

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

        # Sourced From
        if org.source_links:
            st.divider()
            st.subheader("🔗 Sourced From")
            for link in org.source_links:
                source = link.source
                url = link.source_url or source.url
                if url:
                    st.markdown(f"- [{source.name}]({url}) — *{source.source_type}*")
                else:
                    st.markdown(f"- **{source.name}** — *{source.source_type}*")

        # Audit Trail
        st.divider()
        st.subheader("📜 Audit Trail")
        filters = [
            (AuditTrail.canonical_entity_type == "ORGANIZATION") & (AuditTrail.canonical_entity_id == str(org.id)),
            AuditTrail.canonical_entity_type.in_(["COMPETITIVE_ANALYSIS", "COMPETITIVE_ANALYSIS_COMPANY", "COMPETITIVE_ANALYSIS_SEGMENT", "MARKET_SEGMENT_LINK"])
        ]
        if org.company_profile:
            filters.append((AuditTrail.canonical_entity_type == "COMPANY_PROFILE") & (AuditTrail.canonical_entity_id == str(org.company_profile.id)))
            
        role_ids = [str(role.id) for role in org.roles] if org.roles else []
        if role_ids:
            filters.append((AuditTrail.canonical_entity_type == "PERSON_ROLE") & (AuditTrail.canonical_entity_id.in_(role_ids)))
            
        audit = db.query(AuditTrail).options(
            joinedload(AuditTrail.extraction_job).joinedload(ExtractionJob.pipeline_run).joinedload(PipelineRun.source_documents)
        ).filter(
            or_(*filters)
        ).order_by(AuditTrail.created_at.desc()).limit(20).all()
        
        if audit:
            audit_data = []
            from market_comps.utils import format_audit_row
            for a in audit:
                audit_data.append(format_audit_row(a, db))
            st.dataframe(pd.DataFrame(audit_data), use_container_width=True, hide_index=True)
        else:
            st.info("No mutation entries found.")

# Main Page Entry
company_id = st.query_params.get("id")

col_back, _ = st.columns([1, 5])
with col_back:
    st.page_link("pages/14_CRM_Directory.py", label="← Back to Directory")

if company_id:
    display_company_details(company_id)
else:
    st.info("No company ID provided in the URL. Please return to the directory and select a company.")
