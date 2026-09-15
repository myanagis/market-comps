import streamlit as st
from market_comps.ui.style import inject_global_styles
inject_global_styles()
import pandas as pd
from sqlalchemy.orm import joinedload
from market_comps.db.session import get_db_context
from market_comps.db.models import (
    Market, MarketSegment, MarketSegmentCompanyLink, Organization,
    ComparisonSet, MarketComparisonSetLink, ComparisonSetOrganizationLink,
    MetricObservation, MetricType
)
from market_comps.crm.competitor_manager import (
    create_market_segment, get_market_segments
)
from market_comps.integrations.yahoo_finance import YahooFinanceClient

st.set_page_config(page_title="Market Details", page_icon="🗺️", layout="wide")

market_id_str = st.query_params.get("id")

if not market_id_str:
    st.error("No Market ID provided. Please navigate from the Market Directory.")
    st.page_link("pages/21_Market_Map.py", label="Back to Directory", icon="⬅️")
    st.stop()

st.page_link("pages/21_Market_Map.py", label="Back to Directory", icon="⬅️")
st.divider()

with get_db_context() as db:
    market = db.query(Market).get(int(market_id_str))
    
    if not market:
        st.error(f"Market with ID {market_id_str} not found.")
        st.stop()
        
    col_t1, col_t2 = st.columns([0.85, 0.15])
    with col_t1:
        st.markdown('<div class="market-eyebrow">MARKET MAP</div>', unsafe_allow_html=True)
        st.title(market.name)
        
        st.markdown('<div class="market-notes">', unsafe_allow_html=True)
        if market.sectors:
            st.write(f"**Sectors:** {', '.join(market.sectors)}")
        if market.description:
            st.write(market.description)
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col_t2:
        st.markdown('<div class="header-action-container">', unsafe_allow_html=True)
        with st.popover("✏️ Edit Details", use_container_width=True):
            with st.form("edit_market_form"):
                new_name = st.text_input("Name", value=market.name)
                new_sectors = st.text_input("Sectors (comma separated)", value=", ".join(market.sectors) if market.sectors else "")
                new_desc = st.text_area("Description", value=market.description or "")
                if st.form_submit_button("Save Changes"):
                    market.name = new_name
                    market.sectors = [s.strip() for s in new_sectors.split(",")] if new_sectors else []
                    market.description = new_desc
                    db.commit()
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    segments = get_market_segments(db, market.id)
    
    # -------------------------------------------------------------
    # ##### Segments
    # -------------------------------------------------------------
    st.header("Market Segments and Competition")
    
    col_h, col_a = st.columns([0.85, 0.15])
    with col_h:
        st.subheader("Segmentation")
    with col_a:
        st.markdown('<div class="header-action-container">', unsafe_allow_html=True)
        with st.popover("➕ Add Segment"):
            with st.form("new_segment_form_map_head"):
                s_name = st.text_input("Segment Name")
                s_desc = st.text_area("Description")
                s_type = st.text_input("Segment Type (Optional)")
                s_sort = st.number_input("Sort Order", value=0, step=10)
                if st.form_submit_button("Create Segment"):
                    if s_name:
                        seg = create_market_segment(db, market.id, s_name, s_desc, s_type)
                        seg.sort_order = s_sort
                        db.commit()
                        st.success(f"Segment '{s_name}' added!")
                        st.rerun()
                    else:
                        st.error("Segment name is required.")
        st.markdown('</div>', unsafe_allow_html=True)

    if segments:
        h1, h2, h3, h4 = st.columns([2, 3, 2, 0.5])
        h1.markdown("**Segment Name**")
        h2.markdown("**Description**")
        h3.markdown("**Segment Type**")
        h4.markdown("** **")
        
        for seg in segments:
            c1, c2, c3, c4 = st.columns([2, 3, 2, 0.5])
            c1.write(seg.name)
            c2.write(seg.description or "")
            c3.write(seg.segment_type or "")
            with c4:
                with st.popover("✏️"):
                    with st.form(f"edit_seg_{seg.id}"):
                        s_name = st.text_input("Segment Name", value=seg.name)
                        s_desc = st.text_area("Description", value=seg.description or "")
                        s_type = st.text_input("Segment Type", value=seg.segment_type or "")
                        s_sort = st.number_input("Sort Order", value=seg.sort_order or 0, step=10)
                        save_btn = st.form_submit_button("Save")
                        del_btn = st.form_submit_button("🗑️ Delete Segment")
                        
                        if save_btn:
                            s_obj = db.query(MarketSegment).get(seg.id)
                            if s_obj:
                                s_obj.name = s_name
                                s_obj.description = s_desc
                                s_obj.segment_type = s_type
                                s_obj.sort_order = s_sort
                                db.commit()
                                st.rerun()
                        elif del_btn:
                            s_obj = db.query(MarketSegment).get(seg.id)
                            if s_obj:
                                db.delete(s_obj)
                                db.commit()
                                st.rerun()
    else:
        st.info("No segments in this market yet.")

    segment_links = (
        db.query(MarketSegmentCompanyLink)
        .join(MarketSegment, MarketSegmentCompanyLink.market_segment_id == MarketSegment.id)
        .filter(MarketSegment.market_id == market.id)
        .order_by(MarketSegment.sort_order.asc(), MarketSegment.name.asc())
        .all()
    )
    
    if segments:
        from market_comps.db.models import FinancingRound, FinancingRoundFact, MetricObservation, MetricType
        
        grouped_links = {s.name: [] for s in segments}
        for link in segment_links:
            if link.market_segment:
                s_name = link.market_segment.name
                if s_name in grouped_links:
                    grouped_links[s_name].append(link)
            
        for s_name, links in grouped_links.items():
            col_h, col_a = st.columns([0.85, 0.15])
            with col_h:
                st.subheader(f"Companies: {s_name}")
            with col_a:
                st.markdown('<div class="header-action-container">', unsafe_allow_html=True)
                with st.popover("➕ Link Org"):
                    with st.form(f"link_company_map_form_{s_name.replace(' ', '_')}"):
                        all_orgs = db.query(Organization).order_by(Organization.name).all()
                        org_opts = {f"{o.name} ({o.organization_type or 'Company'})": o.id for o in all_orgs}
                        seg_opts = {s.name: s.id for s in segments}
                        seg_idx = list(seg_opts.keys()).index(s_name) if s_name in seg_opts else 0
                        if org_opts and seg_opts:
                            comp_sel = st.selectbox("Organization", options=list(org_opts.keys()), key=f"org_{s_name}")
                            seg_sel = st.selectbox("Segment", options=list(seg_opts.keys()), index=seg_idx, key=f"seg_{s_name}")
                            diff_text = st.text_area("Differentiation", placeholder="How does this organization differentiate in this segment?", key=f"diff_{s_name}")
                            if st.form_submit_button("Link Organization"):
                                if comp_sel and seg_sel:
                                    from market_comps.crm.competitor_manager import add_company_to_segment
                                    add_company_to_segment(db, org_opts[comp_sel], seg_opts[seg_sel], diff_text, False)
                                    db.commit()
                                    st.success("Organization linked to segment!")
                                    st.rerun()
                        else:
                            st.write("Ensure organizations and segments exist.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            h1, h2, h3, h4, h5 = st.columns([2, 3, 1.5, 1.5, 0.5])
            h1.markdown("**Organization**")
            h2.markdown("**Differentiation**")
            h3.markdown("**Total / Last Raised**")
            h4.markdown("**Valuation**")
            h5.markdown("** **")
            
            for link in links:
                comp_org = link.company
                seg_obj = link.market_segment
                if not comp_org or not seg_obj: continue
                
                raised_str = "Unknown"
                val_str = "Unknown"
                
                fin = db.query(FinancingRound).filter_by(company_id=comp_org.id).order_by(FinancingRound.id.desc()).first()
                if fin:
                    raised_fact = db.query(FinancingRoundFact).filter_by(financing_round_id=fin.id, fact_type="amount_raised").first()
                    if raised_fact and raised_fact.value_numeric:
                        val = raised_fact.value_numeric
                        if val >= 1e9: raised_str = f"${val/1e9:.2f}B"
                        elif val >= 1e6: raised_str = f"${val/1e6:.2f}M"
                        else: raised_str = f"${val:,.0f}"
                    
                    val_fact = db.query(FinancingRoundFact).filter_by(financing_round_id=fin.id, fact_type="post_money_valuation").first()
                    if val_fact and val_fact.value_numeric:
                        val = val_fact.value_numeric
                        date_str = f" ({fin.announced_date.strftime('%Y-%m')})" if fin.announced_date else ""
                        if val >= 1e9: val_str = f"${val/1e9:.2f}B{date_str}"
                        elif val >= 1e6: val_str = f"${val/1e6:.2f}M{date_str}"
                        else: val_str = f"${val:,.0f}{date_str}"
                        
                if val_str == "Unknown":
                    mc_type = db.query(MetricType).filter_by(code="market_cap").first()
                    if mc_type:
                        obs = db.query(MetricObservation).filter_by(company_id=comp_org.id, metric_type_id=mc_type.id).order_by(MetricObservation.recorded_at.desc()).first()
                        if obs and obs.value_numeric:
                            val = obs.value_numeric
                            date_str = f" ({obs.recorded_at.strftime('%Y-%m')})" if obs.recorded_at else ""
                            if val >= 1e9: val_str = f"${val/1e9:.2f}B{date_str}"
                            elif val >= 1e6: val_str = f"${val/1e6:.2f}M{date_str}"
                            else: val_str = f"${val:,.0f}{date_str}"
                
                c1, c2, c3, c4, c5 = st.columns([2, 3, 1.5, 1.5, 0.5])
                c1.markdown(f"[{comp_org.name}](/company?id={comp_org.id})")
                c2.write(link.differentiation or "")
                c3.write(raised_str)
                c4.write(val_str)
                
                with c5:
                    with st.popover("✏️"):
                        with st.form(f"edit_comp_{link.company_id}_{link.market_segment_id}"):
                            seg_opts = {s.name: s.id for s in segments}
                            seg_idx = list(seg_opts.values()).index(seg_obj.id) if seg_obj.id in seg_opts.values() else 0
                            new_seg_name = st.selectbox("Segment", options=list(seg_opts.keys()), index=seg_idx)
                            new_diff = st.text_area("Differentiation", value=link.differentiation or "")
                            save_btn = st.form_submit_button("Save")
                            del_btn = st.form_submit_button("🗑️ Unlink from Segment")
                            if save_btn:
                                link_obj = db.query(MarketSegmentCompanyLink).filter_by(
                                    company_id=link.company_id,
                                    market_segment_id=link.market_segment_id
                                ).first()
                                if link_obj:
                                    new_seg_id = seg_opts[new_seg_name]
                                    if new_seg_id != link.market_segment_id:
                                        db.delete(link_obj)
                                        db.flush()
                                        new_link = MarketSegmentCompanyLink(
                                            company_id=link.company_id,
                                            market_segment_id=new_seg_id,
                                            differentiation=new_diff
                                        )
                                        db.add(new_link)
                                    else:
                                        link_obj.differentiation = new_diff
                                    db.commit()
                                    st.rerun()
                            elif del_btn:
                                link_obj = db.query(MarketSegmentCompanyLink).filter_by(
                                    company_id=link.company_id,
                                    market_segment_id=link.market_segment_id
                                ).first()
                                if link_obj:
                                    db.delete(link_obj)
                                    db.commit()
                                    st.rerun()
    else:
        st.info("No segments exist in this market yet. Add a segment to begin mapping organizations.")

    # -------------------------------------------------------------
    # ##### Comparison Sets
    # -------------------------------------------------------------

    market_set_links = db.query(MarketComparisonSetLink).options(
        joinedload(MarketComparisonSetLink.comparison_set).joinedload(ComparisonSet.organization_links).joinedload(ComparisonSetOrganizationLink.organization)
    ).filter_by(market_id=market.id).all()
    
    # Group by set_type
    sets_by_type = {}
    for link in market_set_links:
        cset = link.comparison_set
        if not cset: continue
        # Handle rename of "Investor Comps" to "Investors" for display/logic
        stype = "Investors" if cset.set_type == "Investor Comps" else cset.set_type
        if stype not in sets_by_type:
            sets_by_type[stype] = []
        sets_by_type[stype].append(cset)

    desired_order = ["M&A Precedents", "Public Comps", "Financing Comps", "Investors"]
    all_types = desired_order + [t for t in sets_by_type.keys() if t not in desired_order]

    if not sets_by_type:
        st.info("No comparison sets exist for this market yet.")

    for stype in all_types:
        csets = sets_by_type.get(stype, [])
        if not csets: continue
        
        st.header(stype)
        
        for cset in csets:
            col_c1, col_c2 = st.columns([0.85, 0.15])
            with col_c1:
                st.subheader(f"📚 {cset.name}")
                if cset.description:
                    st.markdown(f"<div class='market-notes' style='margin-top: 0px; margin-bottom: 16px; font-size: 0.9em; padding-left: 10px; border-left: 3px solid #e5e7eb;'>{cset.description}</div>", unsafe_allow_html=True)
            with col_c2:
                st.markdown('<div class="header-action-container">', unsafe_allow_html=True)
                with st.popover("✏️ Edit Section"):
                    with st.form(f"edit_cset_form_{cset.id}"):
                        new_name = st.text_input("Name", value=cset.name)
                        new_desc = st.text_area("Description", value=cset.description or "")
                        if st.form_submit_button("Save"):
                            cset.name = new_name
                            cset.description = new_desc
                            db.commit()
                            st.rerun()
                
                if cset.set_type == "Public Comps" and st.session_state.get("yfinance_enabled", True):
                    if st.button("📈 Pull Market Data", key=f"pull_yf_{cset.id}"):
                        with st.spinner("Fetching data from Yahoo Finance..."):
                            from market_comps.metrics_fetcher import MetricsFetcher
                            from market_comps.db.models import CompanyCandidate, Organization
                            fetcher = MetricsFetcher(max_fetch_workers=4)
                            
                            # Convert to candidates
                            candidates = []
                            for link in cset.organization_links:
                                if link.included and link.organization:
                                    candidates.append(
                                        CompanyCandidate(
                                            name=link.organization.name,
                                            ticker=link.organization.ticker_symbol or "",
                                            exchange=link.organization.stock_exchange,
                                            is_public=True,
                                            confidence=1.0,
                                            reasoning=""
                                        )
                                    )
                            
                            # Run fetcher
                            metrics_list = fetcher.enrich_candidates(candidates)
                            
                            # Update DB
                            for comp_link in cset.organization_links:
                                if not comp_link.included or not comp_link.organization: continue
                                org = comp_link.organization
                                
                                # Find corresponding metrics
                                match = next((m for m in metrics_list if m.ticker == org.ticker_symbol), None)
                                if match and match.data_available:
                                    from market_comps.db.models import MetricType, MetricObservation
                                    import datetime
                                    
                                    # Update ticker/exchange on org
                                    org.stock_exchange = match.exchange
                                    
                                    # Prepare metrics dict
                                    updates = {
                                        "Market Cap": (match.market_cap, "currency"),
                                        "Enterprise Value": (match.enterprise_value, "currency"),
                                        "Revenue (TTM)": (match.revenue_ttm, "currency"),
                                        "Revenue (NTM)": (match.revenue_ntm, "currency"),
                                        "Gross Margin (%)": (match.gross_margin_pct, "percentage"),
                                        "Revenue Growth (YoY)": (match.revenue_growth_yoy_pct, "percentage"),
                                    }
                                    
                                    # Insert/Update MetricObservations
                                    now = datetime.datetime.utcnow()
                                    for m_name, (val, v_type) in updates.items():
                                        if val is None: continue
                                        mt = db.query(MetricType).filter_by(display_name=m_name).first()
                                        if not mt:
                                            mt = MetricType(display_name=m_name, value_type=v_type)
                                            db.add(mt)
                                            db.flush()
                                            
                                        obs = db.query(MetricObservation).filter_by(
                                            company_id=org.id, 
                                            metric_type_id=mt.id,
                                            reporting_basis="trailing_twelve_months"
                                        ).first()
                                        if not obs:
                                            obs = MetricObservation(company_id=org.id, metric_type_id=mt.id, reporting_basis="trailing_twelve_months")
                                            db.add(obs)
                                            
                                        obs.value_numeric = val
                                        obs.recorded_at = now
                                        
                            db.commit()
                            st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)
            
            companies_in_set = [cl.organization for cl in cset.organization_links if cl.included and cl.organization]
            if companies_in_set:
                clink_map = {cl.organization_id: cl for cl in cset.organization_links if cl.included and cl.organization}
                
                if cset.set_type == "M&A Precedents":
                    h1, h2, h3, h4, h5, h6 = st.columns([2, 2, 1.5, 1.5, 3, 0.5])
                    h1.markdown("**Target**")
                    h2.markdown("**Acquirer**")
                    h3.markdown("**Transaction Value**")
                    h4.markdown("**Date**")
                    h5.markdown("**Notes**")
                    h6.markdown("** **")
                    
                    for comp in companies_in_set:
                        from market_comps.db.models import Transaction
                        import datetime
                        tx = db.query(Transaction).filter_by(target_company_id=comp.id, transaction_type="ACQUISITION").order_by(Transaction.id.desc()).first()
                        
                        acq_name = tx.acquirer_company.name if tx and tx.acquirer_company else ""
                        acq_link = f"/company?id={tx.acquirer_company.id}" if tx and tx.acquirer_company else None
                        
                        val_str = "Undisclosed"
                        if tx and tx.transaction_value_numeric:
                            val = tx.transaction_value_numeric
                            if val >= 1e9: val_str = f"${val/1e9:.2f}B"
                            elif val >= 1e6: val_str = f"${val/1e6:.2f}M"
                            else: val_str = f"${val:,.0f}"
                            
                        date_str = tx.announced_date.strftime("%Y-%m-%d") if tx and tx.announced_date else "Unknown"
                        
                        c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 1.5, 1.5, 3, 0.5])
                        c1.markdown(f"[{comp.name}](/company?id={comp.id})")
                        if acq_link: c2.markdown(f"[{acq_name}]({acq_link})")
                        else: c2.write(acq_name)
                        c3.write(val_str)
                        c4.write(date_str)
                        c5.write(clink_map[comp.id].notes or "")
                        
                        with c6:
                            with st.popover("✏️"):
                                with st.form(f"edit_ma_{cset.id}_{comp.id}"):
                                    new_notes = st.text_area("Notes", value=clink_map[comp.id].notes or "", key=f"notes_{cset.id}_{comp.id}")
                                    cur_date = tx.announced_date if tx and tx.announced_date else datetime.date.today()
                                    new_date = st.date_input("Transaction Date", value=cur_date, key=f"date_{cset.id}_{comp.id}")
                                    new_val = st.number_input("Transaction Value ($)", value=float(tx.transaction_value_numeric) if tx and tx.transaction_value_numeric else 0.0, step=1000000.0, key=f"val_{cset.id}_{comp.id}")
                                    save_btn = st.form_submit_button("Save")
                                    del_btn = st.form_submit_button("🗑️ Remove from Set")
                                    if save_btn:
                                        clink_map[comp.id].notes = new_notes
                                        if tx:
                                            tx.announced_date = new_date
                                            if new_val > 0: tx.transaction_value_numeric = new_val
                                        else:
                                            new_tx = Transaction(target_company_id=comp.id, transaction_type="ACQUISITION", announced_date=new_date, transaction_value_numeric=new_val if new_val > 0 else None)
                                            db.add(new_tx)
                                        db.commit()
                                        st.rerun()
                                    elif del_btn:
                                        db.delete(clink_map[comp.id])
                                        db.commit()
                                        st.rerun()

                elif cset.set_type == "Financing Comps":
                    h1, h2, h3, h4, h5, h6 = st.columns([2, 1.5, 1.5, 2, 3, 0.5])
                    h1.markdown("**Organization**")
                    h2.markdown("**Round Name**")
                    h3.markdown("**Amount Raised**")
                    h4.markdown("**Lead Investors**")
                    h5.markdown("**Notes**")
                    h6.markdown("** **")
                    
                    for comp in companies_in_set:
                        from market_comps.db.models import FinancingRound, FinancingRoundFact, RoundInvestor
                        fin = db.query(FinancingRound).filter_by(company_id=comp.id).order_by(FinancingRound.id.desc()).first()
                        round_name = fin.round_name if fin else ""
                        
                        val_str = "Undisclosed"
                        if fin:
                            fact = db.query(FinancingRoundFact).filter_by(financing_round_id=fin.id, fact_type="amount_raised").first()
                            if fact and fact.value_numeric:
                                val = fact.value_numeric
                                if val >= 1e9: val_str = f"${val/1e9:.2f}B"
                                elif val >= 1e6: val_str = f"${val/1e6:.2f}M"
                                else: val_str = f"${val:,.0f}"
                                
                        lead_invs = ""
                        if fin:
                            invs = db.query(RoundInvestor).filter_by(financing_round_id=fin.id, role="lead").all()
                            if invs: lead_invs = ", ".join([inv.investor.name for inv in invs if inv.investor])
                            
                        c1, c2, c3, c4, c5, c6 = st.columns([2, 1.5, 1.5, 2, 3, 0.5])
                        c1.markdown(f"[{comp.name}](/company?id={comp.id})")
                        c2.write(round_name)
                        c3.write(val_str)
                        c4.write(lead_invs)
                        c5.write(clink_map[comp.id].notes or "")
                        
                        with c6:
                            with st.popover("✏️"):
                                with st.form(f"edit_notes_{cset.id}_{comp.id}"):
                                    new_notes = st.text_area("Notes", value=clink_map[comp.id].notes or "")
                                    save_btn = st.form_submit_button("Save")
                                    del_btn = st.form_submit_button("🗑️ Remove from Set")
                                    if save_btn:
                                        clink_map[comp.id].notes = new_notes
                                        db.commit()
                                        st.rerun()
                                    elif del_btn:
                                        db.delete(clink_map[comp.id])
                                        db.commit()
                                        st.rerun()

                else:
                    # Public Comps
                    # Columns: Organization, Ticker, Last Updated, [Metrics], Notes, Edit
                    obs_list_all = db.query(MetricObservation).filter(
                        MetricObservation.company_id.in_([c.id for c in companies_in_set]),
                        MetricObservation.reporting_basis == "trailing_twelve_months"
                    ).all()
                    
                    metric_types = {}
                    for obs in obs_list_all:
                        mt = db.query(MetricType).get(obs.metric_type_id)
                        if mt and mt.display_name not in metric_types:
                            metric_types[mt.display_name] = mt
                            
                    metric_names = list(metric_types.keys())
                    cols = [2, 1] + [1.5] * len(metric_names) + [1, 2, 0.5]
                    header_cols = st.columns(cols)
                    header_cols[0].markdown("**Organization**")
                    header_cols[1].markdown("**Ticker**")
                    for i, m_name in enumerate(metric_names): header_cols[2+i].markdown(f"**{m_name}**")
                    header_cols[2+len(metric_names)].markdown("**Last Updated**")
                    header_cols[3+len(metric_names)].markdown("**Notes**")
                    
                    for comp in companies_in_set:
                        c_cols = st.columns(cols)
                        c_cols[0].markdown(f"[{comp.name}](/company?id={comp.id})")
                        c_cols[1].write(comp.ticker or "")
                        
                        obs_list = db.query(MetricObservation).filter_by(
                            company_id=comp.id, reporting_basis="trailing_twelve_months"
                        ).all()
                        
                        m_values = {m: "" for m in metric_names}
                        last_updated = None
                        
                        for obs in obs_list:
                            mt = db.query(MetricType).get(obs.metric_type_id)
                            if mt:
                                if mt.value_type == "currency":
                                    val = obs.value_numeric
                                    if val:
                                        if val >= 1e9: m_values[mt.display_name] = f"${val/1e9:.2f}B"
                                        elif val >= 1e6: m_values[mt.display_name] = f"${val/1e6:.2f}M"
                                        else: m_values[mt.display_name] = f"${val:,.0f}"
                                elif mt.value_type == "multiple":
                                    m_values[mt.display_name] = f"{obs.value_numeric:.1f}x" if obs.value_numeric else ""
                                
                                if hasattr(obs, 'recorded_at') and obs.recorded_at:
                                    if not last_updated or obs.recorded_at > last_updated: last_updated = obs.recorded_at
                                        
                        for i, m_name in enumerate(metric_names): c_cols[2+i].write(m_values[m_name])
                        c_cols[2+len(metric_names)].write(last_updated.strftime("%Y-%m-%d") if last_updated else "")
                        c_cols[3+len(metric_names)].write(clink_map[comp.id].notes or "")
                        
                        with c_cols[4+len(metric_names)]:
                            with st.popover("✏️"):
                                with st.form(f"edit_notes_{cset.id}_{comp.id}"):
                                    new_notes = st.text_area("Notes", value=clink_map[comp.id].notes or "")
                                    save_btn = st.form_submit_button("Save")
                                    del_btn = st.form_submit_button("🗑️ Remove from Set")
                                    if save_btn:
                                        clink_map[comp.id].notes = new_notes
                                        db.commit()
                                        st.rerun()
                                    elif del_btn:
                                        db.delete(clink_map[comp.id])
                                        db.commit()
                                        st.rerun()

            else:
                st.info("No organizations linked to this Comparison Set.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_c1, col_c2 = st.columns([1, 1])
            with col_c1:
                with st.popover("➕ Add organization"):
                    all_orgs = db.query(Organization).filter(Organization.organization_type != "Investor").order_by(Organization.name).all()
                    org_opts = {o.name: o.id for o in all_orgs}
                    with st.form(f"add_comp_cset_{cset.id}"):
                        comp_sel = st.selectbox("Organization", options=list(org_opts.keys()))
                        if st.form_submit_button("Add to Set"):
                            if comp_sel:
                                clink = ComparisonSetOrganizationLink(comparison_set_id=cset.id, organization_id=org_opts[comp_sel])
                                db.add(clink)
                                db.commit()
                                st.success(f"{comp_sel} added to set!")
                                st.rerun()
            with col_c2:
                if st.button("Unlink Set from Market", key=f"unlink_cset_{cset.id}"):
                    link_to_delete = db.query(MarketComparisonSetLink).filter_by(market_id=market.id, comparison_set_id=cset.id).first()
                    if link_to_delete:
                        db.delete(link_to_delete)
                        db.commit()
                        st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("Set Management")
    with st.popover("🔗 Link Existing Set"):
        with st.form("link_existing_set"):
            all_sets = db.query(ComparisonSet).order_by(ComparisonSet.name).all()
            if all_sets:
                set_opts = {f"{s.name} ({s.set_type})": s.id for s in all_sets}
                set_sel = st.selectbox("Select Set", options=list(set_opts.keys()))
                if st.form_submit_button("Link Set"):
                    if set_sel:
                        new_link = MarketComparisonSetLink(market_id=market.id, comparison_set_id=set_opts[set_sel])
                        db.add(new_link)
                        db.commit()
                        st.success("Set linked!")
                        st.rerun()
            else:
                st.write("No existing sets found.")
                
    with st.popover("➕ Create New Set"):
        with st.form("create_new_set"):
            c_name = st.text_input("Set Name", placeholder="e.g. Small-Cap Animal Health Publics")
            STANDARD_SET_TYPES = ["Public Comps", "Financing Comps", "M&A Precedents", "Competitors", "Investors", "Other"]
            c_type = st.selectbox("Type", STANDARD_SET_TYPES)
            c_desc = st.text_area("Description")
            if st.form_submit_button("Create and Link Set"):
                if c_name:
                    db_stype = "Investor Comps" if c_type == "Investors" else c_type
                    new_set = ComparisonSet(name=c_name, set_type=db_stype, description=c_desc)
                    db.add(new_set)
                    db.flush() # get ID
                    
                    new_link = MarketComparisonSetLink(market_id=market.id, comparison_set_id=new_set.id)
                    db.add(new_link)
                    db.commit()
                    st.success(f"Comparison set '{c_name}' created and linked!")
                    st.rerun()
                else:
                    st.error("Name is required.")
