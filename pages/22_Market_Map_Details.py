import streamlit as st
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
        
    col_t1, col_t2 = st.columns([0.8, 0.2])
    with col_t1:
        st.title(f"🗺️ {market.name}")
        if market.sectors:
            st.write(f"**Sectors:** {', '.join(market.sectors)}")
        if market.description:
            st.caption(market.description)
            
    with col_t2:
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
        
    segments = get_market_segments(db, market.id)
    
    # -------------------------------------------------------------
    # ##### Segments
    # -------------------------------------------------------------
    st.header("Market Players")
    st.markdown("##### Segmentation")

    if segments:
        h1, h2, h3, h4 = st.columns([2, 3, 2, 0.5])
        h1.markdown("**Segment Name**")
        h2.markdown("**Description**")
        h3.markdown("**Segment Type**")
        
        st.markdown("<hr style='margin: 0; padding: 0; margin-bottom: 10px;'>", unsafe_allow_html=True)
        
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
                        if st.form_submit_button("Save"):
                            s_obj = db.query(MarketSegment).get(seg.id)
                            if s_obj:
                                s_obj.name = s_name
                                s_obj.description = s_desc
                                s_obj.segment_type = s_type
                                s_obj.sort_order = s_sort
                                db.commit()
                                st.rerun()
                                
        st.markdown("<br>", unsafe_allow_html=True)
        with st.popover("➕ Add Segment"):
            with st.form("new_segment_form_map"):
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
    else:
        st.info("No segments in this market yet.")
        with st.popover("➕ Add Segment"):
            with st.form("new_segment_form_map_empty"):
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

    # -------------------------------------------------------------
    # ##### Organizations mapped to segments
    # -------------------------------------------------------------
    st.markdown("##### Companies")

    segment_links = (
        db.query(MarketSegmentCompanyLink)
        .join(MarketSegment, MarketSegmentCompanyLink.market_segment_id == MarketSegment.id)
        .filter(MarketSegment.market_id == market.id)
        .all()
    )
    
    if segment_links:
        h1, h2, h3, h4, h5, h6 = st.columns([2, 1.5, 3, 1.5, 1.5, 0.5])
        h1.markdown("**Organization**")
        h2.markdown("**Segment**")
        h3.markdown("**Differentiation**")
        h4.markdown("**Total / Last Raised**")
        h5.markdown("**Valuation**")
        
        st.markdown("<hr style='margin: 0; padding: 0; margin-bottom: 10px;'>", unsafe_allow_html=True)
        
        from market_comps.db.models import FinancingRound, FinancingRoundFact, MetricObservation, MetricType
        
        for link in segment_links:
            comp_org = link.company
            seg_obj = link.market_segment
            if not comp_org or not seg_obj: continue
            
            raised_str = "-"
            val_str = "-"
            
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
                    
            if val_str == "-":
                mc_type = db.query(MetricType).filter_by(code="market_cap").first()
                if mc_type:
                    obs = db.query(MetricObservation).filter_by(company_id=comp_org.id, metric_type_id=mc_type.id).order_by(MetricObservation.recorded_at.desc()).first()
                    if obs and obs.value_numeric:
                        val = obs.value_numeric
                        date_str = f" ({obs.recorded_at.strftime('%Y-%m')})" if obs.recorded_at else ""
                        if val >= 1e9: val_str = f"${val/1e9:.2f}B{date_str}"
                        elif val >= 1e6: val_str = f"${val/1e6:.2f}M{date_str}"
                        else: val_str = f"${val:,.0f}{date_str}"
            
            c1, c2, c3, c4, c5, c6 = st.columns([2, 1.5, 3, 1.5, 1.5, 0.5])
            c1.markdown(f"[{comp_org.name}](/company?id={comp_org.id})")
            c2.write(seg_obj.name)
            c3.write(link.differentiation or "")
            c4.write(raised_str)
            c5.write(val_str)
            
            with c6:
                with st.popover("✏️"):
                    with st.form(f"edit_comp_{link.company_id}_{link.market_segment_id}"):
                        seg_opts = {s.name: s.id for s in segments}
                        seg_idx = list(seg_opts.values()).index(seg_obj.id) if seg_obj.id in seg_opts.values() else 0
                        new_seg_name = st.selectbox("Segment", options=list(seg_opts.keys()), index=seg_idx)
                        new_diff = st.text_area("Differentiation", value=link.differentiation or "")
                        if st.form_submit_button("Save"):
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
                                
        st.markdown("<br>", unsafe_allow_html=True)
        with st.popover("➕ Link Org to Segment"):
            with st.form("link_company_map_form"):
                    all_orgs = db.query(Organization).order_by(Organization.name).all()
                    org_opts = {f"{o.name} ({o.organization_type or 'Company'})": o.id for o in all_orgs}
                    seg_opts = {s.name: s.id for s in segments}
                    if org_opts and seg_opts:
                        comp_sel = st.selectbox("Organization", options=list(org_opts.keys()))
                        seg_sel = st.selectbox("Segment", options=list(seg_opts.keys()))
                        diff_text = st.text_area("Differentiation", placeholder="How does this organization differentiate in this segment?")
                        if st.form_submit_button("Link Organization"):
                            if comp_sel and seg_sel:
                                from market_comps.crm.competitor_manager import add_company_to_segment
                                add_company_to_segment(db, org_opts[comp_sel], seg_opts[seg_sel], diff_text, False)
                                db.commit()
                                st.success("Organization linked to segment!")
                                st.rerun()
                    else:
                        st.write("Ensure organizations and segments exist.")
    else:
        st.info("No organizations linked to segments in this market yet.")
        with st.popover("➕ Link Org to Segment"):
            with st.form("link_company_map_form_empty"):
                all_orgs = db.query(Organization).order_by(Organization.name).all()
                org_opts = {f"{o.name} ({o.organization_type or 'Company'})": o.id for o in all_orgs}
                seg_opts = {s.name: s.id for s in segments}
                if org_opts and seg_opts:
                    comp_sel = st.selectbox("Organization", options=list(org_opts.keys()))
                    seg_sel = st.selectbox("Segment", options=list(seg_opts.keys()))
                    diff_text = st.text_area("Differentiation", placeholder="How does this organization differentiate in this segment?")
                    if st.form_submit_button("Link Organization"):
                        if comp_sel and seg_sel:
                            from market_comps.crm.competitor_manager import add_company_to_segment
                            add_company_to_segment(db, org_opts[comp_sel], seg_opts[seg_sel], diff_text, False)
                            db.commit()
                            st.success("Organization linked to segment!")
                            st.rerun()

    # -------------------------------------------------------------
    # ##### Comparison Sets
    # -------------------------------------------------------------
    st.divider()

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

    all_types = list(sets_by_type.keys())
    if not all_types:
        st.info("No comparison sets exist for this market yet.")

    for stype in all_types:
        csets = sets_by_type.get(stype, [])
        st.markdown(f"##### {stype}")
        
        for cset in csets:
            st.markdown(f"###### 📚 {cset.name}")
            col_c1, col_c2 = st.columns([0.8, 0.2])
            with col_c1:
                if cset.description:
                    st.caption(cset.description)
            with col_c2:
                with st.popover("✏️ Edit Section", use_container_width=True):
                    with st.form(f"edit_cset_form_{cset.id}"):
                        new_name = st.text_input("Name", value=cset.name)
                        new_desc = st.text_area("Description", value=cset.description or "")
                        if st.form_submit_button("Save"):
                            cset.name = new_name
                            cset.description = new_desc
                            db.commit()
                            st.rerun()
            
            companies_in_set = [cl.organization for cl in cset.organization_links if cl.included and cl.organization]
            if companies_in_set:
                st.markdown("<hr style='margin: 0; padding: 0; margin-bottom: 10px;'>", unsafe_allow_html=True)
                clink_map = {cl.organization_id: cl for cl in cset.organization_links if cl.included and cl.organization}
                
                if cset.set_type == "M&A Precedents":
                    h1, h2, h3, h4, h5, h6 = st.columns([2, 2, 1.5, 1.5, 3, 0.5])
                    h1.markdown("**Target**")
                    h2.markdown("**Acquirer**")
                    h3.markdown("**Transaction Value**")
                    h4.markdown("**Date**")
                    h5.markdown("**Notes**")
                    
                    st.markdown("<hr style='margin: 0; padding: 0; margin-bottom: 10px;'>", unsafe_allow_html=True)
                    
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
                            
                        date_str = tx.announced_date.strftime("%Y-%m-%d") if tx and tx.announced_date else ""
                        
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
                                    new_notes = st.text_area("Notes", value=clink_map[comp.id].notes or "")
                                    cur_date = tx.announced_date if tx and tx.announced_date else datetime.date.today()
                                    new_date = st.date_input("Transaction Date", value=cur_date)
                                    new_val = st.number_input("Transaction Value ($)", value=float(tx.transaction_value_numeric) if tx and tx.transaction_value_numeric else 0.0, step=1000000.0)
                                    if st.form_submit_button("Save"):
                                        clink_map[comp.id].notes = new_notes
                                        if tx:
                                            tx.announced_date = new_date
                                            if new_val > 0: tx.transaction_value_numeric = new_val
                                        else:
                                            new_tx = Transaction(target_company_id=comp.id, transaction_type="ACQUISITION", announced_date=new_date, transaction_value_numeric=new_val if new_val > 0 else None)
                                            db.add(new_tx)
                                        db.commit()
                                        st.rerun()

                elif cset.set_type == "Financing Comps":
                    h1, h2, h3, h4, h5, h6 = st.columns([2, 1.5, 1.5, 2, 3, 0.5])
                    h1.markdown("**Organization**")
                    h2.markdown("**Round Name**")
                    h3.markdown("**Amount Raised**")
                    h4.markdown("**Lead Investors**")
                    h5.markdown("**Notes**")
                    
                    st.markdown("<hr style='margin: 0; padding: 0; margin-bottom: 10px;'>", unsafe_allow_html=True)
                    
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
                                    if st.form_submit_button("Save"):
                                        clink_map[comp.id].notes = new_notes
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
                    
                    st.markdown("<hr style='margin: 0; padding: 0; margin-bottom: 10px;'>", unsafe_allow_html=True)
                    
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
                                    if st.form_submit_button("Save"):
                                        clink_map[comp.id].notes = new_notes
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

    st.divider()
    st.markdown("###### Set Management")
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
