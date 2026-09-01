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
    st.markdown("##### Segments")

    if segments:
        seg_map_data = []
        for seg in segments:
            seg_map_data.append({
                "_seg_id": seg.id,
                "Segment Name": seg.name,
                "Description": seg.description or "",
                "Segment Type": seg.segment_type or "",
                "Sort Order": seg.sort_order or 0
            })
        df_seg_map = pd.DataFrame(seg_map_data)
        
        edited_seg_map_df = st.data_editor(
            df_seg_map,
            hide_index=True,
            use_container_width=True,
            column_config={
                "_seg_id": None,
                "Segment Name": st.column_config.TextColumn(disabled=False),
                "Description": st.column_config.TextColumn(disabled=False),
                "Segment Type": st.column_config.TextColumn(disabled=False),
                "Sort Order": st.column_config.NumberColumn(disabled=False, step=10)
            },
            key=f"data_editor_map_seg_{market.id}"
        )
        
        col_sb1, col_sb2 = st.columns([1, 1])
        with col_sb1:
            if st.button("💾 Save Segment Edits", key=f"save_map_seg_btn_{market.id}"):
                for _, row in edited_seg_map_df.iterrows():
                    s_obj = db.query(MarketSegment).get(int(row["_seg_id"]))
                    if s_obj:
                        s_obj.name = row["Segment Name"]
                        s_obj.description = row["Description"]
                        s_obj.segment_type = row["Segment Type"]
                        s_obj.sort_order = int(row["Sort Order"])
                db.commit()
                st.success("Segment edits saved!")
                st.rerun()
        with col_sb2:
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
    st.markdown("##### Organizations in Segments")

    segment_links = (
        db.query(MarketSegmentCompanyLink)
        .join(MarketSegment, MarketSegmentCompanyLink.market_segment_id == MarketSegment.id)
        .filter(MarketSegment.market_id == market.id)
        .all()
    )
    
    if segment_links:
        comp_map_data = []
        for link in segment_links:
            comp_org = link.company
            seg_obj = link.market_segment
            if not comp_org or not seg_obj: continue
            comp_map_data.append({
                "_company_id": link.company_id,
                "_segment_id": link.market_segment_id,
                "Organization (Read Only)": comp_org.name,
                "Type (Read Only)": comp_org.organization_type or "N/A",
                "Segment (Read Only)": seg_obj.name,
                "Differentiation": link.differentiation or ""
            })
        
        df_comp_map = pd.DataFrame(comp_map_data)
        
        edited_comp_map_df = st.data_editor(
            df_comp_map,
            hide_index=True,
            use_container_width=True,
            column_config={
                "_company_id": None,
                "_segment_id": None,
                "Organization (Read Only)": st.column_config.TextColumn(disabled=True),
                "Type (Read Only)": st.column_config.TextColumn(disabled=True),
                "Segment (Read Only)": st.column_config.TextColumn(disabled=True),
                "Differentiation": st.column_config.TextColumn(disabled=False)
            },
            key=f"data_editor_map_comp_{market.id}"
        )
        
        col_cb1, col_cb2 = st.columns([1, 1])
        with col_cb1:
            if st.button("💾 Save Company Differentiation Edits", key=f"save_map_comp_btn_{market.id}"):
                for _, row in edited_comp_map_df.iterrows():
                    link_obj = db.query(MarketSegmentCompanyLink).filter_by(
                        company_id=int(row["_company_id"]),
                        market_segment_id=int(row["_segment_id"])
                    ).first()
                    if link_obj:
                        link_obj.differentiation = row["Differentiation"]
                db.commit()
                st.success("Organization differentiation edits saved!")
                st.rerun()
        with col_cb2:
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
            with st.expander(f"📚 {cset.name}", expanded=True):
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
                    if st.session_state.get("yfinance_enabled", True):
                        if st.button("📈 Pull Market Data (Yahoo Finance)", key=f"pull_yf_{cset.id}"):
                            with st.spinner("Fetching data from Yahoo Finance... (1 request/sec)"):
                                yf_client = YahooFinanceClient()
                                
                                # Ensure metric types exist
                                metric_codes = {
                                    "market_cap": "Market Cap",
                                    "enterprise_value": "Enterprise Value", 
                                    "revenue": "Revenue",
                                    "ebitda": "EBITDA",
                                    "revenue_multiple": "EV / Revenue",
                                    "ebitda_multiple": "EV / EBITDA"
                                }
                                
                                metric_type_map = {}
                                for code, name in metric_codes.items():
                                    mt = db.query(MetricType).filter_by(code=code).first()
                                    if not mt:
                                        mt = MetricType(code=code, display_name=name, value_type="currency" if "multiple" not in code else "multiple")
                                        db.add(mt)
                                        db.commit()
                                    metric_type_map[code] = mt.id
                                
                                for comp in companies_in_set:
                                    if comp.ticker:
                                        data = yf_client.fetch_financial_metrics(comp.ticker)
                                        if data:
                                            for k, v in data.items():
                                                if v is not None and k in metric_type_map:
                                                    # Check if observation exists for TTM
                                                    obs = db.query(MetricObservation).filter_by(
                                                        company_id=comp.id, 
                                                        metric_type_id=metric_type_map[k],
                                                        reporting_basis="trailing_twelve_months"
                                                    ).first()
                                                    
                                                    if not obs:
                                                        obs = MetricObservation(
                                                            company_id=comp.id,
                                                            metric_type_id=metric_type_map[k],
                                                            reporting_basis="trailing_twelve_months",
                                                            observation_status="external_estimate",
                                                            currency_code=data.get("currency", "USD")
                                                        )
                                                        db.add(obs)
                                                    obs.value_numeric = float(v)
                                db.commit()
                                st.success("Financial data updated from Yahoo Finance!")
                                st.rerun()
                    
                    comps_data = []
                    clink_map = {cl.organization_id: cl for cl in cset.organization_links if cl.included and cl.organization}
                    
                    for comp in companies_in_set:
                        row = {
                            "_comp_id": comp.id,
                            "Organization": comp.name,
                            "Notes": clink_map[comp.id].notes or "",
                            "Link": f"/company?id={comp.id}"
                        }
                        
                        if cset.set_type == "M&A Precedents":
                            from market_comps.db.models import Transaction
                            tx = db.query(Transaction).filter_by(target_company_id=comp.id, transaction_type="ACQUISITION").order_by(Transaction.id.desc()).first()
                            if tx:
                                row["Acquirer"] = tx.acquirer_company.name if tx.acquirer_company else "Unknown"
                                if tx.transaction_value_numeric:
                                    row["Transaction Value"] = f"${tx.transaction_value_numeric:,.0f}"
                                else:
                                    row["Transaction Value"] = "Undisclosed"
                                row["Date"] = tx.announced_date.strftime("%Y-%m-%d") if tx.announced_date else ""
                            else:
                                row["Acquirer"] = ""
                                row["Transaction Value"] = ""
                                row["Date"] = ""
                                
                        elif cset.set_type == "Financing Comps":
                            from market_comps.db.models import FinancingRound, FinancingRoundFact, RoundInvestor
                            fin = db.query(FinancingRound).filter_by(company_id=comp.id).order_by(FinancingRound.id.desc()).first()
                            if fin:
                                row["Round Name"] = fin.round_name or ""
                                fact = db.query(FinancingRoundFact).filter_by(financing_round_id=fin.id, fact_type="amount_raised").first()
                                if fact and fact.value_numeric:
                                    val = fact.value_numeric
                                    if val >= 1e9:
                                        row["Amount Raised"] = f"${val/1e9:.2f}B"
                                    elif val >= 1e6:
                                        row["Amount Raised"] = f"${val/1e6:.2f}M"
                                    else:
                                        row["Amount Raised"] = f"${val:,.0f}"
                                else:
                                    row["Amount Raised"] = "Undisclosed"
                                    
                                invs = db.query(RoundInvestor).filter_by(financing_round_id=fin.id, role="lead").all()
                                if invs:
                                    row["Lead Investors"] = ", ".join([inv.investor.name for inv in invs if inv.investor])
                                else:
                                    row["Lead Investors"] = ""
                            else:
                                row["Round Name"] = ""
                                row["Amount Raised"] = ""
                                row["Lead Investors"] = ""
                                
                        else:
                            # Standard public comps display
                            row["Ticker"] = comp.ticker or ""
                            row["Domain"] = comp.primary_domain or ""
                            
                            # Get latest metrics
                            obs_list = db.query(MetricObservation).filter_by(
                                company_id=comp.id, reporting_basis="trailing_twelve_months"
                            ).all()
                            
                            last_updated = None
                            for obs in obs_list:
                                mt = db.query(MetricType).get(obs.metric_type_id)
                                if mt:
                                    if mt.value_type == "currency":
                                        # Format as Millions/Billions
                                        val = obs.value_numeric
                                        if val:
                                            if val >= 1e9:
                                                row[mt.display_name] = f"${val/1e9:.2f}B"
                                            elif val >= 1e6:
                                                row[mt.display_name] = f"${val/1e6:.2f}M"
                                            else:
                                                row[mt.display_name] = f"${val:,.0f}"
                                    elif mt.value_type == "multiple":
                                        row[mt.display_name] = f"{obs.value_numeric:.1f}x" if obs.value_numeric else ""
                                
                                # Track newest update time
                                if hasattr(obs, 'recorded_at') and obs.recorded_at:
                                    if not last_updated or obs.recorded_at > last_updated:
                                        last_updated = obs.recorded_at
                                        
                            if last_updated:
                                row["Last Updated"] = last_updated.strftime("%Y-%m-%d")
                            
                        comps_data.append(row)
                        
                    df_cset = pd.DataFrame(comps_data)
                    
                    col_config = {
                        "_comp_id": None,
                        "Organization": st.column_config.TextColumn(disabled=True),
                        "Notes": st.column_config.TextColumn(disabled=False, width="large"),
                        "Link": st.column_config.LinkColumn("View Profile", disabled=True)
                    }
                    
                    if cset.set_type == "M&A Precedents":
                        col_config.update({
                            "Acquirer": st.column_config.TextColumn(disabled=True),
                            "Transaction Value": st.column_config.TextColumn(disabled=True),
                            "Date": st.column_config.TextColumn(disabled=True)
                        })
                    elif cset.set_type == "Financing Comps":
                        col_config.update({
                            "Round Name": st.column_config.TextColumn(disabled=True),
                            "Amount Raised": st.column_config.TextColumn(disabled=True),
                            "Lead Investors": st.column_config.TextColumn(disabled=True)
                        })
                    else:
                        col_config.update({
                            "Ticker": st.column_config.TextColumn(disabled=True),
                            "Domain": st.column_config.TextColumn(disabled=True)
                        })
                        
                    edited_df_cset = st.data_editor(
                        df_cset, 
                        hide_index=True, 
                        use_container_width=True,
                        column_config=col_config,
                        key=f"data_editor_cset_{cset.id}"
                    )
                    
                    if st.button("💾 Save Notes", key=f"save_notes_cset_{cset.id}"):
                        for _, row in edited_df_cset.iterrows():
                            c_id = int(row["_comp_id"])
                            clink = clink_map.get(c_id)
                            if clink:
                                clink.notes = row["Notes"]
                        db.commit()
                        st.success("Notes saved!")
                        st.rerun()
                        
                else:
                    st.info("No organizations linked to this Comparison Set.")
                
                col_c1, col_c2 = st.columns([1, 1])
                with col_c1:
                    with st.popover("➕ Add Organization to Set"):
                        all_orgs = db.query(Organization).order_by(Organization.name).all()
                        org_opts = {f"{o.name} ({o.organization_type or 'Company'})": o.id for o in all_orgs}
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
