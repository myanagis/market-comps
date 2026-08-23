import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from market_comps.db.session import get_db_context
from market_comps.db.models import (
    Market, MarketSegment, MarketSegmentCompanyLink, Organization,
    ComparisonSet, MarketComparisonSetLink, ComparisonSetCompanyLink
)
from market_comps.crm.competitor_manager import (
    create_market_segment, get_market_segments
)

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
        
    st.title(f"🗺️ {market.name}")
    if market.description:
        st.caption(market.description)
        
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
    # ##### Companies mapped to segments
    # -------------------------------------------------------------
    st.markdown("##### Companies in Segments")

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
                "Company (Read Only)": comp_org.name,
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
                "Company (Read Only)": st.column_config.TextColumn(disabled=True),
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
                st.success("Company differentiation edits saved!")
                st.rerun()
        with col_cb2:
            with st.popover("➕ Link Company to Segment"):
                with st.form("link_company_map_form"):
                    all_orgs = db.query(Organization).order_by(Organization.name).all()
                    org_opts = {o.name: o.id for o in all_orgs}
                    seg_opts = {s.name: s.id for s in segments}
                    if org_opts and seg_opts:
                        comp_sel = st.selectbox("Company", options=list(org_opts.keys()))
                        seg_sel = st.selectbox("Segment", options=list(seg_opts.keys()))
                        diff_text = st.text_area("Differentiation", placeholder="How does this company differentiate in this segment?")
                        if st.form_submit_button("Link Company"):
                            if comp_sel and seg_sel:
                                from market_comps.crm.competitor_manager import add_company_to_segment
                                add_company_to_segment(db, org_opts[comp_sel], seg_opts[seg_sel], diff_text, False)
                                db.commit()
                                st.success("Company linked to segment!")
                                st.rerun()
                    else:
                        st.write("Ensure companies and segments exist.")
    else:
        st.info("No companies linked to segments in this market yet.")
        with st.popover("➕ Link Company to Segment"):
            with st.form("link_company_map_form_empty"):
                all_orgs = db.query(Organization).order_by(Organization.name).all()
                org_opts = {o.name: o.id for o in all_orgs}
                seg_opts = {s.name: s.id for s in segments}
                if org_opts and seg_opts:
                    comp_sel = st.selectbox("Company", options=list(org_opts.keys()))
                    seg_sel = st.selectbox("Segment", options=list(seg_opts.keys()))
                    diff_text = st.text_area("Differentiation", placeholder="How does this company differentiate in this segment?")
                    if st.form_submit_button("Link Company"):
                        if comp_sel and seg_sel:
                            from market_comps.crm.competitor_manager import add_company_to_segment
                            add_company_to_segment(db, org_opts[comp_sel], seg_opts[seg_sel], diff_text, False)
                            db.commit()
                            st.success("Company linked to segment!")
                            st.rerun()

    # -------------------------------------------------------------
    # ##### Comparison Sets
    # -------------------------------------------------------------
    st.divider()
    st.markdown("##### Comparison Sets")
    
    market_set_links = db.query(MarketComparisonSetLink).options(
        joinedload(MarketComparisonSetLink.comparison_set).joinedload(ComparisonSet.company_links).joinedload(ComparisonSetCompanyLink.company)
    ).filter_by(market_id=market.id).all()
    
    if market_set_links:
        for link in market_set_links:
            cset = link.comparison_set
            if not cset: continue
            
            with st.expander(f"📚 {cset.name} ({cset.set_type})", expanded=True):
                if cset.description:
                    st.caption(cset.description)
                
                companies_in_set = [cl.company for cl in cset.company_links if cl.included and cl.company]
                if companies_in_set:
                    comps_data = []
                    for comp in companies_in_set:
                        ticker_str = f" ({comp.exchange}: {comp.ticker})" if comp.ticker and comp.exchange else (f" ({comp.ticker})" if comp.ticker else "")
                        type_str = f"Public{ticker_str}" if comp.ownership_type and comp.ownership_type.upper() == "PUBLIC" else "Private"
                        comps_data.append({
                            "Company": comp.name,
                            "Ownership": type_str,
                            "Domain": comp.primary_domain or "",
                            "Link": f"/company?id={comp.id}"
                        })
                    df_cset = pd.DataFrame(comps_data)
                    st.dataframe(
                        df_cset, 
                        hide_index=True, 
                        use_container_width=True,
                        column_config={
                            "Link": st.column_config.LinkColumn("View Profile")
                        }
                    )
                else:
                    st.info("No companies linked to this Comparison Set.")
                
                col_c1, col_c2 = st.columns([1, 1])
                with col_c1:
                    with st.popover("➕ Add Company to Set"):
                        all_orgs = db.query(Organization).order_by(Organization.name).all()
                        org_opts = {o.name: o.id for o in all_orgs}
                        with st.form(f"add_comp_cset_{cset.id}"):
                            comp_sel = st.selectbox("Company", options=list(org_opts.keys()))
                            if st.form_submit_button("Add to Set"):
                                if comp_sel:
                                    clink = ComparisonSetCompanyLink(comparison_set_id=cset.id, company_id=org_opts[comp_sel])
                                    db.add(clink)
                                    db.commit()
                                    st.success(f"{comp_sel} added to set!")
                                    st.rerun()
                with col_c2:
                    if st.button("Unlink Set from Market", key=f"unlink_cset_{cset.id}"):
                        db.delete(link)
                        db.commit()
                        st.rerun()

    else:
        st.info("No Comparison Sets linked to this market yet.")
        
    st.markdown("###### Add or Create Comparison Set")
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
            c_type = st.selectbox("Type", ["Public Comps", "Financing Comps", "M&A Precedents", "Competitors", "Other"])
            c_desc = st.text_area("Description")
            if st.form_submit_button("Create and Link Set"):
                if c_name:
                    new_set = ComparisonSet(name=c_name, set_type=c_type, description=c_desc)
                    db.add(new_set)
                    db.flush() # get ID
                    
                    new_link = MarketComparisonSetLink(market_id=market.id, comparison_set_id=new_set.id)
                    db.add(new_link)
                    db.commit()
                    st.success(f"Comparison set '{c_name}' created and linked!")
                    st.rerun()
                else:
                    st.error("Name is required.")
