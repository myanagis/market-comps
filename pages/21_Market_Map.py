import streamlit as st
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func

from market_comps.db.session import get_db_context
from market_comps.db.models import (
    Market, MarketSegment, MarketSegmentCompanyLink, Organization
)
from market_comps.crm.competitor_manager import (
    get_all_markets, create_market, create_market_segment, get_market_segments
)

st.set_page_config(page_title="Market Map", page_icon="🗺️", layout="wide")

st.title("🗺️ Market Map")

st.markdown("""
This page provides a top-down view of different markets and their segments, showing which companies are participating where.
""")

with get_db_context() as db:
    markets = get_all_markets(db)
    
    if not markets:
        st.info("No markets defined yet. Create one below!")
        
    # --- UI: Create Market ---
    with st.expander("➕ Create New Market", expanded=not bool(markets)):
        with st.form("new_market_form"):
            m_name = st.text_input("Market Name", placeholder="e.g. Utility Inspection")
            m_desc = st.text_area("Description", placeholder="e.g. Technologies used to inspect...")
            if st.form_submit_button("Create Market"):
                if m_name:
                    create_market(db, m_name, m_desc)
                    db.commit()
                    st.success(f"Market '{m_name}' created!")
                    st.rerun()
                else:
                    st.error("Market name is required.")

    if markets:
        st.divider()
        market_options = {m.name: m for m in markets}
        selected_market_name = st.selectbox("Select a Market", options=list(market_options.keys()))
        selected_market = market_options[selected_market_name]
        
        st.markdown(f"#### Market: {selected_market.name}")
        if selected_market.description:
            st.caption(selected_market.description)
            
        segments = get_market_segments(db, selected_market.id)
        
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
                key=f"data_editor_map_seg_{selected_market.id}"
            )
            
            col_sb1, col_sb2 = st.columns([1, 1])
            with col_sb1:
                if st.button("💾 Save Segment Edits", key=f"save_map_seg_btn_{selected_market.id}"):
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
                        s_type = st.text_input("Segment Type (Optional)", placeholder="e.g. Technology, Competitor class")
                        s_sort = st.number_input("Sort Order", value=0, step=10)
                        if st.form_submit_button("Create Segment"):
                            if s_name:
                                seg = create_market_segment(db, selected_market.id, s_name, s_desc, s_type)
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
                    s_type = st.text_input("Segment Type (Optional)", placeholder="e.g. Technology, Competitor class")
                    s_sort = st.number_input("Sort Order", value=0, step=10)
                    if st.form_submit_button("Create Segment"):
                        if s_name:
                            seg = create_market_segment(db, selected_market.id, s_name, s_desc, s_type)
                            seg.sort_order = s_sort
                            db.commit()
                            st.success(f"Segment '{s_name}' added!")
                            st.rerun()

        # -------------------------------------------------------------
        # ##### Companies
        # -------------------------------------------------------------
        st.markdown("##### Companies")

        segment_links = (
            db.query(MarketSegmentCompanyLink)
            .join(MarketSegment, MarketSegmentCompanyLink.market_segment_id == MarketSegment.id)
            .filter(MarketSegment.market_id == selected_market.id)
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
                key=f"data_editor_map_comp_{selected_market.id}"
            )
            
            col_cb1, col_cb2 = st.columns([1, 1])
            with col_cb1:
                if st.button("💾 Save Company Differentiation Edits", key=f"save_map_comp_btn_{selected_market.id}"):
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
