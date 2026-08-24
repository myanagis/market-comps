import streamlit as st
import pandas as pd
from sqlalchemy.orm import joinedload
from market_comps.db.session import get_db_context
from market_comps.db.models import (
    Market, MarketSegment, MarketSegmentCompanyLink, Organization,
    ComparisonSet, MarketComparisonSetLink, ComparisonSetOrganizationLink
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
    if market.sectors:
        st.write(f"**Sectors:** {', '.join(market.sectors)}")
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
    # ##### Comparison Groups
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

    STANDARD_SET_TYPES = ["Public Comps", "Financing Comps", "M&A Precedents", "Competitors", "Investors", "Other"]
    all_types = list(dict.fromkeys(STANDARD_SET_TYPES + list(sets_by_type.keys())))

    for stype in all_types:
        csets = sets_by_type.get(stype, [])
        st.markdown(f"##### {stype}")
        
        # Flatten companies
        companies_in_set = []
        for cset in csets:
            for cl in cset.organization_links:
                if cl.included and cl.organization and cl.organization not in companies_in_set:
                    companies_in_set.append(cl.organization)
                    
        if companies_in_set:
            comps_data = []
            for comp in companies_in_set:
                ticker_str = f" ({comp.exchange}: {comp.ticker})" if comp.ticker and comp.exchange else (f" ({comp.ticker})" if comp.ticker else "")
                type_str = f"Public{ticker_str}" if comp.ownership_type and comp.ownership_type.upper() == "PUBLIC" else "Private"
                comps_data.append({
                    "Organization": comp.name,
                    "Type": comp.organization_type or "COMPANY",
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
            st.info(f"No organizations added to {stype} yet.")
            
        with st.popover(f"➕ Add to {stype}"):
            all_orgs = db.query(Organization).order_by(Organization.name).all()
            org_opts = {f"{o.name} ({o.organization_type or 'Company'})": o.id for o in all_orgs}
            with st.form(f"add_comp_{stype.replace(' ', '_')}"):
                comp_sel = st.selectbox("Organization", options=list(org_opts.keys()))
                if st.form_submit_button("Add Organization"):
                    if comp_sel:
                        # Find or create a comparison set of this type
                        target_cset = csets[0] if csets else None
                        if not target_cset:
                            # Create new
                            db_stype = "Investor Comps" if stype == "Investors" else stype
                            target_cset = ComparisonSet(name=f"{market.name} - {stype}", set_type=db_stype)
                            db.add(target_cset)
                            db.flush()
                            new_link = MarketComparisonSetLink(market_id=market.id, comparison_set_id=target_cset.id)
                            db.add(new_link)
                            db.flush()
                            
                        # Add organization link
                        clink = ComparisonSetOrganizationLink(comparison_set_id=target_cset.id, organization_id=org_opts[comp_sel])
                        db.add(clink)
                        db.commit()
                        st.success(f"Added to {stype}!")
                        st.rerun()
                        
        st.write("") # spacing
