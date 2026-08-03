import streamlit as st
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func

from market_comps.db.session import get_db_context
from market_comps.db.models import (
    Market, MarketSegment, CompanyMarketSegment, Organization
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
                    st.success(f"Market '{m_name}' created!")
                    st.rerun()
                else:
                    st.error("Market name is required.")

    if markets:
        st.divider()
        market_options = {m.name: m for m in markets}
        selected_market_name = st.selectbox("Select a Market", options=list(market_options.keys()))
        selected_market = market_options[selected_market_name]
        
        st.subheader(f"{selected_market.name}")
        if selected_market.description:
            st.caption(selected_market.description)
            
        segments = get_market_segments(db, selected_market.id)
        
        # --- UI: Create Segment ---
        with st.expander("➕ Add Segment to Market"):
            with st.form("new_segment_form"):
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
        
        st.markdown("### Market Segments")
        if not segments:
            st.info("No segments in this market yet.")
        else:
            # Group companies by segment using a single query
            # We want to mimic the SQL query provided by the user
            segment_data = (
                db.query(MarketSegment.name, Organization.name, CompanyMarketSegment.differentiation)
                .join(CompanyMarketSegment, CompanyMarketSegment.market_segment_id == MarketSegment.id)
                .join(Organization, Organization.id == CompanyMarketSegment.company_id)
                .filter(MarketSegment.market_id == selected_market.id)
                .order_by(MarketSegment.sort_order, Organization.name)
                .all()
            )
            
            if segment_data:
                df = pd.DataFrame(segment_data, columns=["Segment", "Company", "Differentiation"])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No companies linked to segments in this market yet.")
                
            st.write("#### Segment Definitions")
            for seg in segments:
                st.markdown(f"**{seg.name}**")
                if seg.description:
                    st.caption(seg.description)
