import streamlit as st
from sqlalchemy import or_
from market_comps.db.session import get_db_context
from market_comps.db.models import Market
from market_comps.crm.competitor_manager import create_market

st.set_page_config(page_title="Market Map Directory", page_icon="🗺️", layout="wide")

st.title("🗺️ Market Map Directory")
st.markdown("Browse and manage different markets.")

tab_dir, tab_add = st.tabs(["Markets Directory", "Add Market"])

with get_db_context() as db:
    with tab_dir:
        st.markdown("### Search")
        search_query = st.text_input("Search Markets...", placeholder="Search by name or description...")
        st.divider()
        
        q = db.query(Market)
        if search_query:
            search_filter = f"%{search_query}%"
            q = q.filter(
                or_(
                    Market.name.ilike(search_filter),
                    Market.description.ilike(search_filter)
                )
            )
            
        markets = q.order_by(Market.name).all()
        
        if not markets:
            st.info("No markets found.")
        else:
            st.markdown(f"**Showing {len(markets)} markets**")
            
            for m in markets:
                with st.container(border=True):
                    col_info, col_link = st.columns([5, 1])
                    with col_info:
                        st.subheader(m.name)
                        if m.description:
                            st.caption(m.description)
                    with col_link:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f"[👁️ View Market →](/market?id={m.id})")
                        
    with tab_add:
        with st.form("new_market_form", clear_on_submit=True):
            st.subheader("Create New Market")
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
