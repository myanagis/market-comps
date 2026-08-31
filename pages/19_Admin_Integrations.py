import streamlit as st

if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.error("🔒 Unauthorized access. Please return to the homepage to log in.")
    st.stop()

st.set_page_config(page_title="Integrations", page_icon="🔌", layout="wide")

st.title("🔌 System Integrations")
st.markdown("Manage data providers and external API connections.")

st.subheader("Financial Data Providers")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    **Yahoo Finance**  
    Used for pulling public market comparables (Market Cap, Revenue, Multiples, etc.).
    *Note: Throttled to 1 request per second to prevent rate limiting.*
    """)
with col2:
    is_enabled = st.session_state.get("yfinance_enabled", True)
    if st.toggle("Enable Yahoo Finance", value=is_enabled):
        st.session_state["yfinance_enabled"] = True
    else:
        st.session_state["yfinance_enabled"] = False
