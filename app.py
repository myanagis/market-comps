# app.py
"""
Market Comps — Entry point with explicit page navigation.

Run with:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Market Comps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("pages/1_Public_Comps.py",      title="Competition & Public Comps", icon="📊"),
    st.Page("pages/2_PDF_Parser.py",        title="PDF Parser",                 icon="📄"),
    st.Page("pages/3_LLM_Cross_Checker.py", title="Chorus of LLMs",             icon="🎼"),
])

pg.run()
